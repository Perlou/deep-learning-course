"""
ClearMind 模型定义 — HuggingFace PreTrainedModel 架构
=====================================================

单文件包含所有模型组件 (HF 自定义模型惯例):
  - ClearMindRMSNorm:        均方根层归一化
  - ClearMindRotaryEmbedding: 旋转位置编码 (RoPE)
  - ClearMindAttention:       GQA 注意力 + KV Cache
  - ClearMindMLP:             SwiGLU 前馈网络
  - ClearMindDecoderLayer:    Transformer 解码器层
  - ClearMindModel:           基础模型 (embedding + layers + norm)
  - ClearMindForCausalLM:     因果语言模型 (+ LM head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_clearmind import ClearMindConfig


# ============================================================
# ClearMindRMSNorm
# ============================================================


class ClearMindRMSNorm(nn.Module):
    """均方根层归一化

    RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
    相比 LayerNorm 省去均值和 bias，计算更高效。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * self.weight


# ============================================================
# ClearMindRotaryEmbedding
# ============================================================


class ClearMindRotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)

    预计算 cos/sin 频率矩阵并注册为 buffer，
    forward 时根据 position_ids 应用旋转。
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        # 预计算频率: θ_k = 1 / base^(2k/d)
        k = torch.arange(0, head_dim, 2, dtype=torch.float32)
        freqs = 1.0 / (base ** (k / head_dim))
        # 计算 positions × freqs → angles
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # [max_pos, head_dim//2]
        angles = angles.repeat(1, 2)  # [max_pos, head_dim]

        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """应用 RoPE 旋转

        Args:
            x: [batch, n_heads, seq_len, head_dim]
            position_ids: [batch, seq_len]

        Returns:
            旋转后的张量，形状不变
        """
        # 根据 position_ids 索引 cos/sin
        # cos_cached: [max_pos, head_dim]
        # position_ids: [batch, seq_len] → cos: [batch, seq_len, head_dim]
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)

        # 旋转: x * cos + rotate_half(x) * sin
        x_rotated = _rotate_half(x)
        return x * cos + x_rotated * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2, x3, x4] → [-x3, -x4, x1, x2]"""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ============================================================
# ClearMindAttention
# ============================================================


class ClearMindAttention(nn.Module):
    """Multi-Head Attention with GQA, RoPE, KV Cache & Flash Attention

    使用 HF 命名: q_proj, k_proj, v_proj, o_proj
    """

    def __init__(self, config: ClearMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups

        d = config.hidden_size
        self.q_proj = nn.Linear(d, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, d, bias=False)

        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rotary_emb = ClearMindRotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = hidden_states.shape

        # 线性投影 + 分头
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 构建 position_ids (如果未提供)
        if position_ids is None:
            offset = past_key_value[0].shape[2] if past_key_value is not None and past_key_value[0] is not None else 0
            position_ids = torch.arange(offset, offset + seq_len, device=hidden_states.device).unsqueeze(0).expand(batch, -1)

        # 应用 RoPE
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        # KV Cache: 拼接历史 KV
        if past_key_value is not None and past_key_value[0] is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # GQA 扩展: 复制 KV heads 到与 Q heads 相同数量
        if self.num_key_value_groups > 1:
            k_expanded = self._expand_kv(k)
            v_expanded = self._expand_kv(v)
        else:
            k_expanded = k
            v_expanded = v

        # Flash Attention
        kv_len = k_expanded.shape[2]
        if seq_len == kv_len:
            # Prefill / 训练: 使用 causal mask
            if attention_mask is not None:
                output = F.scaled_dot_product_attention(
                    q, k_expanded, v_expanded,
                    attn_mask=attention_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                output = F.scaled_dot_product_attention(
                    q, k_expanded, v_expanded,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True,
                )
        else:
            # Decode: Q 只有 1 个 token
            output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

        # 合并头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(output)
        output = self.resid_dropout(output)

        return output, new_kv_cache

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """将 KV heads 扩展到与 Q heads 相同数量 (GQA)"""
        batch, n_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, self.num_key_value_groups, -1, -1)
        return x.reshape(batch, self.num_attention_heads, seq_len, head_dim)


# ============================================================
# ClearMindMLP
# ============================================================


class ClearMindMLP(nn.Module):
    """SwiGLU 前馈网络

    SwiGLU(x) = SiLU(gate_proj(x)) * up_proj(x) → down_proj()
    """

    def __init__(self, config: ClearMindConfig):
        super().__init__()
        d = config.hidden_size
        ff = config.intermediate_size
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ============================================================
# ClearMindDecoderLayer
# ============================================================


class ClearMindDecoderLayer(nn.Module):
    """单个 Transformer 解码器层 (Pre-Norm 架构)

    x → RMSNorm → Attention → + residual → RMSNorm → MLP → + residual
    """

    def __init__(self, config: ClearMindConfig):
        super().__init__()
        self.input_layernorm = ClearMindRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ClearMindAttention(config)
        self.post_attention_layernorm = ClearMindRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ClearMindMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Attention block + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP block + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


# ============================================================
# ClearMindPreTrainedModel
# ============================================================


class ClearMindPreTrainedModel(PreTrainedModel):
    """ClearMind 模型的基类，提供权重初始化和通用接口"""

    config_class = ClearMindConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ClearMindDecoderLayer"]

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


# ============================================================
# ClearMindModel
# ============================================================


class ClearMindModel(ClearMindPreTrainedModel):
    """ClearMind 基础模型: embedding + N × DecoderLayer + final norm

    返回 BaseModelOutputWithPast，不含 LM head。
    """

    def __init__(self, config: ClearMindConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList(
            [ClearMindDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = ClearMindRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool | None = None,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        # 构建 4D causal attention mask (如果提供了 padding mask)
        causal_mask = None
        if attention_mask is not None:
            batch, seq_len = input_ids.shape
            # [batch, 1, 1, seq_len] — padding 位置为 -inf
            causal_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(hidden_states.dtype).min

        new_kv_caches = [] if use_cache else None

        # 支持 DynamicCache (HF generate 使用) 和 list of tuples
        if isinstance(past_key_values, DynamicCache):
            if past_key_values.get_seq_length() > 0:
                past_kv_list = [
                    (past_key_values.layers[i].keys, past_key_values.layers[i].values)
                    if i < len(past_key_values.layers) and past_key_values.layers[i].keys is not None
                    else None
                    for i in range(len(self.layers))
                ]
            else:
                past_kv_list = [None] * len(self.layers)
        elif past_key_values is not None:
            past_kv_list = past_key_values
        else:
            past_kv_list = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_list[i]
            hidden_states, new_cache = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(new_cache)

        hidden_states = self.norm(hidden_states)

        # 将 list of (K, V) tuples 转换为 DynamicCache 以兼容 HF generate
        past_kv_out = None
        if use_cache and new_kv_caches:
            cache = DynamicCache()
            for layer_idx, (k_val, v_val) in enumerate(new_kv_caches):
                cache.update(k_val, v_val, layer_idx=layer_idx)
            past_kv_out = cache

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_kv_out,
        )


# ============================================================
# ClearMindForCausalLM
# ============================================================


class ClearMindForCausalLM(ClearMindPreTrainedModel, GenerationMixin):
    """ClearMind 因果语言模型

    ClearMindModel + LM head，支持 HF generate() 接口。
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ClearMindConfig):
        super().__init__(config)
        self.model = ClearMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: list | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """为 model.generate() 准备输入

        有 KV Cache 时只保留最后一个 token 作为输入。
        """
        # 判断 past_key_values 是否有实际缓存内容
        has_cache = False
        if past_key_values is not None and len(past_key_values) > 0:
            if isinstance(past_key_values, DynamicCache):
                has_cache = past_key_values.get_seq_length() > 0
            else:
                has_cache = past_key_values[0] is not None

        if has_cache:
            # Decode 阶段: 只传入最后一个 token
            input_ids = input_ids[:, -1:]

        # 构建 position_ids
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if has_cache:
                position_ids = position_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
        }

    def count_parameters(self) -> dict:
        """统计模型的实际参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
            "trainable_millions": trainable / 1e6,
        }
