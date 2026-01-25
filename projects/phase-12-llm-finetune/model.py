"""
模型定义
========

包含 LoRA 层和 TinyLLM 模型的实现。

学习要点：
    1. LoRA 通过低秩分解大幅减少可训练参数
    2. 只训练 LoRA 参数，原始模型权重保持冻结
    3. 推理时可以将 LoRA 权重合并到原始权重中
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, LoRAConfig


# ==================== LoRA 层 ====================


class LoRALinear(nn.Module):
    """带有 LoRA 的线性层

    LoRA 核心思想：
        原始权重 W 固定，只训练低秩增量 ΔW = B × A
        输出 = W × x + ΔW × x = W × x + B × (A × x)

    参数量对比（以 d=4096, r=8 为例）：
        原始: d × d = 16,777,216 (16M)
        LoRA: 2 × d × r = 65,536 (65K)
        比例: 0.4%
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始线性层（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA 参数（可训练）
        # A: 降维矩阵，初始化为正态分布
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        # B: 升维矩阵，初始化为零（初始时 ΔW = 0）
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_output = self.linear(x)

        # LoRA 输出: x @ A @ B * scaling
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return original_output + lora_output

    def merge_weights(self):
        """将 LoRA 权重合并到原始权重中"""
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B * self.scaling).T
            self.linear.weight.add_(delta_w)

    def unmerge_weights(self):
        """从原始权重中移除 LoRA 权重"""
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B * self.scaling).T
            self.linear.weight.sub_(delta_w)


# ==================== 注意力层 ====================


class MultiHeadAttention(nn.Module):
    """多头自注意力层"""

    def __init__(
        self,
        config: ModelConfig,
        use_lora: bool = False,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        assert self.head_dim * config.num_heads == config.hidden_size

        # Q, K, V 投影
        if use_lora and lora_config:
            self.q_proj = LoRALinear(
                config.hidden_size,
                config.hidden_size,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                bias=False,
            )
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = LoRALinear(
                config.hidden_size,
                config.hidden_size,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 重塑为多头格式: (batch, heads, seq, head_dim)
        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # KV Cache
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        new_key_value = (key, value) if use_cache else None

        # 计算注意力分数
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        # 因果掩码（下三角）
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(
                    seq_len, key.size(2), device=hidden_states.device, dtype=torch.bool
                ),
                diagonal=key.size(2) - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, new_key_value


# ==================== FFN 层 ====================


class FeedForward(nn.Module):
    """前馈神经网络层（SwiGLU 激活）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate) * up
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ==================== Transformer 块 ====================


class TransformerBlock(nn.Module):
    """Transformer 解码器块"""

    def __init__(
        self,
        config: ModelConfig,
        use_lora: bool = False,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(config, use_lora, lora_config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.RMSNorm(config.hidden_size)
        self.ffn_norm = nn.RMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-Norm + 注意力
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, new_key_value = self.attention(
            hidden_states, attention_mask, use_cache, past_key_value
        )
        hidden_states = residual + hidden_states

        # Pre-Norm + FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_key_value


# ==================== TinyLLM 模型 ====================


class TinyLLM(nn.Module):
    """小型语言模型（GPT-2 风格）

    架构特点：
        - Decoder-Only
        - Pre-Norm (RMSNorm)
        - SwiGLU 激活函数
        - 可选 LoRA 微调
    """

    def __init__(
        self,
        config: ModelConfig,
        use_lora: bool = False,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.use_lora = use_lora

        # Token 嵌入
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # Transformer 层
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, use_lora, lora_config)
                for _ in range(config.num_layers)
            ]
        )

        # 最终层归一化
        self.norm = nn.RMSNorm(config.hidden_size)

        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享
        self.lm_head.weight = self.embed_tokens.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        # 获取嵌入
        hidden_states = self.embed_tokens(input_ids)

        # 通过 Transformer 层
        new_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = layer(
                hidden_states, attention_mask, use_cache, past_kv
            )
            if use_cache:
                new_key_values.append(new_kv)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 计算 logits
        logits = self.lm_head(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            # 移位预测
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_key_values if use_cache else None,
        }

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取所有 LoRA 参数"""
        lora_params = []
        for name, param in self.named_parameters():
            if "lora_" in name:
                lora_params.append(param)
        return lora_params

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """获取可训练参数数量"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable, total = self.get_trainable_parameters()
        print(f"可训练参数: {trainable:,} / {total:,} = {100 * trainable / total:.2f}%")


def create_model(
    config: ModelConfig, use_lora: bool = True, lora_config: Optional[LoRAConfig] = None
) -> TinyLLM:
    """创建模型

    Args:
        config: 模型配置
        use_lora: 是否使用 LoRA
        lora_config: LoRA 配置

    Returns:
        TinyLLM 模型实例
    """
    model = TinyLLM(config, use_lora, lora_config)

    if use_lora:
        # 冻结非 LoRA 参数
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    return model


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    model = create_model(config.model, use_lora=True, lora_config=config.lora)
    model.print_trainable_parameters()

    # 测试前向传播
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Logits shape: {output['logits'].shape}")
