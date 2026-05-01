"""
GPT — 完整的 Decoder-only Transformer 语言模型
==============================================

这是 ClearMind 的核心模型文件，将所有组件组装成完整的 GPT 模型。

模型结构:
  Token IDs → Embedding → N × TransformerBlock → RMSNorm → LM Head → Logits

组件列表:
  - Token Embedding:  将整数 token ID 映射为 d_model 维向量
  - TransformerBlock: 包含 Attention + FFN + RMSNorm + Residual (×N layers)
  - Final RMSNorm:    最后一层归一化
  - LM Head:          将 d_model 维映射回 vocab_size 维的 logits

与 GPT-4/Gemini 的区别:
  本模型 = GPT-4/Gemini 的完全缩小版
  - 架构 100% 一致 (Decoder-only + RoPE + RMSNorm + SwiGLU + GQA)
  - 区别仅在于参数量 (26M vs 1.8T) 和训练数据量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .normalization import RMSNorm
from .transformer import TransformerBlock


class GPT(nn.Module):
    """GPT 语言模型

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ========== Token Embedding ==========
        # 将 token ID (整数) 映射为 d_model 维连续向量
        # 这是模型的 "单词表", 每个 token 对应一行可学习参数
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ========== Dropout ==========
        self.embedding_dropout = nn.Dropout(config.dropout)

        # ========== N × Transformer Block ==========
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # ========== Final RMSNorm ==========
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # ========== LM Head ==========
        # 将 d_model 维映射回 vocab_size 维, 输出每个 token 的 logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ========== 权重共享 (Weight Tying) ==========
        # 让 Embedding 和 LM Head 共享同一组权重
        # 理由: Embedding 将 token → 向量, LM Head 做反向映射
        # 共享权重能减少参数量且提升效果 (参考 GPT-2 论文)
        self.token_embedding.weight = self.lm_head.weight

        # ========== RoPE 频率（顶层共享 buffer，所有 attention 共用）==========
        # 旧实现：每个 Attention 各自 register_buffer cos/sin → n_layers 份重复
        # 新实现：顶层 register 一次，forward 时透传给每个 Block → Attention
        # state_dict 中也只有一份 cos/sin（persistent=False 时不入 state_dict）
        from .rope import precompute_rope_frequencies

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim,
            config.max_seq_len,
            base=config.rope_theta,            # 默认 1e6（minimind / Qwen3 对齐）
            rope_scaling=config.rope_scaling,  # YaRN 长上下文（None = 关闭）
        )
        # persistent=False：不计入 state_dict（forward 时再生成更省磁盘）
        # 但保持 buffer 行为以便 .to(device) 时自动迁移
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # ========== Causal Mask ==========
        # 不再预分配 (max_seq_len, max_seq_len) 的全局 buffer：
        #   - SDPA `is_causal=True` 路径根本不消费 mask
        #   - SFT/DPO 走的"padding mask + causal mask"路径只需要 (seq_len, seq_len)
        #     大小的临时 mask，按 forward 时的实际 seq_len 动态构造更省（不污染
        #     state_dict、不占用 max_seq_len² 显存）
        # 该改动相比原实现：
        #   small (max_seq_len=1024)  省 4 MB / large (2048+)  省 16 MB+
        #   state_dict 中不再含 causal_mask，发布到 HF 时模型卡更干净

        # ========== Activation Checkpointing（Phase 4） ==========
        # 用 torch.utils.checkpoint 把每个 TransformerBlock 包成"前向时不存激活，
        # 反向时重算"模式：显存峰值降 30-50%（深层模型尤其明显），代价是
        # ~25% 训练吞吐下降。对 plus（24 层）+ 长序列特别有用。
        # 通过 yaml.use_gradient_checkpointing=true 或 model.gradient_checkpointing_enable() 启用。
        self.gradient_checkpointing = False

        # ========== 初始化权重 ==========
        self.apply(self._init_weights)
        # 残差路径输出 proj 的 1/√(2L) 缩放（GPT-2/Llama 标准做法）
        # 详见 _scale_residual_proj 注释
        self._scale_residual_proj()

    def gradient_checkpointing_enable(self) -> None:
        """打开 activation checkpointing（与 HF Trainer 兼容的 API）"""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        """Xavier/He 风格的权重初始化

        不同层使用不同的初始化策略:
        - Linear 层: 正态分布 N(0, 0.02)
        - Embedding: 正态分布 N(0, 0.02)
        - RMSNorm:   已经初始化为 1 (不需要额外处理)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_proj(self):
        """对每个 transformer block 的"残差路径输出 proj"权重再除以 √(2·n_layers)

        理论依据（Karpathy ``minGPT``、GPT-2/3 paper、Llama 实现都做了这一步）：
          - Pre-Norm transformer 中，每层有 2 个残差连接（attn + ffn）
          - 残差累积时方差按 √n_layers 增长 → 深层激活越来越大 → 训练不稳定
          - 把每个残差路径"出口"层的初始权重缩放 1/√(2·n_layers)
          - 让残差累积后的方差与初始保持一致

        具体哪些层是"残差出口"：
          - Attention 的 output projection (``w_o``)
          - FFN 的 down projection (``w_down``)

        这是 minimind 没有做的改进，是 ClearMind 反超 minimind 的关键差异化点之一。
        """
        scale = 1.0 / (2 * self.config.n_layers) ** 0.5
        for layer in self.layers:
            # Attention output proj
            if hasattr(layer, "attention") and hasattr(layer.attention, "w_o"):
                with torch.no_grad():
                    layer.attention.w_o.weight.mul_(scale)
            # FFN down proj
            if hasattr(layer, "feedforward") and hasattr(layer.feedforward, "w_down"):
                with torch.no_grad():
                    layer.feedforward.w_down.weight.mul_(scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list | None]:
        """前向传播

        Args:
            input_ids:      token ID 序列 [batch, seq_len]
            targets:        目标 token ID (用于计算 loss) [batch, seq_len], 可选
            kv_caches:      各层的 KV cache 列表, 长度 = n_layers
            use_cache:      是否返回更新后的 KV caches
            attention_mask: padding mask [batch, seq_len], 1=有效, 0=padding

        Returns:
            logits:         模型输出的 logits [batch, seq_len, vocab_size]
            loss:           交叉熵损失 (仅当提供 targets 时)
            new_kv_caches:  更新后的各层 KV cache 列表 (仅 use_cache=True 时)
        """
        batch, seq_len = input_ids.shape

        # Token Embedding
        x = self.token_embedding(input_ids)  # [batch, seq, d_model]
        x = self.embedding_dropout(x)

        # 构建 attention mask: 将 2D padding mask 与 causal mask 合并
        if attention_mask is not None:
            # [batch, seq_len] (1=valid, 0=pad) → [batch, 1, 1, seq_len] additive mask
            # 关键陷阱：旧实现 `(1.0 - mask) * -inf` 在 valid 位置产生 0*inf=NaN，
            # 必须用 masked_fill / where 安全地把 pad 填 -inf、valid 填 0
            pad_mask = torch.zeros(
                attention_mask.shape[0], 1, 1, attention_mask.shape[1],
                dtype=torch.float32,
                device=attention_mask.device,
            )
            pad_mask = pad_mask.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float("-inf"),
            )
            # 动态构造 (seq_len, seq_len) 的 causal mask（按需大小，不预分配 max_seq_len²）
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"),
                           dtype=torch.float32, device=input_ids.device),
                diagonal=1,
            )
            # 合并: [seq_len, seq_len] + [batch, 1, 1, seq_len]
            combined_mask = causal_mask.unsqueeze(0) + pad_mask
        else:
            # 无 padding：让 attention 走 SDPA `is_causal=True` 内部路径（无需显式 mask）
            combined_mask = None

        # N × Transformer Block (带 KV Cache 透传 + RoPE buffer 共享)
        new_kv_caches = [] if use_cache else None
        # activation checkpointing 仅在训练时 + 不用 KV cache 时生效
        ckpt_active = (
            self.gradient_checkpointing
            and self.training
            and not use_cache
        )
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            if ckpt_active:
                # 用 torch.utils.checkpoint 包裹：前向不存中间激活，反向时重算
                # 注：checkpoint 不支持 keyword-only 参数 + 可变返回值，所以用 closure
                from torch.utils.checkpoint import checkpoint as _ckpt

                def _layer_fwd(x_in, mask_in, rope_cos_in, rope_sin_in, _layer=layer):
                    out, _ = _layer(
                        x_in, mask=mask_in, kv_cache=None, use_cache=False,
                        rope_cos=rope_cos_in, rope_sin=rope_sin_in,
                    )
                    return out

                x = _ckpt(
                    _layer_fwd, x, combined_mask, self.rope_cos, self.rope_sin,
                    use_reentrant=False,
                )
                new_cache = None
            else:
                x, new_cache = layer(
                    x,
                    mask=combined_mask,
                    kv_cache=layer_cache,
                    use_cache=use_cache,
                    rope_cos=self.rope_cos,
                    rope_sin=self.rope_sin,
                )
            if use_cache:
                new_kv_caches.append(new_cache)

        # Final RMSNorm
        x = self.final_norm(x)

        # LM Head → logits
        logits = self.lm_head(x)  # [batch, seq, vocab_size]

        # 计算 loss (如果提供了 targets)
        loss = None
        if targets is not None:
            # 安全聚合（A 方案 Layer 2）：用 reduction='sum' + clamp(valid_count, min=1)
            # 替代默认 reduction='mean'，避免"整个 batch 全是 -100 (assistant 段被尾部
            # 截断砍光) 时分母 0 → NaN"污染 epoch 累加器的经典 SFT 坑。
            #   - 全 -100：sum=0、valid=0→clamp=1 → loss=0.0（无梯度，对训练无害）
            #   - 正常 batch：与 reduction='mean' 数值完全等价
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            loss_sum = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=-100,
                reduction="sum",
            )
            valid_count = (flat_targets != -100).sum().clamp(min=1)
            loss = loss_sum / valid_count

        return logits, loss, new_kv_caches

    def count_parameters(self) -> dict:
        """统计模型的实际参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 注意: 由于 weight tying, embedding 和 lm_head 共享参数
        # parameters() 不会重复计算
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
            "trainable_millions": trainable / 1e6,
        }

if __name__ == "__main__":
    print("=" * 60)
    print("ClearMind GPT 模型验证")
    print("=" * 60)

    from inference.generate import generate

    # ========== Small 配置测试 ==========
    config = ModelConfig.small()
    model = GPT(config)

    params = model.count_parameters()
    print("\n📊 Small 配置:")
    print(f"  总参数量: {params['total_millions']:.1f}M")
    print(f"  可训练参数: {params['trainable_millions']:.1f}M")

    # 前向传播测试
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss, _ = model(input_ids, targets)
    print("\n🔄 前向传播:")
    print(f"  Input:  {input_ids.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss:   {loss.item():.4f}")
    print(
        f"  期望初始 loss ≈ ln(vocab_size) = {torch.tensor(config.vocab_size).float().log().item():.4f}"
    )

    # 生成测试
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = generate(model, prompt, max_new_tokens=20, temperature=0.8, top_k=50, eos_token_id=-1)
    print("\n✨ 文本生成:")
    print(f"  Prompt:    {prompt.shape}")
    print(f"  Generated: {generated.shape}")
    print(f"  新生成 tokens: {generated.shape[1] - prompt.shape[1]}")

    # 反向传播测试
    loss.backward()
    print("\n✅ 反向传播成功!")

    # ========== Medium 配置参数量 ==========
    config_med = ModelConfig.medium()
    model_med = GPT(config_med)
    params_med = model_med.count_parameters()
    print(f"\n📊 Medium 配置参数量: {params_med['total_millions']:.1f}M")

    print(f"\n{'=' * 60}")
    print("✅ 全部验证通过!")
    print("=" * 60)
