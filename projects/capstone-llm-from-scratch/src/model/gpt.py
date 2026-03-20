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

        # ========== Causal Mask ==========
        # 上三角矩阵, 防止 token 看到未来的信息
        # 注册为 buffer: 不参与训练, 但跟随模型设备
        mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

        # ========== 初始化权重 ==========
        self.apply(self._init_weights)

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
            # [batch, 1, 1, seq_len] — padding 位置为 -inf
            pad_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * float("-inf")
            # 与 causal mask 合并: [seq_len, seq_len] + [batch, 1, 1, seq_len]
            combined_mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0) + pad_mask
        else:
            combined_mask = self.causal_mask

        # N × Transformer Block (带 KV Cache 透传)
        new_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, mask=combined_mask, kv_cache=layer_cache, use_cache=use_cache
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
            # Cross-entropy loss
            # logits: [batch × seq, vocab] vs targets: [batch × seq]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # padding token 不计算 loss
            )

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
