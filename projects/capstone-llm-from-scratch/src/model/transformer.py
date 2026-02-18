"""
TransformerBlock — Transformer 解码器层
======================================

一个 Transformer Block 包含:
  1. RMSNorm → Multi-Head Attention → Residual Connection
  2. RMSNorm → SwiGLU FFN → Residual Connection

结构图:
  x ──────────────────┐
  │                   │
  RMSNorm             │
  │                   │
  Attention           │
  │                   │
  + ←─────────────────┘ (residual)
  │
  ──────────────────┐
  │                 │
  RMSNorm           │
  │                 │
  FeedForward       │
  │                 │
  + ←───────────────┘ (residual)
  │
  output

注意 Pre-Norm 架构:
  - 原始 Transformer 使用 Post-Norm: x + Sublayer(LayerNorm(x)) ← 不对
  - 实际原始是: LayerNorm(x + Sublayer(x))
  - 现代 LLM 使用 Pre-Norm: x + Sublayer(RMSNorm(x))
  - Pre-Norm 训练更稳定, 不容易梯度爆炸
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .normalization import RMSNorm
from .attention import Attention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """单个 Transformer 解码器层

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Pre-Norm for Attention
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        # Multi-Head Self Attention
        self.attention = Attention(config)

        # Pre-Norm for FFN
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        # SwiGLU FeedForward
        self.feedforward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    输入张量 [batch, seq_len, d_model]
            mask: 注意力 causal mask

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        # Attention block with residual connection
        # h = x + Attention(RMSNorm(x))
        h = x + self.attention(self.attn_norm(x), mask)

        # FFN block with residual connection
        # out = h + FFN(RMSNorm(h))
        out = h + self.feedforward(self.ffn_norm(h))

        return out


if __name__ == "__main__":
    config = ModelConfig.small()
    block = TransformerBlock(config)

    x = torch.randn(2, 10, config.d_model)
    mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    y = block(x, mask)

    print("TransformerBlock:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  参数量: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")

    # 验证残差连接: 初始时 attention 和 ffn 输出随机, output ≈ x
    block_zero = TransformerBlock(config)
    # 将所有权重设为 0
    for name, param in block_zero.named_parameters():
        if "weight" in name and "norm" not in name:
            param.data.zero_()
    y_zero = block_zero(x, mask)
    print(f"\n  残差连接验证 (权重全0时, 输出 ≈ 输入):")
    print(f"  max diff: {(y_zero - x).abs().max().item():.8f}")
