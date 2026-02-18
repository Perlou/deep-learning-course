"""
FeedForward — 基于 SwiGLU 的前馈网络
=====================================

前馈网络 (FFN) 是 Transformer Block 中与 Attention 并列的核心组件。
MiniMind 使用 SwiGLU 变体替代原始 Transformer 的 ReLU FFN。

结构对比:
  标准 FFN:    FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂     (2 个权重矩阵)
  SwiGLU FFN:  FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up))·W_down  (3 个权重矩阵)

为什么用 3 个矩阵而不是 2 个?
  - W_gate 控制门控 (哪些特征通过)
  - W_up   提取特征  (要传递什么信息)
  - W_down 输出投影  (降维回 d_model)
  - 门控机制使网络有选择性地传递信息，提升表达能力

参数量对比:
  标准 FFN: 2 × d_model × d_ff = 2 × d × 4d = 8d²
  SwiGLU:   3 × d_model × d_ff = 3 × d × (8d/3) ≈ 8d²
  → 调整 d_ff ≈ 2.67×d 使总参数量接近

信号流:
  input (d_model) → W_gate → SiLU → ⊙
                  → W_up   ─────────┘ → W_down → output (d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FeedForward(nn.Module):
    """基于 SwiGLU 的前馈网络

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        d_model = config.d_model
        d_ff = config.d_ff

        # 三个线性变换 (均无 bias, 与 Llama 一致)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # 门控
        self.w_up = nn.Linear(d_model, d_ff, bias=False)  # 上投影
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # 下投影

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        # SwiGLU: SiLU(x·W_gate) ⊙ (x·W_up) → W_down
        gate = F.silu(self.w_gate(x))  # [batch, seq, d_ff] — 门控分支
        up = self.w_up(x)  # [batch, seq, d_ff] — 上投影分支
        hidden = gate * up  # 逐元素门控
        output = self.w_down(hidden)  # [batch, seq, d_model] — 下投影
        return self.dropout(output)


if __name__ == "__main__":
    config = ModelConfig.small()
    ffn = FeedForward(config)

    x = torch.randn(2, 10, config.d_model)
    y = ffn(x)

    print("SwiGLU FeedForward:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")
    print(f"  参数量: {sum(p.numel() for p in ffn.parameters()) / 1e6:.2f}M")
    print(f"  W_gate: {ffn.w_gate.weight.shape}")
    print(f"  W_up:   {ffn.w_up.weight.shape}")
    print(f"  W_down: {ffn.w_down.weight.shape}")

    # 对比标准 FFN
    standard_ffn_params = 2 * config.d_model * (4 * config.d_model)
    swiglu_params = 3 * config.d_model * config.d_ff
    print(f"\n参数量对比 (不含 bias):")
    print(f"  标准 FFN (4×d): {standard_ffn_params / 1e6:.2f}M")
    print(f"  SwiGLU (d_ff={config.d_ff}): {swiglu_params / 1e6:.2f}M")
