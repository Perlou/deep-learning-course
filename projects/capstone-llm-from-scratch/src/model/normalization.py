"""
RMSNorm — 均方根层归一化 (Root Mean Square Layer Normalization)
=============================================================

RMSNorm 是 LayerNorm 的简化版本，被 Llama、Gemini 等现代 LLM 广泛采用。
相比 LayerNorm，RMSNorm 省去了均值的计算和可学习 bias，计算更高效。

数学公式对比:
  LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) · γ + β
  RMSNorm(x)   = x / sqrt(mean(x²) + ε) · γ

关键区别:
  1. RMSNorm 不做中心化 (不减均值), 只做缩放
  2. RMSNorm 没有 bias (β)
  3. RMSNorm 计算量减少约 10%
  4. 实践中效果与 LayerNorm 相当甚至更好

参考论文: Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """均方根层归一化

    Args:
        dim: 归一化的维度大小 (通常为 d_model)
        eps: 防止除零的小常数 (默认 1e-6)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 γ, 初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [..., dim]

        Returns:
            归一化后的张量, 形状不变
        """
        # Step 1: 计算 RMS = sqrt(mean(x²) + ε)
        # x.pow(2): 每个元素平方
        # .mean(-1, keepdim=True): 在最后一维上求均值
        # rsqrt = 1/sqrt: 合并开方和除法为一步
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()

        # Step 2: 归一化并缩放
        # x * rms: x / sqrt(mean(x²) + ε)
        # * self.weight: 乘以可学习参数 γ
        return x * rms * self.weight


if __name__ == "__main__":
    # 对比 RMSNorm 和 LayerNorm
    dim = 512
    batch, seq_len = 2, 10

    rms_norm = RMSNorm(dim)
    layer_norm = nn.LayerNorm(dim)

    x = torch.randn(batch, seq_len, dim)

    # 前向传播
    y_rms = rms_norm(x)
    y_ln = layer_norm(x)

    print("RMSNorm vs LayerNorm 对比:")
    print(f"  输入形状: {x.shape}")
    print(f"  RMSNorm 输出形状: {y_rms.shape}")
    print(f"  LayerNorm 输出形状: {y_ln.shape}")
    print(f"\n  RMSNorm 参数量: {sum(p.numel() for p in rms_norm.parameters())}")
    print(f"  LayerNorm 参数量: {sum(p.numel() for p in layer_norm.parameters())}")
    print(
        f"  RMSNorm 更少参数: {sum(p.numel() for p in rms_norm.parameters()) < sum(p.numel() for p in layer_norm.parameters())}"
    )

    # 验证输出的统计特性
    print(f"\n  RMSNorm 输出 RMS ≈ 1: {y_rms.pow(2).mean(-1).sqrt().mean().item():.4f}")
