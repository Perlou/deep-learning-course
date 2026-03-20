"""
SwiGLU — 门控线性单元激活函数
============================

SwiGLU 是 Llama、Gemini 等现代 LLM 使用的激活函数，
替代了 Transformer 原始论文中的 ReLU 和 GPT-2 中的 GELU。

注意: 这是一个独立的参考实现，用于教学演示。
实际模型中使用的是 feedforward.py 中内联的 F.silu 调用。

数学公式:
  SiLU(x) = x · σ(x)      (也叫 Swish 激活函数)
  σ(x) = 1/(1+e^(-x))     (Sigmoid 函数)

SwiGLU 门控机制:
  SwiGLU(x, W_gate, W_up) = SiLU(x·W_gate) ⊙ (x·W_up)

其中 ⊙ 表示逐元素乘法 (门控)。

参考论文: GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202)
"""

import torch
import torch.nn.functional as F


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU 激活函数

    SwiGLU(gate, up) = SiLU(gate) ⊙ up

    Args:
        gate: 门控分支的输出 [batch, seq, d_ff]
        up:   上投影分支的输出 [batch, seq, d_ff]

    Returns:
        门控后的张量 [batch, seq, d_ff]
    """
    return F.silu(gate) * up


if __name__ == "__main__":
    # 演示 SwiGLU 的效果
    import matplotlib

    matplotlib.use("Agg")

    x = torch.linspace(-5, 5, 100)

    # 对比不同激活函数
    relu = F.relu(x)
    gelu = F.gelu(x)
    silu = F.silu(x)  # SiLU = Swish = x · σ(x)

    print("激活函数对比 (x=2.0):")
    val = torch.tensor(2.0)
    print(f"  ReLU(2.0)  = {F.relu(val).item():.4f}")
    print(f"  GELU(2.0)  = {F.gelu(val).item():.4f}")
    print(f"  SiLU(2.0)  = {F.silu(val).item():.4f}")  # SwiGLU 使用的基础激活

    # SwiGLU 门控演示
    gate = torch.randn(1, 1, 8)
    up = torch.randn(1, 1, 8)
    result = swiglu(gate, up)
    print(f"\nSwiGLU 门控:")
    print(f"  gate:   {gate.squeeze()[:4].tolist()}")
    print(f"  up:     {up.squeeze()[:4].tolist()}")
    print(f"  output: {result.squeeze()[:4].tolist()}")
