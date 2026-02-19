"""
RoPE — 旋转位置编码 (Rotary Position Embedding)
================================================

RoPE 是 Llama、Gemini 等现代 LLM 使用的位置编码方式。
与传统的绝对/正弦位置编码不同，RoPE 将位置信息编码为向量旋转，
使注意力分数自然包含相对位置关系。

核心数学原理:
  1. 将 d 维向量分成 d/2 组二维子空间
  2. 对每组子空间 (x_2k, x_{2k+1})，在位置 pos 处做旋转:
     f(x, pos) = R(pos·θ_k) · x
     其中 θ_k = 1 / 10000^(2k/d)
  3. 旋转矩阵:
     [cos(pos·θ)  -sin(pos·θ)] [x_2k    ]
     [sin(pos·θ)   cos(pos·θ)] [x_{2k+1}]

优势:
  - 通过旋转让 q·k 的内积自然包含相对位置信息
  - 无需学习参数，完全由数学公式决定
  - 支持外推到训练时未见过的序列长度

参考论文: RoFormer (https://arxiv.org/abs/2104.09864)
"""

import torch
import torch.nn as nn


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos 和 sin 频率矩阵

    这个函数在模型初始化时调用一次，之后的 RoPE 应用只需要查表。

    Args:
        head_dim:    每个注意力头的维度 (必须为偶数)
        max_seq_len: 最大序列长度
        base:        频率底数 (默认 10000, 与原始 Transformer 一致)
        device:      计算设备

    Returns:
        cos_cached: [max_seq_len, head_dim] 的 cos 值
        sin_cached: [max_seq_len, head_dim] 的 sin 值
    """
    assert head_dim % 2 == 0, f"head_dim ({head_dim}) 必须为偶数"

    # Step 1: 计算频率 θ_k = 1 / base^(2k/d), k = 0, 1, ..., d/2-1
    # 形状: [head_dim // 2]
    k = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (base ** (k / head_dim))

    # Step 2: 计算位置 × 频率
    # positions: [max_seq_len]
    # freqs:     [head_dim // 2]
    # 外积 → [max_seq_len, head_dim // 2]
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)

    # Step 3: 扩展到完整的 head_dim 维度
    # 每个频率对应两个维度 (cos, sin 对)
    # [max_seq_len, head_dim // 2] → [max_seq_len, head_dim]
    angles = angles.repeat(1, 2)

    cos_cached = angles.cos()
    sin_cached = angles.sin()

    return cos_cached, sin_cached


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """将 RoPE 旋转应用到输入张量上

    旋转公式 (对每个二维子空间):
      x_rot[2k]   = x[2k] · cos(θ) - x[2k+1] · sin(θ)
      x_rot[2k+1] = x[2k] · sin(θ) + x[2k+1] · cos(θ)

    等价于将 (x[2k], x[2k+1]) 视为复数 x[2k] + i·x[2k+1],
    然后乘以 e^(iθ) = cos(θ) + i·sin(θ)

    Args:
        x:      输入张量 [batch, n_heads, seq_len, head_dim]
        cos:    预计算的 cos 值 [max_seq_len, head_dim]
        sin:    预计算的 sin 值 [max_seq_len, head_dim]
        offset: 位置偏移量 (KV Cache decode 时使用)
                prefill: offset=0, 位置 [0, 1, ..., seq_len-1]
                decode:  offset=cache_len, 位置 [cache_len, ..., cache_len+seq_len-1]

    Returns:
        旋转后的张量, 形状与输入相同
    """
    seq_len = x.shape[2]
    # 截取当前序列长度对应的 cos/sin (支持 offset)
    cos = (
        cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    )  # [1, 1, seq_len, head_dim]
    sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

    # 构造 "旋转配对" 的张量: 将 (x0, x1, x2, x3, ...) 变成 (-x1, x0, -x3, x2, ...)
    x_rotated = _rotate_half(x)

    # 应用旋转: x · cos + rotate_half(x) · sin
    return x * cos + x_rotated * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将输入的前后两半交换并取反 (用于 RoPE 旋转)

    [x1, x2, x3, x4] → [-x3, -x4, x1, x2]

    这等价于将每对 (x_2k, x_{2k+1}) 变为 (-x_{2k+1}, x_{2k})
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]  # 前半部分
    x2 = x[..., d // 2 :]  # 后半部分
    return torch.cat((-x2, x1), dim=-1)


if __name__ == "__main__":
    # 演示 RoPE 的效果
    head_dim = 8
    max_seq_len = 16
    batch = 1
    n_heads = 2

    cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)
    print(f"cos shape: {cos.shape}")  # [16, 8]
    print(f"sin shape: {sin.shape}")  # [16, 8]

    # 模拟 query 张量
    q = torch.randn(batch, n_heads, 4, head_dim)
    q_rotated = apply_rope(q, cos, sin)
    print(f"\nInput shape:  {q.shape}")
    print(f"Output shape: {q_rotated.shape}")
    print(f"形状保持不变: {q.shape == q_rotated.shape}")

    # 验证: 相对位置不变时, 内积相同
    q1 = torch.randn(1, 1, 1, head_dim)
    k1 = torch.randn(1, 1, 1, head_dim)

    # 在位置 3 和 5 (相对距离 2)
    cos3, sin3 = cos[3:4].unsqueeze(0).unsqueeze(0), sin[3:4].unsqueeze(0).unsqueeze(0)
    cos5, sin5 = cos[5:6].unsqueeze(0).unsqueeze(0), sin[5:6].unsqueeze(0).unsqueeze(0)
    dot1 = (q1 * cos3 + _rotate_half(q1) * sin3) @ (
        k1 * cos5 + _rotate_half(k1) * sin5
    ).transpose(-2, -1)

    # 在位置 10 和 12 (同样相对距离 2)
    cos10, sin10 = (
        cos[10:11].unsqueeze(0).unsqueeze(0),
        sin[10:11].unsqueeze(0).unsqueeze(0),
    )
    cos12, sin12 = (
        cos[12:13].unsqueeze(0).unsqueeze(0),
        sin[12:13].unsqueeze(0).unsqueeze(0),
    )
    dot2 = (q1 * cos10 + _rotate_half(q1) * sin10) @ (
        k1 * cos12 + _rotate_half(k1) * sin12
    ).transpose(-2, -1)

    print(f"\n验证相对位置编码特性:")
    print(f"  位置 (3,5) 的内积: {dot1.item():.6f}")
    print(f"  位置 (10,12) 的内积: {dot2.item():.6f}")
    print(f"  两者相等 (相对距离相同): {torch.allclose(dot1, dot2, atol=1e-5)}")
