"""
05-gru.py - GRU 简化结构

本节学习:
1. GRU 的设计思想
2. 两个门的作用
3. GRU vs LSTM 对比
4. PyTorch GRU 使用
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第5节: GRU 简化结构")
print("=" * 60)

# =============================================================================
# 1. GRU 核心思想
# =============================================================================
print("""
📚 GRU (Gated Recurrent Unit)

设计思想: 简化 LSTM，减少参数量
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  LSTM: 3 个门 (遗忘、输入、输出) + 2 个状态 (h, C)           │
│  GRU:  2 个门 (重置、更新) + 1 个状态 (h)                    │
│                                                              │
│  关键简化:                                                    │
│    • 合并遗忘门和输入门为更新门                              │
│    • 去掉细胞状态，只保留隐藏状态                            │
│    • 性能与 LSTM 相近，计算更快                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 2. GRU 公式详解
# =============================================================================
print("\n" + "=" * 60)
print("📌 2. GRU 公式详解")
print("-" * 60)

print("""
GRU 的两个门:

🚪 更新门 (Update Gate) - zₜ
─────────────────────────────
作用：决定保留多少旧信息，添加多少新信息

  zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)
  
  zₜ ≈ 1: 保持旧状态 (类似 LSTM 遗忘门=1)
  zₜ ≈ 0: 使用新状态 (类似 LSTM 输入门=1)

🚪 重置门 (Reset Gate) - rₜ  
─────────────────────────────
作用：决定利用多少旧状态来计算新候选值

  rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)
  
  rₜ ≈ 1: 充分利用历史信息
  rₜ ≈ 0: 忽略历史，从头开始

完整公式:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  重置门:   rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)                     │
│  更新门:   zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)                     │
│  候选值:   h̃ₜ = tanh(Wh · [rₜ ⊙ hₜ₋₁, xₜ] + bh)             │
│  新状态:   hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ                  │
│                 └─保留旧─┘     └─添加新─┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 3. 手动实现 GRU
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 手动实现 GRU")
print("-" * 60)


class GRUCell:
    """从零实现 GRU 单元"""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 权重初始化
        combined_size = input_size + hidden_size
        scale = np.sqrt(2.0 / combined_size)

        # 重置门和更新门的权重
        self.Wz = np.random.randn(hidden_size, combined_size) * scale
        self.Wr = np.random.randn(hidden_size, combined_size) * scale
        self.Wh = np.random.randn(hidden_size, combined_size) * scale

        self.bz = np.zeros(hidden_size)
        self.br = np.zeros(hidden_size)
        self.bh = np.zeros(hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev):
        """
        单步前向传播
        Args:
            x: [batch_size, input_size]
            h_prev: [batch_size, hidden_size]
        Returns:
            h_next
        """
        # 拼接输入
        combined = np.concatenate([h_prev, x], axis=1)

        # 更新门
        z = self.sigmoid(combined @ self.Wz.T + self.bz)

        # 重置门
        r = self.sigmoid(combined @ self.Wr.T + self.br)

        # 候选隐藏状态 (使用重置门)
        combined_reset = np.concatenate([r * h_prev, x], axis=1)
        h_tilde = np.tanh(combined_reset @ self.Wh.T + self.bh)

        # 新隐藏状态
        h_next = (1 - z) * h_prev + z * h_tilde

        return h_next, (z, r, h_tilde)


# 测试
print("\n测试手动实现的 GRU:")
batch_size = 2
input_size = 10
hidden_size = 20

gru_cell = GRUCell(input_size, hidden_size)

x = np.random.randn(batch_size, input_size)
h = np.zeros((batch_size, hidden_size))

h_new, gates = gru_cell.forward(x, h)

print(f"  输入形状: {x.shape}")
print(f"  隐藏状态形状: {h_new.shape}")
print(f"  更新门值范围: [{gates[0].min():.3f}, {gates[0].max():.3f}]")
print(f"  重置门值范围: [{gates[1].min():.3f}, {gates[1].max():.3f}]")


# =============================================================================
# 4. PyTorch GRU
# =============================================================================
print("\n" + "=" * 60)
print("📌 4. PyTorch GRU 使用")
print("-" * 60)

gru = nn.GRU(
    input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True
)

seq_len = 15
x = torch.randn(batch_size, seq_len, input_size)
h0 = torch.zeros(1, batch_size, hidden_size)

output, hn = gru(x, h0)

print(f"\nPyTorch GRU:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")
print(f"  最终隐藏状态形状: {hn.shape}")

# 注意：GRU 只返回 h，不返回 c
print(f"\n注意: GRU 只返回隐藏状态 h，没有细胞状态 c")


# =============================================================================
# 5. GRU vs LSTM 对比
# =============================================================================
print("\n" + "=" * 60)
print("📌 5. GRU vs LSTM 对比")
print("-" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│  特性              │  LSTM                │  GRU              │
├─────────────────────────────────────────────────────────────┤
│  门的数量          │  3 (遗忘/输入/输出)  │  2 (重置/更新)    │
│  状态数量          │  2 (h, C)            │  1 (h)            │
│  参数量            │  4 × (i+h) × h       │  3 × (i+h) × h    │
│  计算速度          │  较慢                │  较快 (~25%)      │
│  性能              │  略优于 GRU          │  接近 LSTM        │
│  长序列            │  更好                │  良好             │
│  适用场景          │  复杂任务、长序列    │  资源受限、快速    │
└─────────────────────────────────────────────────────────────┘
""")

# 参数量对比
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
gru = nn.GRU(input_size, hidden_size, batch_first=True)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())

print(f"参数量对比 (input_size={input_size}, hidden_size={hidden_size}):")
print(f"  LSTM: {lstm_params:,} 参数")
print(f"  GRU:  {gru_params:,} 参数")
print(f"  比率: GRU/LSTM = {gru_params / lstm_params:.2%}")


# =============================================================================
# 6. 性能对比实验
# =============================================================================
print("\n" + "=" * 60)
print("📌 6. 性能对比实验")
print("-" * 60)


def compare_speed():
    """对比 RNN, LSTM, GRU 的速度"""
    import time

    input_size = 64
    hidden_size = 128
    seq_len = 100
    batch_size = 32

    models = {
        "RNN": nn.RNN(input_size, hidden_size, batch_first=True),
        "LSTM": nn.LSTM(input_size, hidden_size, batch_first=True),
        "GRU": nn.GRU(input_size, hidden_size, batch_first=True),
    }

    x = torch.randn(batch_size, seq_len, input_size)

    print(f"\n速度对比 (batch={batch_size}, seq_len={seq_len}):")

    for name, model in models.items():
        # 预热
        for _ in range(3):
            _ = model(x)

        # 计时
        start = time.time()
        for _ in range(100):
            _ = model(x)
        elapsed = time.time() - start

        print(f"  {name}: {elapsed:.3f}s (100次)")


compare_speed()


# =============================================================================
# 7. 双向 GRU
# =============================================================================
print("\n" + "=" * 60)
print("📌 7. 双向 GRU")
print("-" * 60)

gru_bidirectional = nn.GRU(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=2,
    batch_first=True,
    bidirectional=True,  # 双向
)

x = torch.randn(batch_size, seq_len, input_size)
# 双向: num_layers * 2
h0 = torch.zeros(2 * 2, batch_size, hidden_size)

output, hn = gru_bidirectional(x, h0)

print(f"双向 GRU (2层):")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}  (hidden_size * 2 因为双向)")
print(f"  隐藏状态形状: {hn.shape}  (num_layers * 2)")
print(f"  参数量: {sum(p.numel() for p in gru_bidirectional.parameters()):,}")


# =============================================================================
# 8. 何时选择 GRU vs LSTM
# =============================================================================
print("\n" + "=" * 60)
print("📌 8. 如何选择 GRU vs LSTM")
print("-" * 60)

print("""
选择指南:

选择 LSTM 当:
  ✓ 序列很长 (>500 步)
  ✓ 任务复杂，需要精细的信息控制
  ✓ 计算资源充足
  ✓ 追求最佳性能

选择 GRU 当:
  ✓ 序列中等长度
  ✓ 需要快速训练和推理
  ✓ 移动端或嵌入式部署
  ✓ 模型参数量受限
  ✓ 快速实验原型

实际建议:
  • 先用 GRU 快速验证想法
  • 如果效果不够好，切换到 LSTM
  • 两者差距通常 <5%
""")


# =============================================================================
# 9. 练习
# =============================================================================
print("\n" + "=" * 60)
print("📝 练习题")
print("-" * 60)

print("""
1. GRU 的更新门相当于 LSTM 的哪些门？
   答：相当于遗忘门和输入门的结合 (1-z ≈ f, z ≈ i)

2. GRU 没有输出门，它如何控制输出？
   答：通过更新门隐式控制，新状态直接作为输出

3. 重置门 r=0 时，候选值 h̃ 只依赖什么？
   答：只依赖当前输入 x，完全忽略历史状态

4. 为什么 GRU 比 LSTM 快？
   答：更少的门 (3 vs 4)，更少的矩阵乘法
""")

print("\n✅ 第5节完成！")
print("下一节：06-bidirectional.py - 双向 RNN")
