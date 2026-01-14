"""
04-lstm.py - LSTM 门控机制详解

本节学习:
1. LSTM 的核心思想
2. 三个门的作用
3. 手动实现 LSTM
4. PyTorch LSTM 使用
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第4节: LSTM 门控机制详解")
print("=" * 60)

# =============================================================================
# 1. LSTM 核心思想
# =============================================================================
print("""
📚 LSTM (Long Short-Term Memory)

核心思想: RNN + 门控机制
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   • 增加"细胞状态 C" 作为信息高速公路                         │
│   • 使用"门"来控制信息的保留与遗忘                           │
│   • 梯度可以无损地长距离传播                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

LSTM 有两个状态:
  • hₜ (hidden state): 短期记忆，直接输出
  • Cₜ (cell state): 长期记忆，信息高速公路
""")


# =============================================================================
# 2. 三个门详解
# =============================================================================
print("\n" + "=" * 60)
print("📌 2. 三个门详解")
print("-" * 60)

print("""
🚪 遗忘门 (Forget Gate)
─────────────────────────
作用：决定从细胞状态中丢弃什么信息

  fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
  
  fₜ ≈ 1: 完全保留旧信息
  fₜ ≈ 0: 完全遗忘旧信息
  
  例子：读到新句子时，遗忘前一个句子的主语

🚪 输入门 (Input Gate)
─────────────────────────
作用：决定什么新信息存入细胞状态

  iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)    ← 决定更新哪些位置
  C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc) ← 生成候选值
  
  新信息 = iₜ ⊙ C̃ₜ
  
  例子：看到新的主语时，将其存入状态

🚪 输出门 (Output Gate)
─────────────────────────
作用：决定细胞状态的哪些部分作为输出

  oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
  hₜ = oₜ ⊙ tanh(Cₜ)
  
  例子：预测下一个词时，只输出与语法相关的部分
""")


# =============================================================================
# 3. 细胞状态更新
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 细胞状态更新公式")
print("-" * 60)

print("""
完整的 LSTM 前向传播公式:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  遗忘门:   fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)                     │
│  输入门:   iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)                     │
│  候选值:   C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)                  │
│  细胞更新: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ                        │
│  输出门:   oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)                     │
│  隐藏状态: hₜ = oₜ ⊙ tanh(Cₜ)                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

信息流:
           遗忘           添加
  Cₜ₋₁ ──→ (×fₜ) ──→ (+iₜ⊙C̃ₜ) ──→ Cₜ ──→ Cₜ₊₁
                                   ↓
                              tanh ──→ (×oₜ) ──→ hₜ
""")


# =============================================================================
# 4. 手动实现 LSTM
# =============================================================================
print("\n" + "=" * 60)
print("📌 4. 手动实现 LSTM (NumPy)")
print("-" * 60)


class LSTMCell:
    """从零实现 LSTM 单元"""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 所有门共享输入，所以合并权重
        # 顺序: [遗忘门, 输入门, 候选值, 输出门]
        combined_size = input_size + hidden_size
        scale = np.sqrt(2.0 / combined_size)

        self.W = np.random.randn(4 * hidden_size, combined_size) * scale
        self.b = np.zeros(4 * hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev, c_prev):
        """
        单步前向传播
        Args:
            x: [batch_size, input_size]
            h_prev: [batch_size, hidden_size]
            c_prev: [batch_size, hidden_size]
        Returns:
            h_next, c_next
        """
        H = self.hidden_size

        # 拼接输入
        combined = np.concatenate([h_prev, x], axis=1)

        # 一次计算所有门
        gates = combined @ self.W.T + self.b

        # 分割四个门
        f = self.sigmoid(gates[:, 0:H])  # 遗忘门
        i = self.sigmoid(gates[:, H : 2 * H])  # 输入门
        c_tilde = np.tanh(gates[:, 2 * H : 3 * H])  # 候选细胞状态
        o = self.sigmoid(gates[:, 3 * H : 4 * H])  # 输出门

        # 更新细胞状态
        c_next = f * c_prev + i * c_tilde

        # 计算隐藏状态
        h_next = o * np.tanh(c_next)

        return h_next, c_next, (f, i, c_tilde, o)


# 测试
print("\n测试手动实现的 LSTM:")
batch_size = 2
input_size = 10
hidden_size = 20

lstm_cell = LSTMCell(input_size, hidden_size)

x = np.random.randn(batch_size, input_size)
h = np.zeros((batch_size, hidden_size))
c = np.zeros((batch_size, hidden_size))

h_new, c_new, gates = lstm_cell.forward(x, h, c)

print(f"  输入形状: {x.shape}")
print(f"  隐藏状态形状: {h_new.shape}")
print(f"  细胞状态形状: {c_new.shape}")
print(f"  遗忘门值范围: [{gates[0].min():.3f}, {gates[0].max():.3f}]")
print(f"  输入门值范围: [{gates[1].min():.3f}, {gates[1].max():.3f}]")


# =============================================================================
# 5. PyTorch LSTM
# =============================================================================
print("\n" + "=" * 60)
print("📌 5. PyTorch LSTM 使用")
print("-" * 60)

# 单层单向 LSTM
lstm = nn.LSTM(
    input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True
)

# 输入: [batch, seq_len, input_size]
seq_len = 15
x = torch.randn(batch_size, seq_len, input_size)

# 初始状态: (h_0, c_0)，各为 [num_layers, batch, hidden_size]
h0 = torch.zeros(1, batch_size, hidden_size)
c0 = torch.zeros(1, batch_size, hidden_size)

# 前向传播
output, (hn, cn) = lstm(x, (h0, c0))

print(f"\nPyTorch LSTM:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")
print(f"  最终隐藏状态形状: {hn.shape}")
print(f"  最终细胞状态形状: {cn.shape}")

# 参数量
print(f"\nLSTM 参数:")
for name, param in lstm.named_parameters():
    print(f"  {name}: {param.shape}")


# =============================================================================
# 6. 可视化门的激活
# =============================================================================
print("\n" + "=" * 60)
print("📌 6. 可视化门的激活")
print("-" * 60)


def visualize_lstm_gates():
    """可视化 LSTM 门的激活模式"""
    # 创建一个简单的 LSTM
    lstm = nn.LSTM(1, 32, batch_first=True)

    # 创建一个有模式的输入序列
    t = torch.linspace(0, 4 * np.pi, 50)
    x = torch.sin(t).unsqueeze(0).unsqueeze(-1)  # [1, 50, 1]

    # 手动获取门的激活值
    # 使用 hook 来捕获
    gate_activations = {"f": [], "i": [], "o": []}

    # 逐步运行以获取门的值
    h = torch.zeros(1, 1, 32)
    c = torch.zeros(1, 1, 32)

    for step in range(50):
        x_step = x[:, step : step + 1, :]

        with torch.no_grad():
            output, (h, c) = lstm(x_step, (h, c))

        # LSTM 内部门的近似可视化 (通过权重和状态推断)
        # 这里简化为记录隐藏状态的变化
        gate_activations["f"].append(h.mean().item())
        gate_activations["i"].append(h.std().item())
        gate_activations["o"].append(c.mean().item())

    return gate_activations, t.numpy(), x.squeeze().numpy()


activations, t, x_signal = visualize_lstm_gates()

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# 输入信号
axes[0].plot(t, x_signal, "b-", linewidth=2, label="输入信号")
axes[0].set_xlabel("时间步")
axes[0].set_ylabel("值")
axes[0].set_title("输入序列")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 隐藏状态变化
axes[1].plot(activations["f"], "r-", label="隐藏状态均值", alpha=0.7)
axes[1].plot(activations["i"], "g-", label="隐藏状态标准差", alpha=0.7)
axes[1].plot(activations["o"], "b-", label="细胞状态均值", alpha=0.7)
axes[1].set_xlabel("时间步")
axes[1].set_ylabel("值")
axes[1].set_title("LSTM 内部状态变化")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/lstm_gates.png", dpi=100)
plt.close()
print("LSTM 门激活可视化已保存: outputs/lstm_gates.png")


# =============================================================================
# 7. 多层 LSTM
# =============================================================================
print("\n" + "=" * 60)
print("📌 7. 多层 LSTM")
print("-" * 60)

# 多层 LSTM
lstm_stacked = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=3,  # 3 层堆叠
    batch_first=True,
    dropout=0.2,  # 层间 dropout
)

x = torch.randn(batch_size, seq_len, input_size)
h0 = torch.zeros(3, batch_size, hidden_size)  # 3 层
c0 = torch.zeros(3, batch_size, hidden_size)

output, (hn, cn) = lstm_stacked(x, (h0, c0))

print(f"多层 LSTM (3 层):")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")
print(f"  最终隐藏状态形状: {hn.shape} (每层一个)")
print(f"  参数量: {sum(p.numel() for p in lstm_stacked.parameters()):,}")


# =============================================================================
# 8. 练习
# =============================================================================
print("\n" + "=" * 60)
print("📝 练习题")
print("-" * 60)

print("""
1. LSTM 的细胞状态 C 和隐藏状态 h 有什么区别？
   答：C 是长期记忆，通过加法更新，梯度传播稳定；
       h 是短期记忆，直接作为输出

2. 遗忘门全为 1 时会发生什么？
   答：旧的细胞状态完全保留，信息可以无损传递

3. 为什么 LSTM 用 tanh 生成候选值而不是 sigmoid？
   答：tanh 输出范围 (-1, 1)，可以增加或减少细胞状态值

4. LSTM 参数量是 RNN 的多少倍？
   答：约 4 倍（4 个门，每个门都有独立的权重）
""")

print("\n✅ 第4节完成！")
print("下一节：05-gru.py - GRU 简化结构")
