"""
02-bptt.py - 时间反向传播 (Backpropagation Through Time)

本节学习:
1. BPTT 的核心思想
2. 梯度计算过程
3. 手动实现 BPTT
4. 截断 BPTT
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第2节: 时间反向传播 (BPTT)")
print("=" * 60)

# =============================================================================
# 1. BPTT 核心思想
# =============================================================================
print("""
📚 BPTT (Backpropagation Through Time)

核心思想：将 RNN 在时间上展开，然后像普通神经网络一样反向传播

时间展开后的反向传播:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│     ∂L/∂y₄   ∂L/∂y₃   ∂L/∂y₂   ∂L/∂y₁                       │
│        ↓        ↓        ↓        ↓                          │
│     ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                      │
│     │  h₄  │←│  h₃  │←│  h₂  │←│  h₁  │                      │
│     └──────┘ └──────┘ └──────┘ └──────┘                      │
│        ↓        ↓        ↓        ↓                          │
│     ∂L/∂x₄   ∂L/∂x₃   ∂L/∂x₂   ∂L/∂x₁                       │
│                                                              │
│  梯度沿时间反向流动！                                         │
└─────────────────────────────────────────────────────────────┘

权重梯度累加:
  ∂L/∂Wₕₕ = Σₜ (∂L/∂hₜ) · (∂hₜ/∂Wₕₕ)
  
注意：所有时间步共享同一套权重，梯度需要累加！
""")


# =============================================================================
# 2. BPTT 梯度推导
# =============================================================================
print("\n" + "=" * 60)
print("📌 2. BPTT 梯度推导")
print("-" * 60)

print("""
前向传播公式:
  hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
  yₜ = Wₕᵧ · hₜ + bᵧ
  L = Σₜ Lₜ(yₜ, target)

反向传播公式:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  步骤1：输出层梯度                                            │
│    ∂L/∂Wₕᵧ = Σₜ (∂Lₜ/∂yₜ) · hₜᵀ                             │
│                                                              │
│  步骤2：隐藏状态梯度（关键！）                                 │
│    ∂L/∂hₜ = (∂Lₜ/∂yₜ) · Wₕᵧ + (∂L/∂hₜ₊₁) · Wₕₕᵀ · diag(1-hₜ₊₁²)│
│             ↑                  ↑                              │
│          当前时刻           来自未来时刻的梯度                  │
│                                                              │
│  步骤3：权重梯度累加                                          │
│    ∂L/∂Wₓₕ = Σₜ (∂L/∂hₜ) · diag(1-hₜ²) · xₜᵀ                │
│    ∂L/∂Wₕₕ = Σₜ (∂L/∂hₜ) · diag(1-hₜ²) · hₜ₋₁ᵀ              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 3. 手动实现 BPTT
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 手动实现 BPTT")
print("-" * 60)


class RNNWithBPTT:
    """带有 BPTT 的 RNN 实现"""

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # 初始化权重
        scale = 0.01
        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, targets, h_prev=None):
        """前向传播并计算损失"""
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))

        # 保存中间结果用于反向传播
        self.inputs = inputs
        self.hs = {-1: h_prev.copy()}
        self.ys = {}
        self.ps = {}

        loss = 0
        for t in range(len(inputs)):
            x = inputs[t].reshape(-1, 1)

            # 隐藏状态
            self.hs[t] = np.tanh(self.Wxh @ x + self.Whh @ self.hs[t - 1] + self.bh)

            # 输出 (使用 softmax)
            self.ys[t] = self.Why @ self.hs[t] + self.by
            self.ps[t] = np.exp(self.ys[t]) / np.sum(np.exp(self.ys[t]))

            # 交叉熵损失
            loss += -np.log(self.ps[t][targets[t], 0])

        self.targets = targets
        return loss

    def backward(self):
        """BPTT 反向传播"""
        # 初始化梯度
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((self.hidden_size, 1))

        # 反向遍历时间步
        for t in reversed(range(len(self.inputs))):
            # 输出层梯度
            dy = self.ps[t].copy()
            dy[self.targets[t]] -= 1  # softmax + cross-entropy 的梯度

            dWhy += dy @ self.hs[t].T
            dby += dy

            # 隐藏层梯度 (关键：包含来自未来的梯度)
            dh = self.Why.T @ dy + dh_next

            # tanh 求导: d(tanh(x))/dx = 1 - tanh²(x)
            dh_raw = (1 - self.hs[t] ** 2) * dh

            # 参数梯度
            x = self.inputs[t].reshape(-1, 1)
            dWxh += dh_raw @ x.T
            dWhh += dh_raw @ self.hs[t - 1].T
            dbh += dh_raw

            # 传递到上一时刻
            dh_next = self.Whh.T @ dh_raw

        # 梯度裁剪
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby


# 测试 BPTT
print("\n测试 BPTT 实现:")
input_size = 10
hidden_size = 20
output_size = 10
seq_len = 5

rnn = RNNWithBPTT(input_size, hidden_size, output_size)

# 创建 one-hot 编码的输入
inputs = [np.eye(input_size)[np.random.randint(0, input_size)] for _ in range(seq_len)]
targets = [np.random.randint(0, output_size) for _ in range(seq_len)]

# 前向 + 反向
loss = rnn.forward(inputs, targets)
grads = rnn.backward()

print(f"  损失: {loss:.4f}")
print(f"  dWxh 范数: {np.linalg.norm(grads[0]):.4f}")
print(f"  dWhh 范数: {np.linalg.norm(grads[1]):.4f}")
print(f"  dWhy 范数: {np.linalg.norm(grads[2]):.4f}")


# =============================================================================
# 4. PyTorch 自动反向传播
# =============================================================================
print("\n" + "=" * 60)
print("📌 4. PyTorch 自动反向传播")
print("-" * 60)


# 用 PyTorch 验证梯度
class SimpleRNNPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


model = SimpleRNNPyTorch(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()

# 创建输入
x = torch.randn(1, seq_len, input_size, requires_grad=True)
targets_pt = torch.randint(0, output_size, (seq_len,))

# 前向传播
output = model(x).squeeze(0)
loss = criterion(output, targets_pt)

# 反向传播
loss.backward()

print(f"  PyTorch 损失: {loss.item():.4f}")
print(f"  输入梯度形状: {x.grad.shape}")
print(f"  输入梯度范数: {x.grad.norm().item():.4f}")


# =============================================================================
# 5. 截断 BPTT
# =============================================================================
print("\n" + "=" * 60)
print("📌 5. 截断 BPTT (Truncated BPTT)")
print("-" * 60)

print("""
问题：完整 BPTT 对于长序列计算量太大

解决方案：截断 BPTT
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  完整序列: x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ ...               │
│                                                              │
│  截断 BPTT (k=3):                                            │
│    段1: x₁ x₂ x₃ → 反向传播 → 更新权重                       │
│    段2: x₄ x₅ x₆ → 反向传播 → 更新权重 (隐藏状态从段1继承)   │
│    段3: x₇ x₈ x₉ → 反向传播 → 更新权重 (隐藏状态从段2继承)   │
│    ...                                                       │
│                                                              │
│  优点：                                                       │
│    ✅ 减少内存使用                                            │
│    ✅ 加速训练                                                │
│                                                              │
│  缺点：                                                       │
│    ❌ 无法捕捉超过 k 步的长期依赖                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


# 截断 BPTT 示例
def truncated_bptt_example():
    """截断 BPTT 示例"""
    model = nn.RNN(10, 20, batch_first=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 长序列
    long_sequence = torch.randn(1, 100, 10)  # 100 个时间步

    # 分段处理
    chunk_size = 20
    h = None
    total_loss = 0

    for i in range(0, 100, chunk_size):
        chunk = long_sequence[:, i : i + chunk_size, :]

        # 前向传播 (保留隐藏状态)
        output, h = model(chunk, h)

        # 计算损失 (这里用简单的 MSE)
        loss = output.sum()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 关键：detach 隐藏状态，断开计算图
        h = h.detach()

        total_loss += loss.item()
        print(f"  段 {i // chunk_size + 1}: 损失 = {loss.item():.4f}")

    print(f"  总损失: {total_loss:.4f}")


print("\n截断 BPTT 示例:")
truncated_bptt_example()


# =============================================================================
# 6. 练习
# =============================================================================
print("\n" + "=" * 60)
print("📝 练习题")
print("-" * 60)

print("""
1. BPTT 和普通反向传播有什么区别？
   答：BPTT 需要沿时间维度展开，梯度从最后时刻反向传播到第一时刻

2. 为什么需要梯度裁剪？
   答：防止梯度爆炸，将梯度限制在合理范围内

3. 截断 BPTT 中，为什么要 detach 隐藏状态？
   答：断开计算图，防止梯度跨段传播，节省内存

4. 如果不做截断，训练 1000 步序列需要保存多少中间状态？
   答：1000 个隐藏状态，内存消耗与序列长度成正比
""")

print("\n✅ 第2节完成！")
print("下一节：03-vanishing-gradient.py - 梯度消失问题")
