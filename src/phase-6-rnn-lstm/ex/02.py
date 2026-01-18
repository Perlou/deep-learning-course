import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第2节: 时间反向传播 (BPTT)")
print("=" * 60)

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
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))

        self.inputs = inputs
        self.hs = {-1: h_prev.copy()}
        self.ys = {}
        self.ps = {}

        loss = 0
        for t in range(len(inputs)):
            x = inputs[t].reshape(-1, 1)

            self.hs[t] = np.tanh(self.Wxh @ x + self.Whh @ self.hs[t - 1] + self.bh)

            self.ys[t] = self.Why @ self.hs[t] + self.by
            self.ps[t] = np.exp(self.ys[t]) / np.sum(np.exp(self.ys[t]))

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

print("\n" + "=" * 60)
print("📌 4. PyTorch 自动反向传播")
print("-" * 60)


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
