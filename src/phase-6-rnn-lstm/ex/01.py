import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第1节: RNN 基础结构与前向传播")
print("=" * 60)


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # 初始化权重 (Xavier)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(hidden_size, input_size) * scale  # 输入到隐藏
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale  # 隐藏到隐藏
        self.Why = np.random.randn(output_size, hidden_size) * scale  # 隐藏到输出

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev=None):
        seq_len = len(inputs)

        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))

        hidden_states = []
        outputs = []

        h = h_prev
        for t in range(seq_len):
            x = inputs[t].reshape(-1, 1)

            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

            y = self.Why @ h + self.by

            hidden_states.append(h)
            outputs.append(y)

        return outputs, hidden_states


# 测试手动实现
print("\n测试手动实现的 RNN:")
input_size = 4
hidden_size = 8
output_size = 3
seq_len = 5

rnn = SimpleRNN(input_size, hidden_size, output_size)

# 创建随机输入序列
inputs = [np.random.randn(input_size) for _ in range(seq_len)]

outputs, hidden_states = rnn.forward(inputs)

print(f"  输入维度: {input_size}")
print(f"  隐藏层维度: {hidden_size}")
print(f"  输出维度: {output_size}")
print(f"  序列长度: {seq_len}")
print(f"  输出形状: {len(outputs)} × {outputs[0].shape}")
print(f"  隐藏状态形状: {len(hidden_states)} × {hidden_states[0].shape}")

print("\n" + "=" * 60)
print("📌 4. PyTorch RNN 使用")
print("-" * 60)

rnn_pytorch = nn.RNN(
    input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True
)

batch_size = 2
x = torch.randn(batch_size, seq_len, input_size)

# 初始隐藏状态: [num_layers, batch_size, hidden_size]
h0 = torch.zeros(1, batch_size, hidden_size)

# 前向传播
output, hn = rnn_pytorch(x, h0)

print(f"\nPyTorch RNN:")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")
print(f"  最终隐藏状态形状: {hn.shape}")

# 查看参数
print(f"\nRNN 参数:")
for name, param in rnn_pytorch.named_parameters():
    print(f"  {name}: {param.shape}")
