"""
04-nn-module.py
Phase 3: PyTorch 核心技能

nn.Module - 神经网络的基础构建块

学习目标：
1. 理解 nn.Module 的结构和作用
2. 掌握自定义模型的方法
3. 了解常用层和参数管理
4. 理解模型的 train/eval 模式
"""

import torch
import torch.nn as nn

print("=" * 60)
print("PyTorch 核心技能 - nn.Module")
print("=" * 60)

print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 创建一个三层 MLP: 784 -> 256 -> 128 -> 10，使用 ReLU 激活

2. 统计模型的总参数量和可训练参数量

3. 使用 nn.Sequential 创建相同的网络

4. 实现一个带 Dropout 和 BatchNorm 的模型

5. 将模型移动到 GPU（如果可用）并进行一次前向传播
""")

# 1
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2
model = MLP()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total}, 可训练: {trainable}")

# 3
model_seq = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

# 4
class RegularizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
x = torch.randn(32, 784).to(device)
y = model(x)
