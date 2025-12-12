"""
02-tensor-operations.py
Phase 3: PyTorch 核心技能

Tensor 运算 - 深度学习计算的基础

学习目标：
1. 掌握 Tensor 的基本运算
2. 理解广播机制
3. 掌握索引和切片
4. 掌握形状变换操作
"""

import torch

print("=" * 60)
print("PyTorch 核心技能 - Tensor 运算")
print("=" * 60)

print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 创建两个 3x3 矩阵，计算它们的矩阵乘法

2. 创建一个 3x4 矩阵，计算每行的均值和每列的最大值

3. 创建形状为 (2, 3, 4) 的张量，将其变换为 (4, 6)

4. 使用广播：给 3x4 矩阵的每一行加上不同的偏置

5. 将三个形状为 (2, 3) 的张量沿新维度拼接成 (3, 2, 3)
""")

# 1
A = torch.randn(3, 3)
B = torch.randn(3, 3)
C = A @ B
print(f"A @ B:\n{C}")

# 2
x = torch.randn(3, 4)
row_mean = x.mean(dim=1)
col_max = x.max(dim=0).values
print(f"每行均值: {row_mean}")
print(f"每列最大: {col_max}")

# 3
x = torch.randn(2, 3, 4)
y = x.reshape(4, 6)
print(f"{x.shape} -> {y.shape}")

# 4
matrix = torch.randn(3, 4)
bias = torch.tensor([[1], [2], [3]])
result = matrix + bias
print(result)

# 5
a, b, c = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
stacked = torch.stack([a, b, c], dim=0)
print(f"形状: {stacked.shape}")
