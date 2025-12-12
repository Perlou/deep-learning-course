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

# =============================================================================
# 1. 基本数学运算
# =============================================================================
print("\n【1. 基本数学运算】")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")

# 元素级运算
print(f"\na + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")  # 元素级乘法
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")

# 函数形式
print(f"\ntorch.add(a, b) = {torch.add(a, b)}")
print(f"torch.mul(a, b) = {torch.mul(a, b)}")

# 原地操作 (in-place, 以 _ 结尾)
c = torch.tensor([1.0, 2.0, 3.0])
print(f"\n原地操作前: c = {c}")
c.add_(10)  # 等价于 c = c + 10
print(f"c.add_(10) 后: c = {c}")

# =============================================================================
# 2. 矩阵运算
# =============================================================================
print("\n" + "=" * 60)
print("【2. 矩阵运算】")

A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"A:\n{A}")
print(f"B:\n{B}")

# 矩阵乘法
print(f"\n矩阵乘法 A @ B:\n{A @ B}")
print(f"torch.matmul(A, B):\n{torch.matmul(A, B)}")
print(f"torch.mm(A, B):\n{torch.mm(A, B)}")  # 只支持 2D

# 批量矩阵乘法
batch_A = torch.randn(2, 3, 4)  # batch=2, 3x4 矩阵
batch_B = torch.randn(2, 4, 5)  # batch=2, 4x5 矩阵
batch_C = torch.bmm(batch_A, batch_B)  # 结果: 2x3x5
print(f"\n批量矩阵乘法:")
print(f"  batch_A: {batch_A.shape}")
print(f"  batch_B: {batch_B.shape}")
print(f"  batch_C = bmm(A, B): {batch_C.shape}")

# 向量点积
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
print(f"\n向量点积:")
print(f"  v1 · v2 = {torch.dot(v1, v2)}")

# 转置
print(f"\nA 的转置:\n{A.T}")
print(f"等价于 A.transpose(0, 1):\n{A.transpose(0, 1)}")

# =============================================================================
# 3. 聚合运算
# =============================================================================
print("\n" + "=" * 60)
print("【3. 聚合运算】")

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(f"x:\n{x}")

print(f"\n全局聚合:")
print(f"  sum: {x.sum()}")
print(f"  mean: {x.mean()}")
print(f"  max: {x.max()}")
print(f"  min: {x.min()}")
print(f"  std: {x.std()}")

print(f"\n按维度聚合 (dim=0, 沿行):")
print(f"  sum: {x.sum(dim=0)}")
print(f"  mean: {x.mean(dim=0)}")

print(f"\n按维度聚合 (dim=1, 沿列):")
print(f"  sum: {x.sum(dim=1)}")
print(f"  mean: {x.mean(dim=1)}")

# 保持维度
print(f"\nkeepdim=True:")
print(f"  sum(dim=1, keepdim=True):\n{x.sum(dim=1, keepdim=True)}")

# argmax/argmin
print(f"\nargmax: {x.argmax()}")
print(f"argmax(dim=1): {x.argmax(dim=1)}")

# =============================================================================
# 4. 广播机制
# =============================================================================
print("\n" + "=" * 60)
print("【4. 广播机制 (Broadcasting)】")

print("""
广播规则：
1. 从后往前对齐维度
2. 维度为 1 的可以广播
3. 不存在的维度视为 1
""")

# 标量广播
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = 10
print(f"矩阵 + 标量:\n{a + b}")

# 向量广播
row = torch.tensor([1, 2, 3])
print(f"\na:\n{a}")
print(f"row: {row}")
print(f"a + row (行广播):\n{a + row}")

col = torch.tensor([[10], [20]])
print(f"\ncol:\n{col}")
print(f"a + col (列广播):\n{a + col}")

# 复杂广播
x = torch.randn(3, 4, 5)
y = torch.randn(4, 5)
z = x + y  # y 自动扩展为 (1, 4, 5) 然后广播
print(f"\n复杂广播:")
print(f"  x: {x.shape} + y: {y.shape} = z: {z.shape}")

# =============================================================================
# 5. 索引和切片
# =============================================================================
print("\n" + "=" * 60)
print("【5. 索引和切片】")

x = torch.arange(12).reshape(3, 4)
print(f"x:\n{x}")

print(f"\n基本索引:")
print(f"  x[0]: {x[0]}")
print(f"  x[0, 0]: {x[0, 0]}")
print(f"  x[-1]: {x[-1]}")

print(f"\n切片:")
print(f"  x[:2]:\n{x[:2]}")
print(f"  x[:, 1:3]:\n{x[:, 1:3]}")
print(f"  x[1:, 2:]:\n{x[1:, 2:]}")

print(f"\n步长切片:")
print(f"  x[::2]:\n{x[::2]}")
print(f"  x[:, ::2]:\n{x[:, ::2]}")

# 布尔索引
print(f"\n布尔索引 (x > 5):")
print(f"  mask: {x > 5}")
print(f"  x[x > 5]: {x[x > 5]}")

# 花式索引
indices = torch.tensor([0, 2])
print(f"\n花式索引:")
print(f"  x[indices]: {x[indices]}")

# =============================================================================
# 6. 形状变换
# =============================================================================
print("\n" + "=" * 60)
print("【6. 形状变换】")

x = torch.arange(12)
print(f"原始 x: {x}, 形状: {x.shape}")

# reshape
y = x.reshape(3, 4)
print(f"\nreshape(3, 4):\n{y}")

# view (要求内存连续)
z = x.view(2, 6)
print(f"view(2, 6):\n{z}")

# -1 自动推断
w = x.reshape(3, -1)
print(f"reshape(3, -1):\n{w}")

# squeeze 和 unsqueeze
print("\n" + "-" * 40)
a = torch.randn(1, 3, 1, 4)
print(f"原始形状: {a.shape}")
print(f"squeeze(): {a.squeeze().shape}")           # 移除所有 1
print(f"squeeze(0): {a.squeeze(0).shape}")         # 移除第 0 维的 1
print(f"squeeze(2): {a.squeeze(2).shape}")         # 移除第 2 维的 1

b = torch.randn(3, 4)
print(f"\n原始形状: {b.shape}")
print(f"unsqueeze(0): {b.unsqueeze(0).shape}")     # 在第 0 维添加 1
print(f"unsqueeze(1): {b.unsqueeze(1).shape}")     # 在第 1 维添加 1
print(f"unsqueeze(-1): {b.unsqueeze(-1).shape}")   # 在最后添加 1

# flatten
c = torch.randn(2, 3, 4)
print(f"\nflatten:")
print(f"  原始: {c.shape}")
print(f"  flatten(): {c.flatten().shape}")
print(f"  flatten(1): {c.flatten(1).shape}")  # 从第 1 维开始展平

# permute (维度重排)
d = torch.randn(2, 3, 4)
print(f"\npermute:")
print(f"  原始: {d.shape}")
print(f"  permute(2, 0, 1): {d.permute(2, 0, 1).shape}")

# =============================================================================
# 7. 拼接和分割
# =============================================================================
print("\n" + "=" * 60)
print("【7. 拼接和分割】")

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

print(f"a:\n{a}")
print(f"b:\n{b}")

# cat (沿现有维度拼接)
print(f"\ntorch.cat([a, b], dim=0):\n{torch.cat([a, b], dim=0)}")
print(f"torch.cat([a, b], dim=1):\n{torch.cat([a, b], dim=1)}")

# stack (创建新维度)
print(f"\ntorch.stack([a, b], dim=0):\n{torch.stack([a, b], dim=0)}")
print(f"  形状: {torch.stack([a, b], dim=0).shape}")

# split
x = torch.arange(10)
print(f"\nsplit:")
print(f"  x = {x}")
parts = torch.split(x, 3)  # 每份 3 个
print(f"  split(x, 3): {[p.tolist() for p in parts]}")

# chunk
chunks = torch.chunk(x, 3)  # 分成 3 份
print(f"  chunk(x, 3): {[c.tolist() for c in chunks]}")

# =============================================================================
# 8. 练习题
# =============================================================================
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

# === 练习答案 ===
# 1
# A = torch.randn(3, 3)
# B = torch.randn(3, 3)
# C = A @ B
# print(f"A @ B:\n{C}")

# 2
# x = torch.randn(3, 4)
# row_mean = x.mean(dim=1)
# col_max = x.max(dim=0).values
# print(f"每行均值: {row_mean}")
# print(f"每列最大: {col_max}")

# 3
# x = torch.randn(2, 3, 4)
# y = x.reshape(4, 6)
# print(f"{x.shape} -> {y.shape}")

# 4
# matrix = torch.randn(3, 4)
# bias = torch.tensor([[1], [2], [3]])  # (3, 1)
# result = matrix + bias
# print(result)

# 5
# a, b, c = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
# stacked = torch.stack([a, b, c], dim=0)
# print(f"形状: {stacked.shape}")  # (3, 2, 3)

print("\n✅ Tensor 运算完成！")
print("下一步：03-tensor-autograd.py - 自动微分")
