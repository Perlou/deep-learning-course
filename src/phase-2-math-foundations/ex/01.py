"""
01-vectors-matrices.py
Phase 2: 深度学习数学基础

向量与矩阵运算 - 深度学习的核心语言

学习目标：
1. 理解向量空间的基本概念
2. 掌握矩阵运算及其几何意义
3. 理解线性变换在深度学习中的作用
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("深度学习数学基础 - 向量与矩阵")
print("=" * 60)

print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 计算向量 a = [1, 2, 3] 和 b = [4, -5, 6] 的点积和夹角

2. 创建一个 3x3 的旋转矩阵（绕 z 轴旋转 30°），并验证它是正交矩阵

3. 对于矩阵 A = [[2, 1], [1, 3]]：
   - 计算行列式
   - 计算逆矩阵
   - 验证 A @ A⁻¹ = I

4. 模拟一个简单的全连接层：
   - 输入: 5 个样本，每个 4 维特征
   - 输出: 3 维
   - 实现 Y = XW + b

请在下方编写代码完成练习...
""")

# 1
a = np.array([1, 2, 4])
b = np.array([4, -5, 6])
dot = np.dot(a, b)
cos_theta = dot / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.arccos(cos_theta)
print(f"点积: {dot}")
print(f"夹角: {np.degrees(theta):.2f}")

# 2
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta), np.cos(theta), 0],
              [0, 0, 1]])
print(f"旋转矩阵:\n{R}")
# 验证正交性: R @ R^T = I
print(f"R @ R^T:\n{R @ R.T}")
print(f"det(R) = {np.linalg.det(R):.4f}")  # 应该为 1

# 3
A = np.array([[2, 1], [1, 3]])
print(f"det(A) = {np.linalg.det(A):.2f}")
A_inv = np.linalg.inv(A)

print(f"A⁻¹:\n{A_inv}")
print(f"验证:\n{A @ A_inv}")

# 4
np.random.seed(42)
X = np.random.randn(5, 4)
W = np.random.randn(4, 3)
b = np.random.randn(3)
Y = X @ W + b
print(f"X shape: {X.shape}")
print(f"W shape: {W.shape}")
print(f"Y shape: {Y.shape}")
