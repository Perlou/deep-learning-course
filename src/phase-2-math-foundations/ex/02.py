"""
02-eigenvalue-svd.py
Phase 2: 深度学习数学基础

特征分解与奇异值分解 (SVD)

学习目标：
1. 理解特征值和特征向量的几何意义
2. 掌握 SVD 分解及其应用
3. 理解 PCA 的数学原理
4. 了解低秩近似在深度学习中的应用（如 LoRA）
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("深度学习数学基础 - 特征分解与 SVD")
print("=" * 60)

print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 对矩阵 A = [[3, 1], [1, 3]] 进行特征分解：
   - 计算特征值和特征向量
   - 验证 A = Q Λ Q^T
   
2. 使用 SVD 对以下矩阵进行秩-2 近似：
   A = [[1, 2, 3, 4],
        [2, 4, 6, 8],
        [3, 6, 9, 12],
        [1, 1, 1, 1]]
   
3. 实现一个简化的 PCA：
   - 生成 100 个 5 维数据点
   - 降维到 2 维
   - 计算保留了多少方差

4. 计算 LoRA 参数量：
   - 原始权重: 4096 × 4096
   - LoRA 秩: r = 16
   - 计算参数压缩比
""")

# 1
A = np.array([[3, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"矩阵 A:\n{A}")
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

Q = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = Q @ Lambda @ Q.T

print(f"验证 A = Q @ Λ @ Q^T:\n{np.round(A_reconstructed, 6)}")

# 2

A = np.array([[1, 2, 3, 4],
              [2, 4, 6, 8],
              [3, 6, 9, 12],
              [1, 1, 1, 1]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"奇异值: {s}")
k = 2
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
print(f"秩-2 近似:\n{np.round(A_k, 4)}")
print(f"相对误差: {np.linalg.norm(A - A_k) / np.linalg.norm(A):.6f}")

# 3
np.random.seed(42)
X = np.random.randn(100, 5)
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T
k = 2
X_2d = X_centered @ V[:, :k]
print(f"原始数据: {X.shape}")
print(f"降维后: {X_2d.shape}")
var_explained = np.sum(s[:k]**2) / np.sum(s**2)
print(f"保留方差比例: {var_explained:.4f} ({var_explained*100:.2f}%)")

# 4
d, k = 4096, 4096
r = 16
original_params = d * k
lora_params = d * r + r * k
compression = original_params / lora_params
print(f"原始权重: {d}×{k} = {original_params:,} 参数")
print(f"LoRA 参数: A({d}×{r}) + B({r}×{k}) = {lora_params:,} 参数")
print(f"压缩比: {compression:.1f}x")

