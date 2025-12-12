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

# =============================================================================
# 1. 特征值与特征向量
# =============================================================================
print("\n【1. 特征值与特征向量】")

print("""
定义：对于方阵 A，如果存在非零向量 v 和标量 λ，使得：
      A @ v = λ × v
则 λ 是特征值，v 是对应的特征向量。

几何意义：
- 特征向量是在变换 A 作用下只发生缩放（不改变方向）的向量
- 特征值是缩放的倍数
""")

# 示例
A = np.array([[4, 2],
              [1, 3]])
print(f"矩阵 A:\n{A}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\n特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证 A @ v = λ × v
print("\n验证 A @ v = λ × v:")
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    λ = eigenvalues[i]
    Av = A @ v
    λv = λ * v
    print(f"  λ_{i+1} = {λ:.2f}")
    print(f"  A @ v_{i+1} = {Av}")
    print(f"  λ × v_{i+1} = {λv}")
    print()

# 可视化特征向量
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 原始空间
ax = axes[0]
# 绘制网格
for i in np.linspace(-2, 2, 9):
    ax.axhline(y=i, color='lightgray', linewidth=0.5)
    ax.axvline(x=i, color='lightgray', linewidth=0.5)
# 绘制特征向量
colors = ['red', 'blue']
for i in range(2):
    v = eigenvectors[:, i]
    ax.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.05, 
             fc=colors[i], ec=colors[i], label=f'v_{i+1} (λ={eigenvalues[i]:.2f})')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_title('原始特征向量', fontsize=12, fontweight='bold')
ax.legend()

# 变换后
ax = axes[1]
for i in np.linspace(-2, 2, 9):
    ax.axhline(y=i, color='lightgray', linewidth=0.5)
    ax.axvline(x=i, color='lightgray', linewidth=0.5)
for i in range(2):
    v = eigenvectors[:, i]
    v_transformed = A @ v
    ax.arrow(0, 0, v_transformed[0], v_transformed[1], head_width=0.1, head_length=0.05,
             fc=colors[i], ec=colors[i], label=f'A @ v_{i+1}')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.set_title('变换后的特征向量（只缩放，不旋转）', fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/02_eigenvectors.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/02_eigenvectors.png")

# =============================================================================
# 2. 对称矩阵的特征分解
# =============================================================================
print("\n" + "=" * 60)
print("【2. 对称矩阵的特征分解】")

# 对称矩阵的特殊性质
A = np.array([[4, 2],
              [2, 3]])
print(f"对称矩阵 A:\n{A}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\n特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 对称矩阵的特征向量正交
print(f"\n特征向量点积（验证正交性）: {eigenvectors[:, 0] @ eigenvectors[:, 1]:.6f}")

# 特征分解: A = Q Λ Q^T
Q = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = Q @ Lambda @ Q.T

print(f"\n特征分解 A = Q Λ Q^T:")
print(f"Q:\n{Q}")
print(f"Λ:\n{Lambda}")
print(f"重构 A:\n{A_reconstructed}")

print("""
💡 深度学习中的应用：
   - 协方差矩阵是对称的 → 特征向量正交
   - PCA 就是对协方差矩阵做特征分解
   - Hessian 矩阵的特征值反映损失函数的曲率
""")

# =============================================================================
# 3. 奇异值分解 (SVD)
# =============================================================================
print("\n" + "=" * 60)
print("【3. 奇异值分解 (SVD)】")

print("""
SVD 将任意矩阵 A (m × n) 分解为：
    A = U @ Σ @ V^T

其中：
- U: 左奇异向量矩阵 (m × m)，列向量正交
- Σ: 奇异值对角矩阵 (m × n)，对角线元素非负递减
- V^T: 右奇异向量矩阵 (n × n)，行向量正交
""")

# 示例
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
print(f"矩阵 A ({A.shape[0]}×{A.shape[1]}):\n{A}")

U, s, Vt = np.linalg.svd(A)
print(f"\nU ({U.shape[0]}×{U.shape[1]}):\n{U}")
print(f"\n奇异值 σ: {s}")
print(f"\nV^T ({Vt.shape[0]}×{Vt.shape[1]}):\n{Vt}")

# 重构矩阵
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, s)
A_reconstructed = U @ Sigma @ Vt
print(f"\n重构 A = U @ Σ @ V^T:\n{A_reconstructed}")

# =============================================================================
# 4. 低秩近似
# =============================================================================
print("\n" + "=" * 60)
print("【4. 低秩近似】")

print("""
SVD 的重要应用：用前 k 个奇异值近似原矩阵
    A ≈ U_k @ Σ_k @ V_k^T

这是最优的秩-k 近似（最小化 Frobenius 范数误差）
""")

# 创建一个有意义的矩阵进行低秩近似
np.random.seed(42)
# 真实数据是低秩的（秩为 2）加噪声
true_rank = 2
A = np.random.randn(10, 2) @ np.random.randn(2, 8) + 0.1 * np.random.randn(10, 8)
print(f"原始矩阵形状: {A.shape}")
print(f"原始矩阵秩: {np.linalg.matrix_rank(A)}")

U, s, Vt = np.linalg.svd(A, full_matrices=False)

# 计算不同秩的近似误差
errors = []
for k in range(1, len(s) + 1):
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
    errors.append(error)
    if k <= 3:
        print(f"秩-{k} 近似相对误差: {error:.4f}")

# 可视化奇异值和误差
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 奇异值
axes[0].bar(range(1, len(s) + 1), s, color='steelblue', edgecolor='black')
axes[0].set_xlabel('奇异值索引')
axes[0].set_ylabel('奇异值大小')
axes[0].set_title('奇异值分布', fontsize=12, fontweight='bold')

# 累积方差解释
cumulative_var = np.cumsum(s**2) / np.sum(s**2)
axes[1].plot(range(1, len(s) + 1), cumulative_var, 'bo-', linewidth=2, markersize=8)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% 方差')
axes[1].set_xlabel('保留的奇异值数量')
axes[1].set_ylabel('累积方差解释比例')
axes[1].set_title('低秩近似的方差解释', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/02_svd_lowrank.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: outputs/02_svd_lowrank.png")

# =============================================================================
# 5. PCA 降维
# =============================================================================
print("\n" + "=" * 60)
print("【5. PCA 降维】")

print("""
PCA (主成分分析) 步骤：
1. 数据中心化：X̃ = X - mean(X)
2. 计算协方差矩阵：C = X̃^T @ X̃ / (n-1)
3. 对 C 做特征分解（或对 X̃ 做 SVD）
4. 选择前 k 个主成分
5. 投影：Z = X̃ @ V_k
""")

# 生成 2D 数据
np.random.seed(42)
n_samples = 200
# 创建有相关性的数据
mean = [3, 5]
cov = [[2, 1.5], [1.5, 1.5]]
X = np.random.multivariate_normal(mean, cov, n_samples)

print(f"原始数据形状: {X.shape}")

# PCA 实现
# 1. 中心化
X_centered = X - X.mean(axis=0)

# 2. SVD（计算主成分）
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T  # 主成分方向

print(f"奇异值: {s}")
print(f"方差解释比例: {s**2 / np.sum(s**2)}")
print(f"第一主成分方向: {V[:, 0]}")

# 3. 投影到第一主成分
X_1d = X_centered @ V[:, 0]
print(f"降维后数据形状: {X_1d.shape}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原始数据和主成分
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], alpha=0.5, c='steelblue')
# 绘制主成分方向
origin = X.mean(axis=0)
for i in range(2):
    ax.arrow(origin[0], origin[1], V[0, i]*s[i]/10, V[1, i]*s[i]/10,
             head_width=0.1, head_length=0.05, fc=['red', 'green'][i], ec=['red', 'green'][i],
             label=f'PC{i+1} ({s[i]**2/np.sum(s**2)*100:.1f}%)')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('原始数据与主成分方向', fontsize=12, fontweight='bold')
ax.legend()
ax.set_aspect('equal')

# 投影到第一主成分
ax = axes[1]
# 重构点
X_reconstructed = X_centered @ V[:, :1] @ V[:, :1].T + X.mean(axis=0)
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, c='steelblue', label='原始')
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.5, c='red', s=10, label='投影')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('投影到第一主成分', fontsize=12, fontweight='bold')
ax.legend()
ax.set_aspect('equal')

# 降维后的分布
ax = axes[2]
ax.hist(X_1d, bins=30, color='coral', edgecolor='white', alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('频次')
ax.set_title('降维后的一维分布', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/02_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/02_pca.png")

# =============================================================================
# 6. LoRA: 深度学习中的低秩适配
# =============================================================================
print("\n" + "=" * 60)
print("【6. LoRA: 深度学习中的低秩适配】")

print("""
LoRA (Low-Rank Adaptation) 是一种参数高效微调方法：
- 原始权重: W (d × k)
- 微调时不直接更新 W，而是学习低秩分解: ΔW = A @ B
  - A: (d × r)
  - B: (r × k)
  - r << min(d, k)
  
优势：
- 参数量从 d×k 降到 d×r + r×k
- 例如 d=4096, k=4096, r=8
  - 原始: 16M 参数
  - LoRA: 65K 参数 (减少 250 倍!)
""")

# 模拟 LoRA
d, k = 1024, 512
r = 4  # 低秩

# 原始权重
W_original = np.random.randn(d, k) * 0.01

# LoRA 增量
A = np.random.randn(d, r) * 0.01
B = np.random.randn(r, k) * 0.01
delta_W = A @ B

# 微调后的权重
W_finetuned = W_original + delta_W

print(f"原始权重 W: {d}×{k} = {d*k:,} 参数")
print(f"LoRA 参数: A({d}×{r}) + B({r}×{k}) = {d*r + r*k:,} 参数")
print(f"参数压缩比: {d*k / (d*r + r*k):.1f}x")

# 验证 delta_W 的秩
print(f"\nΔW 的秩: {np.linalg.matrix_rank(delta_W)}")
print(f"ΔW 的形状: {delta_W.shape}")

# =============================================================================
# 7. 练习题
# =============================================================================
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

# === 练习代码 ===
# 练习 1: 对称矩阵特征分解
# A = np.array([[3, 1], [1, 3]])
# eigenvalues, eigenvectors = np.linalg.eig(A)
# print(f"矩阵 A:\n{A}")
# print(f"特征值: {eigenvalues}")
# print(f"特征向量:\n{eigenvectors}")
# # 验证 A = Q Λ Q^T
# Q = eigenvectors
# Lambda = np.diag(eigenvalues)
# A_reconstructed = Q @ Lambda @ Q.T
# print(f"验证 A = Q @ Λ @ Q^T:\n{np.round(A_reconstructed, 6)}")

# 练习 2: SVD 秩-2 近似
# A = np.array([[1, 2, 3, 4],
#               [2, 4, 6, 8],
#               [3, 6, 9, 12],
#               [1, 1, 1, 1]])
# U, s, Vt = np.linalg.svd(A, full_matrices=False)
# print(f"奇异值: {s}")
# # 秩-2 近似
# k = 2
# A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
# print(f"秩-2 近似:\n{np.round(A_k, 4)}")
# print(f"相对误差: {np.linalg.norm(A - A_k) / np.linalg.norm(A):.6f}")

# 练习 3: PCA 降维
# np.random.seed(42)
# X = np.random.randn(100, 5)  # 100 samples, 5 features
# X_centered = X - X.mean(axis=0)
# U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
# V = Vt.T
# # 降到 2 维
# k = 2
# X_2d = X_centered @ V[:, :k]
# print(f"原始数据: {X.shape}")
# print(f"降维后: {X_2d.shape}")
# var_explained = np.sum(s[:k]**2) / np.sum(s**2)
# print(f"保留方差比例: {var_explained:.4f} ({var_explained*100:.2f}%)")

# 练习 4: LoRA 参数量
# d, k = 4096, 4096
# r = 16
# original_params = d * k
# lora_params = d * r + r * k
# compression = original_params / lora_params
# print(f"原始权重: {d}×{k} = {original_params:,} 参数")
# print(f"LoRA 参数: A({d}×{r}) + B({r}×{k}) = {lora_params:,} 参数")
# print(f"压缩比: {compression:.1f}x")

print("\n✅ 特征分解与 SVD 完成！")
print("下一步：03-derivatives-gradients.py - 偏导数与梯度")
