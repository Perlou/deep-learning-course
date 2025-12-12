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

# =============================================================================
# 1. 向量基础
# =============================================================================
print("\n【1. 向量基础】")

# 1.1 向量表示
v = np.array([3, 4])
print(f"向量 v = {v}")
print(f"向量维度: {v.shape}")

# 向量长度（L2 范数）
norm = np.linalg.norm(v)
print(f"向量长度 ||v|| = √(3² + 4²) = {norm}")

# 单位向量
unit_v = v / norm
print(f"单位向量 v̂ = {unit_v}")
print(f"单位向量长度 = {np.linalg.norm(unit_v):.6f}")

# 1.2 向量运算
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")       # 加法
print(f"a - b = {a - b}")       # 减法
print(f"2 * a = {2 * a}")       # 标量乘法

# 1.3 点积（内积）
dot_product = np.dot(a, b)  # 或 a @ b
print(f"\n点积 a · b = {dot_product}")
print(f"计算过程: 1×4 + 2×5 + 3×6 = {1*4 + 2*5 + 3*6}")

# 点积的几何意义：投影
print("""
💡 点积的意义：
   a · b = ||a|| × ||b|| × cos(θ)
   - 正值：夹角 < 90°
   - 零值：夹角 = 90°（垂直）
   - 负值：夹角 > 90°
   
   在深度学习中：
   - 神经网络的线性层就是点积运算
   - 注意力机制使用点积计算相似度
""")

# 计算夹角
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.arccos(cos_theta)
print(f"向量夹角 θ = {np.degrees(theta):.2f}°")

# =============================================================================
# 2. 矩阵基础
# =============================================================================
print("\n" + "=" * 60)
print("【2. 矩阵基础】")

# 2.1 矩阵创建
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"矩阵 A (2×3):\n{A}")
print(f"形状: {A.shape}")

# 2.2 特殊矩阵
print("\n特殊矩阵:")
I = np.eye(3)
print(f"单位矩阵 I (3×3):\n{I}")

zeros = np.zeros((2, 3))
print(f"\n零矩阵:\n{zeros}")

diag = np.diag([1, 2, 3])
print(f"\n对角矩阵:\n{diag}")

# 2.3 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"\nA:\n{A}")
print(f"B:\n{B}")
print(f"\nA + B:\n{A + B}")
print(f"A * B (元素级):\n{A * B}")
print(f"A @ B (矩阵乘法):\n{A @ B}")

# 矩阵乘法详解
print("""
💡 矩阵乘法 C = A @ B:
   C[i,j] = Σ A[i,k] × B[k,j]
   
   形状规则：
   (m × n) @ (n × p) = (m × p)
   
   在深度学习中：
   - 全连接层: y = Wx + b
   - 注意力: Attention = softmax(QK^T/√d)V
""")

# =============================================================================
# 3. 线性变换
# =============================================================================
print("\n" + "=" * 60)
print("【3. 线性变换】")

# 矩阵作为线性变换
# 旋转矩阵（逆时针旋转 θ 度）
theta = np.pi / 4  # 45度
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
print(f"旋转矩阵 (45°):\n{R}")

# 对向量应用旋转
v = np.array([1, 0])
v_rotated = R @ v
print(f"\n原向量: {v}")
print(f"旋转后: {v_rotated}")

# 缩放矩阵
S = np.array([[2, 0],
              [0, 0.5]])
print(f"\n缩放矩阵 (x方向2倍, y方向0.5倍):\n{S}")

# 常见线性变换可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原始网格点
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()])

# 定义变换矩阵
transforms = {
    '原始': np.eye(2),
    '旋转 45°': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                         [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    '缩放': np.array([[2, 0], [0, 0.5]]),
    '剪切': np.array([[1, 0.5], [0, 1]]),
    '反射 (y轴)': np.array([[-1, 0], [0, 1]]),
    '组合变换': np.array([[1, 1], [0, 1]]) @ np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                                                       [np.sin(np.pi/6), np.cos(np.pi/6)]])
}

for ax, (name, T) in zip(axes.ravel(), transforms.items()):
    transformed = T @ points
    ax.scatter(transformed[0], transformed[1], c='blue', s=30)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(name, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_linear_transforms.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/01_linear_transforms.png")

# =============================================================================
# 4. 矩阵的逆与行列式
# =============================================================================
print("\n" + "=" * 60)
print("【4. 矩阵的逆与行列式】")

A = np.array([[4, 7], [2, 6]])
print(f"矩阵 A:\n{A}")

# 行列式
det = np.linalg.det(A)
print(f"\n行列式 det(A) = {det:.2f}")
print("行列式的意义: 线性变换后面积的缩放因子")

# 逆矩阵
A_inv = np.linalg.inv(A)
print(f"\n逆矩阵 A⁻¹:\n{A_inv}")

# 验证
print(f"\n验证 A @ A⁻¹ = I:\n{A @ A_inv}")

print("""
💡 深度学习中的应用：
   - 行列式：检测矩阵是否可逆（梯度是否会消失）
   - 逆矩阵：求解线性方程组（正规方程）
   - 注意：深度学习更多使用梯度下降而非直接求逆
""")

# =============================================================================
# 5. 矩阵的秩
# =============================================================================
print("\n" + "=" * 60)
print("【5. 矩阵的秩】")

# 满秩矩阵
A = np.array([[1, 2], [3, 4]])
rank_A = np.linalg.matrix_rank(A)
print(f"矩阵 A:\n{A}")
print(f"秩 = {rank_A} (满秩)")

# 降秩矩阵
B = np.array([[1, 2, 3], [2, 4, 6], [1, 2, 3]])
rank_B = np.linalg.matrix_rank(B)
print(f"\n矩阵 B:\n{B}")
print(f"秩 = {rank_B} (降秩，行向量线性相关)")

print("""
💡 秩在深度学习中的意义：
   - 低秩矩阵：LoRA 微调的核心思想
   - 矩阵分解：将大矩阵分解为低秩矩阵的乘积
   - 维度瓶颈：自编码器的压缩层
""")

# =============================================================================
# 6. 深度学习中的矩阵运算
# =============================================================================
print("\n" + "=" * 60)
print("【6. 深度学习中的矩阵运算】")

# 6.1 全连接层
print("\n6.1 全连接层 (Linear Layer)")
batch_size = 4
input_dim = 3
output_dim = 2

# 输入 X: (batch_size, input_dim)
X = np.random.randn(batch_size, input_dim)
# 权重 W: (input_dim, output_dim)
W = np.random.randn(input_dim, output_dim)
# 偏置 b: (output_dim,)
b = np.random.randn(output_dim)

# 前向传播: Y = XW + b
Y = X @ W + b

print(f"输入 X 形状: {X.shape}")
print(f"权重 W 形状: {W.shape}")
print(f"输出 Y 形状: {Y.shape}")
print(f"\nY = X @ W + b")

# 6.2 批量矩阵乘法
print("\n6.2 批量矩阵乘法 (Batch Matrix Multiplication)")
batch = 2
seq_len = 4
d_model = 3

# 模拟注意力计算中的 Q @ K^T
Q = np.random.randn(batch, seq_len, d_model)
K = np.random.randn(batch, seq_len, d_model)

# 批量转置和矩阵乘
attention_scores = Q @ K.transpose(0, 2, 1)  # (batch, seq, d) @ (batch, d, seq)
print(f"Q 形状: {Q.shape}")
print(f"K 形状: {K.shape}")
print(f"K^T 形状: {K.transpose(0, 2, 1).shape}")
print(f"Q @ K^T 形状: {attention_scores.shape}")

# =============================================================================
# 7. 练习题
# =============================================================================
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

# === 练习代码 ===
# 练习 1
# a = np.array([1, 2, 3])
# b = np.array([4, -5, 6])
# dot = np.dot(a, b)
# cos_theta = dot / (np.linalg.norm(a) * np.linalg.norm(b))
# theta = np.arccos(cos_theta)
# print(f"点积: {dot}")
# print(f"夹角: {np.degrees(theta):.2f}°")

# 练习 2
# theta = np.radians(30)
# R = np.array([[np.cos(theta), -np.sin(theta), 0],
#               [np.sin(theta), np.cos(theta), 0],
#               [0, 0, 1]])
# print(f"旋转矩阵:\n{R}")
# # 验证正交性: R @ R^T = I
# print(f"R @ R^T:\n{R @ R.T}")
# print(f"det(R) = {np.linalg.det(R):.4f}")  # 应该为 1

# 练习 3
# A = np.array([[2, 1], [1, 3]])
# print(f"det(A) = {np.linalg.det(A):.2f}")
# A_inv = np.linalg.inv(A)
# print(f"A⁻¹:\n{A_inv}")
# print(f"验证:\n{A @ A_inv}")

# 练习 4
# np.random.seed(42)
# X = np.random.randn(5, 4)  # 5 samples, 4 features
# W = np.random.randn(4, 3)  # 4 input, 3 output
# b = np.random.randn(3)     # 3 bias
# Y = X @ W + b
# print(f"X shape: {X.shape}")
# print(f"W shape: {W.shape}")
# print(f"Y shape: {Y.shape}")

print("\n✅ 向量与矩阵基础完成！")
print("下一步：02-eigenvalue-svd.py - 特征分解与 SVD")
