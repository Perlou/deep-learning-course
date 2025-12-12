"""
03-derivatives-gradients.py
Phase 2: 深度学习数学基础

偏导数与梯度 - 深度学习的优化基础

学习目标：
1. 理解导数和偏导数的概念
2. 掌握梯度的定义和几何意义
3. 理解梯度与函数优化的关系
4. 为反向传播打下基础
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("深度学习数学基础 - 偏导数与梯度")
print("=" * 60)

# =============================================================================
# 1. 导数基础
# =============================================================================
print("\n【1. 导数基础】")

print("""
导数的定义：
    f'(x) = lim[h→0] (f(x+h) - f(x)) / h

几何意义：函数在某点的切线斜率
物理意义：变化率

常见导数公式：
    d/dx (x^n) = n × x^(n-1)
    d/dx (e^x) = e^x
    d/dx (ln x) = 1/x
    d/dx (sin x) = cos x
    d/dx (cos x) = -sin x
""")

# 数值求导
def numerical_derivative(f, x, h=1e-7):
    """使用中心差分法求导"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 示例函数
def f(x):
    return x**3 - 2*x**2 + x

def f_derivative(x):
    """解析导数"""
    return 3*x**2 - 4*x + 1

x = 2.0
print(f"函数: f(x) = x³ - 2x² + x")
print(f"在 x = {x} 处:")
print(f"  解析导数: f'(x) = 3x² - 4x + 1 = {f_derivative(x)}")
print(f"  数值导数: {numerical_derivative(f, x):.6f}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-1, 3, 100)
y = f(x)
dy = f_derivative(x)

ax.plot(x, y, 'b-', linewidth=2, label='f(x) = x³ - 2x² + x')
ax.plot(x, dy, 'r--', linewidth=2, label="f'(x) = 3x² - 4x + 1")
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# 标注切线
x0 = 2
y0 = f(x0)
slope = f_derivative(x0)
tangent = slope * (x - x0) + y0
ax.plot(x, tangent, 'g:', linewidth=2, label=f'x={x0}处切线 (斜率={slope})')
ax.scatter([x0], [y0], color='green', s=100, zorder=5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('函数与其导数', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 15)

plt.tight_layout()
plt.savefig('outputs/03_derivative.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: outputs/03_derivative.png")

# =============================================================================
# 2. 偏导数
# =============================================================================
print("\n" + "=" * 60)
print("【2. 偏导数】")

print("""
对于多变量函数 f(x, y)，偏导数定义为：
    ∂f/∂x = lim[h→0] (f(x+h, y) - f(x, y)) / h
    ∂f/∂y = lim[h→0] (f(x, y+h) - f(x, y)) / h

偏导数：固定其他变量，对一个变量求导
""")

def g(x, y):
    """示例二元函数"""
    return x**2 + 2*x*y + y**2

def g_dx(x, y):
    """对 x 的偏导数"""
    return 2*x + 2*y

def g_dy(x, y):
    """对 y 的偏导数"""
    return 2*x + 2*y

def numerical_partial(f, x, y, var='x', h=1e-7):
    """数值偏导数"""
    if var == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    else:
        return (f(x, y + h) - f(x, y - h)) / (2 * h)

x0, y0 = 1.0, 2.0
print(f"函数: g(x, y) = x² + 2xy + y²")
print(f"在点 ({x0}, {y0}) 处:")
print(f"  ∂g/∂x = 2x + 2y = {g_dx(x0, y0)}")
print(f"  ∂g/∂y = 2x + 2y = {g_dy(x0, y0)}")
print(f"  数值验证 ∂g/∂x: {numerical_partial(g, x0, y0, 'x'):.6f}")
print(f"  数值验证 ∂g/∂y: {numerical_partial(g, x0, y0, 'y'):.6f}")

# =============================================================================
# 3. 梯度
# =============================================================================
print("\n" + "=" * 60)
print("【3. 梯度】")

print("""
梯度是偏导数组成的向量：
    ∇f(x, y) = [∂f/∂x, ∂f/∂y]

几何意义：
- 梯度方向：函数增长最快的方向
- 梯度大小：最大增长率
- 负梯度方向：函数下降最快的方向（梯度下降的核心！）
""")

def h(x, y):
    """碗状函数"""
    return x**2 + y**2

def grad_h(x, y):
    """梯度"""
    return np.array([2*x, 2*y])

# 可视化梯度场
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 等高线图 + 梯度场
ax = axes[0]
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = h(X, Y)

contour = ax.contour(X, Y, Z, levels=15, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# 绘制梯度向量
x_arrows = np.linspace(-2.5, 2.5, 8)
y_arrows = np.linspace(-2.5, 2.5, 8)
X_a, Y_a = np.meshgrid(x_arrows, y_arrows)
U = 2 * X_a  # ∂f/∂x
V = 2 * Y_a  # ∂f/∂y

# 归一化箭头长度（仅表示方向）
magnitude = np.sqrt(U**2 + V**2)
U_norm = U / (magnitude + 1e-8) * 0.3
V_norm = V / (magnitude + 1e-8) * 0.3

ax.quiver(X_a, Y_a, U_norm, V_norm, color='red', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('等高线与梯度方向（红箭头）', fontsize=12, fontweight='bold')
ax.set_aspect('equal')

# 3D 表面
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('f(x, y) = x² + y²', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/03_gradient_field.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/03_gradient_field.png")

# =============================================================================
# 4. 梯度下降预览
# =============================================================================
print("\n" + "=" * 60)
print("【4. 梯度下降预览】")

print("""
梯度下降算法：
    x_new = x_old - lr × ∇f(x_old)

其中 lr 是学习率（步长）
""")

def gradient_descent_demo(f, grad_f, x0, lr=0.1, n_steps=20):
    """简单梯度下降演示"""
    path = [x0.copy()]
    x = x0.copy()
    
    for i in range(n_steps):
        grad = grad_f(x[0], x[1])
        x = x - lr * grad
        path.append(x.copy())
    
    return np.array(path)

# 运行梯度下降
x0 = np.array([2.5, 2.0])
path = gradient_descent_demo(h, grad_h, x0, lr=0.1, n_steps=20)

print(f"起点: {x0}")
print(f"终点: {path[-1]}")
print(f"最小值点: [0, 0]")

# 可视化梯度下降轨迹
fig, ax = plt.subplots(figsize=(10, 8))

x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = h(X, Y)

contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# 绘制路径
ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=8, linewidth=2, label='梯度下降轨迹')
ax.scatter(path[0, 0], path[0, 1], c='green', s=200, marker='*', zorder=5, label='起点')
ax.scatter(path[-1, 0], path[-1, 1], c='red', s=200, marker='*', zorder=5, label='终点')
ax.scatter(0, 0, c='blue', s=200, marker='x', zorder=5, label='最小值')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('梯度下降过程可视化', fontsize=14, fontweight='bold')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/03_gradient_descent.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/03_gradient_descent.png")

# =============================================================================
# 5. 深度学习中的梯度
# =============================================================================
print("\n" + "=" * 60)
print("【5. 深度学习中的梯度】")

print("""
在神经网络中：
- 目标：最小化损失函数 L(θ)，其中 θ 是所有参数
- 方法：计算 ∇L(θ)，然后更新参数

示例：线性回归的梯度
    模型: y = wx + b
    损失: L = (1/n) Σ(y_pred - y_true)²
    
    ∂L/∂w = (2/n) Σ (y_pred - y_true) × x
    ∂L/∂b = (2/n) Σ (y_pred - y_true)
""")

# 简单线性回归的梯度计算
np.random.seed(42)
n = 50
X = np.random.randn(n)
y_true = 2 * X + 1 + 0.2 * np.random.randn(n)

# 初始参数
w, b = 0.0, 0.0
lr = 0.1
n_epochs = 50

losses = []
w_history = [w]
b_history = [b]

for epoch in range(n_epochs):
    # 前向传播
    y_pred = w * X + b
    
    # 计算损失
    loss = np.mean((y_pred - y_true)**2)
    losses.append(loss)
    
    # 计算梯度
    dL_dw = 2 * np.mean((y_pred - y_true) * X)
    dL_db = 2 * np.mean(y_pred - y_true)
    
    # 更新参数
    w = w - lr * dL_dw
    b = b - lr * dL_db
    
    w_history.append(w)
    b_history.append(b)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(f"\n最终参数: w={w:.4f}, b={b:.4f}")
print(f"真实参数: w=2, b=1")

# 可视化训练过程
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 损失曲线
axes[0].plot(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('损失函数下降', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 参数收敛
axes[1].plot(w_history, 'r-', linewidth=2, label='w')
axes[1].plot(b_history, 'g-', linewidth=2, label='b')
axes[1].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='w_true=2')
axes[1].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='b_true=1')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('参数值')
axes[1].set_title('参数收敛过程', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 拟合结果
axes[2].scatter(X, y_true, alpha=0.6, label='数据')
x_line = np.linspace(-3, 3, 100)
axes[2].plot(x_line, w * x_line + b, 'r-', linewidth=2, label=f'拟合: y={w:.2f}x+{b:.2f}')
axes[2].plot(x_line, 2 * x_line + 1, 'g--', linewidth=2, label='真实: y=2x+1')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('线性回归拟合结果', fontsize=12, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/03_linear_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/03_linear_regression.png")

# =============================================================================
# 6. 高维梯度
# =============================================================================
print("\n" + "=" * 60)
print("【6. 高维梯度】")

print("""
神经网络通常有数百万参数：
    θ = [w1, w2, ..., wn, b1, b2, ..., bm]
    
梯度是同样维度的向量：
    ∇L(θ) = [∂L/∂w1, ∂L/∂w2, ..., ∂L/∂wn, ∂L/∂b1, ..., ∂L/∂bm]

实际计算：
- 不是手动求导，而是使用自动微分（autograd）
- PyTorch 的 backward() 自动计算梯度
""")

# 模拟高维参数的梯度
n_params = 1000
theta = np.random.randn(n_params)
grad = np.random.randn(n_params)  # 假设的梯度

print(f"参数数量: {n_params}")
print(f"参数向量范数: {np.linalg.norm(theta):.4f}")
print(f"梯度向量范数: {np.linalg.norm(grad):.4f}")

# 梯度裁剪（防止梯度爆炸）
max_norm = 1.0
grad_norm = np.linalg.norm(grad)
if grad_norm > max_norm:
    grad_clipped = grad * (max_norm / grad_norm)
    print(f"梯度裁剪后范数: {np.linalg.norm(grad_clipped):.4f}")

# =============================================================================
# 7. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 求以下函数的导数：
   - f(x) = 3x⁴ - 2x² + x - 5
   - g(x) = e^(2x) + ln(x)
   - h(x) = sin(x) × cos(x)

2. 对于函数 f(x, y) = x²y + xy² + 3x - 2y：
   - 计算 ∂f/∂x 和 ∂f/∂y
   - 计算在点 (1, 2) 处的梯度
   
3. 实现梯度下降求解 f(x, y) = (x-3)² + (y+2)² 的最小值

4. 实现一个简单的二次损失函数：
   - L(w) = (1/2) ||Xw - y||²
   - 计算梯度 ∂L/∂w = X^T(Xw - y)
   - 用梯度下降求解最优 w
""")

# === 练习代码 ===
# 练习 1: 求导数
# print("f(x) = 3x⁴ - 2x² + x - 5")
# print("f'(x) = 12x³ - 4x + 1")
# print("\ng(x) = e^(2x) + ln(x)")
# print("g'(x) = 2e^(2x) + 1/x")
# print("\nh(x) = sin(x) × cos(x) = (1/2)sin(2x)")
# print("h'(x) = cos(2x)")

# 练习 2: 偏导数和梯度
# print("f(x, y) = x²y + xy² + 3x - 2y")
# print("∂f/∂x = 2xy + y² + 3")
# print("∂f/∂y = x² + 2xy - 2")
# x0, y0 = 1, 2
# df_dx = 2*x0*y0 + y0**2 + 3  # = 11
# df_dy = x0**2 + 2*x0*y0 - 2  # = 3
# print(f"在点 (1, 2) 处: 梯度 = [{df_dx}, {df_dy}]")

# 练习 3: 梯度下降求最小值
# def f(x, y): return (x - 3)**2 + (y + 2)**2
# def grad_f(x, y): return np.array([2*(x - 3), 2*(y + 2)])
# x = np.array([0.0, 0.0])
# lr = 0.1
# for i in range(20):
#     x = x - lr * grad_f(x[0], x[1])
# print(f"最优解: {x}")  # 应接近 [3, -2]

# 练习 4: 最小二乘梯度下降
# np.random.seed(42)
# X = np.random.randn(20, 3)
# y = X @ np.array([2, -1, 0.5]) + 0.1 * np.random.randn(20)
# w = np.zeros(3)
# lr = 0.01
# for i in range(100):
#     grad = X.T @ (X @ w - y)
#     w = w - lr * grad
# print(f"真实 w: [2, -1, 0.5]")
# print(f"求解 w: {np.round(w, 4)}")

print("\n✅ 偏导数与梯度完成！")
print("下一步：04-chain-rule.py - 链式法则")
