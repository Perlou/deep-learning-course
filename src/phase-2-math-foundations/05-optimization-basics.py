"""
05-optimization-basics.py
Phase 2: 深度学习数学基础

优化基础 - 梯度下降及其变体

学习目标：
1. 深入理解梯度下降算法
2. 掌握动量、Adam 等优化器
3. 理解学习率的重要性
4. 了解凸优化与非凸优化的区别
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("深度学习数学基础 - 优化基础")
print("=" * 60)

# =============================================================================
# 1. 梯度下降回顾
# =============================================================================
print("\n【1. 梯度下降回顾】")

print("""
梯度下降算法：
    θ_{t+1} = θ_t - η × ∇f(θ_t)

其中：
- θ: 参数
- η (eta): 学习率
- ∇f: 梯度（函数增长最快的方向）

目标：找到使损失函数最小的参数
""")

def gradient_descent(f, grad_f, x0, lr=0.1, n_iters=50):
    """基础梯度下降"""
    path = [x0.copy()]
    x = x0.copy()
    
    for _ in range(n_iters):
        g = grad_f(x)
        x = x - lr * g
        path.append(x.copy())
    
    return np.array(path)

# 测试函数：凸函数
def f_convex(x):
    return x[0]**2 + 10*x[1]**2

def grad_convex(x):
    return np.array([2*x[0], 20*x[1]])

# =============================================================================
# 2. 学习率的影响
# =============================================================================
print("\n" + "=" * 60)
print("【2. 学习率的影响】")

print("""
学习率太小：收敛太慢
学习率太大：可能不收敛甚至发散
学习率适中：平衡速度和稳定性
""")

x0 = np.array([4.0, 4.0])
learning_rates = [0.01, 0.05, 0.1, 0.2]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, lr in zip(axes.ravel(), learning_rates):
    path = gradient_descent(f_convex, grad_convex, x0, lr=lr, n_iters=50)
    
    # 绘制等高线
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 10*Y**2
    
    ax.contour(X, Y, Z, levels=30, cmap='viridis')
    ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=1)
    ax.scatter(path[0, 0], path[0, 1], c='green', s=100, marker='*', zorder=5)
    ax.scatter(path[-1, 0], path[-1, 1], c='red', s=100, marker='*', zorder=5)
    
    final_loss = f_convex(path[-1])
    ax.set_title(f'学习率 η = {lr}\n最终损失 = {final_loss:.6f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/05_learning_rate.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/05_learning_rate.png")

# =============================================================================
# 3. 动量优化 (Momentum)
# =============================================================================
print("\n" + "=" * 60)
print("【3. 动量优化 (Momentum)】")

print("""
问题：标准梯度下降在陡峭方向震荡

动量法：引入"速度"概念，累积历史梯度
    v_t = β × v_{t-1} + ∇f(θ_t)
    θ_{t+1} = θ_t - η × v_t

其中：
- β: 动量系数 (通常 0.9)
- v: 速度/动量
""")

def gradient_descent_momentum(f, grad_f, x0, lr=0.1, beta=0.9, n_iters=50):
    """带动量的梯度下降"""
    path = [x0.copy()]
    x = x0.copy()
    v = np.zeros_like(x)
    
    for _ in range(n_iters):
        g = grad_f(x)
        v = beta * v + g
        x = x - lr * v
        path.append(x.copy())
    
    return np.array(path)

# 比较标准 GD 和动量 GD
x0 = np.array([4.0, 4.0])
path_gd = gradient_descent(f_convex, grad_convex, x0, lr=0.05, n_iters=50)
path_momentum = gradient_descent_momentum(f_convex, grad_convex, x0, lr=0.05, beta=0.9, n_iters=50)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, path, title in zip(axes, [path_gd, path_momentum], ['标准梯度下降', '动量梯度下降']):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 10*Y**2
    
    ax.contour(X, Y, Z, levels=30, cmap='viridis')
    ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=1)
    ax.scatter(path[0, 0], path[0, 1], c='green', s=100, marker='*', zorder=5, label='起点')
    ax.scatter(path[-1, 0], path[-1, 1], c='red', s=100, marker='*', zorder=5, label='终点')
    ax.set_title(f'{title}\n步数: {len(path)}, 最终损失: {f_convex(path[-1]):.6f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend()

plt.tight_layout()
plt.savefig('outputs/05_momentum.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/05_momentum.png")

# =============================================================================
# 4. RMSprop
# =============================================================================
print("\n" + "=" * 60)
print("【4. RMSprop】")

print("""
问题：不同参数的梯度尺度差异大

RMSprop：自适应学习率
    s_t = β × s_{t-1} + (1-β) × g_t²
    θ_{t+1} = θ_t - η × g_t / √(s_t + ε)

其中：
- s: 梯度平方的指数移动平均
- ε: 防止除零的小常数
- 效果：梯度大的方向学习率减小，梯度小的方向学习率增大
""")

def rmsprop(f, grad_f, x0, lr=0.1, beta=0.99, eps=1e-8, n_iters=50):
    """RMSprop 优化器"""
    path = [x0.copy()]
    x = x0.copy()
    s = np.zeros_like(x)
    
    for _ in range(n_iters):
        g = grad_f(x)
        s = beta * s + (1 - beta) * g**2
        x = x - lr * g / (np.sqrt(s) + eps)
        path.append(x.copy())
    
    return np.array(path)

# =============================================================================
# 5. Adam 优化器
# =============================================================================
print("\n" + "=" * 60)
print("【5. Adam 优化器】")

print("""
Adam = Momentum + RMSprop + 偏差修正

    m_t = β₁ × m_{t-1} + (1-β₁) × g_t      # 一阶矩（动量）
    v_t = β₂ × v_{t-1} + (1-β₂) × g_t²     # 二阶矩（RMSprop）
    
    # 偏差修正
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    
    θ_{t+1} = θ_t - η × m̂_t / (√v̂_t + ε)

默认超参数：
- β₁ = 0.9, β₂ = 0.999, η = 0.001, ε = 1e-8
""")

def adam(f, grad_f, x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_iters=50):
    """Adam 优化器"""
    path = [x0.copy()]
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    for t in range(1, n_iters + 1):
        g = grad_f(x)
        
        # 更新一阶矩和二阶矩
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        
        # 偏差修正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # 更新参数
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())
    
    return np.array(path)

# 比较所有优化器
x0 = np.array([4.0, 4.0])
optimizers = {
    'SGD': gradient_descent(f_convex, grad_convex, x0, lr=0.05, n_iters=100),
    'Momentum': gradient_descent_momentum(f_convex, grad_convex, x0, lr=0.05, beta=0.9, n_iters=100),
    'RMSprop': rmsprop(f_convex, grad_convex, x0, lr=0.5, beta=0.99, n_iters=100),
    'Adam': adam(f_convex, grad_convex, x0, lr=0.5, n_iters=100)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, (name, path) in zip(axes.ravel(), optimizers.items()):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 10*Y**2
    
    ax.contour(X, Y, Z, levels=30, cmap='viridis')
    ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=3, linewidth=1)
    ax.scatter(path[0, 0], path[0, 1], c='green', s=100, marker='*', zorder=5)
    ax.scatter(path[-1, 0], path[-1, 1], c='red', s=100, marker='*', zorder=5)
    ax.set_title(f'{name}\n最终损失: {f_convex(path[-1]):.8f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/05_optimizers_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/05_optimizers_comparison.png")

# 损失曲线对比
fig, ax = plt.subplots(figsize=(10, 6))

for name, path in optimizers.items():
    losses = [f_convex(p) for p in path]
    ax.plot(losses, label=name, linewidth=2)

ax.set_xlabel('迭代次数')
ax.set_ylabel('损失')
ax.set_title('不同优化器的收敛速度', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_optimizers_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/05_optimizers_convergence.png")

# =============================================================================
# 6. 非凸优化与局部最小值
# =============================================================================
print("\n" + "=" * 60)
print("【6. 非凸优化与局部最小值】")

print("""
凸函数：只有一个全局最小值
非凸函数：可能有多个局部最小值

深度学习的损失函数是非凸的！
- 梯度下降可能陷入局部最小值
- 但实践表明：大部分局部最小值质量都不错（鞍点问题更严重）
""")

# 非凸函数示例
def f_nonconvex(x):
    return np.sin(5*x[0]) * np.cos(5*x[1]) / 5 + (x[0]**2 + x[1]**2) / 2

def grad_nonconvex(x):
    dx = np.cos(5*x[0]) * np.cos(5*x[1]) + x[0]
    dy = -np.sin(5*x[0]) * np.sin(5*x[1]) + x[1]
    return np.array([dx, dy])

# 从不同起点优化
np.random.seed(42)
starting_points = [np.random.randn(2) * 2 for _ in range(4)]

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制等高线
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(5*X) * np.cos(5*Y) / 5 + (X**2 + Y**2) / 2

contour = ax.contour(X, Y, Z, levels=30, cmap='viridis')

# 从不同起点优化
colors = ['red', 'blue', 'green', 'orange']
for i, (x0, color) in enumerate(zip(starting_points, colors)):
    path = gradient_descent(f_nonconvex, grad_nonconvex, x0, lr=0.05, n_iters=100)
    ax.plot(path[:, 0], path[:, 1], 'o-', color=color, markersize=3, linewidth=1, label=f'起点 {i+1}')
    ax.scatter(path[0, 0], path[0, 1], c=color, s=100, marker='*', zorder=5)
    ax.scatter(path[-1, 0], path[-1, 1], c=color, s=100, marker='s', zorder=5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('非凸函数的梯度下降（不同起点收敛到不同局部最小值）', fontsize=12, fontweight='bold')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/05_nonconvex.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/05_nonconvex.png")

# =============================================================================
# 7. 学习率调度
# =============================================================================
print("\n" + "=" * 60)
print("【7. 学习率调度】")

print("""
训练过程中调整学习率：

1. Step Decay: 每 N 个 epoch 学习率减半
2. Exponential Decay: lr_t = lr_0 × γ^t
3. Cosine Annealing: lr_t = lr_min + (lr_0 - lr_min) × (1 + cos(πt/T)) / 2
4. Warmup: 开始时逐渐增加学习率
""")

epochs = 100

# 不同调度策略
def step_decay(epoch, lr0=0.1, drop=0.5, epochs_drop=20):
    return lr0 * (drop ** (epoch // epochs_drop))

def exponential_decay(epoch, lr0=0.1, gamma=0.95):
    return lr0 * (gamma ** epoch)

def cosine_annealing(epoch, lr0=0.1, lr_min=0.001, T=100):
    return lr_min + (lr0 - lr_min) * (1 + np.cos(np.pi * epoch / T)) / 2

def warmup_cosine(epoch, lr0=0.1, warmup=10, T=100):
    if epoch < warmup:
        return lr0 * epoch / warmup
    else:
        return cosine_annealing(epoch - warmup, lr0=lr0, T=T-warmup)

schedules = {
    'Step Decay': [step_decay(e) for e in range(epochs)],
    'Exponential Decay': [exponential_decay(e) for e in range(epochs)],
    'Cosine Annealing': [cosine_annealing(e) for e in range(epochs)],
    'Warmup + Cosine': [warmup_cosine(e) for e in range(epochs)]
}

fig, ax = plt.subplots(figsize=(12, 6))

for name, lrs in schedules.items():
    ax.plot(lrs, label=name, linewidth=2)

ax.set_xlabel('Epoch')
ax.set_ylabel('学习率')
ax.set_title('学习率调度策略', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_lr_schedules.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/05_lr_schedules.png")

# =============================================================================
# 8. 优化器总结
# =============================================================================
print("\n" + "=" * 60)
print("【8. 优化器总结】")

print("""
╔═══════════════════════════════════════════════════════════════╗
║                    常用优化器对比                              ║
╠════════════════╦══════════════════════════════════════════════╣
║  优化器        ║  特点                                        ║
╠════════════════╬══════════════════════════════════════════════╣
║  SGD           ║  简单，但可能震荡，收敛慢                     ║
║  SGD+Momentum  ║  加速收敛，减少震荡                          ║
║  RMSprop       ║  自适应学习率，适合 RNN                      ║
║  Adam          ║  通用首选，结合动量和自适应                   ║
║  AdamW         ║  Adam + 权重衰减解耦，目前主流               ║
║  LAMB/LARS     ║  分布式训练大 batch 专用                     ║
╚════════════════╩══════════════════════════════════════════════╝

实践建议：
1. 默认使用 AdamW，lr=1e-3 或 3e-4
2. 微调预训练模型用更小学习率，如 1e-5 或 2e-5
3. 大 batch 训练配合 warmup
4. 训练后期可以切换到 SGD+Momentum 获得更好泛化
""")

# =============================================================================
# 9. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 实现 Nesterov 动量：
   v_t = β × v_{t-1} + ∇f(θ_t - β × v_{t-1})
   θ_{t+1} = θ_t - η × v_t

2. 比较 Adam 和 SGD 在 Rosenbrock 函数上的表现：
   f(x, y) = (1-x)² + 100(y-x²)²

3. 实现 Warmup + Cosine Annealing 学习率调度

4. 研究不同 β₁, β₂ 对 Adam 收敛的影响
""")

# === 练习代码 ===
# 练习 1: Nesterov 动量
# def nesterov_momentum(grad_f, x0, lr=0.1, beta=0.9, n_iters=50):
#     x = x0.copy()
#     v = np.zeros_like(x)
#     for _ in range(n_iters):
#         x_lookahead = x - beta * v
#         g = grad_f(x_lookahead)
#         v = beta * v + lr * g
#         x = x - v
#     return x
# x0 = np.array([4.0, 4.0])
# result = nesterov_momentum(grad_convex, x0)
# print(f"Nesterov 结果: {result}")

# 练习 2: Rosenbrock 函数优化
# def rosenbrock(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
# def grad_rosenbrock(x):
#     dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
#     dy = 200 * (x[1] - x[0]**2)
#     return np.array([dx, dy])
# 
# # SGD
# x_sgd = np.array([-1.0, 1.0])
# for _ in range(1000):
#     x_sgd = x_sgd - 0.001 * grad_rosenbrock(x_sgd)
# print(f"SGD: {x_sgd}, f(x)={rosenbrock(x_sgd):.6f}")
# 
# # Adam
# x_adam = np.array([-1.0, 1.0])
# m, v = np.zeros(2), np.zeros(2)
# for t in range(1, 1001):
#     g = grad_rosenbrock(x_adam)
#     m = 0.9 * m + 0.1 * g
#     v = 0.999 * v + 0.001 * g**2
#     m_hat = m / (1 - 0.9**t)
#     v_hat = v / (1 - 0.999**t)
#     x_adam = x_adam - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
# print(f"Adam: {x_adam}, f(x)={rosenbrock(x_adam):.6f}")

# 练习 3: Warmup + Cosine 调度
# def warmup_cosine_lr(epoch, warmup=10, total=100, lr_max=0.1, lr_min=0.001):
#     if epoch < warmup:
#         return lr_max * epoch / warmup
#     progress = (epoch - warmup) / (total - warmup)
#     return lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * progress)) / 2
# for e in [0, 5, 10, 50, 99]:
#     print(f"Epoch {e}: lr = {warmup_cosine_lr(e):.6f}")

# 练习 4: Adam β 参数影响
# print("β₁ (动量): 控制一阶矩衰减")
# print("  β₁ 大 (0.99): 更平滑但收敛慢")
# print("  β₁ 小 (0.8): 响应快但可能震荡")
# print("β₂ (RMSprop): 控制二阶矩衰减")
# print("  β₂ 大 (0.9999): 学习率调整更稳定")
# print("  β₂ 小 (0.99): 更快适应梯度变化")

print("\n✅ 优化基础完成！")
print("下一步：06-probability-basics.py - 概率论基础")
