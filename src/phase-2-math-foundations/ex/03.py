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

# 1
print("f(x) = 3x⁴ - 2x² + x - 5")
print("f'(x) = 12x³ - 4x + 1")
print("\ng(x) = e^(2x) + ln(x)")
print("g'(x) = 2e^(2x) + 1/x")
print("\nh(x) = sin(x) × cos(x) = (1/2)sin(2x)")
print("h'(x) = cos(2x)")

# 2
print("f(x, y) = x²y + xy² + 3x - 2y")
print("∂f/∂x = 2xy + y² + 3")
print("∂f/∂y = x² + 2xy - 2")
x0, y0 = 1, 2
df_dx = 2*x0*y0 + y0**2 + 3
df_dy = x0**2 + 2*x0*y0 - 2
print(f"在点 (1, 2) 处: 梯度 = [{df_dx}, {df_dy}]")

# 3
def f(x, y):
    return (x - 3)**2 + (y + 2)**2

def grad_f(x, y):
    return np.array([2*(x - 3), 2*(y + 2)])

x = np.array([0.0, 0.0])
lr = 0.1
for i in range(20):
    x = x - lr * grad_f(x[0], x[1])
print(f"最优解: {x}") 

# 4
np.random.seed(42)
X = np.random.randn(20, 3)
y = X @ np.array([2, -1, 0.5]) + 0.1 * np.random.randn(20)
w = np.zeros(3)
lr = 0.01
for i in range(100):
    grad = X.T @ (X @ w - y)
    w = w - lr * grad
print(f"真实 w: [2, -1, 0.5]")
print(f"求解 w: {np.round(w, 4)}")

