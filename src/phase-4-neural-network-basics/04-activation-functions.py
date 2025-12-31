"""
04-activation-functions.py
Phase 4: 神经网络基础

激活函数详解 - 引入非线性的关键

学习目标：
1. 理解常用激活函数的公式和导数
2. 了解各激活函数的优缺点
3. 掌握激活函数的选择原则
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - 激活函数详解")
print("=" * 60)

# =============================================================================
# 1. 为什么需要激活函数
# =============================================================================
print("\n【1. 为什么需要激活函数】")

print("""
如果没有激活函数：
    y = W₂(W₁x + b₁) + b₂
      = W₂W₁x + W₂b₁ + b₂
      = W'x + b'  （等价于单层！）

激活函数的作用：
1. 引入非线性 → 网络可以学习复杂函数
2. 限制输出范围 → 数值稳定
3. 模拟生物神经元 → 阈值激活
""")

# =============================================================================
# 2. Sigmoid
# =============================================================================
print("\n" + "=" * 60)
print("【2. Sigmoid】")

print("""
公式: σ(x) = 1 / (1 + e^(-x))
值域: (0, 1)
导数: σ'(x) = σ(x)(1 - σ(x))

优点:
- 输出在 (0,1)，可解释为概率
- 平滑可微

缺点:
- 梯度消失：导数最大 0.25，深层网络梯度衰减
- 输出不以 0 为中心
- exp 计算较慢
""")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# =============================================================================
# 3. Tanh
# =============================================================================
print("\n" + "=" * 60)
print("【3. Tanh】")

print("""
公式: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
值域: (-1, 1)
导数: tanh'(x) = 1 - tanh²(x)

优点:
- 输出以 0 为中心 → 训练更稳定
- 比 Sigmoid 梯度更大

缺点:
- 仍有梯度消失问题
""")

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# =============================================================================
# 4. ReLU
# =============================================================================
print("\n" + "=" * 60)
print("【4. ReLU (Rectified Linear Unit)】")

print("""
公式: ReLU(x) = max(0, x)
值域: [0, +∞)
导数: 1 if x > 0 else 0

优点:
- 计算简单，速度快
- 缓解梯度消失（正区间梯度恒为 1）
- 稀疏激活 → 正则化效果

缺点:
- 死神经元问题：一旦 x < 0，梯度为 0，永不更新
- 输出无上界
""")

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# =============================================================================
# 5. Leaky ReLU
# =============================================================================
print("\n" + "=" * 60)
print("【5. Leaky ReLU】")

print("""
公式: LeakyReLU(x) = x if x > 0 else αx  (通常 α = 0.01)
值域: (-∞, +∞)

优点:
- 解决死神经元问题
- 负区间也有梯度

变体:
- PReLU: α 可学习
- ELU: x if x > 0 else α(e^x - 1)
""")

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# =============================================================================
# 6. GELU
# =============================================================================
print("\n" + "=" * 60)
print("【6. GELU (Gaussian Error Linear Unit)】")

print("""
公式: GELU(x) = x × Φ(x)  # Φ 是标准正态分布的 CDF
近似: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

特点:
- Transformer 默认激活函数
- 平滑版 ReLU
- 考虑输入的分布

用途:
- BERT, GPT 等 NLP 模型
- Vision Transformer
""")

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# =============================================================================
# 7. SiLU / Swish
# =============================================================================
print("\n" + "=" * 60)
print("【7. SiLU / Swish】")

print("""
公式: Swish(x) = x × σ(x)  # σ 是 Sigmoid
     SiLU(x) = x × σ(x)    # 相同

特点:
- 平滑、非单调
- 自门控机制
- 比 ReLU 表现更好

用途:
- EfficientNet
- 现代 CNN 架构
""")

def silu(x):
    return x * sigmoid(x)

# =============================================================================
# 8. Softmax
# =============================================================================
print("\n" + "=" * 60)
print("【8. Softmax】")

print("""
公式: Softmax(xᵢ) = e^xᵢ / Σⱼe^xⱼ

特点:
- 输出和为 1，可解释为概率分布
- 用于多分类的输出层
- 与 CrossEntropyLoss 配合使用

注意:
- CrossEntropyLoss 内部包含 Softmax
- 不要在输出层重复使用
""")

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 减去最大值防止溢出
    return exp_x / np.sum(exp_x)

# =============================================================================
# 9. 可视化所有激活函数
# =============================================================================
print("\n" + "=" * 60)
print("【9. 激活函数可视化】")

x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2, label='Sigmoid')
axes[0, 0].plot(x, sigmoid_derivative(x), 'r--', linewidth=2, label="Sigmoid'")
axes[0, 0].set_title('Sigmoid')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

# Tanh
axes[0, 1].plot(x, tanh(x), 'b-', linewidth=2, label='Tanh')
axes[0, 1].plot(x, tanh_derivative(x), 'r--', linewidth=2, label="Tanh'")
axes[0, 1].set_title('Tanh')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

# ReLU
axes[0, 2].plot(x, relu(x), 'b-', linewidth=2, label='ReLU')
axes[0, 2].plot(x, relu_derivative(x), 'r--', linewidth=2, label="ReLU'")
axes[0, 2].set_title('ReLU')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
axes[0, 2].axvline(x=0, color='k', linewidth=0.5)

# Leaky ReLU
axes[0, 3].plot(x, leaky_relu(x), 'b-', linewidth=2, label='LeakyReLU')
axes[0, 3].set_title('Leaky ReLU (α=0.01)')
axes[0, 3].legend()
axes[0, 3].grid(True, alpha=0.3)
axes[0, 3].axhline(y=0, color='k', linewidth=0.5)
axes[0, 3].axvline(x=0, color='k', linewidth=0.5)

# GELU
axes[1, 0].plot(x, gelu(x), 'b-', linewidth=2, label='GELU')
axes[1, 0].plot(x, relu(x), 'g--', linewidth=1, alpha=0.7, label='ReLU')
axes[1, 0].set_title('GELU vs ReLU')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)

# SiLU/Swish
axes[1, 1].plot(x, silu(x), 'b-', linewidth=2, label='SiLU/Swish')
axes[1, 1].plot(x, relu(x), 'g--', linewidth=1, alpha=0.7, label='ReLU')
axes[1, 1].set_title('SiLU vs ReLU')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linewidth=0.5)

# 梯度消失对比
axes[1, 2].plot(x, sigmoid_derivative(x), label='Sigmoid', linewidth=2)
axes[1, 2].plot(x, tanh_derivative(x), label='Tanh', linewidth=2)
axes[1, 2].plot(x, relu_derivative(x), label='ReLU', linewidth=2)
axes[1, 2].set_title('导数对比')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_ylim(-0.1, 1.5)

# Softmax
softmax_inputs = [np.array([1, 2, 3]), np.array([1, 1, 1]), np.array([0, 0, 5])]
for i, inp in enumerate(softmax_inputs):
    output = softmax(inp)
    axes[1, 3].bar(np.arange(3) + i*0.25, output, width=0.2, label=f'{inp}')
axes[1, 3].set_title('Softmax 示例')
axes[1, 3].set_xticks([0.25, 1.25, 2.25])
axes[1, 3].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
axes[1, 3].legend()

plt.tight_layout()
plt.savefig('outputs/activation_functions.png', dpi=100)
plt.close()
print("激活函数图已保存: outputs/activation_functions.png")

# =============================================================================
# 10. PyTorch 中使用激活函数
# =============================================================================
print("\n" + "=" * 60)
print("【10. PyTorch 中使用激活函数】")

x = torch.randn(5)
print(f"输入: {x}")

# 模块形式
print("\n模块形式:")
print(f"  ReLU: {nn.ReLU()(x)}")
print(f"  GELU: {nn.GELU()(x)}")
print(f"  SiLU: {nn.SiLU()(x)}")

# 函数形式
print("\n函数形式:")
print(f"  relu: {F.relu(x)}")
print(f"  gelu: {F.gelu(x)}")
print(f"  silu: {F.silu(x)}")

# =============================================================================
# 11. 激活函数选择建议
# =============================================================================
print("\n" + "=" * 60)
print("【11. 激活函数选择建议】")

print("""
╔═══════════════════════════════════════════════════════════════╗
║                   激活函数选择指南                             ║
╠═══════════════════╦═══════════════════════════════════════════╣
║  任务/位置         ║  推荐激活函数                             ║
╠═══════════════════╬═══════════════════════════════════════════╣
║  隐藏层 (默认)     ║  ReLU                                     ║
║  现代 CNN          ║  ReLU, SiLU/Swish                        ║
║  Transformer       ║  GELU                                     ║
║  RNN               ║  Tanh (隐藏状态)                          ║
║  二分类输出        ║  Sigmoid (或用 BCEWithLogitsLoss)        ║
║  多分类输出        ║  无 (CrossEntropyLoss 包含 Softmax)      ║
║  回归输出          ║  无（直接输出）                           ║
║  防止死神经元      ║  Leaky ReLU, ELU                         ║
╚═══════════════════╩═══════════════════════════════════════════╝
""")

# =============================================================================
# 12. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 计算 Sigmoid 和 Tanh 的导数，解释梯度消失

2. 实现 ELU 激活函数及其导数

3. 比较 ReLU 和 GELU 在同一网络上的训练效果

4. 解释为什么 Softmax 不能用于二分类

5. 实现一个带可学习参数的 PReLU
""")

print("\n✅ 激活函数详解完成！")
print("下一步：05-activation-comparison.py - 激活函数对比分析")
