"""
03-forward-backward.py
Phase 4: 神经网络基础

前向传播与反向传播 - 神经网络的核心算法

学习目标：
1. 理解前向传播的计算过程
2. 理解反向传播的链式法则
3. 手动实现一个简单网络的反向传播
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - 前向传播与反向传播")
print("=" * 60)

# =============================================================================
# 1. 前向传播
# =============================================================================
print("\n【1. 前向传播】")

print("""
前向传播 (Forward Propagation):
    从输入到输出的计算过程

单个神经元:
    z = w·x + b        (线性变换)
    a = σ(z)           (激活函数)

多层网络:
    a⁽⁰⁾ = x           (输入)
    z⁽¹⁾ = W⁽¹⁾a⁽⁰⁾ + b⁽¹⁾
    a⁽¹⁾ = σ(z⁽¹⁾)
    z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
    a⁽²⁾ = σ(z⁽²⁾)
    ...
    ŷ = a⁽L⁾           (输出)
""")

# 手动实现前向传播
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_manual(x, W1, b1, W2, b2):
    """手动实现两层网络的前向传播"""
    # 第一层
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)
    
    # 第二层
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    # 保存中间值（反向传播需要）
    cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

# 示例
np.random.seed(42)
x = np.array([0.5, 0.3])
W1 = np.random.randn(3, 2) * 0.1
b1 = np.zeros(3)
W2 = np.random.randn(1, 3) * 0.1
b2 = np.zeros(1)

output, cache = forward_manual(x, W1, b1, W2, b2)
print(f"输入: {x}")
print(f"隐藏层输出 a1: {cache['a1']}")
print(f"最终输出 a2: {output}")

# =============================================================================
# 2. 反向传播
# =============================================================================
print("\n" + "=" * 60)
print("【2. 反向传播】")

print("""
反向传播 (Backpropagation):
    从输出到输入，计算每个参数的梯度

核心：链式法则
    ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

对于损失 L = (y - ŷ)²:
    
    输出层:
    ∂L/∂a² = 2(a² - y)
    ∂L/∂z² = ∂L/∂a² · σ'(z²)
    ∂L/∂W² = ∂L/∂z² · a¹ᵀ
    ∂L/∂b² = ∂L/∂z²
    
    隐藏层:
    ∂L/∂a¹ = W²ᵀ · ∂L/∂z²
    ∂L/∂z¹ = ∂L/∂a¹ · σ'(z¹)
    ∂L/∂W¹ = ∂L/∂z¹ · xᵀ
    ∂L/∂b¹ = ∂L/∂z¹
""")

def sigmoid_derivative(z):
    """Sigmoid 的导数: σ'(z) = σ(z)(1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)

def backward_manual(y, cache, W2):
    """手动实现反向传播"""
    x = cache['x']
    z1, a1 = cache['z1'], cache['a1']
    z2, a2 = cache['z2'], cache['a2']
    
    # 输出层梯度
    dL_da2 = 2 * (a2 - y)              # 损失对输出的导数
    dL_dz2 = dL_da2 * sigmoid_derivative(z2)  # 链式法则
    dL_dW2 = np.outer(dL_dz2, a1)      # 对权重的梯度
    dL_db2 = dL_dz2                     # 对偏置的梯度
    
    # 隐藏层梯度
    dL_da1 = np.dot(W2.T, dL_dz2)      # 传播到隐藏层
    dL_dz1 = dL_da1 * sigmoid_derivative(z1)
    dL_dW1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1
    
    gradients = {
        'dW1': dL_dW1, 'db1': dL_db1,
        'dW2': dL_dW2, 'db2': dL_db2
    }
    return gradients

# 计算梯度
y = np.array([1.0])  # 目标值
grads = backward_manual(y, cache, W2)

print(f"目标值: {y}")
print(f"预测值: {cache['a2']}")
print(f"dL/dW2 形状: {grads['dW2'].shape}")
print(f"dL/dW1 形状: {grads['dW1'].shape}")

# =============================================================================
# 3. 验证梯度（数值梯度检验）
# =============================================================================
print("\n" + "=" * 60)
print("【3. 梯度检验】")

def compute_loss(x, y, W1, b1, W2, b2):
    """计算损失"""
    output, _ = forward_manual(x, W1, b1, W2, b2)
    return np.sum((output - y) ** 2)

def numerical_gradient(x, y, W1, b1, W2, b2, epsilon=1e-5):
    """数值梯度（用于验证）"""
    num_grads = {}
    
    # W2 的数值梯度
    dW2 = np.zeros_like(W2)
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2[i, j] += epsilon
            loss_plus = compute_loss(x, y, W1, b1, W2, b2)
            W2[i, j] -= 2 * epsilon
            loss_minus = compute_loss(x, y, W1, b1, W2, b2)
            W2[i, j] += epsilon
            dW2[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    num_grads['dW2'] = dW2
    
    # W1 的数值梯度
    dW1 = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1[i, j] += epsilon
            loss_plus = compute_loss(x, y, W1, b1, W2, b2)
            W1[i, j] -= 2 * epsilon
            loss_minus = compute_loss(x, y, W1, b1, W2, b2)
            W1[i, j] += epsilon
            dW1[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    num_grads['dW1'] = dW1
    
    return num_grads

# 验证
num_grads = numerical_gradient(x, y, W1.copy(), b1.copy(), W2.copy(), b2.copy())

print("梯度检验 (分析梯度 vs 数值梯度):")
print(f"  dW2 差异: {np.max(np.abs(grads['dW2'] - num_grads['dW2'])):.2e}")
print(f"  dW1 差异: {np.max(np.abs(grads['dW1'] - num_grads['dW1'])):.2e}")

# =============================================================================
# 4. 完整的训练循环
# =============================================================================
print("\n" + "=" * 60)
print("【4. 完整训练循环】")

def train_network(X, Y, hidden_dim=4, learning_rate=0.5, epochs=1000):
    """训练一个两层网络"""
    np.random.seed(42)
    n_features = X.shape[1]
    n_outputs = Y.shape[1]
    
    # 初始化参数
    W1 = np.random.randn(hidden_dim, n_features) * 0.5
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(n_outputs, hidden_dim) * 0.5
    b2 = np.zeros(n_outputs)
    
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for xi, yi in zip(X, Y):
            # 前向传播
            output, cache = forward_manual(xi, W1, b1, W2, b2)
            
            # 计算损失
            loss = np.sum((output - yi) ** 2)
            total_loss += loss
            
            # 反向传播
            grads = backward_manual(yi, cache, W2)
            
            # 更新参数
            W1 -= learning_rate * grads['dW1']
            b1 -= learning_rate * grads['db1']
            W2 -= learning_rate * grads['dW2']
            b2 -= learning_rate * grads['db2']
        
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return W1, b1, W2, b2, losses

# 训练 XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_xor = np.array([[0], [1], [1], [0]])

print("\n训练 XOR (手动反向传播):")
W1, b1, W2, b2, losses = train_network(X_xor, Y_xor, hidden_dim=4, learning_rate=1.0, epochs=1000)

# 测试
print("\n最终预测:")
for xi, yi in zip(X_xor, Y_xor):
    output, _ = forward_manual(xi, W1, b1, W2, b2)
    print(f"  {xi.tolist()} -> 预测: {output[0]:.4f}, 真实: {yi[0]}")

# 可视化损失曲线
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('手动实现反向传播 - XOR 训练损失')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/backprop_loss.png', dpi=100)
plt.close()
print("\n损失曲线已保存: outputs/backprop_loss.png")

# =============================================================================
# 5. 与 PyTorch 对比
# =============================================================================
print("\n" + "=" * 60)
print("【5. 与 PyTorch 对比】")

# PyTorch 实现
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

X_t = torch.FloatTensor(X_xor)
Y_t = torch.FloatTensor(Y_xor)

model = SimpleNet()

# 查看 PyTorch 计算的梯度
model.zero_grad()
output = model(X_t[0:1])
loss = ((output - Y_t[0:1]) ** 2).sum()
loss.backward()

print("PyTorch 自动计算的梯度:")
print(f"  fc1.weight.grad:\n{model.fc1.weight.grad}")
print(f"  fc2.weight.grad:\n{model.fc2.weight.grad}")

# =============================================================================
# 6. 计算图可视化
# =============================================================================
print("\n" + "=" * 60)
print("【6. 计算图理解】")

print("""
计算图 (Computation Graph):

前向传播：从输入到输出
    x → [Linear] → z1 → [Sigmoid] → a1 → [Linear] → z2 → [Sigmoid] → ŷ → [Loss] → L

反向传播：从输出到输入
    x ← dL/dW1 ← dL/dz1 ← dL/da1 ← dL/dW2 ← dL/dz2 ← dL/dŷ ← L

关键点：
1. 每个节点保存输入值（前向时）
2. 每个节点知道自己的局部梯度
3. 链式法则连接所有梯度
""")

# =============================================================================
# 7. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 推导 ReLU 激活函数的反向传播公式

2. 为三层网络实现手动反向传播

3. 实现 Softmax + CrossEntropy 的反向传播

4. 解释为什么 Sigmoid 会导致梯度消失

5. 比较手动实现和 PyTorch 的计算效率
""")

# === 练习答案 ===
# 1 ReLU 导数
# ReLU(z) = max(0, z)
# ReLU'(z) = 1 if z > 0 else 0

# 2 三层网络
# 与两层类似，多一个隐藏层的梯度传播

# 3 Softmax + CE
# ∂L/∂z = softmax(z) - y (非常简洁的结果！)

# 4 梯度消失
# Sigmoid 导数最大值为 0.25
# 多层相乘后梯度指数级衰减

# 5 效率对比
# 手动实现更容易理解
# PyTorch 更高效（底层优化 + GPU 加速）

print("\n✅ 前向传播与反向传播完成！")
print("下一步：04-activation-functions.py - 激活函数详解")
