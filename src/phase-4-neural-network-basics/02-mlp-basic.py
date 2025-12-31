"""
02-mlp-basic.py
Phase 4: 神经网络基础

多层感知机 (MLP) - 解决非线性问题

学习目标：
1. 理解多层网络的结构
2. 理解隐藏层的作用
3. 实现 MLP 解决 XOR 问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - 多层感知机 (MLP)")
print("=" * 60)

# =============================================================================
# 1. MLP 简介
# =============================================================================
print("\n【1. MLP 简介】")

print("""
多层感知机 (Multilayer Perceptron, MLP):
- 由多个全连接层组成
- 层与层之间有非线性激活函数
- 可以学习非线性决策边界

网络结构:
    输入层 → 隐藏层1 → ... → 隐藏层n → 输出层

每层的计算:
    h = σ(Wx + b)
    
关键点:
    - 没有激活函数，多层等价于单层
    - 激活函数引入非线性，使网络更强大
""")

# =============================================================================
# 2. 用 MLP 解决 XOR
# =============================================================================
print("\n" + "=" * 60)
print("【2. 用 MLP 解决 XOR】")

# XOR 数据
X_xor = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = torch.FloatTensor([[0], [1], [1], [0]])

# MLP 模型
class MLP_XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)  # 隐藏层
        self.output = nn.Linear(4, 1)  # 输出层
    
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # 激活函数
        x = torch.sigmoid(self.output(x))
        return x

# 训练
model = MLP_XOR()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1.0)

print("训练 MLP 解决 XOR:")
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_xor)
    loss = criterion(outputs, y_xor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        preds = (outputs > 0.5).float()
        acc = (preds == y_xor).float().mean()
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

# 测试
print("\n最终预测:")
with torch.no_grad():
    preds = model(X_xor)
    for xi, yi, pi in zip(X_xor, y_xor, preds):
        xi_list = [int(x) for x in xi.tolist()]
        print(f"  {xi_list} -> 预测: {pi.item():.4f}, 真实: {int(yi.item())}")

# =============================================================================
# 3. 可视化决策边界
# =============================================================================
print("\n" + "=" * 60)
print("【3. 可视化决策边界】")

def plot_decision_boundary_mlp(model, X, y):
    """可视化 MLP 的决策边界"""
    plt.figure(figsize=(12, 4))
    
    # 子图1：决策边界
    plt.subplot(1, 3, 1)
    
    # 创建网格
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='输出概率')
    
    # 画数据点
    colors = ['red' if yi == 0 else 'blue' for yi in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', zorder=5)
    for xi, yi in zip(X.numpy(), y.numpy()):
        plt.annotate(f'{int(yi[0])}', xy=xi, ha='center', va='center', fontsize=12, color='white')
    
    plt.title('MLP 决策边界 (XOR)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # 子图2：损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True, alpha=0.3)
    
    # 子图3：网络结构
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.text(0.5, 0.9, 'MLP 结构', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.7, '输入层: 2 个神经元', ha='center', fontsize=12)
    plt.text(0.5, 0.5, '隐藏层: 4 个神经元 + Sigmoid', ha='center', fontsize=12)
    plt.text(0.5, 0.3, '输出层: 1 个神经元 + Sigmoid', ha='center', fontsize=12)
    plt.text(0.5, 0.1, f'总参数: {sum(p.numel() for p in model.parameters())}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('outputs/mlp_xor.png', dpi=100)
    plt.close()
    print("结果已保存: outputs/mlp_xor.png")

plot_decision_boundary_mlp(model, X_xor, y_xor)

# =============================================================================
# 4. 隐藏层的作用
# =============================================================================
print("\n" + "=" * 60)
print("【4. 隐藏层的作用】")

print("""
隐藏层的作用：
1. 特征提取：将输入变换到新的特征空间
2. 非线性映射：通过激活函数引入非线性
3. 更复杂的决策边界：多个线性边界的组合

XOR 的隐藏层分析：
- 隐藏层 1：检测 (x1 AND x2)
- 隐藏层 2：检测 (x1 OR x2)  
- 输出层：组合为 (x1 XOR x2) = (x1 OR x2) AND NOT(x1 AND x2)
""")

# 可视化隐藏层输出
print("\n隐藏层特征:")
with torch.no_grad():
    hidden_output = torch.sigmoid(model.hidden(X_xor))
    for xi, hi in zip(X_xor, hidden_output):
        print(f"  输入{xi.tolist()} -> 隐藏层{[f'{h:.2f}' for h in hi.tolist()]}")

# =============================================================================
# 5. 更大的 MLP
# =============================================================================
print("\n" + "=" * 60)
print("【5. 更大的 MLP - 分类螺旋数据】")

# 创建螺旋数据
def generate_spiral_data(n_samples=100, n_classes=2):
    X = []
    y = []
    for i in range(n_classes):
        r = np.linspace(0.1, 1, n_samples)
        theta = np.linspace(i * 4, (i + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        X.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        y.append(np.full(n_samples, i))
    return np.vstack(X), np.hstack(y)

X_spiral, y_spiral = generate_spiral_data(n_samples=100)
X_spiral_t = torch.FloatTensor(X_spiral)
y_spiral_t = torch.LongTensor(y_spiral)

# 更深的 MLP
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

# 训练
model_deep = DeepMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_deep.parameters(), lr=0.01)

print("训练深度 MLP 分类螺旋数据:")
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model_deep(X_spiral_t)
    loss = criterion(outputs, y_spiral_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        _, preds = outputs.max(1)
        acc = (preds == y_spiral_t).float().mean()
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

# 可视化
plt.figure(figsize=(10, 4))

# 原始数据
plt.subplot(1, 2, 1)
plt.scatter(X_spiral[y_spiral == 0, 0], X_spiral[y_spiral == 0, 1], c='blue', label='Class 0')
plt.scatter(X_spiral[y_spiral == 1, 0], X_spiral[y_spiral == 1, 1], c='red', label='Class 1')
plt.title('螺旋数据 (原始)')
plt.legend()
plt.axis('equal')

# 决策边界
plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.arange(-1.5, 1.5, 0.02), np.arange(-1.5, 1.5, 0.02))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
with torch.no_grad():
    Z = model_deep(grid).argmax(1).numpy().reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5, cmap='RdBu')
plt.scatter(X_spiral[y_spiral == 0, 0], X_spiral[y_spiral == 0, 1], c='blue', edgecolors='k')
plt.scatter(X_spiral[y_spiral == 1, 0], X_spiral[y_spiral == 1, 1], c='red', edgecolors='k')
plt.title('MLP 决策边界')
plt.axis('equal')

plt.tight_layout()
plt.savefig('outputs/mlp_spiral.png', dpi=100)
plt.close()
print("螺旋分类结果已保存: outputs/mlp_spiral.png")

# =============================================================================
# 6. 通用近似定理
# =============================================================================
print("\n" + "=" * 60)
print("【6. 通用近似定理】")

print("""
通用近似定理 (Universal Approximation Theorem):

有一个隐藏层的前馈网络，如果隐藏层有足够多的神经元，
可以以任意精度逼近任意连续函数。

关键点：
1. 理论上单隐藏层就够了
2. 但实践中深层网络更高效
3. 深度 vs 宽度的权衡

为什么用深层而不是宽层？
- 深层网络可以层次化学习特征
- 参数效率更高
- 泛化能力更好
""")

# =============================================================================
# 7. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 修改隐藏层神经元数量，观察 XOR 问题的解决效果

2. 对于螺旋数据，比较 2 层 vs 4 层网络的效果

3. 实现一个 MLP 用于 3 分类问题

4. 分析隐藏层输出，理解网络学到了什么特征

5. 不使用激活函数，验证多层网络等价于单层
""")

# === 练习答案 ===
# 1 隐藏层数量
# 2 个神经元可能不够，4 个神经元通常足够
# 太多神经元可能过拟合

# 2 层数比较
# 浅层需要更多神经元
# 深层可以用更少神经元达到同样效果

# 3 三分类 MLP
# class MLP3Class(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(2, 32)
#         self.fc2 = nn.Linear(32, 3)
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)

# 4 隐藏层分析：可视化每个隐藏单元的激活模式

# 5 无激活验证
# y = W2(W1*x + b1) + b2 = (W2*W1)*x + (W2*b1 + b2) = W'x + b'
# 等价于单层线性变换

print("\n✅ 多层感知机完成！")
print("下一步：03-forward-backward.py - 前向传播与反向传播")
