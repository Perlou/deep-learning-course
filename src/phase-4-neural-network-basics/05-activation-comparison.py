"""
05-activation-comparison.py
Phase 4: 神经网络基础

激活函数对比分析 - 实验验证

学习目标：
1. 通过实验对比不同激活函数
2. 观察梯度消失现象
3. 理解激活函数对训练的影响
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - 激活函数对比分析")
print("=" * 60)

# =============================================================================
# 1. 创建测试数据
# =============================================================================
print("\n【1. 创建测试数据】")

# 螺旋数据（非线性分类）
np.random.seed(42)
n_samples = 500

def generate_spiral(n_points, noise=0.2):
    n = n_points // 2
    theta = np.linspace(0, 4*np.pi, n)
    r = np.linspace(0.1, 1, n)
    
    x1 = r * np.cos(theta) + noise * np.random.randn(n)
    y1 = r * np.sin(theta) + noise * np.random.randn(n)
    
    x2 = -r * np.cos(theta) + noise * np.random.randn(n)
    y2 = -r * np.sin(theta) + noise * np.random.randn(n)
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

X, y = generate_spiral(n_samples)
X_train = torch.FloatTensor(X)
y_train = torch.LongTensor(y)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"数据形状: {X_train.shape}, 标签形状: {y_train.shape}")

# =============================================================================
# 2. 定义带不同激活函数的网络
# =============================================================================
print("\n" + "=" * 60)
print("【2. 定义网络】")

class DeepNet(nn.Module):
    def __init__(self, activation_fn, depth=5, width=64):
        super().__init__()
        layers = [nn.Linear(2, width), activation_fn()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
        layers.append(nn.Linear(width, 2))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

# 不同的激活函数
activations = {
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU
}

# =============================================================================
# 3. 训练函数
# =============================================================================

def train_model(model, loader, epochs=100, lr=0.01):
    """训练模型并记录历史"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'acc': [], 'grad_norms': []}
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        grad_norms = []
        
        for x, y in loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            
            # 记录梯度范数
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_norms.append(param.grad.norm().item())
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        history['loss'].append(total_loss / len(loader))
        history['acc'].append(correct / total)
        history['grad_norms'].append(np.mean(grad_norms))
    
    return history

# =============================================================================
# 4. 对比实验
# =============================================================================
print("\n" + "=" * 60)
print("【3. 对比实验】")

results = {}

for name, act_fn in activations.items():
    print(f"\n训练 {name}...")
    torch.manual_seed(42)  # 保证公平比较
    model = DeepNet(act_fn, depth=5, width=64)
    history = train_model(model, loader, epochs=100)
    results[name] = history
    print(f"  最终准确率: {history['acc'][-1]:.4f}")

# =============================================================================
# 5. 可视化结果
# =============================================================================
print("\n" + "=" * 60)
print("【4. 可视化结果】")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 损失曲线
ax = axes[0, 0]
for name, hist in results.items():
    ax.plot(hist['loss'], label=name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('训练损失对比')
ax.legend()
ax.grid(True, alpha=0.3)

# 准确率曲线
ax = axes[0, 1]
for name, hist in results.items():
    ax.plot(hist['acc'], label=name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('训练准确率对比')
ax.legend()
ax.grid(True, alpha=0.3)

# 梯度范数
ax = axes[0, 2]
for name, hist in results.items():
    ax.plot(hist['grad_norms'], label=name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Gradient Norm')
ax.set_title('梯度范数对比')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# 最终准确率柱状图
ax = axes[1, 0]
names = list(results.keys())
accs = [results[n]['acc'][-1] for n in names]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
bars = ax.bar(names, accs, color=colors)
ax.set_ylabel('Final Accuracy')
ax.set_title('最终准确率')
ax.set_ylim(0, 1)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{acc:.3f}', ha='center', fontsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 收敛速度
ax = axes[1, 1]
target_acc = 0.9
epochs_to_target = {}
for name, hist in results.items():
    for i, acc in enumerate(hist['acc']):
        if acc >= target_acc:
            epochs_to_target[name] = i + 1
            break
    else:
        epochs_to_target[name] = 100

ax.bar(epochs_to_target.keys(), epochs_to_target.values(), color=colors)
ax.set_ylabel('Epochs to 90% Accuracy')
ax.set_title(f'达到 {target_acc*100:.0f}% 准确率所需轮数')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 原始数据可视化
ax = axes[1, 2]
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.5, label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.5, label='Class 1')
ax.set_title('螺旋数据')
ax.legend()
ax.axis('equal')

plt.tight_layout()
plt.savefig('outputs/activation_comparison.png', dpi=100)
plt.close()
print("对比结果已保存: outputs/activation_comparison.png")

# =============================================================================
# 6. 梯度消失实验
# =============================================================================
print("\n" + "=" * 60)
print("【5. 梯度消失实验】")

def analyze_gradients(model, x):
    """分析各层梯度"""
    model.zero_grad()
    y = torch.zeros(x.size(0), dtype=torch.long)
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    layer_grads = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_grads.append(param.grad.abs().mean().item())
    
    return layer_grads

# 比较深层网络的梯度
print("\n10 层网络各层梯度对比:")
x_sample = torch.randn(32, 2)

deep_activations = {'Sigmoid': nn.Sigmoid, 'ReLU': nn.ReLU, 'GELU': nn.GELU}
gradient_analysis = {}

for name, act_fn in deep_activations.items():
    torch.manual_seed(42)
    model = DeepNet(act_fn, depth=10, width=32)
    grads = analyze_gradients(model, x_sample)
    gradient_analysis[name] = grads
    print(f"\n{name}:")
    print(f"  第1层梯度: {grads[0]:.6f}")
    print(f"  第5层梯度: {grads[4]:.6f}")
    print(f"  第10层梯度: {grads[-1]:.6f}")
    print(f"  梯度衰减比: {grads[0]/grads[-1]:.2f}x")

# 可视化梯度分布
plt.figure(figsize=(10, 5))
for name, grads in gradient_analysis.items():
    plt.plot(range(1, len(grads)+1), grads, 'o-', label=name, linewidth=2, markersize=8)
plt.xlabel('层编号')
plt.ylabel('平均梯度幅值')
plt.title('10 层网络各层梯度对比')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/gradient_vanishing.png', dpi=100)
plt.close()
print("\n梯度消失图已保存: outputs/gradient_vanishing.png")

# =============================================================================
# 7. 决策边界可视化
# =============================================================================
print("\n" + "=" * 60)
print("【6. 决策边界可视化】")

def plot_decision_boundary(model, ax, title):
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model(grid).argmax(1).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=10, alpha=0.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=10, alpha=0.5)
    ax.set_title(title)
    ax.axis('equal')

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, (name, act_fn) in enumerate(activations.items()):
    torch.manual_seed(42)
    model = DeepNet(act_fn, depth=5, width=64)
    train_model(model, loader, epochs=100)
    plot_decision_boundary(model, axes[i], name)

plt.tight_layout()
plt.savefig('outputs/decision_boundaries.png', dpi=100)
plt.close()
print("决策边界图已保存: outputs/decision_boundaries.png")

# =============================================================================
# 8. 结论
# =============================================================================
print("\n" + "=" * 60)
print("【7. 实验结论】")

print("""
实验结论:

1. Sigmoid/Tanh 在深层网络中梯度衰减严重
   - 不推荐用于深层隐藏层

2. ReLU 系列保持较好的梯度传播
   - 计算简单，收敛快
   - 可能有死神经元问题

3. GELU/SiLU 表现优异
   - 平滑，梯度传播好
   - 现代架构的首选

4. 激活函数选择建议:
   - 默认使用 ReLU
   - NLP/Transformer 使用 GELU
   - 追求性能使用 SiLU/Swish
""")

# =============================================================================
# 9. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 增加网络深度到 20 层，观察不同激活函数的表现

2. 添加 BatchNorm，观察对 Sigmoid 的改善

3. 实现 Mish 激活函数并加入对比

4. 分析不同学习率下激活函数的表现差异

5. 用 CIFAR-10 数据集重复此实验
""")

print("\n✅ 激活函数对比分析完成！")
print("下一步：06-dropout.py - Dropout 原理与实现")
