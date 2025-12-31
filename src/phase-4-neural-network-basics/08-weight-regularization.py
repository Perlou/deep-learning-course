"""
08-weight-regularization.py
Phase 4: 神经网络基础

权重正则化 - L1/L2 正则化

学习目标：
1. 理解正则化的作用
2. 掌握 L1 和 L2 正则化
3. 了解 weight decay 的实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - 权重正则化")
print("=" * 60)

# =============================================================================
# 1. 为什么需要正则化
# =============================================================================
print("\n【1. 为什么需要正则化】")

print("""
过拟合问题:
- 模型在训练集上表现好，测试集上差
- 模型学习了噪声而非真正的模式
- 模型复杂度过高

正则化的作用:
- 限制模型复杂度
- 惩罚过大的权重
- 提高泛化能力

常见正则化方法:
1. L1 正则化 (Lasso)
2. L2 正则化 (Ridge / Weight Decay)
3. Dropout
4. 早停 (Early Stopping)
5. 数据增强
""")

# =============================================================================
# 2. L2 正则化
# =============================================================================
print("\n" + "=" * 60)
print("【2. L2 正则化 (Weight Decay)】")

print("""
L2 正则化公式:
    L_total = L_data + λ · Σwᵢ²
    
梯度:
    ∂L/∂w = ∂L_data/∂w + 2λw
    
更新规则:
    w = w - lr · (∂L_data/∂w + 2λw)
      = w · (1 - 2λ·lr) - lr · ∂L_data/∂w

效果:
- 权重趋向于较小的值
- 不会变成恰好为 0
- 让权重分布更均匀
""")

# =============================================================================
# 3. L1 正则化
# =============================================================================
print("\n" + "=" * 60)
print("【3. L1 正则化 (Lasso)】")

print("""
L1 正则化公式:
    L_total = L_data + λ · Σ|wᵢ|
    
梯度:
    ∂L/∂w = ∂L_data/∂w + λ · sign(w)
    
效果:
- 权重可以恰好变为 0
- 产生稀疏解
- 自动特征选择
""")

# =============================================================================
# 4. 对比实验
# =============================================================================
print("\n" + "=" * 60)
print("【4. L1 vs L2 对比实验】")

# 创建过拟合场景
np.random.seed(42)
n_train = 30
n_test = 100

# 真实函数 + 噪声
def true_fn(x):
    return np.sin(x * 2)

X_train = np.random.uniform(-np.pi, np.pi, n_train)
y_train = true_fn(X_train) + np.random.randn(n_train) * 0.3

X_test = np.linspace(-np.pi, np.pi, n_test)
y_test = true_fn(X_test)

X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test).unsqueeze(1)

# 定义高容量模型
class HighCapacityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_with_regularization(reg_type='none', lambda_reg=0.01, epochs=500):
    torch.manual_seed(42)
    model = HighCapacityModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        
        # 添加正则化
        if reg_type == 'l2':
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + lambda_reg * l2_reg
        elif reg_type == 'l1':
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            loss = loss + lambda_reg * l1_reg
        
        loss.backward()
        optimizer.step()
        
        # 记录
        train_losses.append(loss.item())
        with torch.no_grad():
            test_output = model(X_test_t)
            test_loss = criterion(test_output, torch.FloatTensor(y_test).unsqueeze(1))
            test_losses.append(test_loss.item())
    
    return model, train_losses, test_losses

# 训练三种模型
print("训练中...")
model_none, train_none, test_none = train_with_regularization('none')
model_l2, train_l2, test_l2 = train_with_regularization('l2', lambda_reg=0.001)
model_l1, train_l1, test_l1 = train_with_regularization('l1', lambda_reg=0.0001)

print(f"无正则化: 测试损失 = {test_none[-1]:.4f}")
print(f"L2 正则化: 测试损失 = {test_l2[-1]:.4f}")
print(f"L1 正则化: 测试损失 = {test_l1[-1]:.4f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 预测曲线
ax = axes[0, 0]
ax.scatter(X_train, y_train, c='blue', s=30, label='训练数据', alpha=0.7)
ax.plot(X_test, y_test, 'g-', linewidth=2, label='真实函数')
with torch.no_grad():
    ax.plot(X_test, model_none(X_test_t).numpy(), 'r-', linewidth=2, label='无正则化')
    ax.plot(X_test, model_l2(X_test_t).numpy(), 'm--', linewidth=2, label='L2 正则化')
    ax.plot(X_test, model_l1(X_test_t).numpy(), 'c:', linewidth=2, label='L1 正则化')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('模型预测对比')
ax.legend()
ax.grid(True, alpha=0.3)

# 损失曲线
ax = axes[0, 1]
ax.plot(test_none, label='无正则化', linewidth=2)
ax.plot(test_l2, label='L2 正则化', linewidth=2)
ax.plot(test_l1, label='L1 正则化', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Loss')
ax.set_title('测试损失对比')
ax.legend()
ax.grid(True, alpha=0.3)

# 权重分布
ax = axes[1, 0]
weights_none = torch.cat([p.flatten() for p in model_none.parameters()]).detach().numpy()
weights_l2 = torch.cat([p.flatten() for p in model_l2.parameters()]).detach().numpy()
weights_l1 = torch.cat([p.flatten() for p in model_l1.parameters()]).detach().numpy()

ax.hist(weights_none, bins=50, alpha=0.5, label='无正则化', density=True)
ax.hist(weights_l2, bins=50, alpha=0.5, label='L2', density=True)
ax.hist(weights_l1, bins=50, alpha=0.5, label='L1', density=True)
ax.set_xlabel('权重值')
ax.set_ylabel('密度')
ax.set_title('权重分布对比')
ax.legend()

# 权重稀疏性
ax = axes[1, 1]
threshold = 0.01
sparsity_none = (np.abs(weights_none) < threshold).mean()
sparsity_l2 = (np.abs(weights_l2) < threshold).mean()
sparsity_l1 = (np.abs(weights_l1) < threshold).mean()

methods = ['无正则化', 'L2', 'L1']
sparsities = [sparsity_none, sparsity_l2, sparsity_l1]
bars = ax.bar(methods, sparsities, color=['red', 'magenta', 'cyan'])
ax.set_ylabel(f'稀疏度 (|w| < {threshold})')
ax.set_title('权重稀疏性对比')
for bar, s in zip(bars, sparsities):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{s:.2%}', ha='center')

plt.tight_layout()
plt.savefig('outputs/regularization_comparison.png', dpi=100)
plt.close()
print("正则化对比图已保存: outputs/regularization_comparison.png")

# =============================================================================
# 5. PyTorch 中的 Weight Decay
# =============================================================================
print("\n" + "=" * 60)
print("【5. PyTorch 中的 Weight Decay】")

print("""
PyTorch 优化器内置 weight_decay 参数:

# SGD with weight decay (等价于 L2)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# AdamW (推荐，解耦的 weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

注意:
- Adam + weight_decay ≠ Adam + L2 正则化
- AdamW 是正确的实现
- 一般不对 bias 和 LayerNorm 参数使用 weight decay
""")

# 分离参数组
class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn(self.fc1(x)))
        return self.fc2(x)

model = SampleModel()

# 分离参数
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if 'bias' in name or 'bn' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=0.001)

print(f"带 weight decay 的参数数: {len(decay_params)}")
print(f"不带 weight decay 的参数数: {len(no_decay_params)}")

# =============================================================================
# 6. 正则化强度选择
# =============================================================================
print("\n" + "=" * 60)
print("【6. 正则化强度选择】")

lambdas = [0, 0.0001, 0.001, 0.01, 0.1]
results = {}

print("不同正则化强度:")
for lam in lambdas:
    _, _, test_losses = train_with_regularization('l2', lambda_reg=lam, epochs=300)
    results[lam] = test_losses[-1]
    print(f"  λ={lam}: 测试损失 = {test_losses[-1]:.4f}")

# 可视化
plt.figure(figsize=(8, 5))
plt.semilogx([l if l > 0 else 0.00001 for l in lambdas], list(results.values()), 'bo-', linewidth=2, markersize=10)
plt.xlabel('正则化强度 λ (log scale)')
plt.ylabel('测试损失')
plt.title('正则化强度 vs 测试损失')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/regularization_strength.png', dpi=100)
plt.close()
print("正则化强度图已保存: outputs/regularization_strength.png")

# =============================================================================
# 7. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 推导 L2 正则化的梯度更新公式

2. 解释为什么 L1 可以产生稀疏解

3. 实现 Elastic Net (L1 + L2 组合)

4. 比较 Adam + weight_decay 和 AdamW 的区别

5. 为什么通常不对 bias 使用 weight decay
""")

# === 答案提示 ===
# 1: w_new = w - lr * (grad + 2λw) = (1 - 2λ*lr) * w - lr * grad

# 2: L1 的梯度是 sign(w)，无论 w 多小，梯度都是 ±λ
#    小权重会被快速推向 0

# 3: Elastic Net
# def elastic_net_loss(model, alpha=0.5, lambda_reg=0.01):
#     l1 = sum(p.abs().sum() for p in model.parameters())
#     l2 = sum(p.pow(2).sum() for p in model.parameters())
#     return lambda_reg * (alpha * l1 + (1-alpha) * l2)

# 4: Adam 的 weight decay 被动量影响，不正确
#    AdamW 解耦 weight decay，是正确的实现

# 5: bias 不会导致过拟合
#    bias 只在某个方向平移决策边界

print("\n✅ 权重正则化完成！")
print("下一步：09-weight-init.py - 权重初始化")
