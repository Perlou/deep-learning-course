"""
07-batch-normalization.py
Phase 4: 神经网络基础

批归一化 (Batch Normalization) - 加速训练的关键

学习目标：
1. 理解 BatchNorm 的原理和公式
2. 了解训练和推理时的区别
3. 掌握不同归一化方法的选择
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - 批归一化 (Batch Normalization)")
print("=" * 60)

# =============================================================================
# 1. 内部协变量偏移问题
# =============================================================================
print("\n【1. 内部协变量偏移问题】")

print("""
问题：Internal Covariate Shift
    - 每层的输入分布随着之前层参数的更新而变化
    - 导致每层需要不断适应新的分布
    - 训练变慢，需要更小的学习率

解决方案：Batch Normalization (Ioffe & Szegedy, 2015)
    - 在每层之后对输入进行归一化
    - 保持每层输入的分布稳定
""")

# =============================================================================
# 2. BatchNorm 公式
# =============================================================================
print("\n" + "=" * 60)
print("【2. BatchNorm 公式】")

print("""
给定一个 mini-batch B = {x₁, x₂, ..., xₘ}:

1. 计算均值:
   μ_B = (1/m) Σxᵢ

2. 计算方差:
   σ²_B = (1/m) Σ(xᵢ - μ_B)²

3. 归一化:
   x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

4. 缩放和平移 (可学习参数):
   yᵢ = γ · x̂ᵢ + β

其中:
- ε: 防止除零的小常数 (如 1e-5)
- γ: 缩放参数 (scale)
- β: 平移参数 (shift)
""")

# =============================================================================
# 3. 手动实现 BatchNorm
# =============================================================================
print("\n" + "=" * 60)
print("【3. 手动实现 BatchNorm】")

class ManualBatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # 运行时统计量（用于推理）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
    
    def __call__(self, x):
        if self.training:
            # 训练时：使用 batch 统计量
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            
            # 更新运行时统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理时：使用运行时统计量
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        return self.gamma * x_norm + self.beta

# 演示
np.random.seed(42)
x = np.random.randn(32, 10) * 5 + 10  # 均值 10，标准差 5
bn = ManualBatchNorm1d(10)

print(f"输入统计: mean={x.mean():.2f}, std={x.std():.2f}")

y = bn(x)
print(f"归一化后: mean={y.mean():.4f}, std={y.std():.4f}")

# =============================================================================
# 4. PyTorch BatchNorm
# =============================================================================
print("\n" + "=" * 60)
print("【4. PyTorch BatchNorm】")

x = torch.randn(32, 64) * 5 + 10

bn1d = nn.BatchNorm1d(num_features=64)
bn2d = nn.BatchNorm2d(num_features=64)

# 训练模式
bn1d.train()
y = bn1d(x)
print(f"1D BatchNorm (训练):")
print(f"  输入: mean={x.mean():.2f}, std={x.std():.2f}")
print(f"  输出: mean={y.mean():.4f}, std={y.std():.4f}")

# 评估模式
bn1d.eval()
y = bn1d(x)
print(f"\n1D BatchNorm (评估):")
print(f"  输出: mean={y.mean():.4f}, std={y.std():.4f}")

# 查看参数
print(f"\n可学习参数:")
print(f"  gamma (weight): shape={bn1d.weight.shape}")
print(f"  beta (bias): shape={bn1d.bias.shape}")
print(f"\n运行时统计 (滑动平均):")
print(f"  running_mean: shape={bn1d.running_mean.shape}")
print(f"  running_var: shape={bn1d.running_var.shape}")

# =============================================================================
# 5. BatchNorm 对训练的影响
# =============================================================================
print("\n" + "=" * 60)
print("【5. BatchNorm 对训练的影响】")

from torch.utils.data import DataLoader, TensorDataset

# 创建数据
np.random.seed(42)
X = np.random.randn(500, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).unsqueeze(1)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 无 BatchNorm
class ModelNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 有 BatchNorm
class ModelWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_model(model, loader, lr=0.01, epochs=100):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(loader))
    
    return losses

print("训练对比 (使用较大学习率 lr=0.1):")
torch.manual_seed(42)
model_no_bn = ModelNoBN()
losses_no_bn = train_model(model_no_bn, loader, lr=0.1)

torch.manual_seed(42)
model_with_bn = ModelWithBN()
losses_with_bn = train_model(model_with_bn, loader, lr=0.1)

print(f"无 BatchNorm: 最终 loss = {losses_no_bn[-1]:.4f}")
print(f"有 BatchNorm: 最终 loss = {losses_with_bn[-1]:.4f}")

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses_no_bn, label='无 BatchNorm', linewidth=2)
plt.plot(losses_with_bn, label='有 BatchNorm', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BatchNorm 对训练的影响 (lr=0.1)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 可视化激活分布
def get_activations(model, x):
    activations = []
    for layer in model.net:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            activations.append(x.detach().numpy().flatten())
    return activations

x_sample = X_train[:100]
model_no_bn.eval()
model_with_bn.eval()

acts_no_bn = get_activations(model_no_bn, x_sample)
acts_with_bn = get_activations(model_with_bn, x_sample)

plt.hist(acts_no_bn[0], bins=50, alpha=0.5, label='无 BN - Layer 1', density=True)
plt.hist(acts_with_bn[0], bins=50, alpha=0.5, label='有 BN - Layer 1', density=True)
plt.xlabel('激活值')
plt.ylabel('密度')
plt.title('激活值分布对比')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/batchnorm_effect.png', dpi=100)
plt.close()
print("BatchNorm 效果图已保存: outputs/batchnorm_effect.png")

# =============================================================================
# 6. 其他归一化方法
# =============================================================================
print("\n" + "=" * 60)
print("【6. 其他归一化方法】")

print("""
╔═══════════════════════════════════════════════════════════════════╗
║                      归一化方法对比                                ║
╠═══════════════╦═══════════════════════════════════════════════════╣
║  方法          ║  归一化维度           ║  适用场景                 ║
╠═══════════════╬═══════════════════════╬═══════════════════════════╣
║  BatchNorm    ║  沿 batch 维度        ║  CNN（batch 足够大）      ║
║  LayerNorm    ║  沿特征维度           ║  NLP/Transformer          ║
║  InstanceNorm ║  单样本单通道         ║  风格迁移                 ║
║  GroupNorm    ║  通道分组内           ║  小 batch/检测            ║
╚═══════════════╩═══════════════════════╩═══════════════════════════╝
""")

# 演示不同归一化
x = torch.randn(2, 4, 3, 3)  # [batch, channels, H, W]

bn = nn.BatchNorm2d(4)      # 沿 batch 归一化
ln = nn.LayerNorm([4, 3, 3])  # 沿 [C, H, W] 归一化
ins = nn.InstanceNorm2d(4)  # 每个样本每个通道单独归一化
gn = nn.GroupNorm(2, 4)     # 将 4 通道分成 2 组

print("不同归一化方法的输出形状:")
print(f"  输入: {x.shape}")
print(f"  BatchNorm2d: {bn(x).shape}")
print(f"  LayerNorm: {ln(x).shape}")
print(f"  InstanceNorm2d: {ins(x).shape}")
print(f"  GroupNorm: {gn(x).shape}")

# =============================================================================
# 7. BatchNorm 注意事项
# =============================================================================
print("\n" + "=" * 60)
print("【7. BatchNorm 注意事项】")

print("""
使用建议:

1. 位置:
   - 通常放在激活函数之前: Linear → BN → ReLU
   - 也有放在激活后的做法

2. batch size:
   - 太小会导致统计量不稳定
   - 推荐 16 以上
   - 小 batch 用 GroupNorm 或 LayerNorm

3. 训练/推理:
   - 训练用 batch 统计量
   - 推理用 running 统计量
   - 记得切换 model.train() / model.eval()

4. 冻结 BatchNorm:
   - 迁移学习时可能需要冻结
   - bn.eval() 保持统计量不变
   - bn.track_running_stats = False

5. 与 Dropout 一起使用:
   - 一般顺序: Linear → BN → ReLU → Dropout
""")

# =============================================================================
# 8. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 解释为什么 BatchNorm 可以使用更大的学习率

2. 实现 LayerNorm（不使用 nn.LayerNorm）

3. 分析 BatchNorm 为什么有正则化效果

4. 比较 BatchNorm 和 LayerNorm 在 RNN 上的表现

5. 解释为什么 BatchNorm 在推理时使用 running stats
""")

# === 答案提示 ===
# 1: 归一化后梯度更稳定，不易爆炸/消失

# 2: LayerNorm
# def layer_norm(x, gamma, beta, eps=1e-5):
#     mean = x.mean(dim=-1, keepdim=True)
#     var = x.var(dim=-1, keepdim=True)
#     return gamma * (x - mean) / (var + eps).sqrt() + beta

# 3: 每个样本看到的归一化参数取决于其他样本
#    类似于数据增强

# 4: RNN 每个时间步的统计量不同
#    LayerNorm 更适合

# 5: 推理时 batch=1，无法计算 batch 统计量
#    使用训练时累积的统计量

print("\n✅ BatchNorm 完成！")
print("下一步：08-weight-regularization.py - 权重正则化")
