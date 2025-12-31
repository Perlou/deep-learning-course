"""
06-dropout.py
Phase 4: 神经网络基础

Dropout - 正则化技术

学习目标：
1. 理解 Dropout 的原理
2. 掌握训练和推理时的区别
3. 了解 Dropout 的变体
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - Dropout")
print("=" * 60)

# =============================================================================
# 1. Dropout 原理
# =============================================================================
print("\n【1. Dropout 原理】")

print("""
Dropout (Srivastava et al., 2014):

训练时:
    - 以概率 p 随机将神经元置为 0
    - 保留的神经元输出除以 (1-p)（保持期望不变）
    
    mask = Bernoulli(1-p)
    out = (x * mask) / (1-p)

推理时:
    - 不进行 dropout，使用全部神经元
    
为什么有效:
1. 集成学习: 相当于训练多个子网络的集成
2. 打破共适应: 防止神经元过度依赖特定神经元
3. 正则化效果: 类似于 L2 正则化
""")

# =============================================================================
# 2. 手动实现 Dropout
# =============================================================================
print("\n" + "=" * 60)
print("【2. 手动实现 Dropout】")

def dropout_manual(x, p=0.5, training=True):
    """手动实现 Dropout"""
    if not training or p == 0:
        return x
    
    # 生成掩码
    mask = (torch.rand_like(x) > p).float()
    
    # 应用掩码并缩放
    return x * mask / (1 - p)

# 演示
torch.manual_seed(42)
x = torch.ones(1, 10)
print(f"输入: {x}")

print(f"\n训练时 (p=0.5):")
for i in range(3):
    out = dropout_manual(x, p=0.5, training=True)
    print(f"  尝试 {i+1}: {out}")

print(f"\n推理时:")
out = dropout_manual(x, p=0.5, training=False)
print(f"  输出: {out}")

# =============================================================================
# 3. PyTorch Dropout
# =============================================================================
print("\n" + "=" * 60)
print("【3. PyTorch Dropout】")

dropout = nn.Dropout(p=0.5)

x = torch.ones(1, 10)

# 训练模式
dropout.train()
print(f"训练模式 (model.train()):")
for i in range(3):
    out = dropout(x)
    print(f"  尝试 {i+1}: {out}")

# 评估模式
dropout.eval()
print(f"\n评估模式 (model.eval()):")
out = dropout(x)
print(f"  输出: {out}")

# =============================================================================
# 4. Dropout 对训练的影响
# =============================================================================
print("\n" + "=" * 60)
print("【4. Dropout 对训练的影响】")

# 创建数据（过拟合场景：少量数据）
np.random.seed(42)
n_samples = 50
X = np.random.randn(n_samples, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# 添加噪声
y_noisy = y.copy()
y_noisy[:5] = 1 - y_noisy[:5]  # 10% 噪声

X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y_noisy).unsqueeze(1)

# 定义模型
class ModelWithDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# 训练和比较
def train_and_evaluate(model, X, y, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses

print("训练模型...")
torch.manual_seed(42)
model_no_dropout = ModelWithDropout(p=0.0)
losses_no_dropout = train_and_evaluate(model_no_dropout, X_train, y_train)

torch.manual_seed(42)
model_with_dropout = ModelWithDropout(p=0.5)
losses_with_dropout = train_and_evaluate(model_with_dropout, X_train, y_train)

print(f"无 Dropout 最终损失: {losses_no_dropout[-1]:.4f}")
print(f"有 Dropout 最终损失: {losses_with_dropout[-1]:.4f}")

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses_no_dropout, label='无 Dropout', alpha=0.8)
plt.plot(losses_with_dropout, label='Dropout p=0.5', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 绘制 Dropout 示意图
np.random.seed(42)
neurons = np.ones((4, 6))
mask = np.random.rand(4, 6) < 0.5
neurons_dropped = neurons.copy()
neurons_dropped[mask] = 0

plt.imshow(neurons_dropped, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Dropout 示意 (p=0.5)')
plt.xlabel('神经元')
plt.ylabel('样本')
plt.colorbar(label='激活状态')

plt.tight_layout()
plt.savefig('outputs/dropout_effect.png', dpi=100)
plt.close()
print("Dropout 效果图已保存: outputs/dropout_effect.png")

# =============================================================================
# 5. 不同 Dropout 率
# =============================================================================
print("\n" + "=" * 60)
print("【5. 不同 Dropout 率】")

dropout_rates = [0.0, 0.2, 0.5, 0.7]
results = {}

for p in dropout_rates:
    torch.manual_seed(42)
    model = ModelWithDropout(p=p)
    losses = train_and_evaluate(model, X_train, y_train)
    results[p] = losses
    print(f"Dropout p={p}: 最终损失 = {losses[-1]:.4f}")

# 可视化
plt.figure(figsize=(8, 5))
for p, losses in results.items():
    plt.plot(losses, label=f'p={p}', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同 Dropout 率对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/dropout_rates.png', dpi=100)
plt.close()
print("Dropout 率对比图已保存: outputs/dropout_rates.png")

# =============================================================================
# 6. Dropout 变体
# =============================================================================
print("\n" + "=" * 60)
print("【6. Dropout 变体】")

print("""
1. Dropout2d (Spatial Dropout):
   - 用于 CNN，丢弃整个通道
   - 保持空间相关性
   nn.Dropout2d(p=0.5)

2. DropPath (Stochastic Depth):
   - 随机跳过整个层
   - 用于 ResNet, Vision Transformer
   
3. DropConnect:
   - 丢弃权重而不是激活值
   
4. Alpha Dropout:
   - 用于 SELU 激活函数
   - 保持自归一化性质
   nn.AlphaDropout(p=0.5)
""")

# Dropout2d 演示
print("\nDropout2d 演示:")
x = torch.ones(1, 4, 3, 3)  # [batch, channels, height, width]
dropout2d = nn.Dropout2d(p=0.5)
dropout2d.train()
out = dropout2d(x)
print(f"输入形状: {x.shape}")
print(f"输出（某些通道被完全置零）:")
print(out[0])

# =============================================================================
# 7. 正确使用 Dropout
# =============================================================================
print("\n" + "=" * 60)
print("【7. 正确使用 Dropout】")

print("""
使用建议:

1. 位置:
   - 通常放在全连接层之后
   - CNN 中可用 Dropout2d
   - 不建议放在输出层

2. 概率选择:
   - 一般 p = 0.5
   - BatchNorm 后可降低到 0.2-0.3
   - 小数据集可提高到 0.7

3. 注意事项:
   - 训练时用 model.train()
   - 推理时用 model.eval()
   - 与 BatchNorm 一起用时要注意顺序
""")

# 正确的模型使用示例
class ProperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)  # BN 后用较低的 dropout
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)  # Dropout 放在激活之后
        x = self.fc2(x)
        return x

# =============================================================================
# 8. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 解释为什么训练时要除以 (1-p)

2. 实现 DropConnect（丢弃权重）

3. 比较 Dropout 放在激活前 vs 激活后的效果

4. 分析 Dropout 与 L2 正则化的关系

5. 实现 Stochastic Depth (DropPath)
""")

# === 答案提示 ===
# 1: 保持期望值不变，E[dropout(x)] = E[x]

# 2: DropConnect
# def dropconnect(x, w, p=0.5):
#     mask = (torch.rand_like(w) > p).float()
#     return F.linear(x, w * mask / (1-p))

# 3: 通常激活后更好，但差异不大

# 4: Dropout 近似于 L2 正则化，但更灵活

# 5: DropPath
# def drop_path(x, p):
#     if not training or p == 0:
#         return x
#     keep_prob = 1 - p
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = keep_prob + torch.rand(shape, device=x.device)
#     random_tensor.floor_()
#     return x.div(keep_prob) * random_tensor

print("\n✅ Dropout 完成！")
print("下一步：07-batch-normalization.py - BatchNorm 详解")
