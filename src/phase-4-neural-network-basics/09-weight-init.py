"""
09-weight-init.py
Phase 4: 神经网络基础

权重初始化 - 训练成功的关键

学习目标：
1. 理解权重初始化的重要性
2. 掌握 Xavier 和 He 初始化
3. 了解不同激活函数对应的初始化策略
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("神经网络基础 - 权重初始化")
print("=" * 60)

# =============================================================================
# 1. 为什么权重初始化重要
# =============================================================================
print("\n【1. 为什么权重初始化重要】")

print("""
问题:
- 初始化太小: 信号在前向传播中逐渐消失
- 初始化太大: 信号在前向传播中爆炸

理想情况:
- 保持每层输出的方差稳定
- 保持梯度的方差稳定

不好的初始化后果:
- 训练缓慢或不收敛
- 梯度消失/爆炸
- 陷入局部最优
""")

# =============================================================================
# 2. 验证初始化的影响
# =============================================================================
print("\n" + "=" * 60)
print("【2. 验证初始化的影响】")

def forward_pass_stats(init_fn, n_layers=10, width=256):
    """计算前向传播中激活值的统计量"""
    layers = []
    for i in range(n_layers):
        linear = nn.Linear(width, width, bias=False)
        init_fn(linear.weight)
        layers.append(linear)
    
    x = torch.randn(32, width)
    means = []
    stds = []
    
    for layer in layers:
        x = torch.tanh(layer(x))  # 使用 tanh 激活
        means.append(x.mean().item())
        stds.append(x.std().item())
    
    return means, stds

# 不同初始化方法
print("\n不同初始化的激活值变化:")

# 太小
means_small, stds_small = forward_pass_stats(lambda w: init.normal_(w, std=0.01))
print(f"太小 (std=0.01): 最后一层 std = {stds_small[-1]:.6f}")

# 太大
means_large, stds_large = forward_pass_stats(lambda w: init.normal_(w, std=1.0))
print(f"太大 (std=1.0): 最后一层 std = {stds_large[-1]:.6f}")

# Xavier
means_xavier, stds_xavier = forward_pass_stats(lambda w: init.xavier_uniform_(w))
print(f"Xavier: 最后一层 std = {stds_xavier[-1]:.6f}")

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(stds_small, 'b-o', label='太小 (std=0.01)', markersize=6)
plt.plot(stds_large, 'r-s', label='太大 (std=1.0)', markersize=6)
plt.plot(stds_xavier, 'g-^', label='Xavier', markersize=6)
plt.xlabel('层编号')
plt.ylabel('激活值标准差')
plt.title('前向传播中的激活值变化')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
# 可视化权重分布
w_small = torch.zeros(256, 256)
w_large = torch.zeros(256, 256)
w_xavier = torch.zeros(256, 256)
init.normal_(w_small, std=0.01)
init.normal_(w_large, std=1.0)
init.xavier_uniform_(w_xavier)

plt.hist(w_small.flatten().numpy(), bins=50, alpha=0.5, label='太小', density=True)
plt.hist(w_xavier.flatten().numpy(), bins=50, alpha=0.5, label='Xavier', density=True)
plt.xlabel('权重值')
plt.ylabel('密度')
plt.title('初始权重分布')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/init_comparison.png', dpi=100)
plt.close()
print("初始化比较图已保存: outputs/init_comparison.png")

# =============================================================================
# 3. Xavier 初始化
# =============================================================================
print("\n" + "=" * 60)
print("【3. Xavier 初始化 (Glorot)】")

print("""
Xavier 初始化 (Glorot & Bengio, 2010):

原理: 保持前向和反向传播中方差稳定

公式:
    Uniform: W ~ U[-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))]
    Normal:  W ~ N(0, 2/(fan_in+fan_out))

其中:
    fan_in = 输入神经元数
    fan_out = 输出神经元数

适用:
    - Sigmoid, Tanh 激活函数
    - 线性激活
""")

# 演示
linear = nn.Linear(256, 128)
init.xavier_uniform_(linear.weight)
print(f"Xavier Uniform: std = {linear.weight.std():.4f}")
print(f"理论 std = √(2/(256+128)) = {np.sqrt(2/384):.4f}")

init.xavier_normal_(linear.weight)
print(f"Xavier Normal: std = {linear.weight.std():.4f}")

# =============================================================================
# 4. He 初始化 (Kaiming)
# =============================================================================
print("\n" + "=" * 60)
print("【4. He 初始化 (Kaiming)】")

print("""
He 初始化 (He et al., 2015):

原理: 针对 ReLU 激活函数优化

公式:
    Normal:  W ~ N(0, 2/fan_in)
    Uniform: W ~ U[-√(6/fan_in), √(6/fan_in)]

改进:
    - ReLU 会将一半的激活置零
    - 需要方差加倍补偿

mode 参数:
    - 'fan_in': 保持前向传播方差
    - 'fan_out': 保持反向传播方差

适用:
    - ReLU, Leaky ReLU 激活函数
""")

# 验证 He 初始化对 ReLU 的效果
def forward_pass_relu(init_fn, n_layers=10, width=256):
    layers = []
    for i in range(n_layers):
        linear = nn.Linear(width, width, bias=False)
        init_fn(linear.weight)
        layers.append(linear)
    
    x = torch.randn(32, width)
    stds = []
    
    for layer in layers:
        x = torch.relu(layer(x))  # 使用 ReLU
        stds.append(x.std().item())
    
    return stds

print("\nReLU 网络测试:")
stds_xavier_relu = forward_pass_relu(lambda w: init.xavier_normal_(w))
stds_he_relu = forward_pass_relu(lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'))

print(f"Xavier + ReLU: 最后一层 std = {stds_xavier_relu[-1]:.4f}")
print(f"He + ReLU: 最后一层 std = {stds_he_relu[-1]:.4f}")

# =============================================================================
# 5. PyTorch 初始化函数
# =============================================================================
print("\n" + "=" * 60)
print("【5. PyTorch 初始化函数】")

print("""
常用初始化函数:

init.zeros_(tensor)           # 全零
init.ones_(tensor)            # 全一
init.constant_(tensor, val)   # 常数
init.normal_(tensor, mean, std)   # 正态分布
init.uniform_(tensor, a, b)       # 均匀分布
init.xavier_uniform_(tensor)      # Xavier 均匀
init.xavier_normal_(tensor)       # Xavier 正态
init.kaiming_uniform_(tensor)     # He 均匀
init.kaiming_normal_(tensor)      # He 正态
init.orthogonal_(tensor)          # 正交初始化
init.sparse_(tensor, sparsity)    # 稀疏初始化
""")

# 演示
linear = nn.Linear(256, 128)

# He 初始化
init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
init.zeros_(linear.bias)  # bias 通常初始化为 0
print(f"He Normal (fan_out): std = {linear.weight.std():.4f}")

# =============================================================================
# 6. 自定义初始化
# =============================================================================
print("\n" + "=" * 60)
print("【6. 自定义初始化】")

def init_weights(m):
    """自定义模型初始化函数"""
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(m.weight)
        init.zeros_(m.bias)

# 应用到模型
class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SampleModel()
model.apply(init_weights)  # 应用自定义初始化
print("自定义初始化已应用")

# 验证
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"  {name}: std = {param.std():.4f}")

# =============================================================================
# 7. 初始化对训练的影响
# =============================================================================
print("\n" + "=" * 60)
print("【7. 初始化对训练的影响】")

from torch.utils.data import DataLoader, TensorDataset

# 创建数据
np.random.seed(42)
X = np.random.randn(1000, 128)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
X_t, y_t = torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

def train_with_init(init_name, init_fn, epochs=100):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid()
    )
    
    # 应用初始化
    for m in model:
        if isinstance(m, nn.Linear):
            init_fn(m.weight)
            init.zeros_(m.bias)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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

print("训练不同初始化的模型...")
losses_zero = train_with_init('零初始化', lambda w: init.zeros_(w))
losses_small = train_with_init('太小', lambda w: init.normal_(w, std=0.001))
losses_large = train_with_init('太大', lambda w: init.normal_(w, std=1.0))
losses_xavier = train_with_init('Xavier', lambda w: init.xavier_normal_(w))
losses_he = train_with_init('He', lambda w: init.kaiming_normal_(w, nonlinearity='relu'))

print(f"零初始化: 最终 loss = {losses_zero[-1]:.4f}")
print(f"太小: 最终 loss = {losses_small[-1]:.4f}")
print(f"太大: 最终 loss = {losses_large[-1]:.4f}")
print(f"Xavier: 最终 loss = {losses_xavier[-1]:.4f}")
print(f"He: 最终 loss = {losses_he[-1]:.4f}")

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(losses_zero, label='零初始化', linewidth=2, alpha=0.7)
plt.plot(losses_small, label='太小 (std=0.001)', linewidth=2, alpha=0.7)
plt.plot(losses_large, label='太大 (std=1.0)', linewidth=2, alpha=0.7)
plt.plot(losses_xavier, label='Xavier', linewidth=2)
plt.plot(losses_he, label='He/Kaiming', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同初始化方法的训练曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/init_training.png', dpi=100)
plt.close()
print("训练曲线图已保存: outputs/init_training.png")

# =============================================================================
# 8. 初始化选择指南
# =============================================================================
print("\n" + "=" * 60)
print("【8. 初始化选择指南】")

print("""
╔═══════════════════════════════════════════════════════════════════╗
║                   初始化方法选择指南                               ║
╠═══════════════════╦═══════════════════════════════════════════════╣
║  激活函数          ║  推荐初始化                                   ║
╠═══════════════════╬═══════════════════════════════════════════════╣
║  Sigmoid, Tanh    ║  Xavier (Glorot)                              ║
║  ReLU             ║  He (Kaiming, mode='fan_in')                  ║
║  Leaky ReLU       ║  He (nonlinearity='leaky_relu')               ║
║  SELU             ║  LeCun Normal                                 ║
║  无激活           ║  Xavier                                        ║
╠═══════════════════╬═══════════════════════════════════════════════╣
║  特殊层            ║  初始化                                        ║
╠═══════════════════╬═══════════════════════════════════════════════╣
║  BatchNorm        ║  weight=1, bias=0                             ║
║  Embedding        ║  Normal(0, 1) 或预训练                        ║
║  残差连接最后层   ║  初始化为 0                                    ║
║  Transformer      ║  Xavier + 特殊缩放                            ║
╚═══════════════════╩═══════════════════════════════════════════════╝

PyTorch 默认:
- nn.Linear: Kaiming Uniform (适合 ReLU)
- nn.Conv2d: Kaiming Uniform (适合 ReLU)
""")

# =============================================================================
# 9. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 推导 Xavier 初始化的方差公式

2. 解释为什么 He 初始化要乘以 2

3. 实现正交初始化 (Orthogonal)

4. 分析 LSTM 应该使用什么初始化

5. 测试深层网络 (50层) 不同初始化的梯度传播
""")

# === 答案提示 ===
# 1: Var(y) = n * Var(w) * Var(x)
#    要保持 Var(y) = Var(x)，需要 Var(w) = 1/n

# 2: ReLU 将约一半的激活置零
#    为保持方差，需要 Var(w) = 2/n

# 3: 正交初始化
# Q, _ = torch.linalg.qr(torch.randn(n, n))
# weight.data = Q[:weight.shape[0], :weight.shape[1]]

# 4: LSTM:
#    - 输入和隐藏权重用 Xavier
#    - 遗忘门 bias 初始化为 1-2

# 5: 深层网络实验
# for depth in [10, 30, 50]:
#     test_gradient_flow(depth, init_method)

print("\n✅ 权重初始化完成！")
print("🎉 Phase 4 全部模块完成！")
print("\n下一步：完成房价预测实战项目")
