"""
06-optimizers.py
Phase 3: PyTorch 核心技能

优化器 - 更新模型参数的策略

学习目标：
1. 理解优化器的工作原理
2. 掌握常用优化器的使用
3. 了解学习率调度器
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("PyTorch 核心技能 - 优化器")
print("=" * 60)

# =============================================================================
# 1. 优化器基础
# =============================================================================
print("\n【1. 优化器基础】")

print("""
优化器的职责:
1. 保存模型参数的引用
2. 根据梯度更新参数
3. 管理学习率和动量等超参数

基本用法:
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()   # 清零梯度
    loss.backward()         # 计算梯度
    optimizer.step()        # 更新参数
""")

# 创建一个简单模型
model = nn.Linear(10, 5)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(f"模型参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# =============================================================================
# 2. 常用优化器
# =============================================================================
print("\n" + "=" * 60)
print("【2. 常用优化器】")

# 2.1 SGD
print("\n2.1 SGD (Stochastic Gradient Descent):")
print("    θ = θ - lr × ∇L")
sgd = optim.SGD(model.parameters(), lr=0.01)
print(f"    {sgd}")

# 2.2 SGD with Momentum
print("\n2.2 SGD with Momentum:")
print("    v = momentum × v + ∇L")
print("    θ = θ - lr × v")
sgd_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"    {sgd_momentum}")

# 2.3 SGD with Nesterov
print("\n2.3 SGD with Nesterov:")
sgd_nesterov = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
print(f"    {sgd_nesterov}")

# 2.4 Adam
print("\n2.4 Adam (常用默认选择):")
print("    结合 Momentum 和 RMSprop")
adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
print(f"    {adam}")

# 2.5 AdamW (权重衰减解耦)
print("\n2.5 AdamW (推荐):")
print("    Adam + 正确的权重衰减")
adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print(f"    {adamw}")

# 2.6 RMSprop
print("\n2.6 RMSprop:")
rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
print(f"    {rmsprop}")

# =============================================================================
# 3. 优化器使用示例
# =============================================================================
print("\n" + "=" * 60)
print("【3. 优化器使用示例】")

# 创建简单的训练数据
torch.manual_seed(42)
X = torch.randn(100, 10)
y = torch.randn(100, 5)

# 创建模型和优化器
model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 训练循环
print("\n训练过程:")
for epoch in range(5):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度！
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

# =============================================================================
# 4. 参数组
# =============================================================================
print("\n" + "=" * 60)
print("【4. 参数组 (不同层不同学习率)】")

class TwoPartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 20)
        self.head = nn.Linear(20, 5)
    
    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))

model = TwoPartModel()

# 不同层使用不同学习率
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},  # backbone 较小学习率
    {'params': model.head.parameters(), 'lr': 1e-3}       # head 较大学习率
])

print("参数组配置:")
for i, group in enumerate(optimizer.param_groups):
    print(f"  组 {i}: lr={group['lr']}")

# =============================================================================
# 5. 学习率调度器
# =============================================================================
print("\n" + "=" * 60)
print("【5. 学习率调度器】")

model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 5.1 StepLR
print("\n5.1 StepLR (每 N 个 epoch 衰减):")
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print(f"    每 10 个 epoch 学习率乘以 0.1")

# 5.2 ExponentialLR
print("\n5.2 ExponentialLR:")
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
print(f"    每个 epoch 学习率乘以 0.9")

# 5.3 CosineAnnealingLR
print("\n5.3 CosineAnnealingLR:")
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
print(f"    余弦退火到最小学习率 1e-6")

# 5.4 OneCycleLR
print("\n5.4 OneCycleLR (推荐):")
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, total_steps=1000, 
    pct_start=0.3, anneal_strategy='cos'
)
print(f"    预热 + 余弦退火")

# 5.5 ReduceLROnPlateau
print("\n5.5 ReduceLROnPlateau (根据验证集调整):")
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)
print(f"    验证 loss 不下降时降低学习率")

# 使用示例
print("\n学习率调度器使用:")
model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(15):
    # 训练...
    lr = optimizer.param_groups[0]['lr']
    if epoch % 3 == 0:
        print(f"  Epoch {epoch}: lr = {lr:.6f}")
    
    scheduler.step()  # 更新学习率

# =============================================================================
# 6. 梯度裁剪
# =============================================================================
print("\n" + "=" * 60)
print("【6. 梯度裁剪】")

print("""
梯度裁剪防止梯度爆炸:
- clip_grad_norm_: 按范数裁剪
- clip_grad_value_: 按值裁剪
""")

model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters())

# 模拟训练
x = torch.randn(3, 10)
y = torch.randn(3, 5)
loss = nn.MSELoss()(model(x), y)
loss.backward()

# 裁剪前
total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
print(f"裁剪前梯度范数: {total_norm_before:.4f}")

# 按范数裁剪
loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"裁剪后梯度范数: {min(total_norm, 1.0):.4f}")

# =============================================================================
# 7. 权重衰减
# =============================================================================
print("\n" + "=" * 60)
print("【7. 权重衰减 (L2 正则化)】")

print("""
权重衰减 = L2 正则化
- 防止过拟合
- AdamW 使用解耦的权重衰减（推荐）
""")

# 带权重衰减
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print(f"weight_decay = 0.01")

# =============================================================================
# 8. 优化器状态
# =============================================================================
print("\n" + "=" * 60)
print("【8. 优化器状态保存与加载】")

# 保存
optimizer_state = optimizer.state_dict()
print(f"优化器状态键: {optimizer_state.keys()}")

# 加载
new_optimizer = optim.AdamW(model.parameters(), lr=0.001)
new_optimizer.load_state_dict(optimizer_state)
print("优化器状态已加载")

# =============================================================================
# 9. 优化器选择建议
# =============================================================================
print("\n" + "=" * 60)
print("【9. 优化器选择建议】")

print("""
╔════════════════════════════════════════════════════════════╗
║                    优化器选择指南                          ║
╠════════════════╦═══════════════════════════════════════════╣
║  任务           ║  推荐优化器                              ║
╠════════════════╬═══════════════════════════════════════════╣
║  默认选择       ║  AdamW, lr=1e-3 或 3e-4                  ║
║  CV 任务        ║  SGD + Momentum + CosineAnnealing        ║
║  NLP/微调       ║  AdamW, lr=1e-5 ~ 2e-5                   ║
║  大 batch       ║  LAMB/LARS + Warmup                      ║
║  追求泛化       ║  SGD + Momentum (收敛后切换)             ║
╚════════════════╩═══════════════════════════════════════════╝
""")

# =============================================================================
# 10. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 创建一个模型，使用 AdamW 优化器训练

2. 实现不同层使用不同学习率的配置

3. 使用 CosineAnnealingLR 调度器，观察学习率变化

4. 比较 Adam 和 SGD+Momentum 的收敛速度

5. 解释 AdamW 和 Adam 的区别
""")

# === 练习答案 ===
# 1
# model = nn.Linear(10, 5)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# for epoch in range(10):
#     loss = nn.MSELoss()(model(torch.randn(32, 10)), torch.randn(32, 5))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# 2
# optimizer = optim.Adam([
#     {'params': model.layer1.parameters(), 'lr': 1e-4},
#     {'params': model.layer2.parameters(), 'lr': 1e-3}
# ])

# 3
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# for epoch in range(50):
#     print(f"Epoch {epoch}: lr = {scheduler.get_last_lr()[0]:.6f}")
#     scheduler.step()

# 4
# 对比代码见上面的训练循环示例

# 5
# Adam: weight_decay 是添加到梯度上的 L2
# AdamW: weight_decay 是直接从权重上衰减
# AdamW 的方式更符合 L2 正则化的原意，效果更好

print("\n✅ 优化器完成！")
print("下一步：07-dataset-dataloader.py - 数据加载")
