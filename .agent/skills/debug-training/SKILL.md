---
name: debug-training
description: 诊断和修复深度学习训练中的常见问题
---

# 调试训练问题技能

此技能用于诊断和修复深度学习模型训练中的常见问题。

## 常见问题检查清单

### 1. Loss 不下降

**可能原因**：

- [ ] 学习率过高或过低
- [ ] 数据没有正确加载（标签错乱）
- [ ] 模型结构问题
- [ ] 梯度消失/爆炸
- [ ] 数据预处理问题

**诊断步骤**：

```python
# 1. 检查数据
for batch_idx, (data, target) in enumerate(train_loader):
    print(f"数据形状: {data.shape}")
    print(f"标签形状: {target.shape}")
    print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")
    print(f"标签分布: {torch.bincount(target)}")
    break

# 2. 检查模型输出
model.eval()
with torch.no_grad():
    output = model(data)
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

# 3. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}, grad std={param.grad.std():.6f}")
```

### 2. 梯度爆炸

**症状**：

- Loss 变成 NaN 或 Inf
- 权重变得非常大

**解决方案**：

```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 3. 使用更好的初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

### 3. 梯度消失

**症状**：

- 前层梯度接近零
- 权重几乎不更新

**诊断**：

```python
# 检查各层梯度
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: {grad_norm:.6e}")
            if grad_norm < 1e-7:
                print(f"  ⚠️ 梯度可能消失！")
```

**解决方案**：

- 使用 ReLU 代替 Sigmoid/Tanh
- 添加残差连接
- 使用 Batch Normalization
- 使用 Xavier/He 初始化

### 4. 过拟合

**症状**：

- 训练 loss 持续下降
- 验证 loss 开始上升

**解决方案**：

```python
# 1. Dropout
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 添加 Dropout
    nn.Linear(256, 10)
)

# 2. 权重衰减
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3. 早停
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(epochs):
    val_loss = validate()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

### 5. 欠拟合

**症状**：

- 训练和验证 loss 都很高

**解决方案**：

- 增加模型容量（更多层/神经元）
- 训练更多轮次
- 减少正则化
- 检查数据质量

## 诊断工具函数

```python
def diagnose_training(model, train_loader, criterion, device):
    """
    训练诊断工具
    """
    print("=" * 50)
    print("训练诊断报告")
    print("=" * 50)

    # 1. 模型概览
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数: 总计 {total_params:,} / 可训练 {trainable_params:,}")

    # 2. 数据检查
    data, target = next(iter(train_loader))
    print(f"\n📦 数据形状: {data.shape}")
    print(f"   标签形状: {target.shape}")
    print(f"   数据类型: {data.dtype}")

    # 3. 前向传播检查
    model.eval()
    data = data.to(device)
    target = target.to(device)

    with torch.no_grad():
        output = model(data)
        loss = criterion(output, target)

    print(f"\n🔄 前向传播:")
    print(f"   输出形状: {output.shape}")
    print(f"   初始 Loss: {loss.item():.4f}")

    # 4. 梯度检查
    model.train()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    print(f"\n📈 梯度统计:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"   {name}:")
            print(f"      范围: [{grad.min():.2e}, {grad.max():.2e}]")
            print(f"      均值: {grad.mean():.2e}, 标准差: {grad.std():.2e}")

    print("\n" + "=" * 50)
```

## 调试最佳实践

1. **从简单开始**：先用小数据集、简单模型验证流程
2. **逐步添加复杂性**：一次只改一个东西
3. **记录所有实验**：使用日志记录每次实验的配置和结果
4. **可视化一切**：loss 曲线、梯度分布、权重分布
5. **先过拟合再正则化**：确保模型有能力拟合数据
