---
name: code-review
description: 审查深度学习代码，提供改进建议
---

# 代码审查技能

此技能用于审查深度学习相关代码，确保代码质量和最佳实践。

## 审查维度

### 1. 代码正确性

- [ ] 模型架构是否正确
- [ ] 损失函数选择是否合适
- [ ] 数据预处理是否正确
- [ ] 维度变换是否正确

### 2. PyTorch 最佳实践

```python
# ✅ 正确：使用 model.train() 和 model.eval()
model.train()
for batch in train_loader:
    ...

model.eval()
with torch.no_grad():
    for batch in val_loader:
        ...

# ❌ 错误：忘记切换模式
for batch in train_loader:
    ...  # Dropout/BatchNorm 可能行为异常
```

```python
# ✅ 正确：使用 torch.no_grad() 进行推理
model.eval()
with torch.no_grad():
    output = model(data)

# ❌ 错误：推理时不禁用梯度计算
model.eval()
output = model(data)  # 浪费内存
```

```python
# ✅ 正确：先 zero_grad 再 backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

# ❌ 错误：忘记 zero_grad，梯度会累积
loss.backward()
optimizer.step()
```

### 3. 性能优化

```python
# ✅ 正确：使用 pin_memory 加速 GPU 传输
train_loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=4
)

# ✅ 正确：使用 non_blocking 异步传输
data = data.to(device, non_blocking=True)
target = target.to(device, non_blocking=True)
```

### 4. 内存管理

```python
# ✅ 正确：记录 loss 值而非 tensor
losses = []
for batch in train_loader:
    loss = criterion(output, target)
    losses.append(loss.item())  # 使用 .item() 获取 Python 数值

# ❌ 错误：存储 tensor 导致计算图无法释放
losses = []
for batch in train_loader:
    loss = criterion(output, target)
    losses.append(loss)  # 保持了对 tensor 的引用
```

### 5. 代码风格

- 使用类型提示增加可读性
- 函数和类要有文档字符串
- 关键步骤要有注释
- 保持一致的命名规范

## 审查模板

```markdown
## 代码审查报告

### 📁 文件: {文件名}

### ✅ 优点

1. ...
2. ...

### ⚠️ 建议改进

1. **问题描述**
   - 位置: 第 X 行
   - 当前代码: `...`
   - 建议修改: `...`
   - 原因: ...

### 🐛 潜在问题

1. ...

### 📊 评分

- 正确性: ⭐⭐⭐⭐⭐
- 可读性: ⭐⭐⭐⭐
- 性能: ⭐⭐⭐
- 总体: 4/5
```

## 常见问题清单

### 数据加载

- [ ] DataLoader 是否使用了多进程 (`num_workers > 0`)
- [ ] 是否启用了 `pin_memory`（GPU 训练时）
- [ ] 数据增强是否在正确的位置

### 模型定义

- [ ] 层的顺序是否正确
- [ ] 激活函数的位置是否合适
- [ ] 是否有不必要的 softmax（CrossEntropyLoss 已包含）

### 训练循环

- [ ] 是否调用了 `optimizer.zero_grad()`
- [ ] `loss.backward()` 是否在 `optimizer.step()` 之前
- [ ] 验证时是否使用了 `model.eval()` 和 `torch.no_grad()`

### 保存和加载

- [ ] 保存的是 `model.state_dict()` 还是整个模型
- [ ] 加载时是否考虑了设备迁移

## 自动化检查

```bash
# 使用 flake8 检查代码风格
flake8 your_code.py

# 使用 mypy 检查类型（如果使用了类型提示）
mypy your_code.py

# 使用 black 格式化代码
black your_code.py
```
