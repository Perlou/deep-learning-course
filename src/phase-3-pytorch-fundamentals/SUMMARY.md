# Phase 3 学习总结：PyTorch 核心技能

## 📚 模块概览

| 模块 | 主题        | 核心概念               |
| ---- | ----------- | ---------------------- |
| 01   | Tensor 基础 | 创建、属性、设备管理   |
| 02   | Tensor 运算 | 数学运算、广播、索引   |
| 03   | 自动微分    | Autograd、计算图       |
| 04   | nn.Module   | 模型构建、参数管理     |
| 05   | 损失函数    | 分类、回归损失         |
| 06   | 优化器      | SGD、Adam、学习率调度  |
| 07   | 数据加载    | Dataset、DataLoader    |
| 08   | 数据增强    | 图像、文本增强         |
| 09   | 训练循环    | 完整训练流程           |
| 10   | 模型保存    | state_dict、checkpoint |

---

## 1️⃣ Tensor 基础

### 创建方式

```python
torch.tensor([1, 2, 3])           # 从列表
torch.zeros(3, 4)                 # 全零
torch.ones(3, 4)                  # 全一
torch.randn(3, 4)                 # 标准正态
torch.arange(0, 10, 2)            # 等差序列
torch.eye(3)                      # 单位矩阵
torch.from_numpy(np_array)        # 从 NumPy
```

### 关键属性

```python
x.shape          # 形状
x.dtype          # 数据类型
x.device         # 设备 (cpu/cuda)
x.requires_grad  # 是否需要梯度
x.numel()        # 元素总数
```

### 设备管理

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
model = model.to(device)
```

---

## 2️⃣ Tensor 运算

### 基本运算

```python
a + b, a - b, a * b, a / b    # 元素级运算
a @ b                          # 矩阵乘法
torch.mm(A, B)                 # 2D 矩阵乘法
torch.bmm(A, B)                # 批量矩阵乘法
torch.dot(a, b)                # 向量点积
```

### 广播规则

```
从后往前对齐维度，维度为 1 的可以广播
(3, 4) + (4,) = (3, 4)
(3, 4) + (3, 1) = (3, 4)
```

### 形状变换

```python
x.view(3, 4)           # 改变形状（要求连续）
x.reshape(3, 4)        # 改变形状（更灵活）
x.squeeze()            # 移除维度=1 的维
x.unsqueeze(0)         # 增加维度
x.permute(2, 0, 1)     # 维度重排
torch.cat([a, b], dim=0)  # 拼接
torch.stack([a, b])       # 堆叠
```

---

## 3️⃣ 自动微分 (Autograd)

### 核心概念

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()      # 反向传播
print(x.grad)     # 查看梯度
```

### 关键点

- **梯度累积**：每次 `backward()` 梯度会累加，需要 `zero_grad()`
- **计算图**：动态创建，`backward()` 后释放
- **禁用梯度**：`with torch.no_grad():` 或 `.detach()`

### 常用模式

```python
# 训练
loss.backward()
optimizer.step()
optimizer.zero_grad()

# 推理
model.eval()
with torch.no_grad():
    output = model(x)
```

---

## 4️⃣ nn.Module

### 自定义模型

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### 常用层

| 层               | 用途     |
| ---------------- | -------- |
| `nn.Linear`      | 全连接层 |
| `nn.Conv2d`      | 2D 卷积  |
| `nn.BatchNorm2d` | 批归一化 |
| `nn.Dropout`     | Dropout  |
| `nn.LSTM`        | LSTM     |
| `nn.Embedding`   | 词嵌入   |

### 参数管理

```python
model.parameters()           # 所有参数
model.named_parameters()     # 带名称的参数
model.train()               # 训练模式
model.eval()                # 评估模式
```

---

## 5️⃣ 损失函数

| 损失函数               | 用途     | 输入   |
| ---------------------- | -------- | ------ |
| `nn.CrossEntropyLoss`  | 多分类   | logits |
| `nn.BCEWithLogitsLoss` | 二分类   | logits |
| `nn.MSELoss`           | 回归     | 预测值 |
| `nn.L1Loss`            | 鲁棒回归 | 预测值 |

### 典型用法

```python
# 多分类
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)  # labels 是类别索引

# 二分类
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets.float())
```

---

## 6️⃣ 优化器

### 常用优化器

```python
optim.SGD(params, lr=0.01, momentum=0.9)
optim.Adam(params, lr=0.001)
optim.AdamW(params, lr=0.001, weight_decay=0.01)
```

### 学习率调度

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 每个 epoch 调用
scheduler.step()
```

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 7️⃣ 数据加载

### 自定义 Dataset

```python
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

---

## 8️⃣ 数据增强

### 图像增强

```python
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### 高级增强

- **MixUp**: 混合两张图片
- **CutOut**: 随机遮挡
- **CutMix**: 剪切粘贴

---

## 9️⃣ 训练循环

### 完整流程

```python
for epoch in range(n_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            # 验证...

    scheduler.step()
```

---

## 🔟 模型保存与加载

### 推荐方式

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
```

### 完整 Checkpoint

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

---

## 🔗 核心流程图

```
数据准备                   模型构建                   训练
   │                         │                        │
Dataset ──→ DataLoader   nn.Module ──→ forward()   训练循环
   │                         │                        │
transform                   层组合                 optimizer.zero_grad()
   │                         │                        │
collate_fn                参数初始化              loss.backward()
                             │                        │
                          to(device)             optimizer.step()
                                                      │
                                              scheduler.step()
                                                      │
                                              保存 checkpoint
```

---

## ✅ 自检清单

- [ ] 能创建和操作 Tensor，理解设备管理
- [ ] 理解 requires_grad 和 backward()
- [ ] 能自定义 nn.Module 构建模型
- [ ] 知道常用损失函数的选择
- [ ] 能配置优化器和学习率调度器
- [ ] 能实现自定义 Dataset 和 DataLoader
- [ ] 理解完整训练循环的各个步骤
- [ ] 能保存和加载模型

---

## 📖 推荐资源

1. [PyTorch 官方教程](https://pytorch.org/tutorials/)
2. [PyTorch 速查手册](docs/PYTORCH_HANDBOOK.md)
3. [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch)
