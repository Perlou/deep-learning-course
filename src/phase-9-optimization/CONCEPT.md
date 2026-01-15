# 深度学习训练技巧与优化完全指南

## 📚 目录

- [一、基础概念回顾](#一基础概念回顾)
- [二、优化器详解](#二优化器详解)
- [三、学习率策略](#三学习率策略)
- [四、正则化技术](#四正则化技术)
- [五、权重初始化](#五权重初始化)
- [六、批量大小的影响](#六批量大小的影响)
- [七、梯度问题与解决方案](#七梯度问题与解决方案)
- [八、混合精度训练](#八混合精度训练)
- [九、高级训练技巧](#九高级训练技巧)
- [十、实战调参指南](#十实战调参指南)

---

## 一、基础概念回顾

### 1.1 什么是深度学习训练？

深度学习训练的本质是一个**优化问题**：通过调整模型参数，使得损失函数最小化。

```
训练流程：
┌─────────────────────────────────────────────────────────────┐
│  输入数据 → 前向传播 → 计算损失 → 反向传播 → 更新参数 → 重复  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心公式

**参数更新的基本公式：**

```
θ(t+1) = θ(t) - η · ∇L(θ)
```

- `θ`: 模型参数
- `η`: 学习率
- `∇L(θ)`: 损失函数对参数的梯度

### 1.3 损失函数

| 任务类型 | 常用损失函数  | 公式                             |
| -------- | ------------- | -------------------------------- |
| 回归     | MSE           | L = (1/n)Σ(y - ŷ)²               |
| 回归     | MAE           | L = (1/n)Σ\|y - ŷ\|              |
| 二分类   | BCE           | L = -[y·log(ŷ) + (1-y)·log(1-ŷ)] |
| 多分类   | Cross Entropy | L = -Σyᵢ·log(ŷᵢ)                 |

---

## 二、优化器详解

### 2.1 优化器演进图

```
SGD → Momentum → NAG → AdaGrad → RMSprop → Adam → AdamW
 │                                                    │
 └──────────────────────────────────────────────────────┘
                    优化器发展历程
```

### 2.2 各优化器详解

#### 2.2.1 SGD（随机梯度下降）

**公式：**

```
θ = θ - η · ∇L(θ)
```

**代码示例：**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

**特点：**

- ✅ 简单直观
- ✅ 泛化性能好
- ❌ 收敛速度慢
- ❌ 容易陷入局部最优

---

#### 2.2.2 SGD with Momentum（动量）

**原理：** 引入"惯性"，加速收敛，减少震荡

**公式：**

```
v(t) = γ · v(t-1) + η · ∇L(θ)
θ = θ - v(t)
```

**直观理解：**

```
想象一个球从山坡滚下：

    ○ 起点              没有动量：○ → ○ → ○ (来回震荡)
   ╱
  ╱                     有动量：  ○ → → → ○ (平滑加速)
 ╱    ╲
╱      ╲   ⊙ 最低点
```

**代码示例：**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

#### 2.2.3 Adam（Adaptive Moment Estimation）

**公式：**

```
m(t) = β₁·m(t-1) + (1-β₁)·g(t)        # 一阶矩估计（均值）
v(t) = β₂·v(t-1) + (1-β₂)·g(t)²       # 二阶矩估计（方差）

m̂(t) = m(t) / (1-β₁ᵗ)                 # 偏差修正
v̂(t) = v(t) / (1-β₂ᵗ)

θ = θ - η · m̂(t) / (√v̂(t) + ε)
```

**默认参数：**

- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

**代码示例：**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

**特点：**

- ✅ 自适应学习率
- ✅ 收敛速度快
- ✅ 适合大多数场景
- ❌ 可能泛化不如 SGD

---

#### 2.2.4 AdamW（Adam with Weight Decay）

**改进点：** 将权重衰减从梯度更新中解耦

```python
# Adam中的L2正则化（耦合）
gradient = gradient + weight_decay * param

# AdamW中的权重衰减（解耦）
param = param - lr * weight_decay * param
```

**代码示例：**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**推荐场景：** Transformer、BERT 等大模型训练首选

---

### 2.3 优化器选择指南

```
┌──────────────────────────────────────────────────────┐
│                  如何选择优化器？                      │
├──────────────────────────────────────────────────────┤
│                                                       │
│  新手/快速实验 ─────────────→ Adam                    │
│                                                       │
│  追求最佳泛化性能 ──────────→ SGD + Momentum          │
│                                                       │
│  Transformer/NLP任务 ───────→ AdamW                   │
│                                                       │
│  稀疏数据 ──────────────────→ AdaGrad/RMSprop         │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## 三、学习率策略

### 3.1 学习率的重要性

```
学习率太大：                    学习率太小：                  学习率合适：
     ╱╲    ╱╲                         .                          ╲
    ╱  ╲  ╱  ╲ 震荡发散              . .                          ╲
   ╱    ╲╱    ╲                     . . .                          ╲
  ╱            ╲                   . . . . (收敛太慢)                ⊙ 收敛
```

### 3.2 学习率调度策略

#### 3.2.1 Step Decay（阶梯衰减）

每隔固定 epochs，学习率乘以衰减因子

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,    # 每30个epoch
    gamma=0.1        # 学习率乘以0.1
)

# 训练循环中
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

**可视化：**

```
lr │
   │────┐
   │    │────┐
   │         │────┐
   │              │────
   └──────────────────── epoch
      30   60   90
```

---

#### 3.2.2 Cosine Annealing（余弦退火）

**公式：**

```
η(t) = η_min + 0.5·(η_max - η_min)·(1 + cos(πt/T))
```

**代码示例：**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,      # 周期
    eta_min=1e-6    # 最小学习率
)
```

**可视化：**

```
lr │
   │╲
   │ ╲
   │  ╲     ╱╲
   │   ╲   ╱  ╲    余弦曲线
   │    ╲ ╱    ╲
   │     ╲      ╲
   └────────────────── epoch
```

---

#### 3.2.3 Warmup（预热）

**原理：** 训练初期使用较小学习率，逐步增大到目标学习率

**为什么需要 Warmup：**

- 初始参数随机，梯度方向不稳定
- 大学习率容易导致训练不稳定
- 让模型"热身"后再加速

```python
def warmup_lr_scheduler(optimizer, warmup_epochs, initial_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**可视化：**

```
lr │           ────────────
   │          ╱
   │         ╱   正常训练
   │        ╱
   │       ╱
   │      ╱  Warmup阶段
   │     ╱
   └──────────────────────── epoch
        5
```

---

#### 3.2.4 Warmup + Cosine（常用组合）

```python
# 使用transformers库的实现
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

**可视化：**

```
lr │
   │     ╱╲
   │    ╱  ╲
   │   ╱    ╲
   │  ╱      ╲
   │ ╱        ╲
   │╱          ╲
   └──────────────────── step
      warmup  cosine
```

---

### 3.3 学习率查找器（Learning Rate Finder）

**原理：** 逐步增大学习率，观察 loss 变化，找到最佳学习率范围

```python
# 使用pytorch-lightning的lr_finder
from pytorch_lightning.tuner import Tuner

trainer = Trainer(...)
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model)
lr_finder.plot(suggest=True)
```

**结果解读：**

```
loss │
     │╲
     │ ╲
     │  ╲_____        ← 最佳学习率区间
     │        ╲
     │         ╲
     │          ╲  ← loss开始上升，学习率太大
     └─────────────────── lr (log scale)
        ↑
    选择这里的学习率
```

---

## 四、正则化技术

### 4.1 正则化概览

```
正则化技术
├── 显式正则化
│   ├── L1正则化 (Lasso)
│   ├── L2正则化 (Ridge/Weight Decay)
│   └── Elastic Net
│
├── 隐式正则化
│   ├── Dropout
│   ├── DropConnect
│   ├── DropPath (Stochastic Depth)
│   └── Early Stopping
│
└── 归一化技术
    ├── Batch Normalization
    ├── Layer Normalization
    ├── Group Normalization
    └── Instance Normalization
```

---

### 4.2 L1 和 L2 正则化

#### L2 正则化（Weight Decay）

**公式：**

```
L_total = L_original + λ · Σ(θ²)
```

**效果：** 使权重趋向于较小的值，但不为零

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

#### L1 正则化

**公式：**

```
L_total = L_original + λ · Σ|θ|
```

**效果：** 使部分权重变为 0，产生稀疏解

```python
# PyTorch中需手动实现
l1_lambda = 0.001
l1_reg = sum(param.abs().sum() for param in model.parameters())
loss = loss + l1_lambda * l1_reg
```

**对比：**

```
         L1正则化                    L2正则化

权重分布:  [0, 0.5, 0, 0.8, 0]      [0.1, 0.3, 0.2, 0.4, 0.1]
特点:      稀疏解                    平滑解
用途:      特征选择                  防止过拟合
```

---

### 4.3 Dropout

**原理：** 训练时随机"丢弃"一部分神经元

```
训练时（p=0.5）:                    测试时:

○ ─┬─ ○ ─┬─ ○                     ○ ─┬─ ○ ─┬─ ○
   │     │                            │     │
○ ─┼─ ✕ ─┼─ ○    随机丢弃         ○ ─┼─ ○ ─┼─ ○  全部激活
   │     │                            │     │
○ ─┼─ ○ ─┼─ ✕                     ○ ─┼─ ○ ─┼─ ○
   │     │                            │     │
✕ ─┴─ ○ ─┴─ ○                     ○ ─┴─ ○ ─┴─ ○
```

**代码示例：**

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 只在训练时生效
        x = self.fc2(x)
        return x
```

**Dropout 变体：**

| 变体            | 描述                 | 适用场景                   |
| --------------- | -------------------- | -------------------------- |
| Dropout         | 随机丢弃神经元       | 全连接层                   |
| Spatial Dropout | 丢弃整个特征图通道   | CNN                        |
| DropPath        | 随机丢弃整个残差分支 | ResNet, Vision Transformer |
| DropConnect     | 随机丢弃连接权重     | 全连接层                   |

---

### 4.4 Batch Normalization

**公式：**

```
μ_B = (1/m) · Σxᵢ                    # 计算批次均值
σ²_B = (1/m) · Σ(xᵢ - μ_B)²          # 计算批次方差
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)       # 归一化
yᵢ = γ · x̂ᵢ + β                      # 缩放和平移
```

**作用：**

- ✅ 加速训练收敛
- ✅ 允许使用更大学习率
- ✅ 减少对初始化的敏感性
- ✅ 有一定的正则化效果

**代码示例：**

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # BN层放在Conv之后
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
```

**BN 的位置之争：**

```
方案1（原论文）: Conv → BN → ReLU
方案2（常用）:   Conv → ReLU → BN
方案3: BN → ReLU → Conv (Pre-activation ResNet)
```

---

### 4.5 Layer Normalization

**与 BN 的区别：**

```
Batch Norm:                      Layer Norm:
对每个特征在batch维度归一化         对每个样本在特征维度归一化

  N个样本                           N个样本
┌───┬───┬───┐                    ┌───┬───┬───┐
│ ← │ ← │ ← │ ← 对这一行归一化   │   │   │   │
├───┼───┼───┤                    │ ↓ │ ↓ │ ↓ │ ← 对每列归一化
│ ← │ ← │ ← │                    │ ↓ │ ↓ │ ↓ │
└───┴───┴───┘                    └───┴───┴───┘
  特征维度                          特征维度
```

**适用场景：**

- **Batch Norm**: CNN, 大 batch size
- **Layer Norm**: RNN, Transformer, 小 batch size

```python
# Layer Normalization
self.ln = nn.LayerNorm(hidden_size)

# Group Normalization (BN和LN的折中)
self.gn = nn.GroupNorm(num_groups=32, num_channels=256)
```

---

### 4.6 Early Stopping

**原理：** 监控验证集性能，当不再提升时停止训练

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 使用
early_stopping = EarlyStopping(patience=10)
for epoch in range(epochs):
    train(...)
    val_loss = validate(...)
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

**可视化：**

```
loss │
     │  ╲     训练loss
     │   ╲ ╲
     │    ╲ ╲_____
     │     ╲       ╲____
     │      ╲___        ╲____  训练loss继续下降
     │          ╲___
     │              ╲  验证loss
     │               ╲___
     │                   ╲___────────  验证loss不再下降
     │                         ↑
     └─────────────────────────────── epoch
                          Early Stop点
```

---

## 五、权重初始化

### 5.1 为什么初始化重要？

```
初始化太小:               初始化太大:              良好初始化:
梯度消失                  梯度爆炸                 稳定训练

层1 → 层2 → 层3          层1 → 层2 → 层3         层1 → 层2 → 层3
0.9 → 0.1 → 0.01         2.0 → 10 → 100          1.0 → 1.0 → 1.0
    ↓    ↓                   ↓    ↓                  ↓    ↓
  激活值越来越小            激活值爆炸              激活值稳定
```

### 5.2 常用初始化方法

#### Xavier/Glorot 初始化

**适用于：** Sigmoid, Tanh 激活函数

**公式：**

```
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))  # 均匀分布
W ~ N(0, 2/(n_in + n_out))                         # 正态分布
```

```python
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

#### He/Kaiming 初始化

**适用于：** ReLU 及其变体

**公式：**

```
W ~ N(0, 2/n_in)
```

```python
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### 5.3 初始化选择指南

| 激活函数     | 推荐初始化 | PyTorch 代码                                  |
| ------------ | ---------- | --------------------------------------------- |
| Sigmoid/Tanh | Xavier     | `xavier_uniform_`                             |
| ReLU         | He         | `kaiming_uniform_(nonlinearity='relu')`       |
| Leaky ReLU   | He         | `kaiming_uniform_(nonlinearity='leaky_relu')` |
| SELU         | LeCun      | `normal_(std=1/sqrt(fan_in))`                 |

### 5.4 完整初始化示例

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

# 应用初始化
model.apply(init_weights)
```

---

## 六、批量大小的影响

### 6.1 Batch Size 的权衡

```
小 Batch Size                        大 Batch Size
┌─────────────────┐                  ┌─────────────────┐
│ ✅ 正则化效果好   │                  │ ✅ 训练更稳定     │
│ ✅ 泛化性能更好   │                  │ ✅ GPU利用率高    │
│ ✅ 内存占用少    │                  │ ✅ 并行效率高     │
│ ❌ 训练速度慢    │                  │ ❌ 泛化可能变差    │
│ ❌ 梯度噪声大    │                  │ ❌ 内存占用大     │
└─────────────────┘                  └─────────────────┘
```

### 6.2 Batch Size 与学习率的关系

**线性缩放法则：**

```
当batch size增大k倍时，学习率也应增大k倍

例如: batch_size: 32 → 256 (8倍)
      learning_rate: 0.001 → 0.008 (8倍)
```

**但实践中需要配合 Warmup 使用！**

### 6.3 梯度累积（小显存模拟大 Batch）

```python
accumulation_steps = 4  # 模拟4倍batch size

optimizer.zero_grad()
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # 归一化损失
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**图示：**

```
常规训练 (batch=32):
[Batch1] → backward → update → [Batch2] → backward → update

梯度累积 (实际batch=8, 累积4次=等效batch=32):
[Batch1] → backward ─┬→ [Batch2] → backward ─┬→ ... → [Batch4] → backward → update
                     │                        │
                   累积梯度                  累积梯度
```

---

## 七、梯度问题与解决方案

### 7.1 梯度消失与梯度爆炸

**梯度消失：**

```
深层网络中，梯度经过多次连乘变得极小

层10 ← 层9 ← 层8 ← ... ← 层1 ← Loss
0.0001   0.001  0.01       0.1    1.0

问题：浅层几乎无法更新
```

**梯度爆炸：**

```
梯度经过多次连乘变得极大

层10 ← 层9 ← 层8 ← ... ← 层1 ← Loss
10000   1000   100        10    1.0

问题：参数更新过大，训练不稳定
```

### 7.2 解决方案

#### 7.2.1 梯度裁剪（Gradient Clipping）

**按范数裁剪（推荐）：**

```python
# 当梯度范数超过max_norm时，缩放梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 使用示例
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**按值裁剪：**

```python
# 将梯度裁剪到[-clip_value, clip_value]
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**可视化：**

```
裁剪前:                          裁剪后:
梯度向量: (3, 4)                 梯度向量: (0.6, 0.8)
范数: 5                          范数: 1.0 (max_norm)

    ↑                               ↑
    │    *(3,4)                     │  *(0.6,0.8)
    │   /                           │ /
    │  /                            │/
    └──────→                        └──────→
```

#### 7.2.2 残差连接（Skip Connections）

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # 残差连接
        return F.relu(out)
```

**图示：**

```
输入 x ────────────────────────┐
   │                           │
   ↓                           │
┌──────────┐                   │
│  Conv1   │                   │
│   BN1    │                   │
│  ReLU    │                   │
└────┬─────┘                   │
     │                         │
     ↓                         │
┌──────────┐                   │
│  Conv2   │                   │
│   BN2    │                   │
└────┬─────┘                   │
     │                         │
     ↓                         │
   ─────────────────────────+ ←┘  (相加)
     │
     ↓
   ReLU
     │
     ↓
   输出
```

---

## 八、混合精度训练

### 8.1 什么是混合精度？

```
FP32 (单精度):  ████████████████████████████████  32位
FP16 (半精度):  ████████████████                  16位

混合精度: 前向/反向传播用FP16, 关键计算用FP32
```

### 8.2 优势

| 指标     | FP32 | 混合精度 |
| -------- | ---- | -------- |
| 内存占用 | 基准 | ~50%     |
| 训练速度 | 基准 | 1.5-3x   |
| 精度损失 | -    | 几乎无   |

### 8.3 PyTorch 实现

```python
from torch.cuda.amp import autocast, GradScaler

# 创建GradScaler
scaler = GradScaler()

for inputs, labels in dataloader:
    optimizer.zero_grad()

    # 使用autocast进行混合精度前向传播
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    # 使用scaler进行缩放反向传播
    scaler.scale(loss).backward()

    # 使用scaler更新参数
    scaler.step(optimizer)
    scaler.update()
```

### 8.4 混合精度工作原理

```
                    ┌─────────────────────────────────────┐
                    │           混合精度训练流程            │
                    └─────────────────────────────────────┘
                                     │
        ┌───────────────────────────────────────────────────┐
        │                                                    │
        ↓                                                    ↓
    前向传播                                             反向传播
  ┌─────────┐                                         ┌─────────┐
  │  FP16   │  计算速度快                              │  FP16   │
  │  权重   │  内存占用少                              │  梯度   │
  └─────────┘                                         └─────────┘
        │                                                    │
        │                                                    ↓
        │                                            Loss Scaling
        │                                           (防止梯度下溢)
        │                                                    │
        └─────────────────┐      ┌───────────────────────────┘
                          ↓      ↓
                    ┌─────────────────┐
                    │    FP32权重     │  主权重副本
                    │     更新        │  保证精度
                    └─────────────────┘
```

---

## 九、高级训练技巧

### 9.1 迁移学习（Transfer Learning）

```python
# 加载预训练模型
model = torchvision.models.resnet50(pretrained=True)

# 方法1: 冻结所有层，只训练最后分类层
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)

# 方法2: 差异化学习率
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], lr=1e-5)  # 其他层使用更小的学习率
```

**迁移学习策略：**

```
数据量小 + 相似任务:  冻结大部分层，只训练顶层
数据量小 + 不同任务:  冻结底层，微调顶层
数据量大 + 相似任务:  全网络微调，小学习率
数据量大 + 不同任务:  全网络训练，可使用预训练初始化
```

### 9.2 知识蒸馏（Knowledge Distillation）

**原理：** 用大模型（Teacher）的知识指导小模型（Student）训练

```python
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=3.0, alpha=0.7):
    """
    蒸馏损失 = α * 软标签损失 + (1-α) * 硬标签损失
    """
    # 软标签损失
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # 硬标签损失
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**图示：**

```
                    ┌──────────────┐
                    │   Teacher    │
                    │  (大模型)    │
                    └──────┬───────┘
                           │
                    ┌──────┴──────┐
                    │  Soft Labels │  带温度的软概率分布
                    │ [0.7, 0.2, 0.1] │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ↓            ↓            ↓
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ Student │  │  Loss   │  │ Ground  │
        │ (小模型) │←─│ 结合    │←─│ Truth   │
        └─────────┘  └─────────┘  └─────────┘
```

### 9.3 模型集成（Model Ensemble）

```python
# 方法1: 简单平均
def ensemble_predict(models, x):
    predictions = [model(x) for model in models]
    return torch.mean(torch.stack(predictions), dim=0)

# 方法2: 加权平均
def weighted_ensemble(models, weights, x):
    predictions = [w * model(x) for w, model in zip(weights, models)]
    return sum(predictions)

# 方法3: 投票
def voting_ensemble(models, x):
    predictions = [model(x).argmax(dim=1) for model in models]
    stacked = torch.stack(predictions, dim=1)
    return torch.mode(stacked, dim=1).values
```

### 9.4 标签平滑（Label Smoothing）

**原理：** 避免模型过于自信，提高泛化能力

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)

        # 将硬标签转换为软标签
        # [0, 1, 0, 0] → [0.025, 0.925, 0.025, 0.025]
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))
```

**对比：**

```
硬标签:  [0, 0, 1, 0, 0]  # 只有正确类别为1

软标签 (smoothing=0.1):
         [0.025, 0.025, 0.9, 0.025, 0.025]  # 分散一些概率给其他类
```

### 9.5 Mixup 数据增强

**原理：** 对训练样本和标签进行线性插值混合

```python
def mixup_data(x, y, alpha=0.2):
    """
    混合两个样本
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 训练循环
for x, y in dataloader:
    mixed_x, y_a, y_b, lam = mixup_data(x, y)
    outputs = model(mixed_x)
    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
```

**可视化：**

```
图片A (猫):          图片B (狗):           混合结果:
┌─────────┐         ┌─────────┐          ┌─────────┐
│  🐱    │    +    │  🐕    │    =     │ 🐱+🐕  │
│         │  λ=0.7 │         │ (1-λ)=0.3│ 透明叠加 │
└─────────┘         └─────────┘          └─────────┘
标签:[1,0]          标签:[0,1]           标签:[0.7, 0.3]
```

---

## 十、实战调参指南

### 10.1 调参顺序

```
Step 1: 确保代码正确
        ↓
        在小数据集上过拟合验证
        ↓
Step 2: 调整学习率
        ↓
        使用学习率finder或网格搜索
        ↓
Step 3: 调整batch size
        ↓
        在显存允许范围内尝试不同大小
        ↓
Step 4: 调整网络结构
        ↓
        深度、宽度、正则化
        ↓
Step 5: 调整其他超参数
        ↓
        dropout rate, weight decay等
        ↓
Step 6: 使用高级技巧
        ↓
        学习率调度、数据增强、mixup等
```

### 10.2 超参数搜索策略

```python
# 网格搜索
from sklearn.model_selection import ParameterGrid

param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64],
    'weight_decay': [1e-5, 1e-4, 1e-3]
}

for params in ParameterGrid(param_grid):
    model = create_model()
    train_and_evaluate(model, **params)
```

```python
# 随机搜索（更高效）
import random

def random_search(n_trials=20):
    for _ in range(n_trials):
        params = {
            'lr': 10 ** random.uniform(-5, -2),
            'batch_size': random.choice([16, 32, 64, 128]),
            'weight_decay': 10 ** random.uniform(-6, -3),
            'dropout': random.uniform(0.1, 0.5)
        }
        train_and_evaluate(**params)
```

### 10.3 常见问题诊断

```
问题: 训练loss不下降
├── 检查学习率是否太小
├── 检查数据加载是否正确
├── 检查标签是否有问题
└── 检查模型输出维度是否匹配

问题: 训练loss下降但验证loss不降
├── 过拟合 → 添加正则化
├── 数据泄露 → 检查数据划分
└── 验证集太小 → 增加验证集

问题: loss变成NaN
├── 学习率太大 → 减小学习率
├── 梯度爆炸 → 添加梯度裁剪
├── 数据有问题 → 检查是否有NaN/Inf
└── 数值不稳定 → 添加eps或使用log_softmax

问题: 训练不稳定
├── 添加BatchNorm/LayerNorm
├── 使用warmup
├── 减小学习率
└── 使用梯度裁剪
```

### 10.4 训练监控模板

```python
import wandb  # 或 tensorboard

def train_with_monitoring():
    wandb.init(project='my-project')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 监控梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # 记录每个batch
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'grad_norm': grad_norm,
                    'lr': optimizer.param_groups[0]['lr']
                })

        # 验证
        val_loss, val_acc = validate(model, val_loader)

        # 记录每个epoch
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        scheduler.step()
```

### 10.5 完整训练代码模板

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 混合精度
        self.scaler = GradScaler()

        # Early Stopping
        self.early_stopping = EarlyStopping(patience=config.patience)

        # 最佳模型
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for data, target in self.train_loader:
            data, target = data.to(self.config.device), target.to(self.config.device)

            self.optimizer.zero_grad()

            # 混合精度前向传播
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0

        for data, target in self.val_loader:
            data, target = data.to(self.config.device), target.to(self.config.device)
            output = self.model(data)
            total_loss += self.criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / len(self.val_loader.dataset)

        return val_loss, val_acc

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

            # Early Stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
```

---

## 📝 总结

### 核心要点速查表

| 类别       | 技巧                      | 常用配置              |
| ---------- | ------------------------- | --------------------- |
| **优化器** | AdamW                     | lr=1e-3~1e-4, wd=0.01 |
| **学习率** | Cosine + Warmup           | warmup_ratio=0.1      |
| **正则化** | Dropout + Label Smoothing | p=0.1~0.5, smooth=0.1 |
| **归一化** | BatchNorm/LayerNorm       | 视场景选择            |
| **梯度**   | 梯度裁剪                  | max_norm=1.0          |
| **精度**   | 混合精度                  | FP16                  |
| **初始化** | He 初始化                 | 对 ReLU               |
| **Batch**  | 梯度累积                  | 按显存调整            |

### 训练 Checklist

- [ ] 代码在小数据上能过拟合
- [ ] 使用了适当的权重初始化
- [ ] 选择了合适的优化器（AdamW 推荐）
- [ ] 设置了学习率调度策略
- [ ] 添加了必要的正则化
- [ ] 使用了混合精度训练
- [ ] 添加了梯度裁剪
- [ ] 设置了 Early Stopping
- [ ] 保存了最佳模型 checkpoint
- [ ] 记录了训练日志和曲线

---

_文档版本: v1.0_  
_适用框架: PyTorch_
