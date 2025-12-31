# 卷积神经网络 (CNN) 深度解析

## 📚 目录

- [1. 什么是 CNN](#1-什么是cnn)
- [2. 为什么需要 CNN](#2-为什么需要cnn)
- [3. CNN 核心组件](#3-cnn核心组件)
- [4. 关键概念详解](#4-关键概念详解)
- [5. CNN 完整架构](#5-cnn完整架构)
- [6. 经典 CNN 模型演进](#6-经典cnn模型演进)
- [7. 代码实战](#7-代码实战)
- [8. 应用场景](#8-应用场景)
- [9. 常见问题与优化](#9-常见问题与优化)

---

## 1. 什么是 CNN

### 1.1 定义

**卷积神经网络 (Convolutional Neural Network, CNN)** 是一种专门用于处理具有网格状拓扑结构数据的深度学习模型，特别适用于图像、视频、语音等数据。

### 1.2 灵感来源

CNN 的设计灵感来自于生物学中的**视觉皮层**：

```
人眼视觉处理过程：
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  视网膜  │ → │ 简单细胞 │ → │ 复杂细胞 │ → │ 高级皮层 │
│ (输入)   │    │(边缘检测)│    │(特征组合)│    │(物体识别)│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 1.3 CNN vs 传统神经网络

| 特性       | 传统全连接网络 | CNN       |
| ---------- | -------------- | --------- |
| 参数数量   | 非常多         | 大幅减少  |
| 空间信息   | 丢失           | 保留      |
| 平移不变性 | 无             | 有        |
| 适用场景   | 结构化数据     | 图像/语音 |

---

## 2. 为什么需要 CNN

### 2.1 全连接网络的问题

假设处理一张 `1000×1000×3` 的彩色图片：

```
输入层神经元数: 1000 × 1000 × 3 = 3,000,000
第一隐藏层(1000个神经元): 3,000,000 × 1,000 = 30亿参数！

❌ 参数爆炸
❌ 容易过拟合
❌ 计算资源巨大
❌ 忽略空间结构
```

### 2.2 CNN 的三大核心思想

```
┌────────────────────────────────────────────────────────┐
│                    CNN 三大核心思想                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1️⃣ 局部连接 (Local Connectivity)                      │
│     每个神经元只与输入的局部区域连接                      │
│                                                        │
│  2️⃣ 权重共享 (Weight Sharing)                          │
│     同一个卷积核在整个输入上滑动，共享参数                 │
│                                                        │
│  3️⃣ 池化降采样 (Pooling)                               │
│     降低特征图尺寸，提取主要特征                          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 3. CNN 核心组件

### 3.1 卷积层 (Convolutional Layer)

#### 3.1.1 卷积操作原理

```
输入图像 (5×5)          卷积核 (3×3)           输出特征图 (3×3)

┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐      ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │   │ 1 │ 0 │ 1 │      │ 4 │ 3 │ 4 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┤   →  ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │   │ 0 │ 1 │ 0 │      │ 2 │ 4 │ 3 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┤      ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │   │ 1 │ 0 │ 1 │      │ 2 │ 3 │ 4 │
├───┼───┼───┼───┼───┤   └───┴───┴───┘      └───┴───┴───┘
│ 0 │ 1 │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘
```

#### 3.1.2 卷积计算过程

```
第一个位置计算:

┌───┬───┬───┐       ┌───┬───┬───┐
│ 1 │ 0 │ 1 │   ×   │ 1 │ 0 │ 1 │
├───┼───┼───┤       ├───┼───┼───┤
│ 0 │ 1 │ 0 │   ×   │ 0 │ 1 │ 0 │  = 1×1+0×0+1×1+0×0+1×1+0×0+1×1+0×0+1×1 = 4
├───┼───┼───┤       ├───┼───┼───┤
│ 1 │ 0 │ 1 │   ×   │ 1 │ 0 │ 1 │
└───┴───┴───┘       └───┴───┴───┘
```

#### 3.1.3 数学公式

$$\text{Output}(i,j) = \sum_{m}\sum_{n} \text{Input}(i+m, j+n) \cdot \text{Kernel}(m,n) + \text{bias}$$

#### 3.1.4 常见卷积核效果

```python
# 边缘检测 (Sobel)
sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

# 模糊 (均值滤波)
blur = [[1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]]

# 锐化
sharpen = [[ 0, -1,  0],
           [-1,  5, -1],
           [ 0, -1,  0]]
```

### 3.2 激活函数 (Activation Function)

#### 3.2.1 ReLU (最常用)

```
           │
        ───│────────────/
           │           /
           │          /
        ───│─────────/
           │        │
───────────│────────│───────
           │        0

f(x) = max(0, x)
```

#### 3.2.2 常见激活函数对比

| 激活函数   | 公式                      | 优点                 | 缺点           |
| ---------- | ------------------------- | -------------------- | -------------- |
| ReLU       | max(0, x)                 | 计算快，缓解梯度消失 | 死亡 ReLU 问题 |
| Leaky ReLU | max(0.01x, x)             | 解决死亡 ReLU        | 需要额外参数   |
| Sigmoid    | 1/(1+e^(-x))              | 输出(0,1)概率        | 梯度消失       |
| Tanh       | (e^x-e^(-x))/(e^x+e^(-x)) | 零中心               | 梯度消失       |

### 3.3 池化层 (Pooling Layer)

#### 3.3.1 最大池化 (Max Pooling)

```
输入 (4×4)                    输出 (2×2)
┌────┬────┬────┬────┐        ┌────┬────┐
│ 1  │ 3  │ 2  │ 4  │        │    │    │
├────┼────┼────┼────┤   →    │ 6  │ 8  │
│ 5  │ 6  │ 7  │ 8  │        ├────┼────┤
├────┼────┼────┼────┤        │ 3  │ 4  │
│ 1  │ 2  │ 3  │ 4  │        │    │    │
├────┼────┼────┼────┤        └────┴────┘
│ 1  │ 1  │ 2  │ 2  │        2×2池化，步幅2
└────┴────┴────┴────┘
```

#### 3.3.2 平均池化 (Average Pooling)

```
┌────┬────┐
│ 1  │ 3  │
├────┼────┤  →  (1+3+5+6)/4 = 3.75
│ 5  │ 6  │
└────┴────┘
```

#### 3.3.3 池化的作用

```
✅ 降低特征维度
✅ 减少计算量
✅ 增加平移不变性
✅ 防止过拟合
```

### 3.4 全连接层 (Fully Connected Layer)

```
特征图展平          全连接层          输出层

┌───────┐          ┌───┐           ┌───┐
│       │          │ ○ │──────────→│ 猫 │
│ 特征图 │ ─→ ○○○○○ │ ○ │──────────→│ 狗 │
│       │          │ ○ │──────────→│ 鸟 │
└───────┘          └───┘           └───┘

 Flatten           Dense           Softmax
```

---

## 4. 关键概念详解

### 4.1 步幅 (Stride)

```
Stride = 1:                    Stride = 2:

□□□ → □□□ → □□□               □□□ ─────→ □□□
      ↓       ↓                         ↓
    □□□ → □□□ → □□□                   □□□ ─────→ □□□
          ↓       ↓
        □□□ → □□□ → □□□

输出尺寸较大                    输出尺寸较小
```

### 4.2 填充 (Padding)

#### 4.2.1 Valid Padding (无填充)

```
输入: 5×5, 卷积核: 3×3, 步幅: 1
输出: (5-3)/1 + 1 = 3×3

输入会变小！
```

#### 4.2.2 Same Padding (保持尺寸)

```
       填充后 (7×7)
    ┌─────────────────┐
    │ 0 0 0 0 0 0 0   │
    │ 0 □ □ □ □ □ 0   │
    │ 0 □ □ □ □ □ 0   │   卷积核: 3×3
    │ 0 □ □ □ □ □ 0   │   输出: 5×5 (保持原尺寸)
    │ 0 □ □ □ □ □ 0   │
    │ 0 □ □ □ □ □ 0   │
    │ 0 0 0 0 0 0 0   │
    └─────────────────┘
```

### 4.3 输出尺寸计算公式

$$\text{Output Size} = \lfloor \frac{W - K + 2P}{S} \rfloor + 1$$

其中:

- W = 输入尺寸
- K = 卷积核尺寸
- P = 填充大小
- S = 步幅

```python
# 示例计算
W = 32  # 输入图像 32×32
K = 5   # 卷积核 5×5
P = 2   # 填充 2
S = 1   # 步幅 1

output_size = (32 - 5 + 2*2) // 1 + 1  # = 32
```

### 4.4 感受野 (Receptive Field)

```
层级          感受野可视化

         ┌─────────────────────────────┐
Layer 3  │                             │  感受野: 7×7
         │     ┌─────────────────┐     │
Layer 2  │     │                 │     │  感受野: 5×5
         │     │   ┌─────────┐   │     │
Layer 1  │     │   │         │   │     │  感受野: 3×3
         │     │   │    ○    │   │     │  (单个神经元)
         │     │   │         │   │     │
         │     │   └─────────┘   │     │
         │     └─────────────────┘     │
         └─────────────────────────────┘

随着网络加深，感受野逐渐增大
```

**感受野计算公式:**

$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

### 4.5 多通道卷积

```
RGB 输入 (H×W×3)          3D卷积核                   输出 (单通道)

  ┌─────────┐
  │    R    │  ────┐
  ├─────────┤      │    ┌───────────┐
  │    G    │  ────┼───→│ 3×3×3核   │────→ 单个特征图
  ├─────────┤      │    └───────────┘
  │    B    │  ────┘
  └─────────┘

使用N个卷积核 → 输出N个特征图
```

```python
# PyTorch中的多通道卷积
nn.Conv2d(
    in_channels=3,    # RGB 3通道输入
    out_channels=64,  # 64个卷积核，输出64通道
    kernel_size=3,    # 3×3卷积核
    padding=1
)
# 卷积核实际尺寸: 64 × 3 × 3 × 3
```

---

## 5. CNN 完整架构

### 5.1 经典架构示意图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CNN 完整架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入     卷积层1    池化层1   卷积层2    池化层2   展平    全连接   输出      │
│                                                                             │
│ ┌────┐   ┌────┐    ┌───┐    ┌────┐    ┌───┐    ┌───┐   ┌───┐   ┌───┐     │
│ │    │   │    │    │   │    │    │    │   │    │   │   │   │   │ 猫 │     │
│ │ 🖼️ │→ │    │ → │   │ → │    │ → │   │ → │ → │→ │   │→ │ 狗 │     │
│ │    │   │    │    │   │    │    │    │   │    │   │   │   │   │ 鸟 │     │
│ └────┘   └────┘    └───┘    └────┘    └───┘    └───┘   └───┘   └───┘     │
│ 224×224  32@222    32@111   64@109    64@54   64×54×54  128    3类       │
│ ×3                                            =186624                     │
│                                                                             │
│  提取低级特征  降维  提取高级特征  降维    分类器                              │
│  (边缘纹理)        (形状部件)          (决策)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 特征层级可视化

```
Layer 1: 边缘和颜色
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ ─── │ │  |  │ │  /  │ │  \  │
│ ─── │ │  |  │ │ /   │ │   \ │
│ ─── │ │  |  │ │/    │ │    \│
└─────┘ └─────┘ └─────┘ └─────┘

Layer 2-3: 纹理和图案
┌─────┐ ┌─────┐ ┌─────┐
│░░░░░│ │▓▓▓▓▓│ │╔═══╗│
│░░░░░│ │▓▓▓▓▓│ │║   ║│
│░░░░░│ │▓▓▓▓▓│ │╚═══╝│
└─────┘ └─────┘ └─────┘

Layer 4-5: 物体部件
┌─────┐ ┌─────┐ ┌─────┐
│ 👁️  │ │ 👃  │ │ 👂  │
│     │ │     │ │     │
└─────┘ └─────┘ └─────┘

Layer 6+: 完整物体
┌─────┐ ┌─────┐ ┌─────┐
│ 🐱  │ │ 🐕  │ │ 🚗  │
└─────┘ └─────┘ └─────┘
```

---

## 6. 经典 CNN 模型演进

### 6.1 发展时间线

```
                    CNN 发展历程

1998 ─────── LeNet-5 (Yann LeCun)
             │ 手写数字识别
             │ 7层网络
             ▼
2012 ─────── AlexNet ⭐ ImageNet突破
             │ 8层, 6000万参数
             │ GPU训练, ReLU, Dropout
             ▼
2014 ─────── VGGNet
             │ 更深(19层), 3×3小卷积核
             │
          ── GoogLeNet/Inception
             │ Inception模块, 22层
             ▼
2015 ─────── ResNet ⭐ 里程碑
             │ 残差连接, 152层+
             ▼
2016 ─────── DenseNet
             │ 密集连接
             ▼
2017 ─────── MobileNet / ShuffleNet
             │ 轻量化设计
             ▼
2019 ─────── EfficientNet
             │ 复合缩放
             ▼
2020+ ────── Vision Transformer (ViT)
             │ 注意力机制
             ▼
2022+ ────── ConvNeXt
             将CNN现代化
```

### 6.2 经典模型详解

#### LeNet-5 (1998)

```
输入      C1        S2       C3        S4       C5     F6    输出
32×32   28×28×6   14×14×6  10×10×16  5×5×16  1×1×120  84    10
  │        │         │        │         │        │      │     │
  └──Conv──┴──Pool───┴──Conv──┴──Pool───┴──Conv──┴─FC──┴─FC──┘
     5×5      2×2       5×5      2×2       5×5
```

#### AlexNet (2012)

```python
AlexNet = {
    'conv1': (96, 11×11, stride=4),   # 227→55
    'pool1': (3×3, stride=2),          # 55→27
    'conv2': (256, 5×5),               # 27→27
    'pool2': (3×3, stride=2),          # 27→13
    'conv3': (384, 3×3),               # 13→13
    'conv4': (384, 3×3),               # 13→13
    'conv5': (256, 3×3),               # 13→13
    'pool5': (3×3, stride=2),          # 13→6
    'fc6': 4096,
    'fc7': 4096,
    'fc8': 1000
}
# 创新: ReLU, Dropout, GPU训练, 数据增强
```

#### VGGNet (2014)

```
设计哲学: 使用多个 3×3 小卷积核代替大卷积核

两个 3×3 卷积 = 一个 5×5 卷积的感受野
三个 3×3 卷积 = 一个 7×7 卷积的感受野

优势:
- 更少参数: 3×(3×3) = 27 < 7×7 = 49
- 更多非线性: 3次ReLU vs 1次ReLU
- 更深网络
```

```
VGG-16 架构:
┌─────────────────────────────────────────────────────┐
│ [Conv3-64]×2 → Pool → [Conv3-128]×2 → Pool →        │
│ [Conv3-256]×3 → Pool → [Conv3-512]×3 → Pool →       │
│ [Conv3-512]×3 → Pool → FC-4096 → FC-4096 → FC-1000  │
└─────────────────────────────────────────────────────┘
```

#### ResNet (2015) - 残差网络

```
残差块 (Residual Block):

          ┌──────────────────────┐
          │                      │
    x ────┤                      ├────(+)──→ ReLU ──→ H(x)
          │                      │     ↑
          └─→ Conv → BN → ReLU ─┘     │
              ↓                        │
          Conv → BN ───────────────────┘

          F(x) = H(x) - x
          H(x) = F(x) + x  (恒等快捷连接)
```

**为什么残差有效:**

```
问题: 网络越深,梯度越难传播

普通网络:                   ResNet:

输出 ← 梯度消失 ← ... ← 输入   输出 ← 梯度通过跳连 ← 输入
                               ↑_______直接传递_______↑
```

#### Inception 模块 (GoogLeNet)

```
                    ┌─────────────┐
                    │   Concat    │
                    └─────────────┘
                          ↑
       ┌──────┬──────┬──────┬──────┐
       │      │      │      │      │
    ┌──┴──┐┌──┴──┐┌──┴──┐┌──┴──┐
    │1×1  ││3×3  ││5×5  ││Pool │
    │Conv ││Conv ││Conv ││3×3  │
    └──┬──┘└──┬──┘└──┬──┘└──┬──┘
       │   ┌──┴──┐┌──┴──┐   │
       │   │1×1  ││1×1  │   │
       │   │Conv ││Conv │   │
       │   └──┬──┘└──┬──┘   │
       └──────┴──────┴──────┘
                ↑
          ┌─────────┐
          │  Input  │
          └─────────┘

并行处理多种尺度特征
```

---

## 7. 代码实战

### 7.1 使用 PyTorch 构建 CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    简单的CNN模型用于图像分类
    输入: 3×32×32 (CIFAR-10格式)
    输出: 10个类别
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 卷积层块1
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB输入
            out_channels=32,    # 32个卷积核
            kernel_size=3,      # 3×3卷积核
            padding=1           # 保持尺寸
        )
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积层块2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 卷积层块3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 尺寸变化: 3×32×32

        # 块1: 32×32 → 16×16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 块2: 16×16 → 8×8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 块3: 8×8 → 4×4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平: 128×4×4 → 2048
        x = x.view(-1, 128 * 4 * 4)

        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# 创建模型
model = SimpleCNN(num_classes=10)

# 查看模型结构
print(model)

# 测试前向传播
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)
print(f"输出尺寸: {output.shape}")  # [1, 10]
```

### 7.2 完整训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========================
# 1. 数据预处理
# ========================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

# ========================
# 2. 加载数据
# ========================
train_dataset = datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# ========================
# 3. 训练配置
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# ========================
# 4. 训练循环
# ========================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss/len(loader), 100.*correct/total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.*correct/total

# ========================
# 5. 开始训练
# ========================
num_epochs = 50
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_acc = evaluate(model, test_loader)
    scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Test Acc: {test_acc:.2f}%')
```

### 7.3 实现经典 ResNet 块

```python
class BasicBlock(nn.Module):
    """ResNet基本残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 主路径
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet模型"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 创建不同深度的ResNet
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
```

### 7.4 特征可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_filters(model, layer_name='conv1'):
    """可视化卷积核"""
    # 获取指定层的权重
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data.cpu()
            break

    # 绘制
    n_filters = min(weights.shape[0], 64)
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # 对于多通道,取平均
            filter_img = weights[i].mean(dim=0).numpy()
            ax.imshow(filter_img, cmap='viridis')
        ax.axis('off')

    plt.suptitle(f'{layer_name} Filters')
    plt.tight_layout()
    plt.savefig('filters.png')
    plt.show()


def visualize_feature_maps(model, image, layer_names):
    """可视化特征图"""
    model.eval()
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 前向传播
    with torch.no_grad():
        _ = model(image.unsqueeze(0))

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 绘制特征图
    for name, activation in activations.items():
        act = activation.squeeze().cpu().numpy()
        n_features = min(act.shape[0], 64)

        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < n_features:
                ax.imshow(act[i], cmap='viridis')
            ax.axis('off')

        plt.suptitle(f'Feature Maps: {name}')
        plt.tight_layout()
        plt.savefig(f'feature_maps_{name}.png')
        plt.show()
```

---

## 8. 应用场景

### 8.1 图像分类

```
应用场景:
┌─────────────────────────────────────────────────────┐
│ • 医学影像诊断 (X光/CT/MRI分析)                      │
│ • 自动驾驶 (道路标志识别)                            │
│ • 安防监控 (人脸识别)                               │
│ • 农业 (病虫害检测)                                 │
│ • 零售 (商品识别)                                   │
└─────────────────────────────────────────────────────┘
```

### 8.2 目标检测

```
┌───────────────────────────────────────────┐
│                 目标检测                    │
│                                           │
│    ┌────────┐                             │
│    │  🐕   │ dog: 0.98                   │
│    └────────┘                             │
│           ┌─────────┐                     │
│           │  🐱    │ cat: 0.95           │
│           └─────────┘                     │
│                      ┌──────────┐         │
│                      │  🚗     │ car:0.89│
│                      └──────────┘         │
│                                           │
│ 模型: YOLO, Faster R-CNN, SSD             │
└───────────────────────────────────────────┘
```

### 8.3 语义分割

```
输入图像                    分割结果
┌─────────────────┐       ┌─────────────────┐
│  🌳  🏠  ☁️    │       │ ████ ████ ████  │ 绿色=树
│ 🚗 🚗 🚗       │  →    │ ▓▓▓▓▓▓▓▓▓      │ 灰色=车
│ ▂▂▂▂▂▂▂▂      │       │ ░░░░░░░░░░     │ 白色=道路
└─────────────────┘       └─────────────────┘

模型: U-Net, DeepLab, FCN
```

### 8.4 更多应用

| 领域         | 应用     | 典型模型      |
| ------------ | -------- | ------------- |
| 自然语言处理 | 文本分类 | TextCNN       |
| 语音识别     | 声纹识别 | CNN + RNN     |
| 视频分析     | 动作识别 | 3D CNN, C3D   |
| 生成模型     | 图像生成 | DCGAN         |
| 超分辨率     | 图像增强 | SRCNN, ESRGAN |

---

## 9. 常见问题与优化

### 9.1 过拟合解决方案

```
┌──────────────────────────────────────────────┐
│              防止过拟合策略                    │
├──────────────────────────────────────────────┤
│                                              │
│  1. 数据增强 (Data Augmentation)              │
│     - 随机裁剪、翻转、旋转                     │
│     - 颜色抖动、Mixup、Cutout                  │
│                                              │
│  2. 正则化                                    │
│     - L1/L2 正则化                            │
│     - Dropout (通常 p=0.5)                    │
│     - Batch Normalization                    │
│                                              │
│  3. 早停 (Early Stopping)                    │
│     - 监控验证集损失                           │
│                                              │
│  4. 更多数据                                  │
│     - 迁移学习                                │
│     - 数据采集                                │
│                                              │
└──────────────────────────────────────────────┘
```

### 9.2 常用数据增强

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    # 几何变换
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

    # 颜色变换
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),

    # 转换为张量
    transforms.ToTensor(),

    # 标准化
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    # 高级增强
    transforms.RandomErasing(p=0.5),  # Cutout效果
])
```

### 9.3 训练技巧

```
┌──────────────────────────────────────────────────────┐
│                    训练技巧                           │
├──────────────────────────────────────────────────────┤
│                                                      │
│  📈 学习率策略                                        │
│     • 学习率预热 (Warmup)                             │
│     • 余弦退火 (Cosine Annealing)                     │
│     • 学习率查找 (LR Finder)                          │
│                                                      │
│  🔧 优化器选择                                        │
│     • SGD + Momentum (大数据集)                       │
│     • Adam / AdamW (快速收敛)                         │
│     • LAMB (大batch训练)                              │
│                                                      │
│  📊 Batch Normalization                              │
│     • 放在卷积后,激活前                               │
│     • 加速训练,正则化效果                              │
│                                                      │
│  🎯 权重初始化                                        │
│     • He初始化 (ReLU)                                 │
│     • Xavier初始化 (Sigmoid/Tanh)                    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 9.4 模型部署优化

```python
# 模型量化
import torch.quantization as quant

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# 模型剪枝
import torch.nn.utils.prune as prune

# 剪枝30%的权重
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

---

## 📖 总结

### CNN 核心要点

```
┌────────────────────────────────────────────────────────────┐
│                      CNN 学习路线图                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  基础概念 ──────────────────────────────────────────→       │
│    │                                                       │
│    ├── 卷积操作: 局部连接 + 权重共享                         │
│    ├── 池化操作: 降维 + 平移不变性                           │
│    ├── 激活函数: 非线性变换 (ReLU)                          │
│    └── 全连接层: 特征整合 + 分类                             │
│                                                            │
│  核心模型 ──────────────────────────────────────────→       │
│    │                                                       │
│    ├── LeNet → AlexNet → VGG → Inception                   │
│    └── ResNet → DenseNet → EfficientNet                    │
│                                                            │
│  实践技能 ──────────────────────────────────────────→       │
│    │                                                       │
│    ├── 数据预处理与增强                                     │
│    ├── 模型训练与调优                                       │
│    ├── 迁移学习                                            │
│    └── 模型部署                                            │
│                                                            │
│  进阶方向 ──────────────────────────────────────────→       │
│    │                                                       │
│    ├── 目标检测 (YOLO, Faster R-CNN)                       │
│    ├── 语义分割 (U-Net, DeepLab)                           │
│    ├── 实例分割 (Mask R-CNN)                               │
│    └── 视觉Transformer                                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 推荐学习资源

| 资源类型 | 推荐                                                   |
| -------- | ------------------------------------------------------ |
| 课程     | CS231n (斯坦福), Deep Learning Specialization (吴恩达) |
| 书籍     | 《深度学习》花书, 《动手学深度学习》                   |
| 框架     | PyTorch, TensorFlow, Keras                             |
| 实践     | Kaggle 竞赛, Papers With Code                          |

---

> 📝 **文档版本**: v1.0  
> 📅 **最后更新**: 2024 年  
> 💡 **建议**: 理论结合实践，多动手写代码！
