"""
04-lenet.py - LeNet 网络实现

本节学习内容：
1. LeNet 的历史意义
2. LeNet-5 架构详解
3. PyTorch 实现 LeNet
4. 在 MNIST 上训练和测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("=" * 60)
print("第4节: LeNet 架构")
print("=" * 60)

# ============================================================
# 1. LeNet 历史背景
# ============================================================
print("\n📌 1. LeNet 历史意义")
print("-" * 40)

print("""
LeNet-5 (1998, Yann LeCun)
──────────────────────────

🏆 第一个成功的卷积神经网络
📋 用于手写数字识别（邮政编码识别）
🎯 奠定了现代 CNN 的基础

关键创新:
1. 卷积层提取局部特征
2. 池化层降维
3. 端到端训练

架构图:
  输入      C1        S2       C3        S4       C5      F6     Output
 32×32   28×28×6   14×14×6  10×10×16   5×5×16  1×1×120   84      10
   │        │         │        │         │        │       │       │
   └──Conv──┴──Pool───┴──Conv──┴──Pool───┴──Conv──┴──FC───┴──FC───┘
      5×5     2×2       5×5      2×2       5×5
""")

# ============================================================
# 2. LeNet-5 架构实现
# ============================================================
print("\n📌 2. LeNet-5 PyTorch 实现")
print("-" * 40)

class LeNet5(nn.Module):
    """
    经典 LeNet-5 实现
    输入: 1×32×32 或 1×28×28 (MNIST)
    输出: 10 类
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 保持32×32
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        
        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
        # 池化层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # C1: 卷积层 + 激活 + 池化
        # 输入: 1×32×32 → 输出: 6×14×14
        x = self.pool(torch.tanh(self.conv1(x)))
        
        # C3: 卷积层 + 激活 + 池化
        # 输入: 6×14×14 → 输出: 16×5×5
        x = self.pool(torch.tanh(self.conv2(x)))
        
        # C5: 卷积层 + 激活
        # 输入: 16×5×5 → 输出: 120×1×1
        x = torch.tanh(self.conv3(x))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # F6: 全连接层
        x = torch.tanh(self.fc1(x))
        
        # 输出层
        x = self.fc2(x)
        
        return x

# 打印模型结构
model = LeNet5()
print(model)

# 测试前向传播
dummy_input = torch.randn(1, 1, 32, 32)
output = model(dummy_input)
print(f"\n输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# ============================================================
# 3. 现代化的 LeNet (使用 ReLU 和 MaxPool)
# ============================================================
print("\n📌 3. 现代化 LeNet")
print("-" * 40)

class LeNetModern(nn.Module):
    """
    现代化的 LeNet 实现
    - 使用 ReLU 替代 Tanh
    - 使用 MaxPool 替代 AvgPool
    - 添加 Dropout
    """
    def __init__(self, num_classes=10):
        super(LeNetModern, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model_modern = LeNetModern()
print("现代化 LeNet 结构:")
print(model_modern)

# ============================================================
# 4. 在 MNIST 上训练
# ============================================================
print("\n📌 4. MNIST 数据集准备")
print("-" * 40)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet 需要 32×32 输入
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 检查是否可以加载数据集
try:
    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 查看一些样本
    sample_images, sample_labels = next(iter(train_loader))
    print(f"批次形状: {sample_images.shape}")
    print(f"标签示例: {sample_labels[:10]}")
    
    DATA_LOADED = True
except Exception as e:
    print(f"数据加载失败 (这在首次运行时是正常的): {e}")
    print("请确保网络连接，或手动下载 MNIST 数据集")
    DATA_LOADED = False

# ============================================================
# 5. 训练函数
# ============================================================
print("\n📌 5. 训练与评估函数")
print("-" * 40)

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return total_loss / len(test_loader), 100. * correct / total

# ============================================================
# 6. 运行训练
# ============================================================
print("\n📌 6. 训练 LeNet (简短演示)")
print("-" * 40)

if DATA_LOADED:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = LeNetModern().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练 2 个 epoch 作为演示
    num_epochs = 2
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
else:
    print("跳过训练演示 (数据未加载)")

# ============================================================
# 7. 层次特征可视化
# ============================================================
print("\n📌 7. 理解 LeNet 的特征提取")
print("-" * 40)

print("""
LeNet 的层级特征提取过程:

Layer 1 (Conv1 + Pool1):
┌─────────────────────────────────────────┐
│  输入: 32×32 灰度图                      │
│  输出: 6个 14×14 特征图                  │
│  学习: 边缘、简单纹理                    │
└─────────────────────────────────────────┘
          ↓
Layer 2 (Conv2 + Pool2):
┌─────────────────────────────────────────┐
│  输入: 6×14×14                          │
│  输出: 16个 5×5 特征图                   │
│  学习: 笔画组合、局部形状                │
└─────────────────────────────────────────┘
          ↓
Layer 3 (Conv3):
┌─────────────────────────────────────────┐
│  输入: 16×5×5                           │
│  输出: 120个 1×1 特征                    │
│  学习: 数字的整体特征                    │
└─────────────────────────────────────────┘
          ↓
Classifier (FC):
┌─────────────────────────────────────────┐
│  120 → 84 → 10                          │
│  分类决策                                │
└─────────────────────────────────────────┘
""")

# ============================================================
# 8. LeNet vs 现代 CNN
# ============================================================
print("\n📌 8. LeNet vs 现代 CNN 对比")
print("-" * 40)

print("""
┌───────────────┬─────────────────┬─────────────────┐
│     特性      │    LeNet-5      │    现代 CNN     │
├───────────────┼─────────────────┼─────────────────┤
│   激活函数    │    Tanh         │    ReLU         │
│   池化方式    │    AvgPool      │    MaxPool      │
│   正则化      │    无           │    Dropout/BN   │
│   网络深度    │    5层          │    数十到数百层 │
│   参数量      │    约6万        │    数百万到亿级 │
│   输入尺寸    │    32×32        │    224×224+     │
│   任务        │    手写数字     │    复杂场景     │
└───────────────┴─────────────────┴─────────────────┘

LeNet 的历史贡献:
✅ 证明了 CNN 的有效性
✅ 建立了 Conv-Pool-FC 的经典架构
✅ 启发了后续所有 CNN 的设计
""")

# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("💡 练习题")
print("=" * 60)

print("""
1. 修改 LeNet 使其接受 CIFAR-10 的 3×32×32 RGB 图像

2. 尝试增加卷积层的通道数（如 6→32, 16→64）,
   观察准确率和参数量的变化

3. 在 LeNet 中添加 BatchNorm 层，观察训练效果

4. 将池化层从 AvgPool 改为 MaxPool，比较结果
""")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("📝 本节要点总结")
print("=" * 60)

print("""
1. LeNet-5 是第一个成功的 CNN，用于手写数字识别

2. 架构: Conv → Pool → Conv → Pool → Conv → FC → FC

3. 原始版本使用 Tanh 和 AvgPool

4. 现代改进:
   - ReLU 替代 Tanh
   - MaxPool 替代 AvgPool
   - 添加 Dropout 和 BatchNorm

5. LeNet 参数量约 6 万，在 MNIST 上可达 99%+ 准确率

下一节: AlexNet 架构
""")
