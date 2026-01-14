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
            in_channels=3,  # RGB输入
            out_channels=32,  # 32个卷积核
            kernel_size=3,  # 3×3卷积核
            padding=1,  # 保持尺寸
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
