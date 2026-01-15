"""
07-resnet.py - ResNet 残差网络

本节学习: 残差连接原理、BasicBlock/Bottleneck、ResNet-18/34/50 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("第7节: ResNet 残差网络")
print("=" * 60)

# 残差连接原理
print("""
🏆 ResNet (2015, Kaiming He) - ImageNet 冠军, 解决深度退化问题

残差学习核心思想:
─────────────────────

普通网络学习: H(x) = 目标映射
残差网络学习: F(x) = H(x) - x, 则 H(x) = F(x) + x

     x ─────────────────────┐
     │                      │ (恒等快捷连接)
     ↓                      ↓
  ┌─────┐   ┌─────┐      ┌─────┐
  │Conv1│ → │Conv2│ ─────│  +  │──→ ReLU ──→ 输出
  └─────┘   └─────┘      └─────┘
  
为什么有效:
1. 梯度可以直接通过跳跃连接传播
2. 恒等映射比学习零映射更容易
3. 支持训练超过100层的网络
""")


class BasicBlock(nn.Module):
    """ResNet 基本残差块 (用于 ResNet-18/34)"""

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


class Bottleneck(nn.Module):
    """ResNet 瓶颈残差块 (用于 ResNet-50/101/152)"""

    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion),
            )
        layers = [block(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


# 测试
for name, model_fn in [
    ("ResNet-18", resnet18),
    ("ResNet-34", resnet34),
    ("ResNet-50", resnet50),
]:
    model = model_fn()
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params / 1e6:.1f}M 参数")

print("\n测试前向传播:")
x = torch.randn(2, 3, 224, 224)
print(f"输入: {x.shape} → 输出: {resnet18()(x).shape}")

print("""
📝 要点总结:
1. 残差连接: H(x) = F(x) + x
2. BasicBlock: 两个 3×3 卷积 (ResNet-18/34)
3. Bottleneck: 1×1→3×3→1×1 (ResNet-50+)
4. 使用 GAP 替代全连接层，大幅减少参数
""")
