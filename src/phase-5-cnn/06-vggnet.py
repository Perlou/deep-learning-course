"""
06-vggnet.py - VGGNet 架构

本节学习: VGG 设计哲学、3×3 小卷积核堆叠、VGG-16 实现
"""

import torch
import torch.nn as nn

print("=" * 60)
print("第6节: VGGNet 架构")
print("=" * 60)

# VGG 设计哲学
print("""
VGG 核心思想: 用多个 3×3 小卷积核替代大卷积核

  两个 3×3 = 一个 5×5 的感受野 (参数: 18 < 25)
  三个 3×3 = 一个 7×7 的感受野 (参数: 27 < 49)

设计规则:
- 所有卷积 3×3，padding=1
- 所有池化 2×2，stride=2
- 通道数翻倍: 64→128→256→512
""")

# VGG 配置
cfgs = {
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
}


def make_layers(cfg, batch_norm=False):
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(2, 2)]
        else:
            conv = nn.Conv2d(in_ch, v, 3, padding=1)
            layers += (
                [conv, nn.BatchNorm2d(v), nn.ReLU(True)]
                if batch_norm
                else [conv, nn.ReLU(True)]
            )
            in_ch = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self, num_classes=1000, batch_norm=True):
        super().__init__()
        self.features = make_layers(cfgs["vgg16"], batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# 测试
model = VGG16()
print(f"\nVGG-16 参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"输入: (1,3,224,224) → 输出: {model(torch.randn(1, 3, 224, 224)).shape}")


# CIFAR-10 简化版
class VGGCifar(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


print(f"\nCIFAR-10 VGG 参数量: {sum(p.numel() for p in VGGCifar().parameters()):,}")

print("""
📝 要点总结:
1. 3×3 卷积是黄金标准
2. 通道数逐渐增加，空间尺寸逐渐减小
3. VGG 参数量大 (138M)，大部分在全连接层
""")
