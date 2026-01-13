"""
09-mobilenet.py - 轻量化网络 MobileNet

本节学习: 深度可分离卷积、MobileNet V1/V2 架构
"""
import torch
import torch.nn as nn

print("=" * 60)
print("第9节: MobileNet 轻量化网络")
print("=" * 60)

print("""
📌 为什么需要轻量化网络?
- 移动端/嵌入式设备计算资源有限
- 需要实时推理
- VGG-16: 138M 参数，ResNet-50: 25M 参数
- MobileNet V1: 4.2M 参数，精度接近

核心技术: 深度可分离卷积 (Depthwise Separable Conv)
────────────────────────────────────────────────────

普通卷积: 3×3×C_in×C_out 参数
深度可分离: 3×3×C_in + 1×1×C_in×C_out 参数

例如 3×3, in=64, out=128:
  普通卷积: 3×3×64×128 = 73,728
  可分离:  3×3×64 + 1×1×64×128 = 576 + 8,192 = 8,768
  减少约 8.4 倍!
""")

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 深度卷积: 每个通道单独卷积
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        # 逐点卷积: 1×1 卷积混合通道
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# MobileNet V1 简化实现
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        def c(x): return int(x * width_mult)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, c(32), 3, 2, 1, bias=False),
            nn.BatchNorm2d(c(32)), nn.ReLU6(True),
            DepthwiseSeparableConv(c(32), c(64), 1),
            DepthwiseSeparableConv(c(64), c(128), 2),
            DepthwiseSeparableConv(c(128), c(128), 1),
            DepthwiseSeparableConv(c(128), c(256), 2),
            DepthwiseSeparableConv(c(256), c(256), 1),
            DepthwiseSeparableConv(c(256), c(512), 2),
            *[DepthwiseSeparableConv(c(512), c(512), 1) for _ in range(5)],
            DepthwiseSeparableConv(c(512), c(1024), 2),
            DepthwiseSeparableConv(c(1024), c(1024), 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c(1024), num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))

# MobileNet V2 倒置残差块
class InvertedResidual(nn.Module):
    """MobileNet V2 倒置残差块"""
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                      nn.BatchNorm2d(hidden), nn.ReLU6(True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)

# 测试
model = MobileNetV1(num_classes=1000)
params = sum(p.numel() for p in model.parameters())
print(f"MobileNet V1 参数量: {params/1e6:.1f}M")

x = torch.randn(2, 3, 224, 224)
print(f"输入: {x.shape} → 输出: {model(x).shape}")

# 使用 torchvision
from torchvision import models
mv2 = models.mobilenet_v2(weights=None)
print(f"MobileNet V2 参数量: {sum(p.numel() for p in mv2.parameters())/1e6:.1f}M")

print("""
📝 要点总结:
1. 深度可分离卷积: Depthwise + Pointwise
2. 计算量减少约 8-9 倍
3. MobileNet V2: 倒置残差 + 线性瓶颈
4. 宽度乘子 (width_mult) 控制模型大小
""")
