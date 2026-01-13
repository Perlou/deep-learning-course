"""
08-resnet-variants.py - ResNet 变体与改进

本节学习: ResNet 各版本对比、预训练模型使用、迁移学习
"""
import torch
import torch.nn as nn
from torchvision import models

print("=" * 60)
print("第8节: ResNet 变体与改进")
print("=" * 60)

# ResNet 家族对比
print("""
ResNet 家族对比:
┌──────────────┬────────┬────────┬────────────────────┐
│    模型      │  参数量 │ Top-1  │     架构          │
├──────────────┼────────┼────────┼────────────────────┤
│  ResNet-18   │  11.7M │ 69.8%  │ [2, 2, 2, 2]       │
│  ResNet-34   │  21.8M │ 73.3%  │ [3, 4, 6, 3]       │
│  ResNet-50   │  25.6M │ 76.1%  │ [3, 4, 6, 3] 瓶颈  │
│  ResNet-101  │  44.5M │ 77.4%  │ [3, 4, 23, 3]      │
│  ResNet-152  │  60.2M │ 78.3%  │ [3, 8, 36, 3]      │
└──────────────┴────────┴────────┴────────────────────┘

注: ResNet-50+ 使用 Bottleneck 结构
    expansion=4, 所以参数量和 34 接近但层数翻倍
""")

# 使用预训练模型
print("\n📌 加载预训练模型:")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print(f"ResNet-50 参数量: {sum(p.numel() for p in model.parameters()):,}")

# 迁移学习示例
print("\n📌 迁移学习示例 (10类分类):")

class TransferResNet(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        # 加载预训练 ResNet
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 冻结特征提取层
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换分类头
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

model = TransferResNet(num_classes=10)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable:,} (只训练分类头)")

# 测试
x = torch.randn(2, 3, 224, 224)
print(f"输出形状: {model(x).shape}")

# ResNet 改进版本
print("""
📌 ResNet 改进版本:

1. ResNeXt: 分组卷积，cardinality=32
2. SE-ResNet: 通道注意力机制
3. ResNet-D: 改进下采样路径
4. ResNet-RS: 训练策略改进

使用 torchvision:
  models.resnext50_32x4d()
  models.wide_resnet50_2()
""")

print("""
📝 要点总结:
1. ResNet-50 是最常用的骨干网络 (准确率/速度平衡好)
2. 迁移学习: 冻结骨干，只训练分类头
3. 更深不一定更好，考虑任务复杂度选择合适深度
""")
