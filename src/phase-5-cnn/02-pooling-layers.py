"""
02-pooling-layers.py - 池化层详解

本节学习内容：
1. 池化层的作用
2. 最大池化 (Max Pooling)
3. 平均池化 (Average Pooling)
4. 全局平均池化 (Global Average Pooling)
5. 池化的特性：平移不变性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("第2节: 池化层 (Pooling Layers)")
print("=" * 60)

# ============================================================
# 1. 池化层的作用
# ============================================================
print("\n📌 1. 为什么需要池化层？")
print("-" * 40)

print("""
池化层的主要作用:
┌─────────────────────────────────────┐
│  1. 降低特征图尺寸 → 减少计算量      │
│  2. 增加感受野                       │
│  3. 提供平移不变性                   │
│  4. 防止过拟合                       │
└─────────────────────────────────────┘
""")

# ============================================================
# 2. 最大池化 (Max Pooling)
# ============================================================
print("\n📌 2. 最大池化 (Max Pooling)")
print("-" * 40)

# 创建一个4x4的输入
input_4x4 = torch.tensor([
    [1., 3., 2., 4.],
    [5., 6., 7., 8.],
    [1., 2., 3., 4.],
    [1., 1., 2., 2.]
]).unsqueeze(0).unsqueeze(0)  # [batch, channel, H, W]

print(f"输入 (4×4):\n{input_4x4.squeeze()}")

# 2x2最大池化，步长2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
output = max_pool(input_4x4)

print(f"\n2×2 MaxPooling，stride=2:")
print(f"输出 (2×2):\n{output.squeeze()}")

print("""
计算过程:
┌───┬───┐ ┌───┬───┐     ┌───┬───┐
│ 1 │ 3 │ │ 2 │ 4 │     │   │   │
├───┼───┤ ├───┼───┤  →  │ 6 │ 8 │
│ 5 │ 6 │ │ 7 │ 8 │     ├───┼───┤
└───┴───┘ └───┴───┘     │ 2 │ 4 │
┌───┬───┐ ┌───┬───┐     │   │   │
│ 1 │ 2 │ │ 3 │ 4 │     └───┴───┘
├───┼───┤ ├───┼───┤
│ 1 │ 1 │ │ 2 │ 2 │
└───┴───┘ └───┴───┘

max(1,3,5,6)=6, max(2,4,7,8)=8
max(1,2,1,1)=2, max(3,4,2,2)=4
""")

# ============================================================
# 3. 平均池化 (Average Pooling)
# ============================================================
print("\n📌 3. 平均池化 (Average Pooling)")
print("-" * 40)

avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
output_avg = avg_pool(input_4x4)

print(f"输入 (4×4):\n{input_4x4.squeeze()}")
print(f"\n2×2 AvgPooling，stride=2:")
print(f"输出 (2×2):\n{output_avg.squeeze()}")

print("""
计算过程:
avg(1,3,5,6) = 15/4 = 3.75
avg(2,4,7,8) = 21/4 = 5.25
avg(1,2,1,1) = 5/4  = 1.25
avg(3,4,2,2) = 11/4 = 2.75
""")

# ============================================================
# 4. 最大池化 vs 平均池化
# ============================================================
print("\n📌 4. MaxPool vs AvgPool 对比")
print("-" * 40)

print("""
┌────────────────┬────────────────┬────────────────┐
│     特性       │   Max Pooling  │   Avg Pooling  │
├────────────────┼────────────────┼────────────────┤
│ 保留信息       │  最显著特征     │  区域平均特征   │
│ 对噪声敏感性   │  更敏感         │  更鲁棒         │
│ 常用位置       │  中间层         │  网络末端       │
│ 梯度传播       │  仅最大值位置   │  均匀分布       │
│ 典型应用       │  特征提取       │  分类器前端     │
└────────────────┴────────────────┴────────────────┘
""")

# ============================================================
# 5. 全局平均池化 (Global Average Pooling)
# ============================================================
print("\n📌 5. 全局平均池化 (GAP)")
print("-" * 40)

# 模拟CNN最后一层的特征图
feature_map = torch.randn(1, 512, 7, 7)  # [batch, channels, H, W]
print(f"特征图形状: {feature_map.shape}")

# 全局平均池化
gap = nn.AdaptiveAvgPool2d((1, 1))
output_gap = gap(feature_map)
print(f"GAP输出形状: {output_gap.shape}")

# 展平后可以直接接分类器
output_flat = output_gap.view(1, -1)
print(f"展平后形状: {output_flat.shape}")

print("""
GAP的优势:
1. 大幅减少参数量（替代全连接层）
2. 对空间位置更加鲁棒
3. 防止过拟合
4. 更容易解释（每个通道对应一个类别）

使用方式:
nn.AdaptiveAvgPool2d((1, 1))  # 任意输入 → 1×1 输出
""")

# ============================================================
# 6. 不同池化核大小的效果
# ============================================================
print("\n📌 6. 池化参数的影响")
print("-" * 40)

input_16x16 = torch.randn(1, 1, 16, 16)
print(f"输入形状: {input_16x16.shape}")

# 不同配置
configs = [
    (2, 2),   # kernel=2, stride=2
    (3, 2),   # kernel=3, stride=2 (重叠池化)
    (2, 1),   # kernel=2, stride=1 (密集池化)
    (4, 4),   # kernel=4, stride=4 (更激进的下采样)
]

for kernel_size, stride in configs:
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    output = pool(input_16x16)
    print(f"kernel={kernel_size}, stride={stride}: 输出形状 {output.shape}")

# ============================================================
# 7. 池化的平移不变性
# ============================================================
print("\n📌 7. 平移不变性演示")
print("-" * 40)

# 原始图像（特征在左上角）
img1 = torch.zeros(1, 1, 6, 6)
img1[0, 0, 0:2, 0:2] = 1.0

# 平移后的图像（特征在右边）
img2 = torch.zeros(1, 1, 6, 6)
img2[0, 0, 0:2, 1:3] = 1.0  # 右移1像素

print("原始图像:")
print(img1.squeeze())
print("\n右移1像素后:")
print(img2.squeeze())

# 应用池化
pool = nn.MaxPool2d(2, 2)
out1 = pool(img1)
out2 = pool(img2)

print(f"\n原始图像池化结果:\n{out1.squeeze()}")
print(f"\n平移后池化结果:\n{out2.squeeze()}")
print("\n→ 第一个区域的最大值都保持为1，体现了一定的平移不变性")

# ============================================================
# 8. PyTorch池化层完整API
# ============================================================
print("\n📌 8. PyTorch池化层API")
print("-" * 40)

print("""
1. 最大池化:
   nn.MaxPool2d(kernel_size, stride=None, padding=0)
   
2. 平均池化:
   nn.AvgPool2d(kernel_size, stride=None, padding=0)
   
3. 自适应池化 (输出固定大小):
   nn.AdaptiveMaxPool2d(output_size)   # 如 (7, 7)
   nn.AdaptiveAvgPool2d(output_size)   # 如 (1, 1)

4. 带索引的最大池化 (用于上采样):
   nn.MaxPool2d(kernel_size, return_indices=True)
   nn.MaxUnpool2d(kernel_size)
""")

# 带索引的最大池化示例
pool_with_idx = nn.MaxPool2d(2, 2, return_indices=True)
output, indices = pool_with_idx(input_4x4)
print(f"\n带索引的MaxPool:")
print(f"输出: {output.squeeze()}")
print(f"索引: {indices.squeeze()}")

# ============================================================
# 9. 常见架构中的池化策略
# ============================================================
print("\n📌 9. 经典网络的池化策略")
print("-" * 40)

print("""
┌──────────────┬───────────────────────────────────────┐
│    网络      │           池化策略                     │
├──────────────┼───────────────────────────────────────┤
│   LeNet      │ 2×2 AvgPool                           │
│   AlexNet    │ 3×3 MaxPool, stride=2 (重叠池化)       │
│   VGG        │ 2×2 MaxPool, stride=2                  │
│   GoogLeNet  │ 3×3 MaxPool + Inception模块中的池化    │
│   ResNet     │ 开头7×7conv后3×3 MaxPool + 最后GAP     │
│   MobileNet  │ GAP 替代全连接层                       │
└──────────────┴───────────────────────────────────────┘
""")

# ============================================================
# 10. 实际使用示例
# ============================================================
print("\n📌 10. 实际使用示例：简单CNN块")
print("-" * 40)

class ConvBlock(nn.Module):
    """卷积 + ReLU + 池化"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

# 测试
block = ConvBlock(3, 64)
test_input = torch.randn(1, 3, 32, 32)
test_output = block(test_input)
print(f"输入: {test_input.shape}")
print(f"经过 ConvBlock 后: {test_output.shape}")
print("尺寸变化: 32×32 → 16×16 (池化降采样)")

# ============================================================
# 练习题
# ============================================================
print("\n" + "=" * 60)
print("💡 练习题")
print("=" * 60)

print("""
1. 一个 14×14 的特征图，经过 kernel=2, stride=2 的 MaxPool，
   输出尺寸是多少？

2. 如何用 AdaptiveAvgPool2d 将任意尺寸的特征图转为 [batch, C, 1, 1]？

3. 为什么 ResNet 用 GAP 而不是全连接层？

4. 重叠池化 (kernel > stride) 有什么优缺点？
""")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("📝 本节要点总结")
print("=" * 60)

print("""
1. 池化作用: 降维、增大感受野、平移不变性
2. MaxPool: 保留最显著特征，常用于中间层
3. AvgPool: 保留平均信息，更平滑
4. GAP: 全局平均池化，替代全连接层
5. 池化无可学习参数，不增加模型容量

常用配置:
- 标准: MaxPool2d(2, 2) → 尺寸减半
- 分类器前: AdaptiveAvgPool2d((1, 1))

下一节: 感受野 (Receptive Field)
""")
