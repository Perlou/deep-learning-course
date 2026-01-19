"""
U-Net 架构详解 (U-Net Architecture)
====================================

学习目标：
    1. 理解 U-Net 的架构设计
    2. 掌握 Skip Connection 的作用
    3. 能够从零实现 U-Net
    4. 在医学图像分割任务上应用 U-Net

核心概念：
    - U 形结构: 对称的编码器-解码器
    - Skip Connection: 跳跃连接保留细节
    - 拼接 (Concatenation): 融合多尺度特征
    - 医学图像分割: U-Net 的主要应用

前置知识：
    - 04-semantic-segmentation.py: 语义分割基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：U-Net 概述 ====================


def introduction():
    """U-Net 概述"""
    print("=" * 60)
    print("第一部分：U-Net 概述")
    print("=" * 60)

    print("""
U-Net 简介：

    U-Net 由 Ronneberger 等人于 2015 年提出，
    最初用于医学图像分割，现已广泛应用于各种分割任务。

    为什么叫 "U-Net"？
    - 网络结构呈 U 形

U-Net 架构图：

    输入 ────────────────────────────────────────────────► 输出
    572×572                                              388×388
      │                                                     ↑
      ▼                       Skip Connection               │
    ┌────┐ ──────────────────────────────────────────────► ┌────┐
    │ 64 │    64 channels                                  │ 64 │
    └────┘                                                 └────┘
      │ ↓ MaxPool                                    ↑ UpConv │
    ┌────┐ ──────────────────────────────────────────────► ┌────┐
    │128 │    128 channels                                 │128 │
    └────┘                                                 └────┘
      │ ↓                                              ↑       │
    ┌────┐ ──────────────────────────────────────────────► ┌────┐
    │256 │    256 channels                                 │256 │
    └────┘                                                 └────┘
      │ ↓                                              ↑       │
    ┌────┐ ──────────────────────────────────────────────► ┌────┐
    │512 │    512 channels                                 │512 │
    └────┘                                                 └────┘
      │ ↓                                              ↑       │
      └──────────────► ┌────┐ ──────────────────────────┘
                       │1024│   瓶颈层 (Bottleneck)
                       └────┘

关键特点：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 对称结构: 编码器和解码器对称                           │
    │ 2. Skip Connection: 将编码器特征直接传递给解码器          │
    │ 3. 拼接操作: 将低层特征与高层特征拼接                      │
    │ 4. 少量数据也能训练: 适合医学图像                         │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：Skip Connection ====================


def skip_connection():
    """Skip Connection 详解"""
    print("\n" + "=" * 60)
    print("第二部分：Skip Connection 的作用")
    print("=" * 60)

    print("""
为什么需要 Skip Connection？

    问题: 编码器下采样过程中丢失空间细节信息

    ┌─────────────────────────────────────────────────────────┐
    │  没有 Skip Connection:                                  │
    │                                                         │
    │  输入 → 编码器 → 瓶颈 → 解码器 → 输出                     │
    │                   ↑                                     │
    │              细节信息丢失                                │
    │              边界模糊                                    │
    │                                                         │
    │  有 Skip Connection:                                    │
    │                                                         │
    │  输入 → 编码器 ──────────────→ 解码器 → 输出             │
    │            │         拼接         ↑                     │
    │            └───────────→─────────┘                      │
    │                保留细节信息                              │
    │                边界清晰                                  │
    └─────────────────────────────────────────────────────────┘

Skip Connection 的具体操作：

    编码器特征 (256 channels, 64×64)
         │
         └─────────────────────────────────┐
                                           │ 拼接 (Concatenate)
    解码器上采样后 (256 channels, 64×64) ──┘
         │
         ▼
    拼接后 (512 channels, 64×64)
         │
    卷积 (512 → 256)
         │
         ▼
    输出 (256 channels, 64×64)

对比: ResNet 的跳跃连接是加法，U-Net 是拼接
    - 加法: H(x) = F(x) + x
    - 拼接: H(x) = Concat(F(x), x)
    """)


# ==================== 第三部分：U-Net 完整实现 ====================


def unet_implementation():
    """U-Net 完整实现"""
    print("\n" + "=" * 60)
    print("第三部分：U-Net 完整实现")
    print("=" * 60)

    class DoubleConv(nn.Module):
        """双卷积块: (Conv → BN → ReLU) × 2"""

        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels

            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        """下采样块: MaxPool → DoubleConv"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        """上采样块: UpConv → Concat → DoubleConv"""

        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()

            if bilinear:
                self.up = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(
                    in_channels, in_channels // 2, kernel_size=2, stride=2
                )
                self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)

            # 处理尺寸不一致
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

            # 拼接
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class OutConv(nn.Module):
        """输出卷积: 1×1 卷积"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        """完整的 U-Net"""

        def __init__(self, n_channels=3, n_classes=2, bilinear=True):
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            # 编码器
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)

            # 解码器
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)

            # 输出层
            self.outc = OutConv(64, n_classes)

        def forward(self, x):
            # 编码
            x1 = self.inc(x)  # 64
            x2 = self.down1(x1)  # 128
            x3 = self.down2(x2)  # 256
            x4 = self.down3(x3)  # 512
            x5 = self.down4(x4)  # 1024

            # 解码 (带 Skip Connection)
            x = self.up1(x5, x4)  # 512
            x = self.up2(x, x3)  # 256
            x = self.up3(x, x2)  # 128
            x = self.up4(x, x1)  # 64

            # 输出
            logits = self.outc(x)
            return logits

    # 测试
    print("U-Net 模型测试:\n")

    model = UNet(n_channels=3, n_classes=2)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 前向传播测试
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")

    return UNet


# ==================== 第四部分：U-Net 变体 ====================


def unet_variants():
    """U-Net 变体"""
    print("\n" + "=" * 60)
    print("第四部分：U-Net 变体")
    print("=" * 60)

    print("""
U-Net 变体：

    ┌──────────────────────────────────────────────────────────┐
    │  变体名称        改进点                     应用场景      │
    ├──────────────────────────────────────────────────────────┤
    │  U-Net++        嵌套跳跃连接                医学分割     │
    │  Attention U-Net 注意力门控                 医学分割     │
    │  R2U-Net        循环残差卷积                医学分割     │
    │  3D U-Net       处理体素数据                CT/MRI分割   │
    │  ResU-Net       ResNet作为编码器            通用分割     │
    │  EfficientU-Net  EfficientNet作为编码器    高效分割     │
    └──────────────────────────────────────────────────────────┘

U-Net++ 结构：

    使用密集跳跃连接，逐步融合特征

    X0,0 ─────────────────────────────────→ X0,4
      │                                      ↑
    X1,0 ───────────────────→ X1,3 ─────────┘
      │                        ↑
    X2,0 ───────→ X2,2 ───────┘
      │            ↑
    X3,0 → X3,1 ──┘
      │    ↑
    X4,0 ──┘

Attention U-Net：

    在跳跃连接中加入注意力机制：
    - 抑制不相关区域
    - 增强目标区域特征
    """)

    # Attention Gate 示例
    print("示例: Attention Gate\n")

    class AttentionGate(nn.Module):
        """注意力门控"""

        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
                nn.BatchNorm2d(F_int),
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
                nn.BatchNorm2d(F_int),
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            """
            g: 来自解码器的特征 (gating signal)
            x: 来自编码器的特征
            """
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    print("Attention Gate 作用:")
    print("  - 根据解码器特征 g 生成注意力权重")
    print("  - 对编码器特征 x 进行加权")
    print("  - 抑制不相关背景区域")


# ==================== 第五部分：医学图像分割 ====================


def medical_segmentation():
    """医学图像分割"""
    print("\n" + "=" * 60)
    print("第五部分：医学图像分割应用")
    print("=" * 60)

    print("""
医学图像分割特点：

    ┌─────────────────────────────────────────────────────────┐
    │ 1. 数据稀缺: 标注成本高，需要专业知识                      │
    │ 2. 类别不平衡: 病变区域通常很小                           │
    │ 3. 边界模糊: 需要精确的边界分割                           │
    │ 4. 多模态: CT, MRI, X-ray 等不同模态                     │
    └─────────────────────────────────────────────────────────┘

常见应用：
    - 器官分割: 肝脏、肾脏、脾脏等
    - 病变检测: 肿瘤、病灶分割
    - 细胞分割: 显微镜图像中的细胞
    - 视网膜分割: 血管、病变区域

数据增强技术：

    医学图像特别需要数据增强来缓解数据不足问题:
    - 随机旋转、翻转
    - 弹性变形 (Elastic Deformation)
    - 随机裁剪和缩放
    - 亮度和对比度调整
    - Mixup 和 CutMix
    """)

    print("示例: 医学图像数据增强\n")

    import torchvision.transforms as T

    # 定义增强
    train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
        ]
    )

    print("训练时数据增强:")
    print("  - 水平/垂直翻转")
    print("  - 随机旋转 (±30°)")
    print("  - 随机平移 (10%)")
    print("  - 亮度/对比度调整")


# ==================== 第六部分：训练技巧 ====================


def training_tips():
    """训练技巧"""
    print("\n" + "=" * 60)
    print("第六部分：U-Net 训练技巧")
    print("=" * 60)

    print("""
U-Net 训练技巧：

    1. 损失函数选择
       - 二分类: BCEWithLogitsLoss + Dice Loss
       - 多分类: CrossEntropyLoss + Dice Loss

    2. 学习率策略
       - Warmup + CosineAnnealing
       - ReduceLROnPlateau

    3. 数据增强
       - 在线增强
       - 弹性变形对医学图像特别有效

    4. 深度监督 (Deep Supervision)
       - 在多个尺度上计算损失
       - 加速收敛，提升性能

    5. 后处理
       - 形态学操作 (开运算、闭运算)
       - 连通区域分析
       - CRF (条件随机场)
    """)

    print("示例: 组合损失函数\n")

    class CombinedLoss(nn.Module):
        """BCE + Dice 组合损失"""

        def __init__(self, bce_weight=0.5, dice_weight=0.5):
            super().__init__()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.bce = nn.BCEWithLogitsLoss()

        def dice_loss(self, pred, target, smooth=1e-5):
            pred = torch.sigmoid(pred)
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)

            intersection = (pred_flat * target_flat).sum()
            return 1 - (2.0 * intersection + smooth) / (
                pred_flat.sum() + target_flat.sum() + smooth
            )

        def forward(self, pred, target):
            bce = self.bce(pred, target)
            dice = self.dice_loss(pred, target)
            return self.bce_weight * bce + self.dice_weight * dice

    criterion = CombinedLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = criterion(pred, target)
    print(f"组合损失: {loss.item():.4f}")


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：实现 U-Net
    任务: 从零实现完整的 U-Net
    测试: 在随机数据上验证前向传播

练习 2：医学图像分割
    任务: 在 Kaggle 的医学图像数据集上训练 U-Net
    推荐: Carvana 汽车分割或肺部 CT 分割

练习 3：实现 Attention U-Net
    任务: 在标准 U-Net 上添加注意力门控
    对比: 有无注意力机制的效果差异

练习 4：深度监督
    任务: 实现深度监督版本的 U-Net
    在多个尺度输出预测并计算损失

练习 5：后处理优化
    任务: 实现形态学后处理
    包含: 开闭运算、连通区域分析

思考题 1：U-Net 为什么特别适合医学图像？
    考虑数据量、边界精度等因素

思考题 2：Skip Connection 拼接 vs 加法?
    各有什么优缺点？

思考题 3：如何处理 3D 医学图像 (如 CT)?
    3D U-Net 的设计要点
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    skip_connection()
    unet_implementation()
    unet_variants()
    medical_segmentation()
    training_tips()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 06-instance-segmentation.py: 实例分割

关键要点回顾：
    ✓ U-Net 是对称的编码器-解码器架构
    ✓ Skip Connection 保留空间细节信息
    ✓ 拼接操作融合多尺度特征
    ✓ 组合损失 (BCE + Dice) 效果更好
    ✓ U-Net 特别适合医学图像分割
    """)


if __name__ == "__main__":
    main()
