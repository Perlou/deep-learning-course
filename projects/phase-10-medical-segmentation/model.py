"""
U-Net 模型定义
==============

实现标准 U-Net 和 Attention U-Net，用于医学图像分割。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块: (Conv → BN → ReLU) × 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

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
    """
    标准 U-Net 实现

    Args:
        n_channels: 输入通道数 (1=灰度, 3=RGB)
        n_classes: 输出类别数 (1=二分类, N=多分类)
        bilinear: 是否使用双线性上采样
        features: 各层特征数列表
    """

    def __init__(self, n_channels=1, n_classes=1, bilinear=True, features=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if features is None:
            features = [64, 128, 256, 512]

        # 编码器
        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)

        # 解码器
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)

        # 输出层
        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        # 编码
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码 (带 Skip Connection)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出
        logits = self.outc(x)
        return logits


class AttentionGate(nn.Module):
    """注意力门控模块"""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net 实现"""

    def __init__(self, n_channels=1, n_classes=1, features=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        if features is None:
            features = [64, 128, 256, 512]

        # 编码器
        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3] * 2)

        # 注意力门控
        self.att4 = AttentionGate(features[3] * 2, features[3], features[3])
        self.att3 = AttentionGate(features[3], features[2], features[2])
        self.att2 = AttentionGate(features[2], features[1], features[1])
        self.att1 = AttentionGate(features[1], features[0], features[0])

        # 解码器
        self.up4 = Up(features[3] * 2, features[3], bilinear=False)
        self.up3 = Up(features[3], features[2], bilinear=False)
        self.up2 = Up(features[2], features[1], bilinear=False)
        self.up1 = Up(features[1], features[0], bilinear=False)

        # 输出层
        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        # 编码
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码 (带注意力门控)
        x4_att = self.att4(x5, x4)
        d4 = self.up4(x5, x4_att)

        x3_att = self.att3(d4, x3)
        d3 = self.up3(d4, x3_att)

        x2_att = self.att2(d3, x2)
        d2 = self.up2(d3, x2_att)

        x1_att = self.att1(d2, x1)
        d1 = self.up1(d2, x1_att)

        # 输出
        logits = self.outc(d1)
        return logits


# ===========================================
# 损失函数
# ===========================================


class DiceLoss(nn.Module):
    """Dice 损失"""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class CombinedLoss(nn.Module):
    """BCE + Dice 组合损失"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ===========================================
# 工厂函数
# ===========================================


def create_model(model_name="unet", **kwargs):
    """
    创建模型

    Args:
        model_name: "unet" 或 "attention_unet"
        **kwargs: 模型参数
    """
    if model_name == "unet":
        return UNet(**kwargs)
    elif model_name == "attention_unet":
        return AttentionUNet(**kwargs)
    else:
        raise ValueError(f"未知模型: {model_name}")


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================
# 测试
# ===========================================

if __name__ == "__main__":
    # 测试 U-Net
    print("=" * 50)
    print("测试 U-Net")
    print("=" * 50)

    model = UNet(n_channels=1, n_classes=1)
    x = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        y = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"参数量: {count_parameters(model):,}")

    # 测试 Attention U-Net
    print("\n" + "=" * 50)
    print("测试 Attention U-Net")
    print("=" * 50)

    att_model = AttentionUNet(n_channels=1, n_classes=1)

    with torch.no_grad():
        y_att = att_model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_att.shape}")
    print(f"参数量: {count_parameters(att_model):,}")

    # 测试损失函数
    print("\n" + "=" * 50)
    print("测试损失函数")
    print("=" * 50)

    criterion = CombinedLoss()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    loss = criterion(y, target)
    print(f"组合损失: {loss.item():.4f}")
