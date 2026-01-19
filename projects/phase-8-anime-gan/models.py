"""
DCGAN 模型定义

包含生成器和判别器网络
"""

import torch
import torch.nn as nn


def weights_init(m):
    """
    DCGAN 权重初始化

    根据 DCGAN 论文的建议:
    - 卷积层: 均值 0，标准差 0.02 的正态分布
    - BatchNorm: γ=1, β=0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN 生成器

    将噪声向量转换为图像
    输入: (batch, latent_dim, 1, 1)
    输出: (batch, channels, 64, 64)
    """

    def __init__(self, latent_dim=100, channels=3, feature_maps=64):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            # 输入: (batch, latent_dim, 1, 1)
            # 第一层: 1x1 → 4x4
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # (batch, 512, 4, 4)
            # 第二层: 4x4 → 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # (batch, 256, 8, 8)
            # 第三层: 8x8 → 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # (batch, 128, 16, 16)
            # 第四层: 16x16 → 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # (batch, 64, 32, 32)
            # 输出层: 32x32 → 64x64
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # (batch, channels, 64, 64)
        )

        # 初始化权重
        self.apply(weights_init)

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    DCGAN 判别器

    将图像转换为真假概率
    输入: (batch, channels, 64, 64)
    输出: (batch, 1, 1, 1)
    """

    def __init__(self, channels=3, feature_maps=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 输入: (batch, channels, 64, 64)
            # 第一层: 64x64 → 32x32
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, 64, 32, 32)
            # 第二层: 32x32 → 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, 128, 16, 16)
            # 第三层: 16x16 → 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, 256, 8, 8)
            # 第四层: 8x8 → 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch, 512, 4, 4)
            # 输出层: 4x4 → 1x1
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # (batch, 1, 1, 1)
        )

        # 初始化权重
        self.apply(weights_init)

    def forward(self, img):
        return self.main(img)


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("DCGAN 模型测试")
    print("=" * 50)

    latent_dim = 100
    batch_size = 4

    # 创建模型
    G = Generator(latent_dim=latent_dim)
    D = Discriminator()

    print(f"\n生成器参数量: {sum(p.numel() for p in G.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in D.parameters()):,}")

    # 测试生成器
    z = torch.randn(batch_size, latent_dim, 1, 1)
    fake_images = G(z)
    print(f"\n噪声输入形状: {z.shape}")
    print(f"生成图像形状: {fake_images.shape}")

    # 测试判别器
    scores = D(fake_images)
    print(f"判别器输出形状: {scores.shape}")
    print(f"判别器输出值: {scores.view(-1).detach().numpy()}")

    print("\n✓ 模型测试通过!")
