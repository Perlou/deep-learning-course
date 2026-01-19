"""
深度卷积生成对抗网络 (DCGAN)
==============================

学习目标：
    1. 理解 DCGAN 的架构设计原则
    2. 掌握卷积和转置卷积在生成模型中的应用
    3. 从零实现 DCGAN
    4. 了解生成高质量图像的技巧

核心概念：
    - 转置卷积 (Transposed Convolution): 上采样操作
    - 批归一化 (BatchNorm): 稳定训练
    - 步长卷积 (Strided Convolution): 替代池化层
    - 渐进式生成: 从小分辨率到高分辨率

前置知识：
    - Phase 5: 卷积神经网络
    - 01-gan-basics.py: GAN 基础
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# ==================== 第一部分：DCGAN 架构介绍 ====================


def introduction():
    """
    DCGAN 架构介绍

    DCGAN (Deep Convolutional GAN) 是首个成功使用卷积网络的 GAN。
    """
    print("=" * 60)
    print("第一部分：DCGAN 架构介绍")
    print("=" * 60)

    print("""
DCGAN 的关键创新：

    使用卷积神经网络替代全连接网络，实现更高质量的图像生成。

DCGAN 架构设计原则（来自原论文）：

    1. 使用步长卷积 (strided convolutions) 替代池化层
       - 生成器: 使用转置卷积进行上采样
       - 判别器: 使用步长卷积进行下采样

    2. 在生成器和判别器中使用批归一化
       - 帮助稳定训练
       - 但在生成器输出层和判别器输入层不使用

    3. 移除全连接隐藏层
       - 使用全局平均池化或展平连接

    4. 生成器使用 ReLU 激活（输出层用 Tanh）
       判别器使用 LeakyReLU 激活

生成器架构（从噪声到图像）：
    ┌─────────────────────────────────────────────────────────┐
    │  z (100,)                                               │
    │      ↓                                                  │
    │  线性变换 + Reshape → (512, 4, 4)                        │
    │      ↓                                                  │
    │  转置卷积 → (256, 8, 8)   [上采样 2x]                    │
    │      ↓                                                  │
    │  转置卷积 → (128, 16, 16) [上采样 2x]                    │
    │      ↓                                                  │
    │  转置卷积 → (64, 32, 32)  [上采样 2x]                    │
    │      ↓                                                  │
    │  转置卷积 → (3, 64, 64)   [输出 RGB 图像]                │
    └─────────────────────────────────────────────────────────┘

判别器架构（从图像到概率）：
    ┌─────────────────────────────────────────────────────────┐
    │  图像 (3, 64, 64)                                       │
    │      ↓                                                  │
    │  卷积 → (64, 32, 32)   [下采样 2x]                       │
    │      ↓                                                  │
    │  卷积 → (128, 16, 16) [下采样 2x]                        │
    │      ↓                                                  │
    │  卷积 → (256, 8, 8)   [下采样 2x]                        │
    │      ↓                                                  │
    │  卷积 → (512, 4, 4)   [下采样 2x]                        │
    │      ↓                                                  │
    │  展平 + 线性 → 1      [真假概率]                         │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：转置卷积详解 ====================


def transposed_conv_demo():
    """转置卷积演示"""
    print("\n" + "=" * 60)
    print("第二部分：转置卷积详解")
    print("=" * 60)

    print("""
转置卷积 (Transposed Convolution) / 反卷积 (Deconvolution)：

    用于上采样，增加特征图尺寸。

    计算公式:
    output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding

    常用配置:
    - kernel=4, stride=2, padding=1 → 尺寸翻倍
    """)

    # 演示转置卷积
    print("\n示例 1: 转置卷积上采样\n")

    # 创建转置卷积层
    trans_conv = nn.ConvTranspose2d(
        in_channels=256,
        out_channels=128,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False,
    )

    # 输入特征图
    x = torch.randn(1, 256, 8, 8)
    print(f"输入形状: {x.shape}")

    # 上采样
    y = trans_conv(x)
    print(f"输出形状: {y.shape}")
    print("尺寸从 8x8 → 16x16 (翻倍)")

    # 对比普通卷积
    print("\n对比: 普通卷积 vs 转置卷积\n")

    print("普通卷积 (下采样):")
    conv = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
    x_down = torch.randn(1, 128, 16, 16)
    y_down = conv(x_down)
    print(f"  {x_down.shape} → {y_down.shape}")

    print("\n转置卷积 (上采样):")
    print(f"  {x.shape} → {y.shape}")


# ==================== 第三部分：DCGAN 实现 ====================


def weights_init(m):
    """
    权重初始化

    DCGAN 论文建议:
    - 卷积层: 均值 0，标准差 0.02 的正态分布
    - BatchNorm: γ=1, β=0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANGenerator(nn.Module):
    """
    DCGAN 生成器

    将噪声向量转换为图像
    输入: (batch, latent_dim, 1, 1)
    输出: (batch, channels, 64, 64)
    """

    def __init__(self, latent_dim=100, channels=1, feature_maps=64):
        super(DCGANGenerator, self).__init__()

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

    def forward(self, x):
        return self.main(x)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN 判别器

    将图像转换为真假概率
    输入: (batch, channels, 64, 64)
    输出: (batch, 1, 1, 1)
    """

    def __init__(self, channels=1, feature_maps=64):
        super(DCGANDiscriminator, self).__init__()

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

    def forward(self, x):
        return self.main(x)


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第三部分：DCGAN 实现")
    print("=" * 60)

    print("\n示例 1: 创建 DCGAN 网络\n")

    # 参数设置
    latent_dim = 100
    channels = 1  # MNIST 灰度图像
    feature_maps = 64

    # 创建网络
    generator = DCGANGenerator(latent_dim, channels, feature_maps)
    discriminator = DCGANDiscriminator(channels, feature_maps)

    # 初始化权重
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print(f"生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")

    print("\n示例 2: 验证网络维度\n")

    # 测试生成器
    z = torch.randn(4, latent_dim, 1, 1)
    print(f"噪声输入形状: {z.shape}")

    fake_images = generator(z)
    print(f"生成图像形状: {fake_images.shape}")

    # 测试判别器
    scores = discriminator(fake_images)
    print(f"判别器输出形状: {scores.shape}")
    print(f"判别器输出值: {scores.view(-1).detach().numpy()}")

    print("\n生成器各层输出形状:")
    print_layer_shapes(generator, z)


def print_layer_shapes(model, x):
    """打印模型各层的输出形状"""
    for i, layer in enumerate(model.main):
        x = layer(x)
        layer_name = layer.__class__.__name__
        print(f"  {i:2d}. {layer_name:25s} → {list(x.shape)}")


# ==================== 第四部分：DCGAN 训练 ====================


def train_dcgan(
    generator,
    discriminator,
    dataloader,
    num_epochs=20,
    latent_dim=100,
    lr=0.0002,
    device="cpu",
):
    """
    DCGAN 训练函数

    注意：为了适配 MNIST (28x28)，需要将图像 resize 到 64x64
    """
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 移动到设备
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 记录
    g_losses = []
    d_losses = []
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    print(f"\n开始训练，设备: {device}")

    for epoch in range(num_epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0

        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # 标签
            real_labels = torch.ones(batch_size, 1, 1, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=device)

            # ===== 训练判别器 =====
            optimizer_D.zero_grad()

            # 真实图像
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, real_labels)

            # 假图像
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ===== 训练生成器 =====
            optimizer_G.zero_grad()

            output = discriminator(fake_images)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        g_losses.append(g_loss_epoch / len(dataloader))
        d_losses.append(d_loss_epoch / len(dataloader))

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"D_loss: {d_losses[-1]:.4f} "
                f"G_loss: {g_losses[-1]:.4f}"
            )

    return g_losses, d_losses, fixed_noise


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第四部分：DCGAN 训练示例")
    print("=" * 60)

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 准备数据（resize 到 64x64）
    print("\n加载并预处理 MNIST 数据集...")

    transform = transforms.Compose(
        [
            transforms.Resize(64),  # DCGAN 需要 64x64
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )

    try:
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
        print(f"数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"无法下载 MNIST: {e}")
        print("使用随机数据进行演示...")
        fake_data = torch.randn(1000, 1, 64, 64)
        fake_labels = torch.zeros(1000, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 创建模型
    latent_dim = 100
    generator = DCGANGenerator(latent_dim, channels=1, feature_maps=64)
    discriminator = DCGANDiscriminator(channels=1, feature_maps=64)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print("\n开始训练 DCGAN (5 epochs 用于演示)...")

    g_losses, d_losses, fixed_noise = train_dcgan(
        generator,
        discriminator,
        dataloader,
        num_epochs=5,
        latent_dim=latent_dim,
        lr=0.0002,
        device=device,
    )

    # 可视化结果
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("DCGAN 训练损失")
    plt.grid(True)

    # 生成的图像
    plt.subplot(1, 2, 2)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise[:16]).cpu()

    # 创建网格
    grid = np.zeros((4 * 64, 4 * 64))
    for i in range(4):
        for j in range(4):
            img = fake_images[i * 4 + j, 0].numpy()
            img = (img + 1) / 2  # 反归一化
            grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64] = img

    plt.imshow(grid, cmap="gray")
    plt.title("DCGAN 生成的图像")
    plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "dcgan_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 结果已保存到: {output_path}")


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：修改输出分辨率
    任务: 修改 DCGAN 生成 32x32 或 128x128 的图像
    提示: 调整转置卷积层的数量

练习 2：训练 CIFAR-10
    任务: 在 CIFAR-10 数据集上训练 DCGAN
    注意:
    - 修改 channels=3 (RGB)
    - CIFAR-10 图像是 32x32

练习 3：特征插值
    任务: 实现潜在空间插值
    - 选择两个随机噪声 z1 和 z2
    - 在它们之间进行线性插值
    - 观察生成图像的平滑过渡

练习 4：分析特征空间
    任务: 探索 DCGAN 学到的语义特征
    - 尝试向量运算: z_glasses - z_no_glasses + z_woman
    - 观察是否能生成"戴眼镜的女性"

思考题 1：为什么 DCGAN 使用 LeakyReLU？
    提示: 考虑判别器中的梯度流动

思考题 2：BatchNorm 的作用
    为什么在生成器输出层和判别器输入层不使用 BatchNorm？

思考题 3：步长卷积 vs 池化
    为什么用步长卷积替代池化层？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    transposed_conv_demo()
    basic_implementation()
    advanced_examples()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 03-wgan.py: Wasserstein GAN

关键要点回顾：
    ✓ DCGAN 使用卷积神经网络实现 GAN
    ✓ 转置卷积用于上采样
    ✓ 步长卷积替代池化层
    ✓ BatchNorm 帮助稳定训练
    ✓ 正确的权重初始化很重要
    """)


if __name__ == "__main__":
    main()
