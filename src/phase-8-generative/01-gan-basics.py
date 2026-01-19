"""
生成对抗网络基础 (GAN Basics)
==============================

学习目标：
    1. 理解 GAN 的核心思想：生成器与判别器的对抗博弈
    2. 掌握 GAN 的目标函数和训练过程
    3. 从零实现一个简单的 GAN
    4. 了解 GAN 训练的常见问题和技巧

核心概念：
    - 生成器 (Generator): 从噪声生成假样本
    - 判别器 (Discriminator): 区分真假样本
    - 对抗训练: 两个网络相互博弈
    - 极小极大博弈: min_G max_D 的优化目标

前置知识：
    - Phase 3: PyTorch 基础
    - Phase 4: 神经网络基础 (MLP)
    - Phase 5: 卷积神经网络（可选）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# ==================== 第一部分：GAN 原理介绍 ====================


def introduction():
    """
    GAN 原理介绍

    GAN 的核心思想是通过两个神经网络的对抗训练来生成逼真数据。
    """
    print("=" * 60)
    print("第一部分：GAN 原理介绍")
    print("=" * 60)

    print("""
GAN 的直觉理解：

想象一个造假者和一个鉴定专家：
- 造假者 (Generator): 试图伪造名画
- 鉴定专家 (Discriminator): 试图辨别真假

┌────────────────────────────────────────────────────────────┐
│                        GAN 架构                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   随机噪声 z ──→ [Generator] ──→ 假图像                      │
│         │                           │                      │
│         │                           ↓                      │
│         │                     [Discriminator] ──→ 真/假?   │
│         │                           ↑                      │
│         │          真实图像 ─────────┘                      │
│         │                                                  │
│         └─────── 通过反馈不断改进 ─────────────────         │
│                                                            │
└────────────────────────────────────────────────────────────┘

目标函数（极小极大博弈）：
    min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

    - 判别器 D 想要最大化这个目标（正确区分真假）
    - 生成器 G 想要最小化这个目标（骗过判别器）

训练过程：
    1. 固定 G，训练 D 几步
    2. 固定 D，训练 G 几步
    3. 重复直到收敛（Nash 均衡）
    """)


# ==================== 第二部分：简单 GAN 实现 ====================


class Generator(nn.Module):
    """
    生成器网络

    输入: 随机噪声向量 z (latent_dim,)
    输出: 假图像 (img_dim,)
    """

    def __init__(self, latent_dim=100, img_dim=784):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # 第一层：升维
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            # 第二层
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            # 第三层
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            # 输出层：映射到图像维度
            nn.Linear(1024, img_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """
    判别器网络

    输入: 图像 (img_dim,)
    输出: 真假概率 (1,)
    """

    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 第一层：降维
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            # 第二层
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            # 输出层
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 输出概率
        )

    def forward(self, img):
        return self.model(img)


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第二部分：基础实现")
    print("=" * 60)

    print("\n示例 1: 创建 GAN 网络\n")

    # 设置参数
    latent_dim = 100  # 噪声向量维度
    img_dim = 28 * 28  # MNIST 图像维度

    # 创建网络
    generator = Generator(latent_dim, img_dim)
    discriminator = Discriminator(img_dim)

    print(f"生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")

    print("\n示例 2: 生成假图像\n")

    # 生成随机噪声
    batch_size = 16
    z = torch.randn(batch_size, latent_dim)
    print(f"噪声向量形状: {z.shape}")

    # 生成假图像
    fake_images = generator(z)
    print(f"假图像形状: {fake_images.shape}")

    # 判别器判断
    fake_scores = discriminator(fake_images)
    print(f"判别器输出形状: {fake_scores.shape}")
    print(f"判别器输出（未训练）: {fake_scores[:5].detach().numpy().flatten()}")

    print("\n示例 3: 损失函数计算\n")

    # 创建标签
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # 创建真实数据（随机模拟）
    real_images = torch.randn(batch_size, img_dim)

    # 计算判别器损失
    criterion = nn.BCELoss()

    # 真实数据损失
    real_scores = discriminator(real_images)
    d_loss_real = criterion(real_scores, real_labels)

    # 假数据损失
    fake_scores = discriminator(fake_images.detach())
    d_loss_fake = criterion(fake_scores, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    print(f"判别器损失: {d_loss.item():.4f}")
    print(f"  - 真实数据损失: {d_loss_real.item():.4f}")
    print(f"  - 假数据损失: {d_loss_fake.item():.4f}")

    # 计算生成器损失
    fake_scores = discriminator(fake_images)
    g_loss = criterion(fake_scores, real_labels)  # 生成器希望骗过判别器
    print(f"生成器损失: {g_loss.item():.4f}")


# ==================== 第三部分：完整 GAN 训练 ====================


def train_gan(
    generator,
    discriminator,
    dataloader,
    num_epochs=50,
    latent_dim=100,
    lr=0.0002,
    device="cpu",
):
    """
    完整的 GAN 训练函数

    Args:
        generator: 生成器网络
        discriminator: 判别器网络
        dataloader: 数据加载器
        num_epochs: 训练轮数
        latent_dim: 噪声维度
        lr: 学习率
        device: 设备
    """
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 移动到设备
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 记录损失
    g_losses = []
    d_losses = []

    # 固定噪声用于可视化
    fixed_noise = torch.randn(64, latent_dim, device=device)

    print(f"\n开始训练，设备: {device}")
    print("-" * 40)

    for epoch in range(num_epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0

        for batch_idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(device)

            # 创建标签
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ==================== 训练判别器 ====================
            optimizer_D.zero_grad()

            # 真实数据损失
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)

            # 生成假数据
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z)

            # 假数据损失
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            # 总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ==================== 训练生成器 ====================
            optimizer_G.zero_grad()

            # 生成假数据并计算损失
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images)

            # 生成器损失（希望判别器认为是真的）
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        # 记录平均损失
        g_losses.append(g_loss_epoch / len(dataloader))
        d_losses.append(d_loss_epoch / len(dataloader))

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"D_loss: {d_losses[-1]:.4f} "
                f"G_loss: {g_losses[-1]:.4f}"
            )

    return g_losses, d_losses, fixed_noise


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第三部分：完整训练示例")
    print("=" * 60)

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 准备数据
    print("\n加载 MNIST 数据集...")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
        ]
    )

    # 创建数据目录
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )
    os.makedirs(data_dir, exist_ok=True)

    try:
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
        print(f"数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"无法下载 MNIST: {e}")
        print("使用随机数据进行演示...")
        # 创建假数据
        fake_data = torch.randn(1000, 1, 28, 28)
        fake_labels = torch.zeros(1000, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 创建模型
    latent_dim = 100
    img_dim = 28 * 28

    generator = Generator(latent_dim, img_dim)
    discriminator = Discriminator(img_dim)

    print("\n开始训练 GAN (10 epochs 用于演示)...")
    print("注意: 完整训练需要更多 epochs (如 100-200)")

    g_losses, d_losses, fixed_noise = train_gan(
        generator,
        discriminator,
        dataloader,
        num_epochs=10,  # 演示用，实际需要更多
        latent_dim=latent_dim,
        lr=0.0002,
        device=device,
    )

    # 可视化训练过程
    plt.figure(figsize=(12, 4))

    # 1. 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN 训练损失曲线")
    plt.grid(True)

    # 2. 生成的图像
    plt.subplot(1, 2, 2)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise[:16]).view(-1, 28, 28).cpu()

    # 创建图像网格
    grid_img = np.zeros((4 * 28, 4 * 28))
    for i in range(4):
        for j in range(4):
            grid_img[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = fake_images[
                i * 4 + j
            ].numpy()

    plt.imshow(grid_img, cmap="gray")
    plt.title("生成的图像样本")
    plt.axis("off")

    plt.tight_layout()

    # 保存图像
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "gan_training_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 训练结果已保存到: {output_path}")


# ==================== 第四部分：GAN 训练技巧与常见问题 ====================


def training_tips():
    """GAN 训练技巧"""
    print("\n" + "=" * 60)
    print("第四部分：GAN 训练技巧与常见问题")
    print("=" * 60)

    print("""
常见问题及解决方案：

1. 模式崩塌 (Mode Collapse)
   ┌─────────────────────────────────────────────────────────┐
   │ 问题: 生成器只生成少数几种样本                            │
   │ 原因: 生成器找到了"骗过"判别器的捷径                      │
   │ 解决:                                                    │
   │   - 使用 Mini-batch discrimination                       │
   │   - 尝试 WGAN 或 WGAN-GP                                 │
   │   - 增加噪声维度                                          │
   └─────────────────────────────────────────────────────────┘

2. 训练不稳定
   ┌─────────────────────────────────────────────────────────┐
   │ 问题: 损失震荡，生成质量不提升                            │
   │ 解决:                                                    │
   │   - 使用 LeakyReLU 而非 ReLU                             │
   │   - 使用 BatchNorm                                       │
   │   - 使用 Adam 优化器，β1=0.5                             │
   │   - 学习率不要太大 (推荐 0.0002)                          │
   └─────────────────────────────────────────────────────────┘

3. 判别器太强/太弱
   ┌─────────────────────────────────────────────────────────┐
   │ 太强: D_loss ≈ 0, 生成器学不到东西                       │
   │ 太弱: D_loss 很大，生成器没有好的反馈                     │
   │ 解决:                                                    │
   │   - 平衡网络容量                                          │
   │   - 标签平滑: 真实标签用 0.9 而非 1.0                     │
   │   - 调整 D 和 G 的训练步数比例                            │
   └─────────────────────────────────────────────────────────┘

训练技巧汇总：

    1. 网络架构:
       - 使用 LeakyReLU (斜率 0.2)
       - 生成器使用 BatchNorm
       - 判别器使用 Dropout

    2. 优化器:
       - Adam, lr=0.0002, β1=0.5, β2=0.999
       - 或 RMSprop

    3. 标签处理:
       - 标签平滑
       - 翻转标签（随机）

    4. 输入处理:
       - 归一化到 [-1, 1]
       - 使用 Tanh 作为生成器输出激活

    5. 监控指标:
       - 查看生成样本
       - 检查 D(x) 和 D(G(z)) 的值
       - 使用 FID 或 Inception Score
    """)


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：修改网络结构
    尝试修改生成器和判别器的网络结构:
    - 增加/减少层数
    - 调整隐藏层维度
    - 观察对生成质量的影响

练习 2：实现条件 GAN (cGAN)
    在 GAN 中加入类别条件:
    - 生成器输入: z + class_label
    - 判别器输入: image + class_label
    任务: 能够生成指定数字的 MNIST 图像

练习 3：实现标签平滑
    修改训练代码，使用标签平滑:
    - 真实标签: 0.9 (而非 1.0)
    - 假标签: 0.1 (而非 0.0)
    观察对训练稳定性的影响

练习 4：可视化训练过程
    每 N 个 epoch 保存生成的图像，创建训练动画:
    - 观察生成质量如何随时间变化
    - 何时开始出现有意义的结构

思考题 1：为什么 GAN 训练困难？
    提示：考虑以下几点:
    - 极小极大博弈的收敛性
    - 梯度消失问题
    - 模式崩塌

思考题 2：生成器和判别器的平衡
    如果判别器太强，生成器会怎样？
    如果生成器太强，又会怎样？

思考题 3：GAN vs VAE
    两种生成模型的主要区别是什么？
    各自的优缺点是什么？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    basic_implementation()
    advanced_examples()
    training_tips()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 02-dcgan.py: 深度卷积 GAN
    - 03-wgan.py: Wasserstein GAN

关键要点回顾：
    ✓ GAN 由生成器和判别器组成
    ✓ 通过对抗训练学习生成逼真数据
    ✓ 目标函数是极小极大博弈
    ✓ 训练技巧对 GAN 至关重要
    ✓ 模式崩塌是常见问题
    """)


if __name__ == "__main__":
    main()
