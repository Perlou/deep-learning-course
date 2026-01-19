"""
自编码器 (Autoencoder)
======================

学习目标：
    1. 理解自编码器的原理和结构
    2. 掌握编码器-解码器架构
    3. 了解不同类型的自编码器
    4. 实现基础自编码器和卷积自编码器

核心概念：
    - 编码器 (Encoder): 将输入压缩到低维隐空间
    - 解码器 (Decoder): 从隐空间重构输入
    - 瓶颈层 (Bottleneck): 隐空间的低维表示
    - 重构损失: 衡量重构质量

前置知识：
    - Phase 4: 神经网络基础
    - Phase 5: 卷积神经网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# ==================== 第一部分：自编码器原理 ====================


def introduction():
    """
    自编码器原理介绍
    """
    print("=" * 60)
    print("第一部分：自编码器原理")
    print("=" * 60)

    print("""
什么是自编码器？

    自编码器是一种无监督学习模型，
    目标是学习数据的压缩表示（编码），并能从中重构原始数据。

基本架构：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   输入 x ──→ [Encoder] ──→ 隐向量 z ──→ [Decoder] ──→ x'│
    │   (784)      压缩         (32)         解压      (784) │
    │                            ↓                           │
    │                        瓶颈层                           │
    │                    (低维表示)                           │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    目标: 最小化 ||x - x'||² (重构误差)

为什么需要自编码器？

    1. 数据压缩
       - 将高维数据压缩到低维空间
       - 例如: 784维图像 → 32维向量

    2. 特征学习
       - 隐空间学习数据的有意义表示
       - 可用于下游任务

    3. 降噪
       - 去噪自编码器可以恢复损坏的数据
       - 学习数据的本质结构

    4. 生成模型基础
       - VAE 基于自编码器
       - 学习生成新样本

自编码器类型：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 基础自编码器 (Vanilla AE)                            │
    │    - 简单的编码器-解码器结构                              │
    │                                                         │
    │ 2. 稀疏自编码器 (Sparse AE)                             │
    │    - 添加稀疏性约束，学习稀疏特征                          │
    │                                                         │
    │ 3. 去噪自编码器 (Denoising AE)                          │
    │    - 输入加噪声，训练恢复原始数据                          │
    │                                                         │
    │ 4. 卷积自编码器 (Convolutional AE)                      │
    │    - 使用卷积层，适合图像数据                              │
    │                                                         │
    │ 5. 变分自编码器 (VAE) ← 下一课                          │
    │    - 学习隐空间的概率分布                                 │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：基础自编码器实现 ====================


class SimpleAutoencoder(nn.Module):
    """
    简单的全连接自编码器

    用于 MNIST 数据集
    """

    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super(SimpleAutoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def encode(self, x):
        """编码: 输入 → 隐向量"""
        return self.encoder(x)

    def decode(self, z):
        """解码: 隐向量 → 重构"""
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第二部分：基础自编码器实现")
    print("=" * 60)

    print("\n示例 1: 创建自编码器\n")

    # 参数
    input_dim = 784  # 28x28 MNIST
    hidden_dim = 256
    latent_dim = 32

    model = SimpleAutoencoder(input_dim, hidden_dim, latent_dim)

    print(f"输入维度: {input_dim}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"隐空间维度: {latent_dim}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n示例 2: 前向传播\n")

    # 测试数据
    x = torch.randn(4, input_dim)
    print(f"输入形状: {x.shape}")

    x_recon, z = model(x)
    print(f"隐向量形状: {z.shape}")
    print(f"重构输出形状: {x_recon.shape}")

    # 计算重构损失
    recon_loss = nn.MSELoss()(x_recon, x)
    print(f"\n重构损失 (MSE): {recon_loss.item():.4f}")

    print("\n示例 3: 编码器和解码器分离使用\n")

    # 编码
    z = model.encode(x)
    print(f"编码结果: {z.shape}")

    # 解码
    x_recon = model.decode(z)
    print(f"解码结果: {x_recon.shape}")


# ==================== 第三部分：卷积自编码器 ====================


class ConvAutoencoder(nn.Module):
    """
    卷积自编码器

    适用于图像数据，保留空间结构
    """

    def __init__(self, channels=1, latent_dim=64):
        super(ConvAutoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 14x14 → 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 7x7 → 4x4 (大约)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # 展平后的维度: 128 * 4 * 4 = 2048
        self.fc_encode = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

        # 解码器
        self.decoder = nn.Sequential(
            # 4x4 → 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 7x7 → 14x14
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 14x14 → 28x28
            nn.ConvTranspose2d(
                32, channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """编码: 图像 → 隐向量"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # 展平
        z = self.fc_encode(h)
        return z

    def decode(self, z):
        """解码: 隐向量 → 图像"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 4, 4)  # 恢复形状
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def conv_autoencoder_demo():
    """卷积自编码器演示"""
    print("\n" + "=" * 60)
    print("第三部分：卷积自编码器")
    print("=" * 60)

    print("\n示例: 卷积自编码器结构\n")

    model = ConvAutoencoder(channels=1, latent_dim=64)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试
    x = torch.randn(4, 1, 28, 28)
    print(f"输入形状: {x.shape}")

    x_recon, z = model(x)
    print(f"隐向量形状: {z.shape}")
    print(f"重构输出形状: {x_recon.shape}")


# ==================== 第四部分：自编码器训练 ====================


def train_autoencoder(model, dataloader, num_epochs=10, lr=0.001, device="cpu"):
    """
    训练自编码器

    Args:
        model: 自编码器模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    print(f"\n开始训练，设备: {device}")

    for epoch in range(num_epochs):
        epoch_loss = 0

        for images, _ in dataloader:
            images = images.to(device)

            # 如果是全连接模型，需要展平
            if isinstance(model, SimpleAutoencoder):
                images = images.view(images.size(0), -1)

            # 前向传播
            recon, z = model(images)

            # 计算损失
            loss = criterion(recon, images)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.6f}")

    return losses


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第四部分：自编码器训练示例")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 准备数据
    transform = transforms.Compose([transforms.ToTensor()])

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )

    try:
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    except Exception:
        print("使用随机数据进行演示...")
        fake_data = torch.rand(1000, 1, 28, 28)
        fake_labels = torch.zeros(1000, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        test_loader = dataloader

    # 创建并训练模型
    model = SimpleAutoencoder(784, 256, 32)

    print("\n训练简单自编码器 (5 epochs)...")
    losses = train_autoencoder(model, dataloader, num_epochs=5, lr=0.001, device=device)

    # 可视化结果
    plt.figure(figsize=(12, 4))

    # 1. 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练损失")
    plt.grid(True)

    # 2. 重构对比
    plt.subplot(1, 3, 2)
    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_flat = test_images.view(test_images.size(0), -1).to(device)
        recon, _ = model(test_flat)
        recon = recon.cpu()

    # 原始 vs 重构
    comparison = torch.zeros(2 * 4, 28, 28)
    for i in range(4):
        comparison[i * 2] = test_images[i, 0]
        comparison[i * 2 + 1] = recon[i].view(28, 28)

    grid = np.zeros((2 * 28, 4 * 28))
    for i in range(4):
        grid[:28, i * 28 : (i + 1) * 28] = comparison[i * 2].numpy()
        grid[28:, i * 28 : (i + 1) * 28] = comparison[i * 2 + 1].numpy()

    plt.imshow(grid, cmap="gray")
    plt.title("原始 (上) vs 重构 (下)")
    plt.axis("off")

    # 3. 隐空间可视化
    plt.subplot(1, 3, 3)
    with torch.no_grad():
        # 获取更多数据的隐向量
        all_z = []
        all_labels = []
        for images, labels in test_loader:
            images_flat = images.view(images.size(0), -1).to(device)
            z = model.encode(images_flat).cpu()
            all_z.append(z)
            all_labels.append(labels)
            if len(all_z) * 16 >= 500:
                break

        all_z = torch.cat(all_z, dim=0)[:500]
        all_labels = torch.cat(all_labels, dim=0)[:500]

    # 使用前两个维度可视化
    scatter = plt.scatter(
        all_z[:, 0].numpy(),
        all_z[:, 1].numpy(),
        c=all_labels.numpy(),
        cmap="tab10",
        alpha=0.5,
        s=10,
    )
    plt.colorbar(scatter, label="数字类别")
    plt.xlabel("隐维度 1")
    plt.ylabel("隐维度 2")
    plt.title("隐空间可视化 (前2维)")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "autoencoder_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 结果已保存到: {output_path}")


# ==================== 第五部分：去噪自编码器 ====================


class DenoisingAutoencoder(nn.Module):
    """去噪自编码器"""

    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super(DenoisingAutoencoder, self).__init__()

        # 与普通自编码器结构相同
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def denoising_demo():
    """去噪自编码器演示"""
    print("\n" + "=" * 60)
    print("第五部分：去噪自编码器")
    print("=" * 60)

    print("""
去噪自编码器 (Denoising Autoencoder, DAE)：

    原理:
    1. 对输入添加噪声: x_noisy = x + noise
    2. 训练模型从 x_noisy 重构原始 x
    3. 迫使模型学习数据的本质结构

    训练过程:
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   x ──→ 添加噪声 ──→ x_noisy ──→ [AE] ──→ x_recon      │
    │   ↑                                          ↓          │
    │   └────────── Loss = ||x - x_recon||² ──────┘          │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    常用噪声类型:
    - 高斯噪声: x + N(0, σ²)
    - 擦除噪声: 随机置零部分像素
    - 椒盐噪声: 随机将像素设为 0 或 1
    """)

    # 演示噪声添加
    print("\n示例: 噪声添加\n")

    # 创建测试图像
    x = torch.rand(1, 784)

    # 高斯噪声
    noise_factor = 0.3
    x_noisy = x + noise_factor * torch.randn_like(x)
    x_noisy = torch.clamp(x_noisy, 0, 1)

    print(f"原始数据范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"加噪后范围: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：尝试不同的隐空间维度
    任务: 训练 latent_dim = 8, 16, 32, 64 的自编码器
    观察:
    - 重构质量如何变化？
    - 隐空间表示如何变化？

练习 2：实现稀疏自编码器
    添加稀疏性约束:
    sparsity_loss = λ * ||z||₁  (L1 正则化)
    观察隐向量的稀疏性

练习 3：训练去噪自编码器
    步骤:
    1. 对输入添加噪声
    2. 训练模型恢复原始输入
    3. 测试在真实噪声数据上的效果

练习 4：使用自编码器进行异常检测
    思路:
    1. 在正常数据上训练自编码器
    2. 用重构误差判断异常
    3. 高重构误差 → 可能是异常

练习 5：隐空间插值
    任务: 在两个数据点的隐向量之间插值
    z_interp = α * z1 + (1-α) * z2
    观察解码后的过渡效果

思考题 1：为什么需要瓶颈层？
    如果隐空间维度等于输入维度会怎样？

思考题 2：自编码器 vs PCA
    两者有什么相似之处？
    自编码器的优势是什么？

思考题 3：为什么自编码器不能直接用于生成？
    提示：隐空间的分布是什么样的？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    basic_implementation()
    conv_autoencoder_demo()
    advanced_examples()
    denoising_demo()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 05-vae.py: 变分自编码器

关键要点回顾：
    ✓ 自编码器学习数据的压缩表示
    ✓ 编码器压缩，解码器重构
    ✓ 瓶颈层迫使学习重要特征
    ✓ 卷积自编码器适合图像数据
    ✓ 去噪自编码器学习数据的本质结构
    """)


if __name__ == "__main__":
    main()
