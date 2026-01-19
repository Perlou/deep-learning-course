"""
变分自编码器 (VAE)
==================

学习目标：
    1. 理解 VAE 与普通自编码器的区别
    2. 掌握变分推断的基本思想
    3. 理解重参数化技巧 (Reparameterization Trick)
    4. 实现 VAE 并理解 ELBO 损失

核心概念：
    - 潜在空间分布: 学习 z 的概率分布而非单点
    - 变分推断: 用简单分布近似复杂后验
    - 重参数化技巧: 使随机采样可导
    - ELBO: 证据下界，VAE 的优化目标

前置知识：
    - 04-autoencoder.py: 自编码器基础
    - 概率论基础（高斯分布、KL散度）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# ==================== 第一部分：VAE 原理介绍 ====================


def introduction():
    """
    VAE 原理介绍
    """
    print("=" * 60)
    print("第一部分：VAE 原理介绍")
    print("=" * 60)

    print("""
为什么需要 VAE？

    普通自编码器的问题：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 隐空间不连续                                          │
    │    - 相似数据的隐向量可能距离很远                          │
    │    - 无法在隐空间中平滑插值                               │
    │                                                         │
    │ 2. 无法生成新样本                                        │
    │    - 随机采样的隐向量可能落在"空洞"区域                    │
    │    - 解码结果可能是无意义的噪声                            │
    └─────────────────────────────────────────────────────────┘

VAE 的核心思想：

    不再学习单点隐向量，而是学习隐空间的概率分布！

    普通 AE:  x → z (确定性)     → x'
    VAE:      x → μ, σ → z~N(μ,σ²) → x'

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   输入 x ──→ 编码器 ──→ μ, σ ──→ 采样 z ──→ 解码器 ──→ x' │
    │                            ↓                           │
    │                      z = μ + σ * ε                      │
    │                      ε ~ N(0, 1)                        │
    │                                                         │
    │                  ↓ 正则化 ↓                              │
    │             约束 q(z|x) 接近 N(0, 1)                     │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

VAE 损失函数 (ELBO - 证据下界)：

    L = 重构损失 + β * KL散度

    1. 重构损失：
       E[log p(x|z)]  → 重构质量

    2. KL 散度：
       D_KL(q(z|x) || p(z))  → 隐空间正则化
       
       强制 q(z|x) = N(μ, σ²) 接近 p(z) = N(0, 1)

    ┌─────────────────────────────────────────────────────────┐
    │ KL散度的解析形式（两个高斯分布）：                         │
    │                                                         │
    │ D_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)                 │
    │                                                         │
    │ 当 μ=0, σ=1 时，KL = 0（完美匹配先验）                   │
    └─────────────────────────────────────────────────────────┘

重参数化技巧 (Reparameterization Trick)：

    问题：采样操作 z ~ N(μ, σ²) 不可导，无法反向传播

    解决：将随机性转移到与参数无关的变量上

    原本：z ~ N(μ, σ²)  ← 不可导
    改为：ε ~ N(0, 1)
          z = μ + σ * ε  ← 可导！
    """)


# ==================== 第二部分：VAE 实现 ====================


class VAE(nn.Module):
    """
    变分自编码器

    编码器输出: μ (均值) 和 log(σ²) (对数方差)
    使用重参数化技巧进行采样
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # 编码器
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 对数方差

        # 解码器
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        编码器：输入 → 隐变量分布参数

        Args:
            x: 输入 (batch, input_dim)

        Returns:
            mu: 均值 (batch, latent_dim)
            logvar: 对数方差 (batch, latent_dim)
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # log(σ²) 更稳定
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：使采样可导

        z = μ + σ * ε, where ε ~ N(0, 1)

        Args:
            mu: 均值
            logvar: 对数方差

        Returns:
            z: 采样的隐向量
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)  # ε ~ N(0, 1)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        解码器：隐向量 → 重构输出

        Args:
            z: 隐向量 (batch, latent_dim)

        Returns:
            x_recon: 重构输出 (batch, input_dim)
        """
        h = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE 损失函数 = 重构损失 + β * KL散度

    Args:
        x_recon: 重构输出
        x: 原始输入
        mu: 编码器输出的均值
        logvar: 编码器输出的对数方差
        beta: KL散度的权重 (β-VAE)

    Returns:
        loss: 总损失
        recon_loss: 重构损失
        kl_loss: KL散度
    """
    # 重构损失 (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

    # KL 散度: D_KL(N(μ, σ²) || N(0, 1))
    # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第二部分：VAE 实现")
    print("=" * 60)

    print("\n示例 1: 创建 VAE\n")

    input_dim = 784
    hidden_dim = 400
    latent_dim = 20

    vae = VAE(input_dim, hidden_dim, latent_dim)

    print(f"输入维度: {input_dim}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"隐空间维度: {latent_dim}")
    print(f"参数量: {sum(p.numel() for p in vae.parameters()):,}")

    print("\n示例 2: 前向传播\n")

    x = torch.rand(4, input_dim)
    print(f"输入形状: {x.shape}")

    x_recon, mu, logvar = vae(x)
    print(f"重构输出形状: {x_recon.shape}")
    print(f"均值 μ 形状: {mu.shape}")
    print(f"对数方差 log(σ²) 形状: {logvar.shape}")

    print("\n示例 3: 损失计算\n")

    loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

    print(f"总损失: {loss.item():.2f}")
    print(f"重构损失: {recon_loss.item():.2f}")
    print(f"KL 散度: {kl_loss.item():.2f}")

    print("\n示例 4: 重参数化技巧验证\n")

    # 验证采样的可导性
    mu = torch.randn(4, latent_dim, requires_grad=True)
    logvar = torch.randn(4, latent_dim, requires_grad=True)

    z = vae.reparameterize(mu, logvar)
    loss = z.sum()
    loss.backward()

    print(f"μ 的梯度存在: {mu.grad is not None}")
    print(f"log(σ²) 的梯度存在: {logvar.grad is not None}")
    print("✓ 重参数化技巧使采样可导！")


# ==================== 第三部分：VAE 训练 ====================


def train_vae(model, dataloader, num_epochs=10, lr=0.001, beta=1.0, device="cpu"):
    """
    训练 VAE

    Args:
        model: VAE 模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        beta: KL 散度权重
        device: 设备
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"total": [], "recon": [], "kl": []}

    print(f"\n开始训练 VAE，设备: {device}, β={beta}")

    for epoch in range(num_epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)

            # 前向传播
            x_recon, mu, logvar = model(images)

            # 计算损失
            loss, recon_loss, kl_loss = vae_loss(x_recon, images, mu, logvar, beta)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += images.size(0)

        # 记录平均损失
        history["total"].append(total_loss / num_batches)
        history["recon"].append(total_recon / num_batches)
        history["kl"].append(total_kl / num_batches)

        if (epoch + 1) % 2 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {history['total'][-1]:.4f} "
                f"Recon: {history['recon'][-1]:.4f} "
                f"KL: {history['kl'][-1]:.4f}"
            )

    return history


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第三部分：VAE 训练与可视化")
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
    except Exception:
        print("使用随机数据进行演示...")
        fake_data = torch.rand(1000, 1, 28, 28)
        fake_labels = torch.zeros(1000, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 创建并训练 VAE
    latent_dim = 2  # 使用 2 维便于可视化
    vae = VAE(784, 400, latent_dim)

    print(f"\n使用 {latent_dim} 维隐空间便于可视化")
    print("训练 VAE (10 epochs)...")

    history = train_vae(
        vae, dataloader, num_epochs=10, lr=0.001, beta=1.0, device=device
    )

    # 可视化
    plt.figure(figsize=(16, 4))

    # 1. 损失曲线
    plt.subplot(1, 4, 1)
    plt.plot(history["total"], label="Total")
    plt.plot(history["recon"], label="Recon")
    plt.plot(history["kl"], label="KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VAE 训练损失")
    plt.grid(True)

    # 2. 隐空间可视化
    plt.subplot(1, 4, 2)
    vae.eval()
    all_z = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(images.size(0), -1).to(device)
            mu, _ = vae.encode(images)
            all_z.append(mu.cpu())
            all_labels.append(labels)
            if len(all_z) * 128 >= 2000:
                break

    all_z = torch.cat(all_z, dim=0)[:2000]
    all_labels = torch.cat(all_labels, dim=0)[:2000]

    scatter = plt.scatter(
        all_z[:, 0].numpy(),
        all_z[:, 1].numpy(),
        c=all_labels.numpy(),
        cmap="tab10",
        alpha=0.5,
        s=5,
    )
    plt.colorbar(scatter, label="数字")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("隐空间分布")

    # 3. 从隐空间采样生成
    plt.subplot(1, 4, 3)
    with torch.no_grad():
        # 从标准正态分布采样
        z_sample = torch.randn(16, latent_dim).to(device)
        generated = vae.decode(z_sample).cpu().view(-1, 28, 28)

    grid = np.zeros((4 * 28, 4 * 28))
    for i in range(4):
        for j in range(4):
            grid[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = generated[
                i * 4 + j
            ].numpy()

    plt.imshow(grid, cmap="gray")
    plt.title("随机采样生成")
    plt.axis("off")

    # 4. 隐空间网格采样
    plt.subplot(1, 4, 4)
    n = 10
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)

    canvas = np.zeros((n * 28, n * 28))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                x_decoded = vae.decode(z).cpu().view(28, 28)
                canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = x_decoded.numpy()

    plt.imshow(canvas, cmap="gray")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("隐空间网格采样")
    plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "vae_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 结果已保存到: {output_path}")


# ==================== 第四部分：β-VAE 和隐空间解纠缠 ====================


def beta_vae_demo():
    """β-VAE 演示"""
    print("\n" + "=" * 60)
    print("第四部分：β-VAE 和隐空间解纠缠")
    print("=" * 60)

    print("""
β-VAE：

    通过调整 KL 散度的权重 β，可以控制隐空间的特性。

    L = 重构损失 + β * KL散度

    β 的作用：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │ β < 1: 更关注重构质量，隐空间可能不够规整                  │
    │                                                         │
    │ β = 1: 标准 VAE                                         │
    │                                                         │
    │ β > 1: 更关注隐空间正则化，促进解纠缠                      │
    │        但可能牺牲重构质量                                 │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

隐空间解纠缠 (Disentanglement)：

    目标: 让隐空间的每个维度对应一个独立的语义因素

    例如人脸数据:
    ┌─────────────────────────────────────────────────────────┐
    │ z₁ 控制: 表情                                           │
    │ z₂ 控制: 姿态                                           │
    │ z₃ 控制: 发色                                           │
    │ ...                                                     │
    └─────────────────────────────────────────────────────────┘

    β-VAE 通过增大 β 来鼓励这种解纠缠
    """)


# ==================== 第五部分：卷积 VAE ====================


class ConvVAE(nn.Module):
    """卷积变分自编码器"""

    def __init__(self, channels=1, latent_dim=32):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 展平后的维度
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # 解码器
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def conv_vae_demo():
    """卷积 VAE 演示"""
    print("\n" + "=" * 60)
    print("第五部分：卷积 VAE")
    print("=" * 60)

    print("\n示例: 卷积 VAE 结构\n")

    model = ConvVAE(channels=1, latent_dim=32)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试
    x = torch.rand(4, 1, 28, 28)
    x_recon, mu, logvar = model(x)

    print(f"输入形状: {x.shape}")
    print(f"重构形状: {x_recon.shape}")
    print(f"隐向量形状: {mu.shape}")


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：调整 β 参数
    任务: 训练 β = 0.5, 1.0, 4.0 的 VAE
    观察:
    - 重构质量如何变化？
    - 隐空间分布如何变化？

练习 2：实现条件 VAE (CVAE)
    在 VAE 中加入类别条件:
    - 编码器输入: x + one_hot(label)
    - 解码器输入: z + one_hot(label)
    任务: 生成指定数字

练习 3：隐空间插值
    任务:
    1. 选择两张不同数字的图像
    2. 编码获得 z1 和 z2
    3. 在 z1 和 z2 之间线性插值
    4. 解码并观察过渡过程

练习 4：隐空间算术
    任务: 尝试类似 word2vec 的向量运算
    例如: z_8 - z_0 + z_1 ≈ z_9?

练习 5：实现卷积 VAE 并训练
    使用 ConvVAE 训练 MNIST
    对比全连接 VAE 的效果

思考题 1：KL 散度的作用
    如果没有 KL 散度约束会怎样？
    VAE 会退化成什么？

思考题 2：VAE vs GAN
    两者的生成机制有什么本质区别？
    各自的优缺点是什么？

思考题 3：重参数化的必要性
    为什么不能直接对 N(μ, σ²) 采样后进行反向传播？
    重参数化技巧的数学原理是什么？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    basic_implementation()
    advanced_examples()
    beta_vae_demo()
    conv_vae_demo()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 06-diffusion-theory.py: 扩散模型理论

关键要点回顾：
    ✓ VAE 学习隐空间的概率分布
    ✓ 编码器输出均值和方差
    ✓ 重参数化技巧使采样可导
    ✓ ELBO = 重构损失 + KL散度
    ✓ β-VAE 通过调整 β 促进解纠缠
    ✓ 隐空间连续，可以平滑插值
    """)


if __name__ == "__main__":
    main()
