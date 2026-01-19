"""
Wasserstein GAN (WGAN)
======================

学习目标：
    1. 理解原始 GAN 的训练问题
    2. 掌握 Wasserstein 距离的概念
    3. 理解 WGAN 和 WGAN-GP 的改进
    4. 实现 WGAN-GP 并对比训练稳定性

核心概念：
    - Wasserstein 距离: 更好的分布距离度量
    - Lipschitz 约束: 保证训练稳定
    - 梯度惩罚 (Gradient Penalty): WGAN-GP 的核心
    - Critic: WGAN 中判别器的新角色

前置知识：
    - 01-gan-basics.py: GAN 基础
    - 02-dcgan.py: DCGAN 架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# ==================== 第一部分：原始 GAN 的问题 ====================


def introduction():
    """
    原始 GAN 的问题和 WGAN 的改进
    """
    print("=" * 60)
    print("第一部分：原始 GAN 的问题")
    print("=" * 60)

    print("""
原始 GAN 的主要问题：

    1. 训练不稳定
       ┌─────────────────────────────────────────────────────┐
       │ 问题: 损失震荡，模式崩塌                             │
       │ 原因: JS 散度在分布不重叠时为常数，梯度消失           │
       └─────────────────────────────────────────────────────┘

    2. 难以判断训练进度
       ┌─────────────────────────────────────────────────────┐
       │ 问题: 损失值与生成质量不相关                         │
       │ 原因: 判别器太强时，生成器损失不反映真实情况          │
       └─────────────────────────────────────────────────────┘

    3. 需要精心平衡 G 和 D
       ┌─────────────────────────────────────────────────────┐
       │ 问题: 网络能力、学习率都需要仔细调整                 │
       │ 后果: 超参数敏感，难以训练                           │
       └─────────────────────────────────────────────────────┘

WGAN 的解决方案：

    使用 Wasserstein 距离代替 JS 散度！

    Wasserstein 距离（Earth Mover's Distance, EMD）：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   W(P_r, P_g) = inf E[||x - y||]                       │
    │                                                         │
    │   直觉: 将分布 P_g 变成 P_r 需要"搬运"多少"土"          │
    │                                                         │
    │   例如：                                                │
    │   P_r: ●●●●○○○○    P_g: ○○○○●●●●                       │
    │   Wasserstein 距离 = 移动所有 ● 的总距离                │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    优点：
    - 即使分布不重叠，也能提供有意义的梯度
    - 损失值与生成质量相关
    - 训练更稳定

WGAN 的关键改变：

    1. 移除判别器的 Sigmoid（输出实数分数，而非概率）
    2. 不再使用 BCE 损失，而是 Wasserstein 损失
    3. 添加 Lipschitz 约束（通过权重裁剪或梯度惩罚）
    4. 使用 RMSprop 优化器（而非 Adam）
    """)


# ==================== 第二部分：Wasserstein 距离 ====================


def wasserstein_distance_demo():
    """Wasserstein 距离可视化"""
    print("\n" + "=" * 60)
    print("第二部分：Wasserstein 距离直觉")
    print("=" * 60)

    print("""
Wasserstein 距离 vs JS 散度：

    场景: 两个不重叠的 1D 高斯分布

    分布 1: N(0, 1)
    分布 2: N(μ, 1)  其中 μ 从 0 变化到 10

    JS 散度:
    - 当 μ > 0 且足够大时，JS ≈ log(2) = 常数
    - 梯度 = 0，无法优化！

    Wasserstein 距离:
    - W = μ（线性增长）
    - 始终有有意义的梯度

    ┌─────────────────────────────────────────────────────────┐
    │  距离                                                   │
    │   ▲                                                     │
    │   │     Wasserstein ／                                  │
    │   │              ／                                     │
    │   │           ／                                        │
    │   │        ／                                           │
    │   │     ／                                              │
    │   │  ／  JS ─────────────────── (饱和)                  │
    │   │／                                                   │
    │   └──────────────────────────────────────────────→ μ   │
    └─────────────────────────────────────────────────────────┘
    """)

    # 可视化演示
    plt.figure(figsize=(10, 4))

    # 1. 两种距离的对比
    plt.subplot(1, 2, 1)
    mu_values = np.linspace(0, 10, 100)

    # Wasserstein 距离 (1D 高斯分布)
    wasserstein = mu_values

    # JS 散度近似 (简化演示)
    js_divergence = np.ones_like(mu_values) * np.log(2)
    js_divergence[:10] = np.linspace(0, np.log(2), 10)  # 过渡区域

    plt.plot(mu_values, wasserstein, label="Wasserstein", linewidth=2)
    plt.plot(mu_values, js_divergence, label="JS Divergence", linewidth=2)
    plt.xlabel("分布间距 μ")
    plt.ylabel("距离")
    plt.title("Wasserstein vs JS 散度")
    plt.legend()
    plt.grid(True)

    # 2. 梯度对比
    plt.subplot(1, 2, 2)
    grad_wasserstein = np.ones_like(mu_values)
    grad_js = np.zeros_like(mu_values)
    grad_js[:10] = np.linspace(0.1, 0.0, 10)

    plt.plot(mu_values, grad_wasserstein, label="Wasserstein 梯度", linewidth=2)
    plt.plot(mu_values, grad_js, label="JS 梯度", linewidth=2)
    plt.xlabel("分布间距 μ")
    plt.ylabel("梯度大小")
    plt.title("梯度对比")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "wasserstein_vs_js.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 对比图已保存到: {output_path}")


# ==================== 第三部分：WGAN-GP 实现 ====================


class WGANGenerator(nn.Module):
    """WGAN 生成器（与 DCGAN 相同）"""

    def __init__(self, latent_dim=100, channels=1, feature_maps=64):
        super(WGANGenerator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


class WGANCritic(nn.Module):
    """
    WGAN Critic（评论家）

    注意：
    1. 没有 Sigmoid（输出任意实数）
    2. 不使用 BatchNorm（会影响 Lipschitz 约束）
    3. 使用 LayerNorm 或不使用归一化
    """

    def __init__(self, channels=1, feature_maps=64):
        super(WGANCritic, self).__init__()

        self.main = nn.Sequential(
            # 不使用 BatchNorm
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: 没有 Sigmoid
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x)


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    计算梯度惩罚（WGAN-GP 的核心）

    惩罚函数: λ * E[(||∇_x D(x̂)||_2 - 1)²]
    其中 x̂ 是真实和假样本之间的随机插值

    Args:
        critic: Critic 网络
        real_samples: 真实样本
        fake_samples: 假样本
        device: 设备

    Returns:
        gradient_penalty: 梯度惩罚项
    """
    # 随机插值系数
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # 创建插值样本
    interpolates = (
        alpha * real_samples + (1 - alpha) * fake_samples.detach()
    ).requires_grad_(True)

    # 计算 Critic 输出
    d_interpolates = critic(interpolates)

    # 创建全 1 向量用于计算梯度
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    # 计算梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第三部分：WGAN-GP 实现")
    print("=" * 60)

    print("\n示例 1: 创建 WGAN 网络\n")

    latent_dim = 100
    generator = WGANGenerator(latent_dim, channels=1)
    critic = WGANCritic(channels=1)

    print(f"生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Critic 参数量: {sum(p.numel() for p in critic.parameters()):,}")

    print("\n示例 2: Wasserstein 损失计算\n")

    # 创建假数据
    z = torch.randn(4, latent_dim, 1, 1)
    fake_images = generator(z)
    real_images = torch.randn(4, 1, 64, 64)

    # Critic 评分
    real_scores = critic(real_images)
    fake_scores = critic(fake_images)

    print(f"真实样本评分: {real_scores.view(-1).detach().numpy()}")
    print(f"假样本评分: {fake_scores.view(-1).detach().numpy()}")

    # Wasserstein 损失
    c_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
    g_loss = -torch.mean(fake_scores)

    print(f"\nCritic 损失: {c_loss.item():.4f}")
    print(f"生成器损失: {g_loss.item():.4f}")

    print("\n示例 3: 梯度惩罚计算\n")

    device = torch.device("cpu")
    gradient_penalty = compute_gradient_penalty(
        critic, real_images, fake_images, device
    )
    print(f"梯度惩罚: {gradient_penalty.item():.4f}")

    lambda_gp = 10
    total_loss = c_loss + lambda_gp * gradient_penalty
    print(f"总 Critic 损失 (含惩罚): {total_loss.item():.4f}")


# ==================== 第四部分：WGAN-GP 训练 ====================


def train_wgan_gp(
    generator,
    critic,
    dataloader,
    num_epochs=20,
    latent_dim=100,
    lr=0.0001,
    n_critic=5,
    lambda_gp=10,
    device="cpu",
):
    """
    WGAN-GP 训练函数

    Args:
        n_critic: 每训练一次 G，训练多少次 Critic
        lambda_gp: 梯度惩罚系数
    """
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    generator = generator.to(device)
    critic = critic.to(device)

    g_losses = []
    c_losses = []
    w_distances = []
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    print(f"\n开始训练 WGAN-GP，设备: {device}")
    print(f"Critic 训练步数比: {n_critic}")

    for epoch in range(num_epochs):
        g_loss_epoch = 0
        c_loss_epoch = 0
        w_dist_epoch = 0

        for batch_idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # ===== 训练 Critic =====
            for _ in range(n_critic):
                optimizer_C.zero_grad()

                # 真实样本评分
                real_scores = critic(real_images)

                # 假样本评分
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = generator(z)
                fake_scores = critic(fake_images.detach())

                # 梯度惩罚
                gp = compute_gradient_penalty(critic, real_images, fake_images, device)

                # WGAN-GP 损失
                c_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
                c_loss_total = c_loss + lambda_gp * gp

                c_loss_total.backward()
                optimizer_C.step()

            # ===== 训练生成器 =====
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            fake_scores = critic(fake_images)

            g_loss = -torch.mean(fake_scores)
            g_loss.backward()
            optimizer_G.step()

            # Wasserstein 距离估计
            w_distance = torch.mean(real_scores) - torch.mean(fake_scores)

            g_loss_epoch += g_loss.item()
            c_loss_epoch += c_loss_total.item()
            w_dist_epoch += w_distance.item()

        g_losses.append(g_loss_epoch / len(dataloader))
        c_losses.append(c_loss_epoch / len(dataloader))
        w_distances.append(w_dist_epoch / len(dataloader))

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"C_loss: {c_losses[-1]:.4f} "
                f"G_loss: {g_losses[-1]:.4f} "
                f"W_dist: {w_distances[-1]:.4f}"
            )

    return g_losses, c_losses, w_distances, fixed_noise


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第四部分：WGAN-GP 训练示例")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 准备数据
    transform = transforms.Compose(
        [
            transforms.Resize(64),
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
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    except Exception:
        print("使用随机数据进行演示...")
        fake_data = torch.randn(500, 1, 64, 64)
        fake_labels = torch.zeros(500, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 创建模型
    latent_dim = 100
    generator = WGANGenerator(latent_dim, channels=1)
    critic = WGANCritic(channels=1)

    print("\n开始训练 WGAN-GP (5 epochs 用于演示)...")

    g_losses, c_losses, w_distances, fixed_noise = train_wgan_gp(
        generator,
        critic,
        dataloader,
        num_epochs=5,
        latent_dim=latent_dim,
        lr=0.0001,
        n_critic=5,
        lambda_gp=10,
        device=device,
    )

    # 可视化
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(g_losses, label="Generator")
    plt.plot(c_losses, label="Critic")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("WGAN-GP 损失")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(w_distances, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Wasserstein Distance")
    plt.title("Wasserstein 距离（越小越好）")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise[:16]).cpu()

    grid = np.zeros((4 * 64, 4 * 64))
    for i in range(4):
        for j in range(4):
            img = fake_images[i * 4 + j, 0].numpy()
            img = (img + 1) / 2
            grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64] = img

    plt.imshow(grid, cmap="gray")
    plt.title("WGAN-GP 生成的图像")
    plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "wgan_gp_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 结果已保存到: {output_path}")


# ==================== 第五部分：WGAN vs 原始 GAN 对比 ====================


def comparison():
    """WGAN vs 原始 GAN 对比"""
    print("\n" + "=" * 60)
    print("第五部分：WGAN vs 原始 GAN 对比")
    print("=" * 60)

    print("""
┌────────────────┬─────────────────────┬─────────────────────┐
│     特性       │     原始 GAN         │       WGAN-GP       │
├────────────────┼─────────────────────┼─────────────────────┤
│ 距离度量       │ JS 散度              │ Wasserstein 距离    │
├────────────────┼─────────────────────┼─────────────────────┤
│ 判别器输出     │ 概率 [0, 1]          │ 任意实数            │
├────────────────┼─────────────────────┼─────────────────────┤
│ 损失函数       │ BCE Loss             │ Wasserstein Loss    │
├────────────────┼─────────────────────┼─────────────────────┤
│ 约束方法       │ 无                   │ 梯度惩罚            │
├────────────────┼─────────────────────┼─────────────────────┤
│ 优化器         │ Adam                 │ Adam (β1=0)         │
├────────────────┼─────────────────────┼─────────────────────┤
│ 训练稳定性     │ 差                   │ 好                  │
├────────────────┼─────────────────────┼─────────────────────┤
│ 损失与质量相关 │ 不一定               │ 是                  │
├────────────────┼─────────────────────┼─────────────────────┤
│ 模式崩塌       │ 常见                 │ 较少                │
├────────────────┼─────────────────────┼─────────────────────┤
│ 需要平衡 G/D   │ 是                   │ 否                  │
└────────────────┴─────────────────────┴─────────────────────┘

WGAN 变体总结：

    1. WGAN (权重裁剪)
       - 使用 weight clipping 约束 Lipschitz
       - 简单但效果不好（会丢失模型容量）

    2. WGAN-GP (梯度惩罚) ✓ 推荐
       - 使用梯度惩罚约束 Lipschitz
       - 训练更稳定，效果更好

    3. WGAN-SN (谱归一化)
       - 使用谱归一化约束
       - 计算更高效
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：实现权重裁剪版 WGAN
    在原始 WGAN 中，使用权重裁剪来满足 Lipschitz 约束:
    for p in critic.parameters():
        p.data.clamp_(-0.01, 0.01)
    对比与 WGAN-GP 的效果

练习 2：调整梯度惩罚系数
    尝试不同的 λ_gp 值 (1, 5, 10, 20)
    观察对训练稳定性和生成质量的影响

练习 3：对比 Wasserstein 距离曲线
    在 WGAN-GP 训练中，记录 Wasserstein 距离
    验证它是否与生成质量正相关

练习 4：实现谱归一化
    使用 PyTorch 的 spectral_norm:
    from torch.nn.utils import spectral_norm
    对比与梯度惩罚的效果

思考题 1：为什么梯度惩罚比权重裁剪更好？
    提示：考虑模型容量和梯度行为

思考题 2：n_critic 的作用
    为什么要让 Critic 训练多步，而 Generator 只训练一步？

思考题 3：WGAN 的理论基础
    Kantorovich-Rubinstein 对偶是什么？
    为什么 Lipschitz 约束是必要的？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    wasserstein_distance_demo()
    basic_implementation()
    advanced_examples()
    comparison()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 04-autoencoder.py: 自编码器基础
    - 05-vae.py: 变分自编码器

关键要点回顾：
    ✓ WGAN 使用 Wasserstein 距离替代 JS 散度
    ✓ Wasserstein 距离提供更好的梯度信号
    ✓ WGAN-GP 使用梯度惩罚约束 Lipschitz
    ✓ Critic 输出任意实数，不使用 Sigmoid
    ✓ 训练更稳定，损失与质量相关
    """)


if __name__ == "__main__":
    main()
