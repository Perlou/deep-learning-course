"""
扩散模型实现 (DDPM Implementation)
==================================

学习目标：
    1. 实现完整的 DDPM 训练和采样流程
    2. 理解 UNet 噪声预测网络的结构
    3. 掌握扩散模型的训练技巧
    4. 能够在 MNIST 上训练简单的扩散模型

核心概念：
    - 简化 UNet: 适用于小规模数据集
    - 训练循环: 随机采样时间步和噪声
    - 采样循环: 逐步去噪生成图像
    - 条件生成: 类别条件扩散

前置知识：
    - 06-diffusion-theory.py: 扩散模型理论
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


# ==================== 第一部分：扩散调度器 ====================


class DiffusionScheduler:
    """
    扩散过程的调度器

    管理 β、α、ᾱ 等参数
    """

    def __init__(
        self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        # 计算 α 和 ᾱ
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # 前一时刻的 ᾱ
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

        # 用于采样的系数
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

        # 用于后验分布
        self.posterior_variance = (
            self.betas * (1 - self.alpha_bars_prev) / (1 - self.alpha_bars)
        )

    def add_noise(self, x_0, t, noise=None):
        """
        前向过程: 给 x_0 添加噪声得到 x_t

        x_t = √ᾱₜ x_0 + √(1-ᾱₜ) ε

        Args:
            x_0: 原始数据 (batch, C, H, W)
            t: 时间步 (batch,)
            noise: 噪声，如果为 None 则随机采样

        Returns:
            x_t: 加噪后的数据
            noise: 使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 获取对应时间步的系数
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def sample_timesteps(self, batch_size):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)


def scheduler_demo():
    """调度器演示"""
    print("=" * 60)
    print("第一部分：扩散调度器")
    print("=" * 60)

    scheduler = DiffusionScheduler(num_timesteps=1000)

    print("\n调度器参数:")
    print(f"  时间步数: {scheduler.num_timesteps}")
    print(f"  β 范围: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
    print(f"  ᾱ 范围: [{scheduler.alpha_bars[-1]:.6f}, {scheduler.alpha_bars[0]:.6f}]")

    # 测试加噪
    print("\n测试前向过程:")
    x_0 = torch.randn(4, 1, 28, 28)
    t = torch.tensor([0, 250, 500, 999])
    x_t, noise = scheduler.add_noise(x_0, t)

    print(f"  输入形状: {x_0.shape}")
    print(f"  时间步: {t.tolist()}")
    print(f"  输出形状: {x_t.shape}")


# ==================== 第二部分：简化 UNet ====================


class Block(nn.Module):
    """基础卷积块"""

    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # 第一个卷积
        h = self.bn1(self.relu(self.conv1(x)))

        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)
        h = h + time_emb

        # 第二个卷积
        h = self.bn2(self.relu(self.conv2(h)))

        # 上/下采样
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class SimpleUNet(nn.Module):
    """
    简化的 UNet 用于 MNIST

    适合 28x28 图像
    """

    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32):
        super().__init__()

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # 下采样路径
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)  # 28 -> 28
        self.down1 = Block(64, 128, time_emb_dim)  # 28 -> 14
        self.down2 = Block(128, 256, time_emb_dim)  # 14 -> 7

        # 中间层
        self.mid_conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv2 = nn.Conv2d(256, 256, 3, padding=1)

        # 上采样路径
        self.up1 = Block(256, 128, time_emb_dim, up=True)  # 7 -> 14
        self.up2 = Block(128, 64, time_emb_dim, up=True)  # 14 -> 28

        # 输出层
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        """
        Args:
            x: 加噪图像 (batch, C, H, W)
            t: 时间步 (batch,)

        Returns:
            预测的噪声 (batch, C, H, W)
        """
        # 时间嵌入
        t = t.float()
        t_emb = self.time_mlp(t)

        # 初始卷积
        x0 = F.relu(self.conv0(x))

        # 下采样
        x1 = self.down1(x0, t_emb)
        x2 = self.down2(x1, t_emb)

        # 中间层
        x_mid = F.relu(self.mid_conv1(x2))
        x_mid = F.relu(self.mid_conv2(x_mid))

        # 上采样 (带跳跃连接)
        x = self.up1(torch.cat([x_mid, x2], dim=1), t_emb)
        x = self.up2(torch.cat([x, x1], dim=1), t_emb)

        # 输出
        return self.output(x)


def unet_demo():
    """UNet 演示"""
    print("\n" + "=" * 60)
    print("第二部分：简化 UNet")
    print("=" * 60)

    model = SimpleUNet(in_channels=1, out_channels=1)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))

    pred_noise = model(x, t)

    print(f"输入形状: {x.shape}")
    print(f"时间步: {t.tolist()}")
    print(f"输出形状: {pred_noise.shape}")


# ==================== 第三部分：训练循环 ====================


def train_diffusion(model, scheduler, dataloader, num_epochs=10, lr=1e-3, device="cpu"):
    """
    训练扩散模型

    Args:
        model: UNet 噪声预测网络
        scheduler: 扩散调度器
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
        num_batches = 0

        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.shape[0]

            # 1. 随机采样时间步
            t = scheduler.sample_timesteps(batch_size)

            # 2. 添加噪声
            x_t, noise = scheduler.add_noise(images, t)

            # 3. 预测噪声
            pred_noise = model(x_t, t)

            # 4. 计算损失
            loss = criterion(pred_noise, noise)

            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.6f}")

    return losses


# ==================== 第四部分：采样循环 ====================


@torch.no_grad()
def sample(model, scheduler, num_samples=16, image_size=28, channels=1, device="cpu"):
    """
    从扩散模型采样

    Args:
        model: 训练好的 UNet
        scheduler: 扩散调度器
        num_samples: 采样数量
        image_size: 图像尺寸
        channels: 通道数
        device: 设备

    Returns:
        生成的图像 (num_samples, channels, H, W)
    """
    model.eval()

    # 从纯噪声开始
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    # 逐步去噪
    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # 预测噪声
        pred_noise = model(x, t_batch)

        # 计算系数
        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alpha_bars[t]
        beta = scheduler.betas[t]

        # 计算均值
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        # 去噪步骤
        x = (
            1 / torch.sqrt(alpha) * (x - beta / torch.sqrt(1 - alpha_bar) * pred_noise)
            + torch.sqrt(beta) * noise
        )

    return x


def sampling_demo():
    """采样演示"""
    print("\n" + "=" * 60)
    print("第四部分：采样过程")
    print("=" * 60)

    print("""
采样算法：

    for t = T-1, T-2, ..., 0:
        1. 预测噪声: ε̂ = model(xₜ, t)
        2. 计算均值: μ = (1/√αₜ)(xₜ - βₜ/√(1-ᾱₜ) * ε̂)
        3. 添加噪声: x_{t-1} = μ + σₜz (t>0时)

    注意: 最后一步 (t=0) 不添加噪声
    """)


# ==================== 第五部分：完整训练示例 ====================


def full_training_example():
    """完整训练示例"""
    print("\n" + "=" * 60)
    print("第五部分：完整训练示例")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 准备数据
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
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
    except Exception:
        print("使用随机数据进行演示...")
        fake_data = torch.randn(1000, 1, 28, 28)
        fake_labels = torch.zeros(1000, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 创建模型和调度器
    model = SimpleUNet(in_channels=1, out_channels=1)
    scheduler = DiffusionScheduler(num_timesteps=1000, device=device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("训练扩散模型 (5 epochs 用于演示)...")
    print("注意: 完整训练需要更多 epochs (如 50-100)")

    # 训练
    losses = train_diffusion(
        model, scheduler, dataloader, num_epochs=5, lr=1e-3, device=device
    )

    # 采样
    print("\n从模型采样...")
    print("(使用较少步数加速演示，可能质量较低)")

    # 使用简化的调度器进行快速采样
    fast_scheduler = DiffusionScheduler(num_timesteps=100, device=device)
    samples = sample(model, fast_scheduler, num_samples=16, device=device)

    # 可视化
    plt.figure(figsize=(12, 4))

    # 1. 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("扩散模型训练损失")
    plt.grid(True)

    # 2. 生成的样本
    plt.subplot(1, 2, 2)
    samples = samples.cpu()
    samples = (samples + 1) / 2  # 反归一化到 [0, 1]
    samples = samples.clamp(0, 1)

    grid = np.zeros((4 * 28, 4 * 28))
    for i in range(4):
        for j in range(4):
            grid[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = samples[
                i * 4 + j, 0
            ].numpy()

    plt.imshow(grid, cmap="gray")
    plt.title("扩散模型生成的样本")
    plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "diffusion_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 结果已保存到: {output_path}")


# ==================== 第六部分：可视化扩散过程 ====================


def visualize_diffusion_process():
    """可视化扩散过程"""
    print("\n" + "=" * 60)
    print("第六部分：可视化扩散过程")
    print("=" * 60)

    device = torch.device("cpu")
    scheduler = DiffusionScheduler(num_timesteps=1000, device=device)

    # 加载一张图像
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )

    try:
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        image, label = dataset[0]
        image = image.unsqueeze(0)  # 添加 batch 维度
    except Exception:
        print("使用随机数据进行演示...")
        image = torch.randn(1, 1, 28, 28)
        label = 0

    # 在不同时间步添加噪声
    timesteps = [0, 100, 200, 400, 600, 800, 999]

    plt.figure(figsize=(14, 2))

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t])
        x_t, _ = scheduler.add_noise(image, t_tensor)

        plt.subplot(1, len(timesteps), i + 1)
        img = x_t[0, 0].numpy()
        img = (img + 1) / 2  # 反归一化
        plt.imshow(img, cmap="gray")
        plt.title(f"t={t}")
        plt.axis("off")

    plt.suptitle("前向扩散过程 (逐步加噪)")
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "diffusion_process.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ 扩散过程可视化已保存到: {output_path}")


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：增加训练轮数
    任务: 训练 50+ epochs 并观察生成质量变化
    记录不同 epoch 的采样结果

练习 2：实现 DDIM 采样
    DDIM 可以用更少的步数生成图像:
    - 原始 DDPM: 1000 步
    - DDIM: 可以 50-100 步
    任务: 实现 DDIM 采样器

练习 3：实现类别条件生成
    在 UNet 中加入类别嵌入:
    - 类别 → embedding → 加到时间嵌入上
    任务: 生成指定数字

练习 4：尝试不同的 UNet 架构
    - 增加/减少层数
    - 添加注意力模块
    - 使用残差连接

练习 5：在 CIFAR-10 上训练
    任务: 修改代码以支持 32x32 RGB 图像

思考题 1：采样步数与质量
    减少采样步数会带来什么影响？
    有什么方法可以加速采样？

思考题 2：Classifier-Free Guidance
    什么是 CFG？
    它如何提高条件生成的质量？

思考题 3：扩散模型的局限性
    扩散模型的主要缺点是什么？
    如何解决？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    scheduler_demo()
    unet_demo()
    sampling_demo()
    visualize_diffusion_process()
    full_training_example()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 08-stable-diffusion-intro.py: Stable Diffusion 架构

关键要点回顾：
    ✓ 调度器管理噪声参数
    ✓ UNet 预测噪声
    ✓ 训练: 随机 t, 加噪, 预测, MSE 损失
    ✓ 采样: 从噪声逐步去噪
    ✓ 可以用更少步数加速采样 (DDIM)
    """)


if __name__ == "__main__":
    main()
