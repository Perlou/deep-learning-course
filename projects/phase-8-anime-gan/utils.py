"""
DCGAN 工具函数

包含数据加载、可视化等工具
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


class ImageFolderDataset(Dataset):
    """
    自定义图像文件夹数据集

    支持从任意文件夹加载图像
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # 支持的图像格式
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

        # 遍历目录获取所有图像
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, file))

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {root_dir} 中没有找到图像文件")

        print(f"找到 {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # 返回 (image, label)，label 不使用


class RandomNoiseDataset(Dataset):
    """
    随机噪声数据集（用于 Demo 模式）
    """

    def __init__(self, size=1000, image_size=64, channels=3):
        self.size = size
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 生成随机彩色图像
        image = torch.randn(self.channels, self.image_size, self.image_size)
        return image, 0


def get_dataloader(
    data_dir=None, batch_size=64, image_size=64, demo=False, num_workers=0
):
    """
    获取数据加载器

    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        image_size: 图像大小
        demo: 是否使用随机噪声数据（Demo 模式）
        num_workers: 数据加载线程数

    Returns:
        DataLoader 对象
    """
    if demo:
        print("使用 Demo 模式（随机噪声数据）")
        dataset = RandomNoiseDataset(size=1000, image_size=image_size)
    else:
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")

        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # 归一化到 [-1, 1]
            ]
        )

        dataset = ImageFolderDataset(data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloader


def save_samples(images, save_path, nrow=8):
    """
    保存生成的样本图像

    Args:
        images: 图像张量 (batch, C, H, W)，范围 [-1, 1]
        save_path: 保存路径
        nrow: 每行图像数量
    """
    # 反归一化到 [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    # 创建网格
    batch_size = images.size(0)
    ncol = min(batch_size, nrow)
    nrow_actual = (batch_size + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 1.5, nrow_actual * 1.5))

    if nrow_actual == 1:
        axes = [axes]
    if ncol == 1:
        axes = [[ax] for ax in axes]

    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i][j] if isinstance(axes[i], list) else axes[j]
            ax.axis("off")

            if idx < batch_size:
                img = images[idx].cpu().numpy().transpose(1, 2, 0)
                ax.imshow(img)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_losses(g_losses, d_losses, save_path):
    """
    绘制训练损失曲线

    Args:
        g_losses: 生成器损失列表
        d_losses: 判别器损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label="Generator", alpha=0.8)
    plt.plot(d_losses, label="Discriminator", alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # 使用移动平均平滑
    window = min(50, len(g_losses) // 10 + 1)
    if len(g_losses) > window:
        g_smooth = np.convolve(g_losses, np.ones(window) / window, mode="valid")
        d_smooth = np.convolve(d_losses, np.ones(window) / window, mode="valid")
        plt.plot(g_smooth, label="Generator (smooth)", alpha=0.8)
        plt.plot(d_smooth, label="Discriminator (smooth)", alpha=0.8)
    else:
        plt.plot(g_losses, label="Generator", alpha=0.8)
        plt.plot(d_losses, label="Discriminator", alpha=0.8)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Smoothed Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def interpolate_latent(z1, z2, steps=10):
    """
    在两个隐向量之间进行球面线性插值 (SLERP)

    Args:
        z1: 起始隐向量
        z2: 终止隐向量
        steps: 插值步数

    Returns:
        插值后的隐向量序列
    """
    z1 = z1.view(-1)
    z2 = z2.view(-1)

    # 归一化
    z1 = z1 / z1.norm()
    z2 = z2 / z2.norm()

    # 计算夹角
    omega = torch.acos(torch.clamp(torch.dot(z1, z2), -1, 1))

    interpolated = []
    for t in np.linspace(0, 1, steps):
        if omega.abs() < 1e-6:
            z = z1 * (1 - t) + z2 * t
        else:
            z = (torch.sin((1 - t) * omega) / torch.sin(omega)) * z1 + (
                torch.sin(t * omega) / torch.sin(omega)
            ) * z2
        interpolated.append(z)

    return torch.stack(interpolated)


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("工具函数测试")
    print("=" * 50)

    # 测试 Demo 数据加载器
    print("\n测试 Demo 数据加载器:")
    loader = get_dataloader(demo=True, batch_size=16)
    images, _ = next(iter(loader))
    print(f"批次形状: {images.shape}")

    # 测试保存样本
    print("\n测试保存样本:")
    save_samples(images, "test_samples.png", nrow=4)
    print("✓ 样本已保存到 test_samples.png")

    # 清理测试文件
    if os.path.exists("test_samples.png"):
        os.remove("test_samples.png")

    print("\n✓ 工具函数测试通过!")
