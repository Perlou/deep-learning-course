"""
DCGAN 图像生成脚本

使用方法:
    # 生成图像
    python generate.py --checkpoint outputs/checkpoints/latest.pth --num_images 16

    # 隐空间插值
    python generate.py --checkpoint outputs/checkpoints/latest.pth --interpolate
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import Generator
from utils import save_samples, interpolate_latent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DCGAN 图像生成脚本")

    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--num_images", type=int, default=64, help="生成图像数量")
    parser.add_argument("--latent_dim", type=int, default=100, help="噪声向量维度")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/generated", help="输出目录"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--interpolate", action="store_true", help="生成隐空间插值")
    parser.add_argument("--interpolate_steps", type=int, default=10, help="插值步数")

    return parser.parse_args()


def generate_images(args):
    """生成图像"""
    print("=" * 60)
    print("DCGAN 图像生成")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"随机种子: {args.seed}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点文件不存在: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)

    G = Generator(latent_dim=args.latent_dim).to(device)
    G.load_state_dict(checkpoint["generator"])
    G.eval()

    print(f"模型来自 epoch: {checkpoint.get('epoch', 'unknown') + 1}")

    if args.interpolate:
        # 隐空间插值
        generate_interpolation(G, args, device)
    else:
        # 生成随机图像
        generate_random(G, args, device)


def generate_random(G, args, device):
    """生成随机图像"""
    print(f"\n生成 {args.num_images} 张随机图像...")

    with torch.no_grad():
        z = torch.randn(args.num_images, args.latent_dim, 1, 1, device=device)
        fake_images = G(z)

    # 保存网格图像
    grid_path = os.path.join(args.output_dir, "generated_grid.png")
    save_samples(fake_images, grid_path, nrow=8)
    print(f"✓ 网格图像已保存到: {grid_path}")

    # 保存单独的图像
    print("\n保存单独的图像...")
    fake_images = (fake_images + 1) / 2  # 反归一化到 [0, 1]
    fake_images = fake_images.clamp(0, 1)

    for i in range(min(args.num_images, 16)):  # 最多保存 16 张
        img = fake_images[i].cpu().numpy().transpose(1, 2, 0)
        img_path = os.path.join(args.output_dir, f"image_{i + 1:03d}.png")
        plt.imsave(img_path, img)

    print(f"✓ 单独图像已保存到: {args.output_dir}")


def generate_interpolation(G, args, device):
    """生成隐空间插值"""
    print(f"\n生成隐空间插值 ({args.interpolate_steps} 步)...")

    with torch.no_grad():
        # 随机选择两个起始点
        z1 = torch.randn(args.latent_dim, device=device)
        z2 = torch.randn(args.latent_dim, device=device)

        # 进行插值
        z_interp = interpolate_latent(z1, z2, steps=args.interpolate_steps)
        z_interp = z_interp.view(-1, args.latent_dim, 1, 1).to(device)

        # 生成图像
        fake_images = G(z_interp)

    # 可视化
    fig, axes = plt.subplots(
        1, args.interpolate_steps, figsize=(args.interpolate_steps * 1.5, 2)
    )

    fake_images = (fake_images + 1) / 2
    fake_images = fake_images.clamp(0, 1)

    for i in range(args.interpolate_steps):
        ax = axes[i] if args.interpolate_steps > 1 else axes
        img = fake_images[i].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{i / (args.interpolate_steps - 1):.1f}")

    plt.suptitle("隐空间线性插值", y=1.05)
    plt.tight_layout()

    interp_path = os.path.join(args.output_dir, "interpolation.png")
    plt.savefig(interp_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ 插值图像已保存到: {interp_path}")

    # 生成多组插值
    print("\n生成多组插值...")
    num_rows = 4

    fig, axes = plt.subplots(
        num_rows,
        args.interpolate_steps,
        figsize=(args.interpolate_steps * 1.5, num_rows * 1.5),
    )

    with torch.no_grad():
        for row in range(num_rows):
            z1 = torch.randn(args.latent_dim, device=device)
            z2 = torch.randn(args.latent_dim, device=device)
            z_interp = interpolate_latent(z1, z2, steps=args.interpolate_steps)
            z_interp = z_interp.view(-1, args.latent_dim, 1, 1).to(device)
            fake_images = G(z_interp)
            fake_images = (fake_images + 1) / 2
            fake_images = fake_images.clamp(0, 1)

            for col in range(args.interpolate_steps):
                ax = axes[row, col] if num_rows > 1 else axes[col]
                img = fake_images[col].cpu().numpy().transpose(1, 2, 0)
                ax.imshow(img)
                ax.axis("off")

    plt.suptitle("多组隐空间插值", y=1.02)
    plt.tight_layout()

    multi_interp_path = os.path.join(args.output_dir, "multi_interpolation.png")
    plt.savefig(multi_interp_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ 多组插值图像已保存到: {multi_interp_path}")


if __name__ == "__main__":
    args = parse_args()
    generate_images(args)
