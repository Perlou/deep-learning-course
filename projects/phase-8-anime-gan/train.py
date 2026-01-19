"""
DCGAN 训练脚本

使用方法:
    # Demo 模式（随机数据）
    python train.py --demo --epochs 5

    # 真实数据训练
    python train.py --data_dir data/images --epochs 50
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import Generator, Discriminator
from utils import get_dataloader, save_samples, plot_losses


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DCGAN 训练脚本")

    # 数据参数
    parser.add_argument(
        "--data_dir", type=str, default="data/images", help="图像数据目录"
    )
    parser.add_argument("--demo", action="store_true", help="使用随机数据进行演示")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")

    # 模型参数
    parser.add_argument("--latent_dim", type=int, default=100, help="噪声向量维度")
    parser.add_argument("--image_size", type=int, default=64, help="图像大小")
    parser.add_argument("--channels", type=int, default=3, help="图像通道数")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument(
        "--save_interval", type=int, default=5, help="保存间隔（epoch）"
    )
    parser.add_argument(
        "--sample_interval", type=int, default=100, help="采样间隔（iteration）"
    )

    # 恢复训练
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")

    return parser.parse_args()


def train(args):
    """训练 DCGAN"""
    print("=" * 60)
    print("DCGAN 动漫人脸生成 - 训练")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 创建输出目录
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    try:
        dataloader = get_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            demo=args.demo,
        )
        print(f"批次数量: {len(dataloader)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请检查数据目录是否正确，或使用 --demo 模式")
        return

    # 创建模型
    print("\n创建模型...")
    G = Generator(latent_dim=args.latent_dim, channels=args.channels).to(device)
    D = Discriminator(channels=args.channels).to(device)

    print(f"生成器参数量: {sum(p.numel() for p in G.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in D.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # 恢复训练
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        G.load_state_dict(checkpoint["generator"])
        D.load_state_dict(checkpoint["discriminator"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"从 epoch {start_epoch} 继续训练")

    # 固定噪声用于可视化
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    # 训练记录
    g_losses = []
    d_losses = []

    # 训练循环
    print(f"\n开始训练 {args.epochs} epochs...")
    print("-" * 60)

    total_iterations = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_g_loss = 0
        epoch_d_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (real_images, _) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # 标签（使用标签平滑）
            real_labels = torch.ones(batch_size, 1, 1, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=device) + 0.1

            # ==================== 训练判别器 ====================
            optimizer_D.zero_grad()

            # 真实图像
            output_real = D(real_images)
            loss_real = criterion(output_real, real_labels)

            # 假图像
            z = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = G(z)
            output_fake = D(fake_images.detach())
            loss_fake = criterion(output_fake, fake_labels)

            # 判别器总损失
            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ==================== 训练生成器 ====================
            optimizer_G.zero_grad()

            # 生成器希望判别器认为假图像是真的
            output = D(fake_images)
            g_loss = criterion(output, torch.ones_like(output))

            g_loss.backward()
            optimizer_G.step()

            # 记录损失
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            # 更新进度条
            pbar.set_postfix(
                {
                    "D_loss": f"{d_loss.item():.4f}",
                    "G_loss": f"{g_loss.item():.4f}",
                }
            )

            # 定期保存样本
            total_iterations += 1
            if total_iterations % args.sample_interval == 0:
                G.eval()
                with torch.no_grad():
                    fake_samples = G(fixed_noise)
                save_samples(
                    fake_samples,
                    os.path.join(sample_dir, f"iter_{total_iterations:06d}.png"),
                    nrow=8,
                )
                G.train()

        # 计算 epoch 平均损失
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1:3d} | "
            f"D_loss: {avg_d_loss:.4f} | "
            f"G_loss: {avg_g_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint = {
                "epoch": epoch,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1:03d}.pth"),
            )
            torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pth"))

            # 保存 epoch 样本
            G.eval()
            with torch.no_grad():
                fake_samples = G(fixed_noise)
            save_samples(
                fake_samples,
                os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"),
                nrow=8,
            )
            G.train()

    # 训练完成
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"\n训练完成!")
    print(f"总时间: {total_time / 60:.1f} 分钟")
    print(f"模型保存位置: {checkpoint_dir}")
    print(f"样本保存位置: {sample_dir}")

    # 保存损失曲线
    plot_losses(g_losses, d_losses, os.path.join(args.output_dir, "losses.png"))
    print(f"损失曲线已保存到: {os.path.join(args.output_dir, 'losses.png')}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
