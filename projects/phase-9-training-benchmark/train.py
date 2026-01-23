"""
训练入口
"""

import argparse
import matplotlib.pyplot as plt

from config import get_config, TrainConfig
from model import get_model
from dataset import get_dataloaders
from trainer import Trainer


def plot_history(history, save_path="outputs/logs/training_history.png"):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["test_loss"], label="Test")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["test_acc"], label="Test")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Accuracy Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning Rate
    axes[1, 0].plot(history["lr"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].grid(True)

    # Final Stats
    axes[1, 1].axis("off")
    best_acc = max(history["test_acc"])
    final_acc = history["test_acc"][-1]
    stats_text = f"""
    训练结果统计
    ─────────────────────
    最佳测试准确率: {best_acc:.2f}%
    最终测试准确率: {final_acc:.2f}%
    最终训练损失: {history["train_loss"][-1]:.4f}
    最终测试损失: {history["test_loss"][-1]:.4f}
    """
    axes[1, 1].text(
        0.1,
        0.5,
        stats_text,
        fontsize=14,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存到 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="训练 ResNet on CIFAR-10")
    parser.add_argument(
        "--config",
        type=str,
        default="full_optimization",
        choices=["baseline", "adamw_cosine", "adamw_onecycle", "full_optimization"],
        help="预设配置名称",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    args = parser.parse_args()

    # 获取配置
    config = get_config(args.config)

    # 覆盖参数
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr

    print("=" * 60)
    print("Phase 9 实战项目: 训练优化基准测试")
    print("=" * 60)
    print(f"\n配置: {args.config}")
    print(f"  优化器: {config.optimizer}")
    print(f"  学习率: {config.lr}")
    print(f"  调度器: {config.scheduler}")
    print(f"  混合精度: {config.use_amp}")
    print(f"  梯度累积: {config.gradient_accumulation_steps}")
    print(f"  梯度裁剪: {config.grad_clip}")
    print(f"  Warmup: {config.warmup_epochs} epochs")
    print()

    # 数据
    train_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # 模型
    model = get_model(config.model_name, config.num_classes)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型: {config.model_name}, 参数量: {param_count:,}")

    # 训练器
    trainer = Trainer(model, config, train_loader, test_loader)

    # 训练
    history = trainer.train()

    # 绘制曲线
    plot_history(history, f"{config.output_dir}/logs/training_history.png")


if __name__ == "__main__":
    main()
