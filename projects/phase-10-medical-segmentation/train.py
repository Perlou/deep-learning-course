"""
训练脚本
========

训练 U-Net 进行肺部分割。
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

from config import (
    DEVICE,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    DATASET_CONFIG,
)
from model import create_model, CombinedLoss, count_parameters
from dataset import get_dataloaders
from utils import (
    save_checkpoint,
    dice_coefficient,
    iou_score,
    plot_training_history,
    visualize_batch,
)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算指标
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            dice = dice_coefficient(preds, masks)

        total_loss += loss.item()
        total_dice += dice.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        preds = torch.sigmoid(outputs)
        dice = dice_coefficient(preds, masks)
        iou = iou_score(preds, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    return avg_loss, avg_dice, avg_iou


def train(
    data_dir=None,
    model_name="unet",
    epochs=None,
    batch_size=None,
    learning_rate=None,
    device=None,
    save_dir=None,
):
    """
    完整训练流程

    Args:
        data_dir: 数据目录
        model_name: 模型名称 ("unet" 或 "attention_unet")
        epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        device: 设备
        save_dir: 模型保存目录
    """
    # 使用默认配置
    epochs = epochs or TRAIN_CONFIG["epochs"]
    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    learning_rate = learning_rate or TRAIN_CONFIG["learning_rate"]
    device = device or DEVICE
    save_dir = Path(save_dir) if save_dir else MODELS_DIR

    print("=" * 60)
    print("🏥 医学图像分割训练")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"批量大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print("=" * 60)

    # 数据加载
    print("\n📊 加载数据...")
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        image_size=DATASET_CONFIG["image_size"],
        batch_size=batch_size,
        num_workers=4,
        augmentation=TRAIN_CONFIG["augmentation"],
    )

    # 创建模型
    print("\n🔧 创建模型...")
    model = create_model(
        model_name=model_name,
        n_channels=MODEL_CONFIG["in_channels"],
        n_classes=MODEL_CONFIG["out_channels"],
        bilinear=MODEL_CONFIG["bilinear"],
        features=MODEL_CONFIG["features"],
    )
    model = model.to(device)
    print(f"参数量: {count_parameters(model):,}")

    # 损失函数和优化器
    criterion = CombinedLoss(
        bce_weight=TRAIN_CONFIG["bce_weight"], dice_weight=TRAIN_CONFIG["dice_weight"]
    )

    optimizer = Adam(
        model.parameters(), lr=learning_rate, weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    # 学习率调度器
    if TRAIN_CONFIG["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif TRAIN_CONFIG["scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)

    # 训练历史
    history = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
    }

    # 早停
    best_dice = 0
    patience = TRAIN_CONFIG["early_stopping_patience"]
    patience_counter = 0

    # 训练循环
    print("\n🚀 开始训练...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # 训练
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        # 打印结果
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(
            f"Val   Loss: {val_loss:.4f}, Val   Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}"
        )
        print(f"Learning Rate: {current_lr:.6f}")

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0

            save_checkpoint(
                model, optimizer, epoch, val_loss, val_dice, save_dir / "best_model.pth"
            )
            print(f"✓ 保存最佳模型 (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"\n⚠ 早停触发 (连续 {patience} 轮未提升)")
            break

    # 保存最终模型
    save_checkpoint(
        model, optimizer, epoch, val_loss, val_dice, save_dir / "final_model.pth"
    )

    # 保存训练曲线
    plot_training_history(history, RESULTS_DIR / "training_history.png")

    print("\n" + "=" * 60)
    print("✓ 训练完成!")
    print(f"最佳验证 Dice: {best_dice:.4f}")
    print(f"模型保存于: {save_dir}")
    print("=" * 60)

    return model, history


def quick_train(data_dir=None, epochs=5, model_name="unet"):
    """快速训练（用于演示）"""
    print("\n⚡ 快速训练模式")
    return train(
        data_dir=data_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练肺部分割模型")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "attention_unet"],
        help="模型类型",
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--quick", action="store_true", help="快速训练模式 (5 轮)")

    args = parser.parse_args()

    if args.quick:
        quick_train(data_dir=args.data, model_name=args.model)
    else:
        train(
            data_dir=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
