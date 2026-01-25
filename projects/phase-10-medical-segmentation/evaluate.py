"""
评估脚本
========

评估分割模型的性能。
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from config import DEVICE, MODELS_DIR, RESULTS_DIR, MODEL_CONFIG, DATASET_CONFIG
from model import create_model
from dataset import get_dataloaders
from utils import (
    dice_coefficient,
    iou_score,
    load_checkpoint,
    visualize_batch,
    postprocess_mask,
)


@torch.no_grad()
def evaluate(
    model_path=None,
    data_dir=None,
    device=None,
    save_visualization=True,
):
    """
    评估模型性能

    Args:
        model_path: 模型权重路径
        data_dir: 数据目录
        device: 设备
        save_visualization: 是否保存可视化结果

    Returns:
        评估指标字典
    """
    device = device or DEVICE

    if model_path is None:
        model_path = MODELS_DIR / "best_model.pth"
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"⚠ 模型不存在: {model_path}")
        print("请先训练模型: python train.py")
        return None

    print("=" * 60)
    print("📊 模型评估")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"设备: {device}")
    print("=" * 60)

    # 加载数据
    print("\n📊 加载数据...")
    _, val_loader = get_dataloaders(
        data_dir=data_dir,
        image_size=DATASET_CONFIG["image_size"],
        batch_size=8,
        num_workers=4,
        augmentation=False,
    )

    # 加载模型
    print("\n🔧 加载模型...")
    model = create_model(
        model_name=MODEL_CONFIG["name"],
        n_channels=MODEL_CONFIG["in_channels"],
        n_classes=MODEL_CONFIG["out_channels"],
        bilinear=MODEL_CONFIG["bilinear"],
        features=MODEL_CONFIG["features"],
    )

    # 加载权重
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"加载 epoch {checkpoint.get('epoch', 'N/A')} 的权重")

    # 评估指标
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    sample_images = []
    sample_masks_true = []
    sample_masks_pred = []

    print("\n🔍 评估中...")
    for batch_idx, (images, masks) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        masks = masks.to(device)

        # 预测
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds_binary = (preds > 0.5).float()

        # 逐样本计算指标
        for i in range(images.size(0)):
            pred = preds_binary[i]
            target = masks[i]

            # Dice
            dice = dice_coefficient(pred, target)
            all_dice.append(dice.item())

            # IoU
            iou = iou_score(pred, target)
            all_iou.append(iou.item())

            # Precision & Recall
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)

            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()

            precision = (tp / (tp + fp + 1e-7)).item()
            recall = (tp / (tp + fn + 1e-7)).item()

            all_precision.append(precision)
            all_recall.append(recall)

        # 保存一些样本用于可视化
        if batch_idx == 0:
            sample_images = images.cpu()
            sample_masks_true = masks.cpu()
            sample_masks_pred = preds_binary.cpu()

    # 计算平均指标
    metrics = {
        "dice": np.mean(all_dice),
        "dice_std": np.std(all_dice),
        "iou": np.mean(all_iou),
        "iou_std": np.std(all_iou),
        "precision": np.mean(all_precision),
        "recall": np.mean(all_recall),
        "f1": 2
        * np.mean(all_precision)
        * np.mean(all_recall)
        / (np.mean(all_precision) + np.mean(all_recall) + 1e-7),
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("📈 评估结果")
    print("=" * 60)
    print(f"Dice Coefficient: {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"IoU (Jaccard):    {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print("=" * 60)

    # 保存可视化
    if save_visualization and len(sample_images) > 0:
        save_path = RESULTS_DIR / "evaluation_samples.png"
        visualize_batch(
            sample_images,
            sample_masks_true,
            sample_masks_pred,
            save_path=save_path,
            max_samples=4,
        )
        print(f"\n📸 可视化结果保存于: {save_path}")

    return metrics


def evaluate_with_postprocess(model_path=None, data_dir=None, device=None):
    """评估带后处理的结果"""
    device = device or DEVICE

    if model_path is None:
        model_path = MODELS_DIR / "best_model.pth"
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"⚠ 模型不存在: {model_path}")
        return None

    print("\n🔍 评估 (带后处理)...")

    # 加载数据和模型
    _, val_loader = get_dataloaders(
        data_dir=data_dir,
        image_size=DATASET_CONFIG["image_size"],
        batch_size=8,
        num_workers=4,
        augmentation=False,
    )

    model = create_model(
        model_name=MODEL_CONFIG["name"],
        n_channels=MODEL_CONFIG["in_channels"],
        n_classes=MODEL_CONFIG["out_channels"],
    )

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_dice = []
    all_dice_post = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for i in range(images.size(0)):
                pred = preds[i].squeeze().cpu().numpy()
                target = masks[i].squeeze().cpu().numpy()

                # 无后处理
                pred_binary = (pred > 0.5).astype(np.float32)
                dice = dice_coefficient(
                    torch.from_numpy(pred_binary), torch.from_numpy(target)
                )
                all_dice.append(dice.item())

                # 有后处理
                pred_post = postprocess_mask(pred).astype(np.float32)
                dice_post = dice_coefficient(
                    torch.from_numpy(pred_post), torch.from_numpy(target)
                )
                all_dice_post.append(dice_post.item())

    print("\n" + "=" * 60)
    print("📈 后处理对比")
    print("=" * 60)
    print(f"无后处理 Dice: {np.mean(all_dice):.4f}")
    print(f"有后处理 Dice: {np.mean(all_dice_post):.4f}")
    print(f"提升: {(np.mean(all_dice_post) - np.mean(all_dice)) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估分割模型")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--data", type=str, default=None, help="数据目录")
    parser.add_argument("--postprocess", action="store_true", help="评估后处理效果")

    args = parser.parse_args()

    if args.postprocess:
        evaluate_with_postprocess(model_path=args.model, data_dir=args.data)
    else:
        evaluate(model_path=args.model, data_dir=args.data)
