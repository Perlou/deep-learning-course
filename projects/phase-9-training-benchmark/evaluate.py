"""
评估脚本
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import get_config
from model import get_model
from dataset import get_dataloaders, CIFAR10_CLASSES


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估中"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, accuracy, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵 (归一化)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"混淆矩阵已保存到 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="projects/phase-9-training-benchmark/outputs/models/best_model.pth",
        help="模型路径",
    )
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    config = checkpoint.get("config", get_config())
    model = get_model(config.model_name, config.num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 数据
    _, test_loader = get_dataloaders(batch_size=100)

    # 评估
    print("\n" + "=" * 60)
    print("开始评估")
    print("=" * 60)

    loss, accuracy, preds, labels = evaluate_model(model, test_loader, device)

    print(f"\n测试损失: {loss:.4f}")
    print(f"测试准确率: {accuracy:.2f}%")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=CIFAR10_CLASSES))

    # 混淆矩阵
    plot_confusion_matrix(
        labels,
        preds,
        CIFAR10_CLASSES,
        "projects/phase-9-training-benchmark/outputs/logs/confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
