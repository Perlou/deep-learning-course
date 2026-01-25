"""
评估脚本
========

评估情感分析模型。
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

from config import DEVICE, MODELS_DIR, RESULTS_DIR, LABELS, NUM_CLASSES
from model import create_model
from dataset import get_dataloaders


@torch.no_grad()
def evaluate(
    model_type="textcnn",
    model_path=None,
    data_dir=None,
    device=None,
):
    """
    评估模型

    Args:
        model_type: 模型类型
        model_path: 模型权重路径
        data_dir: 数据目录
        device: 设备
    """
    device = device or DEVICE

    if model_path is None:
        model_path = MODELS_DIR / f"best_{model_type}.pth"
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"⚠ 模型不存在: {model_path}")
        print(f"请先训练模型: python train.py --model {model_type}")
        return None

    print("=" * 60)
    print("📊 情感分析模型评估")
    print("=" * 60)
    print(f"模型: {model_type}")
    print(f"设备: {device}")
    print("=" * 60)

    # 加载检查点
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    vocab = checkpoint.get("vocab")

    # 加载数据
    print("\n📊 加载数据...")
    tokenizer = None
    if model_type == "bert":
        try:
            from transformers import AutoTokenizer
            from config import BERT_CONFIG

            tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["model_name"])
        except ImportError:
            print("⚠ 未安装 transformers")
            return None

    _, val_loader, vocab_size, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        model_type=model_type,
        tokenizer=tokenizer,
    )

    # 创建模型
    print("\n🔧 加载模型...")
    if model_type == "textcnn":
        from config import TEXTCNN_CONFIG

        model = create_model(
            model_type, vocab_size=vocab_size, num_classes=NUM_CLASSES, **TEXTCNN_CONFIG
        )
    elif model_type == "lstm":
        from config import LSTM_CONFIG

        model = create_model(
            model_type, vocab_size=vocab_size, num_classes=NUM_CLASSES, **LSTM_CONFIG
        )
    elif model_type == "bert":
        from config import BERT_CONFIG

        model = create_model(model_type, num_classes=NUM_CLASSES, **BERT_CONFIG)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"加载 epoch {checkpoint.get('epoch', 'N/A')} 的权重")

    # 预测
    all_preds = []
    all_labels = []

    print("\n🔍 预测中...")
    for batch in val_loader:
        if model_type == "bert":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            outputs = model(input_ids, attention_mask)
        else:
            texts, labels = batch
            texts = texts.to(device)
            outputs = model(texts)

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 打印结果
    print("\n" + "=" * 60)
    print("📈 评估结果")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("\n分类报告:")
    print(
        classification_report(all_labels, all_preds, target_names=list(LABELS.values()))
    )

    # 保存混淆矩阵图
    save_confusion_matrix(
        cm, list(LABELS.values()), RESULTS_DIR / f"{model_type}_confusion_matrix.png"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, class_names, save_path):
    """保存混淆矩阵图"""
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 标注数值
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n📸 混淆矩阵保存于: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估情感分析模型")
    parser.add_argument(
        "--model",
        type=str,
        default="textcnn",
        choices=["textcnn", "lstm", "bert"],
        help="模型类型",
    )
    parser.add_argument("--model-path", type=str, default=None, help="模型权重路径")

    args = parser.parse_args()
    evaluate(model_type=args.model, model_path=args.model_path)
