"""
训练脚本
========

训练情感分析模型。
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from config import (
    DEVICE,
    MODELS_DIR,
    RESULTS_DIR,
    TEXTCNN_CONFIG,
    LSTM_CONFIG,
    BERT_CONFIG,
    TRAIN_CONFIG,
    NUM_CLASSES,
)
from model import create_model, count_parameters
from dataset import get_dataloaders


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, model_type="textcnn"
):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        if model_type == "bert":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
        else:
            texts, labels = batch
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"}
        )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device, model_type="textcnn"):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in val_loader:
        if model_type == "bert":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
        else:
            texts, labels = batch
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)

        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train(
    model_type="textcnn",
    data_dir=None,
    epochs=None,
    batch_size=None,
    learning_rate=None,
    device=None,
):
    """
    训练模型

    Args:
        model_type: 模型类型 ("textcnn", "lstm", "bert")
        data_dir: 数据目录
        epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        device: 设备
    """
    epochs = epochs or TRAIN_CONFIG["epochs"]
    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    device = device or DEVICE

    if learning_rate is None:
        learning_rate = (
            TRAIN_CONFIG["bert_learning_rate"]
            if model_type == "bert"
            else TRAIN_CONFIG["learning_rate"]
        )

    print("=" * 60)
    print("📝 情感分析模型训练")
    print("=" * 60)
    print(f"模型: {model_type}")
    print(f"设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"批量大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print("=" * 60)

    # 数据加载
    print("\n📊 加载数据...")
    tokenizer = None
    if model_type == "bert":
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["model_name"])
        except ImportError:
            print("⚠ 未安装 transformers，请运行: pip install transformers")
            return None, None

    train_loader, val_loader, vocab_size, vocab = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        model_type=model_type,
        tokenizer=tokenizer,
    )

    # 创建模型
    print("\n🔧 创建模型...")
    if model_type == "textcnn":
        model = create_model(
            model_type, vocab_size=vocab_size, num_classes=NUM_CLASSES, **TEXTCNN_CONFIG
        )
    elif model_type == "lstm":
        model = create_model(
            model_type, vocab_size=vocab_size, num_classes=NUM_CLASSES, **LSTM_CONFIG
        )
    elif model_type == "bert":
        model = create_model(model_type, num_classes=NUM_CLASSES, **BERT_CONFIG)

    model = model.to(device)
    print(f"参数量: {count_parameters(model):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    if model_type == "bert":
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=TRAIN_CONFIG["weight_decay"],
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练历史
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # 早停
    best_acc = 0
    patience = TRAIN_CONFIG["early_stopping_patience"]
    patience_counter = 0

    # 训练循环
    print("\n🚀 开始训练...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, model_type
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device, model_type)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0

            save_path = MODELS_DIR / f"best_{model_type}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "vocab": vocab if model_type != "bert" else None,
                },
                save_path,
            )
            print(f"✓ 保存最佳模型 (Acc: {best_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n⚠ 早停触发 (连续 {patience} 轮未提升)")
            break

    print("\n" + "=" * 60)
    print("✓ 训练完成!")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"模型保存于: {MODELS_DIR}")
    print("=" * 60)

    return model, history


def quick_train(model_type="textcnn", epochs=5):
    """快速训练"""
    print("\n⚡ 快速训练模式")
    return train(
        model_type=model_type,
        epochs=epochs,
        batch_size=32,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练情感分析模型")
    parser.add_argument(
        "--model",
        type=str,
        default="textcnn",
        choices=["textcnn", "lstm", "bert"],
        help="模型类型",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--quick", action="store_true", help="快速训练模式")

    args = parser.parse_args()

    if args.quick:
        quick_train(model_type=args.model)
    else:
        train(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
