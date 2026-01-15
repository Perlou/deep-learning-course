"""
训练BERT情感分类模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
import time
import os
from bert_classifier import BERTClassifier, create_sample_data


class SentimentDataset(Dataset):
    """情感分类数据集"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("训练BERT情感分类模型")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    texts, labels = create_sample_data()

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 编码
    encodings = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    labels = torch.LongTensor(labels)

    # 创建数据集
    dataset = SentimentDataset(encodings, labels)

    # 划分训练集和验证集（简单示例：8:2）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # 创建模型
    print("\n创建模型...")
    model = BERTClassifier(num_classes=2).to(device)

    # 优化器和损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # 训练
    num_epochs = 10
    print(f"\n开始训练 {num_epochs} epochs...")
    print("-" * 60)

    best_val_acc = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                "outputs/checkpoints/best_model.pth",
            )

    print("-" * 60)
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型已保存到: outputs/checkpoints/best_model.pth")

    # 测试预测
    print("\n测试预测:")
    model.eval()
    test_texts = ["这部电影真的很棒，非常推荐！", "太无聊了，完全是浪费时间。"]

    test_encodings = tokenizer(
        test_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    with torch.no_grad():
        input_ids = test_encodings["input_ids"].to(device)
        attention_mask = test_encodings["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    for i, text in enumerate(test_texts):
        sentiment = "正面" if preds[i] == 1 else "负面"
        confidence = probs[i][preds[i]].item()
        print(f"\n  文本: {text}")
        print(f"  预测: {sentiment} (置信度: {confidence:.4f})")


if __name__ == "__main__":
    main()
