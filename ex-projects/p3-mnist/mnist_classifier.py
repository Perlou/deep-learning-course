"""
MNIST 手写数字分类器
Phase 3 实战项目

学习目标：
1. 完整的深度学习项目流程
2. CNN 模型构建和训练
3. 模型评估和可视化
4. 模型保存和推理
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("Phase 3 实战项目：MNIST 手写数字分类")
print("=" * 60)


# =============================================================================
# 1. 配置
# =============================================================================
class Config:
    # 路径
    data_dir = "./data"
    save_dir = "./outputs"

    # 训练参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # 模型参数
    num_classes = 10

    # 设备
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


config = Config()
os.makedirs(config.save_dir, exist_ok=True)
print(f"\n使用设备: {config.device}")

# =============================================================================
# 2. 数据准备
# =============================================================================
print("\n" + "=" * 60)
print("【1. 数据准备】")

# 数据变换
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# 下载和加载数据集
train_dataset = datasets.MNIST(
    root=config.data_dir, train=True, download=True, transform=train_transform
)

test_dataset = datasets.MNIST(
    root=config.data_dir, train=False, download=True, transform=test_transform
)

# 划分训练集和验证集
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

print(f"训练集: {len(train_dataset)} 样本")
print(f"验证集: {len(val_dataset)} 样本")
print(f"测试集: {len(test_dataset)} 样本")

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True if config.device.type == "cuda" else False,
)

val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# =============================================================================
# 3. 可视化样本
# =============================================================================
print("\n" + "=" * 60)
print("【2. 可视化样本】")


def visualize_samples(dataset, num_samples=16):
    """可视化数据样本"""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img, label = (
                dataset.dataset[dataset.indices[i]]
                if hasattr(dataset, "indices")
                else dataset[i]
            )
            # 反归一化
            img = img * 0.3081 + 0.1307
            ax.imshow(img.squeeze(), cmap="gray")
            ax.set_title(f"标签: {label}")
        ax.axis("off")
    plt.suptitle("MNIST 样本示例", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "samples.png"), dpi=100)
    plt.close()
    print(f"样本图片已保存: {config.save_dir}/samples.png")


visualize_samples(train_dataset)

# =============================================================================
# 4. 模型定义
# =============================================================================
print("\n" + "=" * 60)
print("【3. 模型定义】")


class CNN(nn.Module):
    """卷积神经网络分类器"""

    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一个卷积块: (1, 28, 28) -> (32, 14, 14)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 第二个卷积块: (32, 14, 14) -> (64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 第三个卷积块: (64, 7, 7) -> (128, 3, 3)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# 创建模型
model = CNN(num_classes=config.num_classes).to(config.device)
print(f"\n模型结构:\n{model}")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

# =============================================================================
# 5. 训练设置
# =============================================================================
print("\n" + "=" * 60)
print("【4. 训练设置】")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

print(f"损失函数: CrossEntropyLoss")
print(f"优化器: AdamW (lr={config.learning_rate})")
print(f"学习率调度: CosineAnnealing")

# =============================================================================
# 6. 训练和验证函数
# =============================================================================


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="训练中", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"}
        )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


# =============================================================================
# 7. 训练循环
# =============================================================================
print("\n" + "=" * 60)
print("【5. 开始训练】")

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

best_val_acc = 0
best_model_state = None
start_time = time.time()

for epoch in range(config.num_epochs):
    print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, config.device
    )

    # 验证
    val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)

    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # 记录历史
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"  ✓ 新的最佳模型！")

    print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"  学习率: {current_lr:.6f}")

elapsed_time = time.time() - start_time
print(f"\n训练完成! 用时: {elapsed_time:.2f}s")
print(f"最佳验证准确率: {best_val_acc:.4f}")

# =============================================================================
# 8. 保存模型
# =============================================================================
print("\n" + "=" * 60)
print("【6. 保存模型】")

# 保存最佳模型
model_path = os.path.join(config.save_dir, "mnist_cnn_best.pth")
torch.save(best_model_state, model_path)
print(f"最佳模型已保存: {model_path}")

# 保存完整 checkpoint
checkpoint_path = os.path.join(config.save_dir, "mnist_checkpoint.pth")
torch.save(
    {
        "epoch": config.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
    },
    checkpoint_path,
)
print(f"Checkpoint 已保存: {checkpoint_path}")

# =============================================================================
# 9. 测试评估
# =============================================================================
print("\n" + "=" * 60)
print("【7. 测试评估】")

# 加载最佳模型
model.load_state_dict(best_model_state)

# 测试
test_loss, test_acc = evaluate(model, test_loader, criterion, config.device)
print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集 Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# =============================================================================
# 10. 混淆矩阵和分类报告
# =============================================================================
print("\n" + "=" * 60)
print("【8. 详细评估】")


@torch.no_grad()
def get_predictions(model, loader, device):
    """获取所有预测结果"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


preds, labels, probs = get_predictions(model, test_loader, config.device)

# 混淆矩阵
from collections import Counter


def plot_confusion_matrix(preds, labels, num_classes):
    """绘制混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")

    # 标签
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵")

    # 添加数值
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "confusion_matrix.png"), dpi=100)
    plt.close()
    print(f"混淆矩阵已保存: {config.save_dir}/confusion_matrix.png")


plot_confusion_matrix(preds, labels, config.num_classes)

# 每个类别的准确率
print("\n每个数字的分类准确率:")
for i in range(config.num_classes):
    mask = labels == i
    acc = (preds[mask] == labels[mask]).mean()
    print(f"  数字 {i}: {acc:.4f} ({acc * 100:.2f}%)")

# =============================================================================
# 11. 训练曲线
# =============================================================================
print("\n" + "=" * 60)
print("【9. 训练曲线】")


def plot_training_curves(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="训练")
    axes[0].plot(history["val_loss"], label="验证")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="训练")
    axes[1].plot(history["val_acc"], label="验证")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("准确率曲线")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(history["lr"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("学习率变化")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "training_curves.png"), dpi=100)
    plt.close()
    print(f"训练曲线已保存: {config.save_dir}/training_curves.png")


plot_training_curves(history)

# =============================================================================
# 12. 可视化预测结果
# =============================================================================
print("\n" + "=" * 60)
print("【10. 预测可视化】")


def visualize_predictions(model, dataset, device, num_samples=16):
    """可视化预测结果"""
    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        img, label = dataset[idx]

        # 预测
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(1).item()
            confidence = prob[0, pred].item()

        # 显示图片
        img_show = img * 0.3081 + 0.1307
        ax.imshow(img_show.squeeze(), cmap="gray")

        # 标题 (正确为绿色，错误为红色)
        color = "green" if pred == label else "red"
        ax.set_title(
            f"预测: {pred} ({confidence:.2%})\n真实: {label}", color=color, fontsize=10
        )
        ax.axis("off")

    plt.suptitle("模型预测结果", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "predictions.png"), dpi=100)
    plt.close()
    print(f"预测结果已保存: {config.save_dir}/predictions.png")


visualize_predictions(model, test_dataset, config.device)

# =============================================================================
# 13. 错误分析
# =============================================================================
print("\n" + "=" * 60)
print("【11. 错误分析】")


def analyze_errors(model, dataset, device, num_samples=16):
    """分析错误预测"""
    model.eval()
    errors = []

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(1).item()

        if pred != label:
            prob = torch.softmax(output, dim=1)
            errors.append(
                {
                    "idx": idx,
                    "true": label,
                    "pred": pred,
                    "confidence": prob[0, pred].item(),
                }
            )

    print(
        f"总错误数: {len(errors)} / {len(dataset)} ({len(errors) / len(dataset) * 100:.2f}%)"
    )

    # 最容易混淆的数字对
    confusion_pairs = Counter()
    for e in errors:
        pair = (e["true"], e["pred"])
        confusion_pairs[pair] += 1

    print("\n最容易混淆的数字对:")
    for (true, pred), count in confusion_pairs.most_common(5):
        print(f"  {true} -> {pred}: {count}次")

    # 可视化一些错误样本
    if errors:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(errors):
                e = errors[i]
                img, _ = dataset[e["idx"]]
                img_show = img * 0.3081 + 0.1307
                ax.imshow(img_show.squeeze(), cmap="gray")
                ax.set_title(
                    f"真实: {e['true']}, 预测: {e['pred']}\n置信度: {e['confidence']:.2%}",
                    color="red",
                    fontsize=9,
                )
            ax.axis("off")
        plt.suptitle("错误预测样本", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(config.save_dir, "error_samples.png"), dpi=100)
        plt.close()
        print(f"错误样本已保存: {config.save_dir}/error_samples.png")


analyze_errors(model, test_dataset, config.device)

# =============================================================================
# 14. 总结
# =============================================================================
print("\n" + "=" * 60)
print("【项目总结】")
print("=" * 60)

print(f"""
📊 训练结果:
   - 训练集准确率: {history["train_acc"][-1]:.4f} ({history["train_acc"][-1] * 100:.2f}%)
   - 验证集准确率: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)
   - 测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)

📁 生成的文件:
   - {config.save_dir}/samples.png           (数据样本)
   - {config.save_dir}/training_curves.png   (训练曲线)
   - {config.save_dir}/confusion_matrix.png  (混淆矩阵)
   - {config.save_dir}/predictions.png       (预测结果)
   - {config.save_dir}/error_samples.png     (错误分析)
   - {config.save_dir}/mnist_cnn_best.pth    (最佳模型)
   - {config.save_dir}/mnist_checkpoint.pth  (完整检查点)

✅ Phase 3 实战项目完成！
""")
