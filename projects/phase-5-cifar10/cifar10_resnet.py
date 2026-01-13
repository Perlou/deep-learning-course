"""
CIFAR-10 图像分类 - ResNet 实现
Phase 5 实战项目

学习目标：
1. 卷积神经网络图像分类
2. 残差连接原理与实现
3. 数据增强技术
4. 学习率调度策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import time

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("Phase 5 实战项目：CIFAR-10 图像分类")
print("=" * 60)


# =============================================================================
# 1. 配置
# =============================================================================
class Config:
    # 数据
    data_dir = "./data"
    batch_size = 128
    num_workers = 0  # macOS 兼容

    # 模型
    num_classes = 10

    # 训练
    num_epochs = 100
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # 保存
    save_dir = "./outputs"

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
os.makedirs(config.data_dir, exist_ok=True)
print(f"\n使用设备: {config.device}")

# CIFAR-10 类别名称
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# =============================================================================
# 2. 数据准备
# =============================================================================
print("\n" + "=" * 60)
print("【1. 数据准备】")

# CIFAR-10 标准化参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# 训练集数据增强
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)

# 测试集变换
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
)

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir, train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir, train=False, download=True, transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
)

print(f"训练集: {len(train_dataset)} 样本")
print(f"测试集: {len(test_dataset)} 样本")
print(f"类别数: {config.num_classes}")
print(f"批次大小: {config.batch_size}")


# =============================================================================
# 3. 可视化样本
# =============================================================================
print("\n" + "=" * 60)
print("【2. 样本可视化】")


def visualize_samples(dataset, num_samples=16):
    """可视化数据样本"""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        img, label = dataset[i]
        # 反标准化
        img = img.numpy().transpose(1, 2, 0)
        img = img * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(CLASSES[label])
        ax.axis("off")

    plt.suptitle("CIFAR-10 样本", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/samples.png", dpi=100)
    plt.close()


# 使用原始数据集可视化（不带增强）
vis_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir,
    train=True,
    download=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    ),
)
visualize_samples(vis_dataset)
print(f"样本图已保存: {config.save_dir}/samples.png")


# =============================================================================
# 4. ResNet 模型定义
# =============================================================================
print("\n" + "=" * 60)
print("【3. 模型定义】")


class BasicBlock(nn.Module):
    """ResNet 基本残差块"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet for CIFAR-10

    与 ImageNet 版本的区别：
    - 第一层用 3×3 卷积（而非 7×7）
    - 移除第一个 MaxPool（因为图像只有 32×32）
    """

    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # 第一层：3×3 卷积，不降采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """创建残差层"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    1,
                    stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cifar(num_classes=10):
    """ResNet-18 for CIFAR-10"""
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes)


# 创建模型
model = resnet18_cifar(num_classes=config.num_classes).to(config.device)
print(f"\n模型: ResNet-18 for CIFAR-10")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

# 测试前向传播
dummy_input = torch.randn(2, 3, 32, 32).to(config.device)
dummy_output = model(dummy_input)
print(f"输入: {dummy_input.shape} → 输出: {dummy_output.shape}")


# =============================================================================
# 5. 损失函数和优化器
# =============================================================================
print("\n" + "=" * 60)
print("【4. 训练配置】")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=config.learning_rate,
    momentum=config.momentum,
    weight_decay=config.weight_decay,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

print(f"损失函数: CrossEntropyLoss")
print(
    f"优化器: SGD (lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})"
)
print(f"学习率调度: CosineAnnealingLR")


# =============================================================================
# 6. 训练和验证函数
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# =============================================================================
# 7. 训练循环
# =============================================================================
print("\n" + "=" * 60)
print("【5. 开始训练】")

history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}

best_acc = 0.0
best_model_state = None
start_time = time.time()

for epoch in range(config.num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, config.device
    )

    # 验证
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, config.device)

    # 记录历史
    current_lr = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["lr"].append(current_lr)

    # 更新学习率
    scheduler.step()

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_state = model.state_dict().copy()

    # 打印进度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1:3d}/{config.num_epochs} | "
            f"Train: {train_acc * 100:.2f}% | Test: {test_acc * 100:.2f}% | "
            f"LR: {current_lr:.4f}"
        )

elapsed_time = time.time() - start_time
print(f"\n训练完成! 用时: {elapsed_time / 60:.1f} 分钟")
print(f"最佳测试准确率: {best_acc * 100:.2f}%")


# =============================================================================
# 8. 绘制训练曲线
# =============================================================================
print("\n" + "=" * 60)
print("【6. 训练可视化】")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 损失曲线
ax1 = axes[0]
ax1.plot(history["train_loss"], label="Train", color="blue")
ax1.plot(history["test_loss"], label="Test", color="orange")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("损失曲线")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 准确率曲线
ax2 = axes[1]
ax2.plot([acc * 100 for acc in history["train_acc"]], label="Train", color="blue")
ax2.plot([acc * 100 for acc in history["test_acc"]], label="Test", color="orange")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("准确率曲线")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% 目标")

# 学习率曲线
ax3 = axes[2]
ax3.plot(history["lr"], color="green")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.set_title("学习率调度")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{config.save_dir}/training_curves.png", dpi=100)
plt.close()
print(f"训练曲线已保存: {config.save_dir}/training_curves.png")


# =============================================================================
# 9. 最终评估
# =============================================================================
print("\n" + "=" * 60)
print("【7. 最终评估】")

# 加载最佳模型
model.load_state_dict(best_model_state)

# 最终评估
test_loss, test_acc, all_preds, all_labels = evaluate(
    model, test_loader, criterion, config.device
)

print(f"\n最终测试集结果:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc * 100:.2f}%")

# 是否达标
if test_acc >= 0.90:
    print(f"\n🎉 恭喜！已达到 90%+ 准确率目标！")
else:
    print(f"\n⚠️ 未达到 90% 目标，可尝试增加训练轮数或调整超参数")


# =============================================================================
# 10. 混淆矩阵
# =============================================================================
print("\n" + "=" * 60)
print("【8. 混淆矩阵】")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("混淆矩阵")
plt.colorbar()

tick_marks = np.arange(len(CLASSES))
plt.xticks(tick_marks, CLASSES, rotation=45, ha="right")
plt.yticks(tick_marks, CLASSES)

# 添加数字标注
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8,
        )

plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.tight_layout()
plt.savefig(f"{config.save_dir}/confusion_matrix.png", dpi=100)
plt.close()
print(f"混淆矩阵已保存: {config.save_dir}/confusion_matrix.png")

# 每类准确率
print("\n各类别准确率:")
for i, class_name in enumerate(CLASSES):
    class_acc = cm[i, i] / cm[i].sum() * 100
    print(f"  {class_name:12s}: {class_acc:.1f}%")


# =============================================================================
# 11. 预测可视化
# =============================================================================
print("\n" + "=" * 60)
print("【9. 预测可视化】")


def visualize_predictions(model, dataset, device, num_samples=16):
    """可视化预测结果"""
    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        image, true_label = dataset[idx]

        # 预测
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            _, pred_label = output.max(1)
            pred_label = pred_label.item()

        # 反标准化显示
        img = image.numpy().transpose(1, 2, 0)
        img = img * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        # 标题颜色
        color = "green" if pred_label == true_label else "red"
        ax.set_title(
            f"True: {CLASSES[true_label]}\nPred: {CLASSES[pred_label]}",
            color=color,
            fontsize=9,
        )
        ax.axis("off")

    plt.suptitle("预测结果 (绿色=正确, 红色=错误)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/predictions.png", dpi=100)
    plt.close()


visualize_predictions(model, test_dataset, config.device)
print(f"预测可视化已保存: {config.save_dir}/predictions.png")


# =============================================================================
# 12. 保存模型
# =============================================================================
print("\n" + "=" * 60)
print("【10. 保存模型】")

model_path = f"{config.save_dir}/best_model.pth"
torch.save(
    {
        "model_state_dict": best_model_state,
        "test_acc": best_acc,
        "config": {"num_classes": config.num_classes, "architecture": "ResNet-18"},
    },
    model_path,
)
print(f"模型已保存: {model_path}")


# =============================================================================
# 13. 总结
# =============================================================================
print("\n" + "=" * 60)
print("【项目总结】")
print("=" * 60)

print(f"""
应用的 Phase 5 知识点:
  ✅ 卷积层 (Conv2d) - 特征提取
  ✅ 池化层 (AdaptiveAvgPool2d) - 降维
  ✅ 残差连接 - 解决梯度消失
  ✅ BatchNorm - 加速训练、稳定梯度
  ✅ 数据增强 - 提升泛化能力
  ✅ Kaiming 初始化 - 适合 ReLU
  ✅ CosineAnnealing 学习率调度

CIFAR-10 版 ResNet 修改:
  • 第一层: 3×3 卷积 (非 7×7)
  • 移除第一个 MaxPool

结果:
  • 最佳测试准确率: {best_acc * 100:.2f}%
  • 训练时间: {elapsed_time / 60:.1f} 分钟

生成文件:
  📊 {config.save_dir}/samples.png
  📊 {config.save_dir}/training_curves.png
  📊 {config.save_dir}/confusion_matrix.png
  📊 {config.save_dir}/predictions.png
  💾 {config.save_dir}/best_model.pth
""")

print("✅ Phase 5 实战项目完成！")
