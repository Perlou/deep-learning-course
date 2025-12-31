# Phase 3 实战项目：MNIST 手写数字分类

## 📋 项目概述

使用卷积神经网络 (CNN) 对 MNIST 手写数字进行分类，实现完整的深度学习项目流程。

## 🎯 学习目标

- 数据加载和预处理
- CNN 模型设计和实现
- 完整训练循环
- 模型评估和可视化
- 模型保存和加载

## 🏗️ 模型架构

```
输入: (1, 28, 28)
    ↓
Conv2d(1→32) + BN + ReLU + MaxPool
    ↓ (32, 14, 14)
Conv2d(32→64) + BN + ReLU + MaxPool
    ↓ (64, 7, 7)
Conv2d(64→128) + BN + ReLU + MaxPool
    ↓ (128, 3, 3)
Flatten
    ↓ (1152,)
Linear(1152→256) + ReLU + Dropout(0.5)
    ↓
Linear(256→10)
    ↓
输出: (10,) logits
```

## 🚀 运行方式

```bash
# 进入项目目录
cd projects/phase-3-mnist

# 激活虚拟环境
source ../../.venv/bin/activate

# 运行项目
python mnist_classifier.py
```

## 📁 生成文件

| 文件                           | 说明           |
| ------------------------------ | -------------- |
| `outputs/samples.png`          | 数据样本可视化 |
| `outputs/training_curves.png`  | 训练曲线       |
| `outputs/confusion_matrix.png` | 混淆矩阵       |
| `outputs/predictions.png`      | 预测结果       |
| `outputs/error_samples.png`    | 错误分析       |
| `outputs/mnist_cnn_best.pth`   | 最佳模型权重   |
| `outputs/mnist_checkpoint.pth` | 完整检查点     |

## 📊 预期结果

- 测试集准确率: ~99%
- 训练时间: ~2-5 分钟 (取决于硬件)

## ✅ 核心代码要点

### 数据增强

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### 训练循环

```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # 验证...

    scheduler.step()
```

### 模型保存

```python
torch.save(model.state_dict(), 'model.pth')
```
