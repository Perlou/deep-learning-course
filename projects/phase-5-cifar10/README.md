# Phase 5 实战项目：CIFAR-10 图像分类

## 📋 项目概述

使用 ResNet-18 实现 CIFAR-10 图像分类，目标达到 **90%+ 测试准确率**。

## 🎯 学习目标

- 卷积神经网络在图像分类中的应用
- 残差连接解决梯度消失问题
- 数据增强提升模型泛化
- 学习率调度策略

## 📊 数据集

**CIFAR-10**：

- 60,000 张 32×32 彩色图像
- 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

## 🏗️ 模型架构

**ResNet-18 for CIFAR-10**（针对 32×32 图像优化）：

```
输入: (3, 32, 32)
    ↓
Conv2d(3→64, 3×3) + BN + ReLU  # 注：不使用 7×7 卷积
    ↓
Layer1: 2× BasicBlock(64)
    ↓
Layer2: 2× BasicBlock(128, stride=2)
    ↓
Layer3: 2× BasicBlock(256, stride=2)
    ↓
Layer4: 2× BasicBlock(512, stride=2)
    ↓
AdaptiveAvgPool2d(1×1) + Linear(512→10)
```

> ⚠️ **CIFAR-10 版修改**：移除第一个 MaxPool，第一层用 3×3 卷积

## 🚀 运行方式

```bash
cd projects/phase-5-cifar10
source ../../.venv/bin/activate
python cifar10_resnet.py
```

## 📁 生成文件

| 文件                           | 说明         |
| ------------------------------ | ------------ |
| `outputs/training_curves.png`  | 训练曲线     |
| `outputs/confusion_matrix.png` | 混淆矩阵     |
| `outputs/predictions.png`      | 预测样例     |
| `outputs/best_model.pth`       | 最佳模型权重 |

## 📈 预期结果

- 测试集准确率: ≥ 90%
- 训练时间: ~10-20 分钟 (GPU) / ~1-2 小时 (CPU)

## ✅ 关键知识点

### 残差连接

```python
output = F(x) + x  # 恒等快捷连接
```

梯度可直接通过跳跃连接传播，支持训练超深网络。

### 数据增强

```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```
