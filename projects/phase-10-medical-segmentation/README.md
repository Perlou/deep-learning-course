# Phase 10 实战项目：医学图像分割系统

## 项目简介

本项目是 Phase 10 计算机视觉应用阶段的医学方向实战项目，使用 **U-Net** 对肺部 X 光片进行分割。

## 技术栈

- Python 3.10+
- PyTorch 2.x
- U-Net / Attention U-Net
- 医学图像预处理

## 项目结构

```
phase-10-medical-segmentation/
├── README.md           # 项目说明
├── config.py           # 配置参数
├── dataset.py          # 数据集加载与增强
├── model.py            # U-Net 模型定义
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── inference.py        # 推理脚本
├── utils.py            # 工具函数
├── main.py             # 主入口
└── outputs/
    ├── models/         # 保存的模型权重
    └── results/        # 分割结果可视化
```

## 快速开始

### 1. 环境准备

```bash
# 在项目根目录
source venv/bin/activate
```

### 2. 运行演示

```bash
python projects/phase-10-medical-segmentation/main.py demo
```

### 3. 快速训练

```bash
# 快速训练 (5 轮)
python projects/phase-10-medical-segmentation/main.py train --quick

# 完整训练
python projects/phase-10-medical-segmentation/main.py train --epochs 50
```

### 4. 评估模型

```bash
python projects/phase-10-medical-segmentation/main.py eval
```

### 5. 推理预测

```bash
python projects/phase-10-medical-segmentation/main.py predict --source chest_xray.png
```

## 核心功能

### LungSegmentationDataset 类

```python
from dataset import LungSegmentationDataset, get_dataloaders

# 自动下载/准备数据集
train_loader, val_loader = get_dataloaders(batch_size=8)

for images, masks in train_loader:
    print(f"图像: {images.shape}")  # [B, 1, 256, 256]
    print(f"掩码: {masks.shape}")   # [B, 1, 256, 256]
```

### U-Net 模型

```python
from model import UNet, AttentionUNet, CombinedLoss

# 标准 U-Net
model = UNet(n_channels=1, n_classes=1)

# Attention U-Net
model = AttentionUNet(n_channels=1, n_classes=1)

# 组合损失 (BCE + Dice)
criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
```

### 推理预测

```python
from inference import predict, predict_and_visualize

# 单张图像预测
mask, prob = predict("chest_xray.png")

# 预测并可视化
predict_and_visualize(
    "chest_xray.png",
    output_path="result.png",
    show=True
)
```

## 命令行接口

| 命令                              | 描述         |
| --------------------------------- | ------------ |
| `main.py demo`                    | 快速演示     |
| `main.py train --quick`           | 快速训练     |
| `main.py train --epochs 50`       | 完整训练     |
| `main.py eval`                    | 评估模型     |
| `main.py predict --source <路径>` | 推理预测     |
| `main.py info`                    | 显示系统信息 |

## 模型架构

### U-Net

```
输入 (1, 256, 256)
    │
    ▼ DoubleConv
┌───────┐                           ┌───────┐
│  64   │ ─────────────────────────→│  64   │
└───────┘                           └───────┘
    │ MaxPool                   UpConv │
┌───────┐                           ┌───────┐
│  128  │ ─────────────────────────→│  128  │
└───────┘                           └───────┘
    │                                   │
┌───────┐                           ┌───────┐
│  256  │ ─────────────────────────→│  256  │
└───────┘                           └───────┘
    │                                   │
┌───────┐                           ┌───────┐
│  512  │ ─────────────────────────→│  512  │
└───────┘                           └───────┘
    │                                   │
    └─────────→ ┌───────┐ ←──────────┘
                │  1024 │ (瓶颈层)
                └───────┘
                    │
                    ▼
            输出 (1, 256, 256)
```

## 评估指标

| 指标      | 说明                                 |
| --------- | ------------------------------------ |
| Dice      | 重叠率，越高越好 (0-1)               |
| IoU       | 交并比，越高越好 (0-1)               |
| Precision | 精确率，预测为正类中真正为正类的比例 |
| Recall    | 召回率，真实正类中被正确预测的比例   |

## 学习要点

1. **医学图像预处理**
   - CLAHE 对比度增强
   - 归一化处理
   - 医学图像特有的数据增强

2. **U-Net 架构**
   - Skip Connection 保留空间细节
   - 编码器-解码器对称结构
   - 特征拼接融合多尺度信息

3. **分割损失函数**
   - Dice Loss 处理类别不平衡
   - BCE + Dice 组合损失

4. **后处理技术**
   - 形态学开闭运算
   - 连通区域分析
   - 平滑边界

## 参考资料

- [U-Net 论文](https://arxiv.org/abs/1505.04597)
- [Attention U-Net 论文](https://arxiv.org/abs/1804.03999)
- [Phase 10 课程笔记](../../src/phase-10-cv-applications/05-unet.py)
