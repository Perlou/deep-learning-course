# Phase 10 实战项目：YOLOv8 目标检测系统

## 项目简介

本项目是 Phase 10 计算机视觉应用阶段的实战项目，综合运用目标检测知识，使用 YOLOv8 构建完整的目标检测系统。

## 技术栈

- Python 3.10+
- PyTorch 2.x
- ultralytics (YOLOv8)
- OpenCV
- matplotlib

## 项目结构

```
phase-10-object-detection/
├── README.md           # 项目说明
├── config.py           # 配置参数
├── dataset.py          # 数据集处理
├── model.py            # 模型封装
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── inference.py        # 推理脚本
├── utils.py            # 工具函数
├── main.py             # 主入口
└── outputs/
    ├── models/         # 保存的模型
    └── results/        # 检测结果
```

## 快速开始

### 1. 环境准备

```bash
# 在项目根目录
source venv/bin/activate

# 安装 ultralytics
pip install ultralytics
```

### 2. 运行演示

```bash
python projects/phase-10-object-detection/main.py demo
```

### 3. 图片检测

```bash
# 使用预训练模型检测
python projects/phase-10-object-detection/main.py predict \
    --source image.jpg \
    --show
```

### 4. 实时摄像头检测

```bash
python projects/phase-10-object-detection/main.py predict --source 0
```

### 5. 训练模型

```bash
# 使用 COCO128 数据集快速训练
python projects/phase-10-object-detection/main.py train \
    --data coco128.yaml \
    --epochs 100

# 快速训练模式 (10 轮)
python projects/phase-10-object-detection/main.py train --quick
```

### 6. 评估模型

```bash
python projects/phase-10-object-detection/main.py eval \
    --model outputs/models/train/weights/best.pt \
    --data coco128.yaml
```

## 核心功能

### ObjectDetector 类

```python
from model import ObjectDetector

# 创建检测器
detector = ObjectDetector(model_size="n")

# 检测图片
detections = detector.predict("image.jpg")
for det in detections:
    print(f"检测到 {len(det['boxes'])} 个物体")
    for name, score in zip(det['class_names'], det['scores']):
        print(f"  {name}: {score:.2f}")

# 检测并绘制
result_image = detector.predict_and_draw(
    "image.jpg",
    save_path="result.jpg"
)

# 视频检测
detector.predict_video(0)  # 摄像头
detector.predict_video("video.mp4", save_path="result.mp4")

# 导出模型
detector.export("onnx")
```

### 命令行接口

| 命令                              | 描述         |
| --------------------------------- | ------------ |
| `main.py demo`                    | 快速演示     |
| `main.py predict --source <路径>` | 目标检测     |
| `main.py train --data <yaml>`     | 训练模型     |
| `main.py eval --model <模型路径>` | 评估模型     |
| `main.py info`                    | 显示系统信息 |

## 自定义数据集训练

### 1. 准备数据集目录

```bash
python -c "
from dataset import create_custom_dataset_structure
create_custom_dataset_structure(
    'data/my_dataset',
    ['cat', 'dog', 'bird']
)
"
```

### 2. 添加图片和标注

将图片放入 `images/train` 和 `images/val`，标注放入 `labels/train` 和 `labels/val`。

标注格式 (每行一个物体):

```
<class_id> <x_center> <y_center> <width> <height>
```

### 3. 训练

```bash
python projects/phase-10-object-detection/train.py \
    --data data/my_dataset/dataset.yaml \
    --model n \
    --epochs 100
```

## 学习要点

1. **YOLOv8 核心概念**
   - 单阶段检测器 (One-Stage Detector)
   - Anchor-Free 设计
   - 多尺度特征融合 (FPN + PAN)
   - 解耦头 (Decoupled Head)

2. **目标检测流程**
   - 数据准备与标注
   - 模型训练与微调
   - 模型评估 (mAP, Precision, Recall)
   - 推理与部署

3. **实用技巧**
   - 数据增强 (Mosaic, MixUp)
   - 迁移学习
   - 混合精度训练
   - 模型导出 (ONNX, TensorRT)

## 实验结果

使用 YOLOv8n 在 COCO128 上训练:

| 指标         | 值             |
| ------------ | -------------- |
| mAP@0.5      | ~45%           |
| mAP@0.5:0.95 | ~30%           |
| 推理速度     | ~200 FPS (GPU) |

## 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLO 论文系列](https://arxiv.org/abs/1506.02640)
- [Phase 10 课程笔记](../../src/phase-10-cv-applications/CONCEPT.md)
