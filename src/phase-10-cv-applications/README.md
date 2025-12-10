# Phase 10: 计算机视觉应用

> **目标**：掌握 CV 核心任务  
> **预计时长**：2 周  
> **前置条件**：Phase 1-9 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解目标检测的工作原理
2. 掌握 YOLO 和 R-CNN 系列模型
3. 理解图像分割的不同类型
4. 使用 U-Net 进行语义分割
5. 了解姿态估计和人脸识别基础

---

## 📚 核心概念

### 目标检测

从图像中定位和识别物体：

- **单阶段检测器**: YOLO, SSD（快速）
- **两阶段检测器**: R-CNN 系列（准确）

### 图像分割

- **语义分割**: 每个像素分类（不区分实例）
- **实例分割**: 区分同类的不同实例
- **全景分割**: 语义 + 实例

### 关键架构

| 任务     | 模型         | 特点          |
| -------- | ------------ | ------------- |
| 目标检测 | YOLOv8       | 快速，端到端  |
| 目标检测 | Faster R-CNN | 准确，两阶段  |
| 语义分割 | U-Net        | 编码器-解码器 |
| 实例分割 | Mask R-CNN   | 检测 + 分割   |

---

## 📁 文件列表

| 文件                          | 描述         | 状态 |
| ----------------------------- | ------------ | ---- |
| `01-detection-basics.py`      | 检测任务概述 | ⏳   |
| `02-yolo-v8.py`               | YOLO 系列    | ⏳   |
| `03-rcnn-family.py`           | R-CNN 系列   | ⏳   |
| `04-semantic-segmentation.py` | 语义分割     | ⏳   |
| `05-unet.py`                  | U-Net 架构   | ⏳   |
| `06-instance-segmentation.py` | 实例分割     | ⏳   |
| `07-pose-estimation.py`       | 姿态估计     | ⏳   |
| `08-face-recognition.py`      | 人脸识别     | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-10-cv-applications/01-detection-basics.py
python src/phase-10-cv-applications/02-yolo-v8.py
```

---

## 📖 推荐资源

- [YOLO 官方文档](https://docs.ultralytics.com/)
- 论文：Faster R-CNN, YOLO, U-Net, Mask R-CNN

---

## ✅ 完成检查

- [ ] 理解 IoU 和 NMS
- [ ] 能够使用 YOLOv8 进行检测
- [ ] 理解 Anchor 的概念
- [ ] 能够实现 U-Net 进行分割
- [ ] 理解实例分割和语义分割的区别
- [ ] 完成目标检测项目
