"""
R-CNN 系列检测器 (R-CNN Family)
================================

学习目标：
    1. 理解两阶段检测器的工作原理
    2. 掌握 R-CNN → Fast R-CNN → Faster R-CNN 的演进
    3. 理解 RPN 和 RoI Pooling/Align 的原理
    4. 使用 PyTorch 预训练的 Faster R-CNN

核心概念：
    - 区域提议 (Region Proposal): 候选区域生成
    - RPN (Region Proposal Network): 用网络生成候选区域
    - RoI Pooling/Align: 固定特征图大小
    - 两阶段检测: 先提议后分类

前置知识：
    - 01-detection-basics.py: 目标检测基础
    - 02-yolo-v8.py: YOLO 检测器
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np


# ==================== 第一部分：R-CNN 演进 ====================


def introduction():
    """R-CNN 演进"""
    print("=" * 60)
    print("第一部分：R-CNN 系列演进")
    print("=" * 60)

    print("""
R-CNN 发展历程：

    ┌─────────────────────────────────────────────────────────┐
    │  R-CNN (2014) → Fast R-CNN (2015) → Faster R-CNN (2015) │
    │     ↓              ↓                    ↓               │
    │    慢             较快                  快               │
    │  (47s/图)       (2s/图)             (0.2s/图)            │
    └─────────────────────────────────────────────────────────┘

各版本特点：

    ┌──────────────────────────────────────────────────────────┐
    │  版本           关键创新                  瓶颈            │
    ├──────────────────────────────────────────────────────────┤
    │  R-CNN          CNN提取特征+SVM分类      2000次CNN前向    │
    │  Fast R-CNN     共享特征图+RoI Pooling   选择性搜索慢     │
    │  Faster R-CNN   RPN生成候选区域          端到端训练      │
    └──────────────────────────────────────────────────────────┘

两阶段检测 vs 单阶段检测：

    两阶段 (R-CNN系列):
    ├── 第一阶段: 生成候选区域 (Region Proposals)
    ├── 第二阶段: 对候选区域进行分类和回归
    ├── 优点: 精度高，特别是小目标
    └── 缺点: 速度较慢

    单阶段 (YOLO系列):
    ├── 直接预测边界框和类别
    ├── 优点: 速度快
    └── 缺点: 小目标检测相对较弱
    """)


# ==================== 第二部分：R-CNN 原理 ====================


def rcnn_basics():
    """R-CNN 原理"""
    print("\n" + "=" * 60)
    print("第二部分：R-CNN 原理")
    print("=" * 60)

    print("""
R-CNN (Regions with CNN features) 流程：

    ┌─────────────────────────────────────────────────────────┐
    │  1. 输入图像                                             │
    │      ↓                                                   │
    │  2. 选择性搜索 (Selective Search) 生成 ~2000 个候选区域   │
    │      ↓                                                   │
    │  3. 每个区域缩放到固定大小 (如 227×227)                   │
    │      ↓                                                   │
    │  4. 送入 CNN 提取特征 (如 AlexNet)                       │
    │      ↓                                                   │
    │  5. SVM 分类 + 边界框回归                                │
    └─────────────────────────────────────────────────────────┘

选择性搜索 (Selective Search)：
    - 基于颜色、纹理、大小等合并相邻区域
    - 生成多尺度的候选区域
    - 问题: 计算量大，无法端到端训练

R-CNN 的问题：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 每个候选区域都要过一次 CNN → 2000次前向传播!           │
    │ 2. 选择性搜索在 CPU 上运行，很慢                         │
    │ 3. 训练分多阶段: CNN、SVM、回归器分别训练                 │
    │ 4. 需要大量磁盘空间存储特征                              │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第三部分：Fast R-CNN ====================


def fast_rcnn():
    """Fast R-CNN"""
    print("\n" + "=" * 60)
    print("第三部分：Fast R-CNN")
    print("=" * 60)

    print("""
Fast R-CNN 的改进：

    核心改进: 整张图只过一次 CNN，在特征图上提取区域特征

    ┌─────────────────────────────────────────────────────────┐
    │  1. 输入图像                                             │
    │      ↓                                                   │
    │  2. CNN 提取整张图的特征图 (一次前向传播!)                │
    │      ↓                                                   │
    │  3. 选择性搜索生成候选区域 (在原图上)                     │
    │      ↓                                                   │
    │  4. RoI Pooling: 从特征图上提取固定大小的特征             │
    │      ↓                                                   │
    │  5. 全连接层 → 分类 + 边界框回归 (端到端训练)             │
    └─────────────────────────────────────────────────────────┘

RoI Pooling (Region of Interest Pooling)：

    将任意大小的 RoI 转换为固定大小的特征

    输入: 特征图上的 RoI 区域 (大小不固定)
    输出: 固定大小的特征图 (如 7×7)

    ┌─────────────────┐         ┌───┬───┐
    │    RoI 区域     │  ───→   │   │   │  固定 2×2
    │   (任意大小)    │ 划分+池化│   │   │
    │                 │         ├───┼───┤
    │                 │         │   │   │
    └─────────────────┘         └───┴───┘

    步骤:
    1. 将 RoI 划分为 H×W 个格子
    2. 对每个格子做 Max Pooling
    3. 得到 H×W 的固定输出
    """)

    # RoI Pooling 示例
    print("示例: RoI Pooling\n")

    # 模拟特征图
    feature_map = torch.randn(1, 256, 50, 50)  # [B, C, H, W]

    # 定义 RoI (x1, y1, x2, y2) - 在特征图坐标系中
    # 格式: [batch_index, x1, y1, x2, y2]
    rois = torch.tensor([[0, 10.0, 10.0, 30.0, 30.0], [0, 5.0, 5.0, 25.0, 40.0]])

    # 使用 torchvision 的 RoI Pooling
    from torchvision.ops import roi_pool

    output = roi_pool(feature_map, rois, output_size=(7, 7), spatial_scale=1.0)

    print(f"特征图大小: {feature_map.shape}")
    print(f"RoI 数量: {len(rois)}")
    print(f"RoI Pooling 输出: {output.shape}")


# ==================== 第四部分：Faster R-CNN ====================


def faster_rcnn():
    """Faster R-CNN"""
    print("\n" + "=" * 60)
    print("第四部分：Faster R-CNN")
    print("=" * 60)

    print("""
Faster R-CNN 核心创新: RPN (Region Proposal Network)

    用神经网络替代选择性搜索，实现端到端训练！

    ┌─────────────────────────────────────────────────────────┐
    │                    Faster R-CNN 架构                    │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  输入图像                                                │
    │      ↓                                                  │
    │  Backbone (ResNet/VGG) → 特征图                         │
    │      ↓                                                  │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  RPN (Region Proposal Network)                  │    │
    │  │  - 在特征图上滑动 3×3 窗口                        │    │
    │  │  - 每个位置预测 k 个 Anchor                      │    │
    │  │  - 输出: 前景/背景得分 + 边界框调整               │    │
    │  └─────────────────────────────────────────────────┘    │
    │      ↓                                                  │
    │  候选区域 (约 300 个)                                    │
    │      ↓                                                  │
    │  RoI Pooling → FC → 分类 + 精细边界框回归                │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

RPN 详解：

    特征图上每个位置:
    - 预设 k 个 Anchor (不同尺度和比例)
    - 预测每个 Anchor: 是否包含物体 (2类) + 边界框调整 (4值)

    Anchor 设计:
    ┌──────────────────────────────────────────────────────────┐
    │  尺度 (scales): [128, 256, 512]                          │
    │  比例 (ratios): [0.5, 1, 2]                              │
    │  共 3×3=9 个 Anchor                                      │
    │                                                          │
    │  ┌─┐  ┌──┐  ┌───┐                                       │
    │  │ │  │  │  │   │  不同比例                              │
    │  └─┘  └──┘  └───┘                                       │
    │  2:1   1:1   1:2                                        │
    └──────────────────────────────────────────────────────────┘

RoI Align vs RoI Pooling：

    RoI Pooling 问题: 量化误差
    - 将浮点坐标量化为整数会丢失精度
    - 对分割任务影响特别大

    RoI Align 解决:
    - 使用双线性插值，不进行量化
    - 保持更精确的空间对应关系
    """)


# ==================== 第五部分：Faster R-CNN 使用 ====================


def faster_rcnn_usage():
    """Faster R-CNN 使用"""
    print("\n" + "=" * 60)
    print("第五部分：使用预训练 Faster R-CNN")
    print("=" * 60)

    print("示例1: 加载预训练模型并推理\n")

    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # COCO 类别名称
    COCO_CLASSES = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    print(f"模型加载成功！")
    print(f"支持的类别数: {len(COCO_CLASSES)}")

    # 模拟推理
    print("\n模拟推理:")
    dummy_image = torch.randn(3, 480, 640)  # [C, H, W]

    with torch.no_grad():
        predictions = model([dummy_image])

    print(f"输入大小: {dummy_image.shape}")
    print(f"输出键: {predictions[0].keys()}")
    print(f"  - boxes: 边界框坐标")
    print(f"  - labels: 类别标签")
    print(f"  - scores: 置信度分数")


def custom_faster_rcnn():
    """自定义 Faster R-CNN"""
    print("\n示例2: 自定义类别数\n")

    def get_model(num_classes):
        """获取自定义类别数的 Faster R-CNN"""
        # 加载预训练模型
        model = fasterrcnn_resnet50_fpn(pretrained=True)

        # 替换分类头
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    # 创建 5 类别的模型 (包括背景)
    model = get_model(num_classes=5)
    print("自定义 Faster R-CNN 创建成功!")
    print(f"类别数: 5 (包括背景)")

    # 查看模型结构
    print("\n模型关键组件:")
    print(f"  Backbone: ResNet50 + FPN")
    print(f"  RPN Anchor 生成器")
    print(f"  RoI Heads: 分类 + 边界框回归")


# ==================== 第六部分：训练自定义数据集 ====================


def training_custom():
    """训练自定义数据集"""
    print("\n" + "=" * 60)
    print("第六部分：训练自定义数据集")
    print("=" * 60)

    print("""
自定义数据集类：
    """)

    from torch.utils.data import Dataset

    class CustomDetectionDataset(Dataset):
        """自定义目标检测数据集"""

        def __init__(self, images, annotations, transforms=None):
            self.images = images
            self.annotations = annotations
            self.transforms = transforms

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            # 加载图像
            image = self.images[idx]

            # 获取标注
            ann = self.annotations[idx]
            boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
            labels = torch.as_tensor(ann["labels"], dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

            if self.transforms:
                image = self.transforms(image)

            return image, target

    print("数据集类定义完成!")

    print("""
训练循环示例:

>>> model = get_model(num_classes=5)
>>> model.to(device)
>>> 
>>> optimizer = torch.optim.SGD(
>>>     model.parameters(),
>>>     lr=0.005,
>>>     momentum=0.9,
>>>     weight_decay=0.0005
>>> )
>>> 
>>> for epoch in range(num_epochs):
>>>     model.train()
>>>     for images, targets in dataloader:
>>>         images = [img.to(device) for img in images]
>>>         targets = [{k: v.to(device) for k, v in t.items()} 
>>>                    for t in targets]
>>>         
>>>         # 前向传播 (训练模式返回损失)
>>>         loss_dict = model(images, targets)
>>>         losses = sum(loss_dict.values())
>>>         
>>>         # 反向传播
>>>         optimizer.zero_grad()
>>>         losses.backward()
>>>         optimizer.step()
>>>         
>>>     print(f'Epoch {epoch}, Loss: {losses.item():.4f}')
    """)


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：Faster R-CNN 推理
    任务: 加载预训练模型，对真实图片进行检测
    要求: 绘制检测框和类别标签

练习 1 答案：
    import torch
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision import transforms
    
    # 加载模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # 加载图片
    image = cv2.imread('image.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换为 tensor
    transform = transforms.ToTensor()
    img_tensor = transform(image_rgb)
    
    # 推理
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # 可视化
    fig, ax = plt.subplots(1)
    ax.imshow(image_rgb)
    
    for box, label, score in zip(predictions[0]['boxes'],
                                  predictions[0]['labels'],
                                  predictions[0]['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=2, edgecolor='r',
                                      facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{COCO_CLASSES[label]}: {score:.2f}')
    
    plt.savefig('faster_rcnn_result.png')

练习 2：理解 RPN
    任务: 打印 RPN 的 Anchor 数量和分布
    提示: 查看 model.rpn.anchor_generator

练习 2 答案：
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 查看 Anchor 生成器
    anchor_generator = model.rpn.anchor_generator
    
    # 打印尺度和比例
    print('Anchor sizes:', anchor_generator.sizes)
    # ((32,), (64,), (128,), (256,), (512,))
    
    print('Anchor ratios:', anchor_generator.aspect_ratios)
    # ((0.5, 1.0, 2.0), ...) 每个尺度 3 种比例
    
    # 计算总 Anchor 数量
    # 假设特征图大小为 [H, W]
    # 每个位置 3 个 Anchor (3种比例)
    # 5 个尺度的特征图 (FPN)
    # 总数 = sum(H_i * W_i * 3)

练习 3：自定义训练
    任务: 在 Pascal VOC 数据集上微调 Faster R-CNN
    要求: 记录训练损失和 mAP

练习 3 答案：
    from torchvision.datasets import VOCDetection
    from torch.utils.data import DataLoader
    
    # 1. 准备数据集
    dataset = VOCDetection(root='./data', year='2012',
                           image_set='train', download=True)
    
    # 2. 自定义 collate_fn
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=True, collate_fn=collate_fn)
    
    # 3. 修改模型 (VOC 20类 + 背景)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)
    
    # 4. 训练循环
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                 momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(10):
        model.train()
        for images, targets in dataloader:
            # 转换数据格式...
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

练习 4：RoI Align vs RoI Pooling
    任务: 对比两种方法的输出差异
    使用 torchvision.ops.roi_align 和 roi_pool

练习 4 答案：
    from torchvision.ops import roi_align, roi_pool
    
    # 创建特征图
    feature_map = torch.randn(1, 256, 50, 50)
    
    # 定义 RoI (浮点坐标)
    rois = torch.tensor([[0, 10.5, 10.5, 30.7, 30.7]])
    
    # RoI Pooling (会量化坐标)
    out_pool = roi_pool(feature_map, rois,
                        output_size=(7, 7), spatial_scale=1.0)
    
    # RoI Align (保持浮点精度)
    out_align = roi_align(feature_map, rois,
                          output_size=(7, 7), spatial_scale=1.0,
                          sampling_ratio=2)
    
    # 比较差异
    diff = (out_pool - out_align).abs()
    print(f'平均差异: {diff.mean():.4f}')
    print(f'最大差异: {diff.max():.4f}')
    
    # RoI Align 在边界附近差异更大
    # 对分割任务影响显著

练习 5：检测速度测试
    任务: 测量 Faster R-CNN 和 YOLOv8 的推理速度
    比较: 不同输入分辨率下的 FPS

练习 5 答案：
    import time
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from ultralytics import YOLO
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Faster R-CNN
    faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    
    # YOLOv8
    yolo = YOLO('yolov8n.pt')
    
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    
    for h, w in resolutions:
        # Faster R-CNN 速度
        img = torch.randn(3, h, w).to(device)
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                faster_rcnn([img])
        frcnn_fps = 50 / (time.time() - start)
        
        # YOLOv8 速度
        img_np = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        start = time.time()
        for _ in range(50):
            yolo(img_np, verbose=False)
        yolo_fps = 50 / (time.time() - start)
        
        print(f'{w}x{h}: Faster R-CNN={frcnn_fps:.1f}fps, YOLO={yolo_fps:.1f}fps')

思考题 1：为什么 Faster R-CNN 比 YOLO 更准确？
    特别是在小目标检测上

思考题 1 答案：
    1. 两阶段设计
       - RPN 先筛选候选区域
       - 第二阶段精细分类和回归
       - 相当于对潜在目标进行两次处理
    
    2. RoI 级别的处理
       - 每个候选区域单独处理
       - 可以对小目标进行放大处理
       - 不会因为全局下采样丢失信息
    
    3. 更精细的特征
       - RoI Align 保持空间精度
       - 独立的分类和回归分支
    
    4. 阈值设计
       - RPN 可以使用较低阈值保证召回
       - 第二阶段再精细筛选

思考题 2：RPN 训练时如何定义正负样本？
    IoU 阈值是多少？

思考题 2 答案：
    RPN 的正负样本定义：
    
    正样本 (标签=1)：
    - 与任意 GT 的 IoU > 0.7 的 Anchor
    - 或者与某个 GT 有最大 IoU 的 Anchor
    
    负样本 (标签=0)：
    - 与所有 GT 的 IoU < 0.3 的 Anchor
    
    忽略样本 (标签=-1)：
    - IoU 在 0.3 到 0.7 之间
    - 不参与损失计算
    
    采样策略：
    - 每张图采样 256 个 Anchor
    - 正负比例约 1:1
    - 正样本不足时用负样本补充

思考题 3：Faster R-CNN 的损失函数包含哪些部分？
    RPN 损失 + Detection 损失

思考题 3 答案：
    Faster R-CNN 的多任务损失：
    
    L = L_rpn + L_detection
    
    RPN 损失：
    - L_rpn_cls: 二分类交叉熵 (前景/背景)
    - L_rpn_reg: Smooth L1 (边界框回归)
    
    Detection Head 损失：
    - L_cls: 多分类交叉熵 (N+1 类)
    - L_reg: Smooth L1 (精细边界框回归)
    
    总损失公式：
    L = λ1 * L_rpn_cls + λ2 * L_rpn_reg + 
        λ3 * L_cls + λ4 * L_reg
    
    通常 λ1 = λ2 = λ3 = λ4 = 1
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    rcnn_basics()
    fast_rcnn()
    faster_rcnn()
    faster_rcnn_usage()
    custom_faster_rcnn()
    training_custom()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 04-semantic-segmentation.py: 语义分割

关键要点回顾：
    ✓ R-CNN 系列是两阶段检测器
    ✓ Faster R-CNN 用 RPN 替代选择性搜索
    ✓ RoI Pooling 将不同大小的区域转为固定大小
    ✓ RoI Align 使用双线性插值避免量化误差
    ✓ torchvision 提供预训练的 Faster R-CNN
    """)


if __name__ == "__main__":
    main()
