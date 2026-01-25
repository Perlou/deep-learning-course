"""
实例分割 (Instance Segmentation)
==================================

学习目标：
    1. 理解实例分割与语义分割的区别
    2. 掌握 Mask R-CNN 的工作原理
    3. 了解 RoI Align 的作用
    4. 使用预训练模型进行实例分割

核心概念：
    - 实例分割: 检测 + 分割，区分不同实例
    - Mask R-CNN: Faster R-CNN + 分割分支
    - RoI Align: 精确的区域特征提取
    - 实例掩码: 每个实例单独的分割掩码

前置知识：
    - 03-rcnn-family.py: R-CNN 系列
    - 04-semantic-segmentation.py: 语义分割
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：实例分割概述 ====================


def introduction():
    """实例分割概述"""
    print("=" * 60)
    print("第一部分：实例分割概述")
    print("=" * 60)

    print("""
实例分割 = 目标检测 + 语义分割

    ┌────────────────────────────────────────────────────────┐
    │  任务对比:                                              │
    │                                                        │
    │  目标检测:                                              │
    │  ┌───────────────────┐                                 │
    │  │ ┌────┐    ┌────┐  │                                 │
    │  │ │dog1│    │dog2│  │  只有边界框                     │
    │  │ └────┘    └────┘  │                                 │
    │  └───────────────────┘                                 │
    │                                                        │
    │  语义分割:                                              │
    │  ┌───────────────────┐                                 │
    │  │ 🐕🐕🐕  🐕🐕🐕   │  像素级分类                      │
    │  │ 🐕🐕🐕  🐕🐕🐕   │  但不区分实例                    │
    │  └───────────────────┘                                 │
    │                                                        │
    │  实例分割:                                              │
    │  ┌───────────────────┐                                 │
    │  │ 🔴🔴🔴  🔵🔵🔵   │  像素级分类                      │
    │  │ 🔴🔴🔴  🔵🔵🔵   │  AND 区分不同实例                │
    │  └───────────────────┘  (狗1红色, 狗2蓝色)             │
    └────────────────────────────────────────────────────────┘

应用场景：
    - 自动驾驶: 区分不同的车辆和行人
    - 机器人: 物体抓取需要知道每个物体的精确轮廓
    - 医学: 区分每个细胞或器官
    - 图像编辑: 精确选择特定物体

主流方法：
    ┌──────────────────────────────────────────────────────────┐
    │  方法              原理                    代表模型       │
    ├──────────────────────────────────────────────────────────┤
    │  自顶向下          先检测后分割             Mask R-CNN    │
    │  自底向上          先分割后聚合             YOLACT        │
    │  单阶段            直接预测掩码             SOLO, YOLACT  │
    └──────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：Mask R-CNN ====================


def mask_rcnn():
    """Mask R-CNN"""
    print("\n" + "=" * 60)
    print("第二部分：Mask R-CNN 架构")
    print("=" * 60)

    print("""
Mask R-CNN = Faster R-CNN + 分割分支

    ┌─────────────────────────────────────────────────────────┐
    │                    Mask R-CNN 架构                      │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  输入图像                                                │
    │      ↓                                                  │
    │  Backbone (ResNet + FPN)                                │
    │      ↓                                                  │
    │  RPN → 候选区域                                          │
    │      ↓                                                  │
    │  RoI Align (而非 RoI Pooling!)                          │
    │      ↓                                                  │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │           RoI 特征 (14×14×256)                  │    │
    │  └────────────────┬──────────────────────────────┬─┘    │
    │                   ↓                              ↓      │
    │            ┌──────────────┐              ┌──────────────┐
    │            │  Box Head    │              │  Mask Head   │
    │            │  分类 + 回归 │              │  分割预测    │
    │            └──────────────┘              └──────────────┘
    │                   ↓                              ↓      │
    │            边界框 + 类别                  28×28 掩码    │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

Mask Head 结构：

    RoI 特征 (14×14×256)
         ↓
    4 × Conv3×3 (256 通道)
         ↓
    ConvTranspose 2×2 (上采样)
         ↓
    Conv 1×1 (K 类)
         ↓
    输出掩码 (28×28×K)

    每个类别预测一个独立的二值掩码
    """)


# ==================== 第三部分：RoI Align ====================


def roi_align():
    """RoI Align 详解"""
    print("\n" + "=" * 60)
    print("第三部分：RoI Align vs RoI Pooling")
    print("=" * 60)

    print("""
RoI Pooling 的问题：

    量化 (Quantization) 导致空间信息丢失

    例如: RoI 在原图上是 [10.5, 20.3, 50.8, 100.2]

    RoI Pooling 步骤:
    1. 量化为整数 → [10, 20, 50, 100]  (丢失精度!)
    2. 划分成网格 → 每个格子再量化
    3. Max Pooling

    对分类影响不大，但对分割影响很大！

RoI Align 解决方案：

    使用双线性插值，不进行量化

    步骤:
    1. 保持浮点坐标
    2. 在每个输出格子中采样 4 个点
    3. 使用双线性插值计算采样点的值
    4. 对 4 个采样点取平均或最大

    ┌─────────────────────────────────────────────────────────┐
    │  RoI 区域 (浮点坐标)                                    │
    │  ┌──────────────────┐                                   │
    │  │  ●  ●  │  ●  ●  │  ← 每个格子采样 4 个点            │
    │  │  ●  ●  │  ●  ●  │    使用双线性插值                 │
    │  ├────────┼────────┤                                   │
    │  │  ●  ●  │  ●  ●  │                                   │
    │  │  ●  ●  │  ●  ●  │                                   │
    │  └──────────────────┘                                   │
    └─────────────────────────────────────────────────────────┘
    """)

    # 代码示例
    print("示例: 使用 RoI Align\n")

    from torchvision.ops import roi_align

    # 特征图
    feature_map = torch.randn(1, 256, 50, 50)

    # RoI (batch_idx, x1, y1, x2, y2)
    rois = torch.tensor(
        [[0, 10.5, 10.5, 30.7, 30.7], [0, 5.2, 5.2, 25.8, 40.3]], dtype=torch.float32
    )

    # RoI Align
    output = roi_align(
        feature_map,
        rois,
        output_size=(14, 14),  # 输出大小
        spatial_scale=1.0,  # 特征图相对原图的缩放
        sampling_ratio=2,  # 每个格子采样点数
    )

    print(f"特征图: {feature_map.shape}")
    print(f"RoI 数量: {len(rois)}")
    print(f"RoI Align 输出: {output.shape}")


# ==================== 第四部分：使用预训练模型 ====================


def pretrained_model():
    """使用预训练模型"""
    print("\n" + "=" * 60)
    print("第四部分：使用预训练 Mask R-CNN")
    print("=" * 60)

    print("加载预训练模型:\n")

    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn

        # 加载模型
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        print("Mask R-CNN-ResNet50-FPN 加载成功!")

        # 推理示例
        print("\n推理示例:\n")

        # 模拟输入
        dummy_image = torch.randn(3, 480, 640)

        with torch.no_grad():
            predictions = model([dummy_image])

        pred = predictions[0]
        print(f"输出键: {pred.keys()}")
        print(f"  - boxes: 边界框 {pred['boxes'].shape}")
        print(f"  - labels: 类别 {pred['labels'].shape}")
        print(f"  - scores: 置信度 {pred['scores'].shape}")
        print(f"  - masks: 实例掩码 {pred['masks'].shape}")

        print("""
掩码格式:
    masks shape: [N, 1, H, W]
    - N: 检测到的实例数
    - 1: 单通道 (二值掩码)
    - H, W: 与输入图像相同分辨率

    掩码值为 [0, 1] 的概率
    通常用 0.5 作为阈值二值化
        """)

    except Exception as e:
        print(f"加载模型失败: {e}")


# ==================== 第五部分：自定义训练 ====================


def custom_training():
    """自定义训练"""
    print("\n" + "=" * 60)
    print("第五部分：自定义数据集训练")
    print("=" * 60)

    print("""
实例分割数据集格式：

    每个图像需要:
    - 边界框: [N, 4]
    - 类别标签: [N]
    - 实例掩码: [N, H, W]
    """)

    from torch.utils.data import Dataset

    class InstanceSegDataset(Dataset):
        """实例分割数据集示例"""

        def __init__(self, images, annotations, transforms=None):
            self.images = images
            self.annotations = annotations
            self.transforms = transforms

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            ann = self.annotations[idx]

            # 构建 target
            target = {}
            target["boxes"] = torch.as_tensor(ann["boxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(ann["labels"], dtype=torch.int64)
            target["masks"] = torch.as_tensor(ann["masks"], dtype=torch.uint8)
            target["image_id"] = torch.tensor([idx])

            if self.transforms:
                image = self.transforms(image)

            return image, target

    print("数据集类定义完成!")

    print("""
训练循环:

>>> from torchvision.models.detection import maskrcnn_resnet50_fpn
>>> 
>>> model = maskrcnn_resnet50_fpn(pretrained=True)
>>> 
>>> # 修改分类头
>>> in_features = model.roi_heads.box_predictor.cls_score.in_features
>>> model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
>>> 
>>> # 修改 Mask 头
>>> in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
>>> model.roi_heads.mask_predictor = MaskRCNNPredictor(
>>>     in_features_mask, 256, num_classes
>>> )
>>> 
>>> # 训练
>>> for images, targets in dataloader:
>>>     loss_dict = model(images, targets)
>>>     losses = sum(loss_dict.values())
>>>     
>>>     optimizer.zero_grad()
>>>     losses.backward()
>>>     optimizer.step()
    """)


# ==================== 第六部分：可视化 ====================


def visualization():
    """结果可视化"""
    print("\n" + "=" * 60)
    print("第六部分：结果可视化")
    print("=" * 60)

    print("""
可视化实例分割结果:
    """)

    def visualize_instance_segmentation(
        image, boxes, masks, labels, scores, class_names
    ):
        """可视化实例分割结果"""
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 显示原图
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        ax.imshow(image)

        # 随机颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(boxes)))

        for i, (box, mask, label, score) in enumerate(
            zip(boxes, masks, labels, scores)
        ):
            if score < 0.5:
                continue

            color = colors[i]

            # 绘制边界框
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            # 绘制类别和分数
            ax.text(
                x1,
                y1 - 5,
                f"{class_names[label]}: {score:.2f}",
                color="white",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

            # 绘制掩码
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            mask = mask > 0.5  # 二值化
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = (*color[:3], 0.5)  # RGBA
            ax.imshow(colored_mask)

        ax.axis("off")
        return fig

    print("可视化函数定义完成!")
    print(
        "使用方法: visualize_instance_segmentation(image, boxes, masks, labels, scores, class_names)"
    )


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：使用 Mask R-CNN
    任务: 加载预训练模型，对真实图像进行实例分割
    要求: 可视化边界框和掩码

练习 1 答案：
    import torch
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision import transforms
    
    # 加载模型
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # 加载图片
    image = cv2.imread('image.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 推理
    transform = transforms.ToTensor()
    img_tensor = transform(image_rgb)
    
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # 可视化
    pred = predictions[0]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    for i in range(len(pred['boxes'])):
        if pred['scores'][i] < 0.5:
            continue
        
        # 边界框
        box = pred['boxes'][i].numpy()
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                    fill=False, color='red', linewidth=2))
        
        # 掩码
        mask = pred['masks'][i, 0].numpy()
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0.5] = [1, 0, 0, 0.5]  # 红色半透明
        ax.imshow(colored_mask)
    
    plt.savefig('mask_rcnn_result.png')

练习 2：理解 RoI Align
    任务: 对比 RoI Pooling 和 RoI Align 的输出差异
    使用 torchvision.ops 中的函数

练习 2 答案：
    from torchvision.ops import roi_align, roi_pool
    import torch
    
    # 创建特征图
    feature_map = torch.randn(1, 256, 100, 100)
    
    # RoI 使用浮点坐标
    rois = torch.tensor([
        [0, 10.3, 20.7, 40.8, 60.2],  # 浮点坐标
        [0, 50.1, 30.9, 80.6, 70.4]
    ], dtype=torch.float32)
    
    # RoI Pooling
    out_pool = roi_pool(feature_map, rois, output_size=(14, 14), spatial_scale=1.0)
    
    # RoI Align
    out_align = roi_align(feature_map, rois, output_size=(14, 14), 
                          spatial_scale=1.0, sampling_ratio=2)
    
    # 对比
    diff = (out_pool - out_align).abs()
    print(f'RoI Pooling 输出: {out_pool.shape}')
    print(f'RoI Align 输出: {out_align.shape}')
    print(f'平均差异: {diff.mean():.4f}')
    print(f'最大差异: {diff.max():.4f}')
    
    # 结论: RoI Align 保持了更精确的空间对应

练习 3：自定义训练
    任务: 在 COCO 子集上微调 Mask R-CNN
    记录: 训练损失、mAP、Mask AP

练习 3 答案：
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    def get_instance_segmentation_model(num_classes):
        # 加载预训练模型
        model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # 替换 Box 预测头
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 替换 Mask 预测头
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        return model
    
    # 训练循环
    model = get_instance_segmentation_model(num_classes=3)  # 背景+2类
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                 momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(10):
        model.train()
        for images, targets in dataloader:
            # targets 需包含 boxes, labels, masks
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print(f'Epoch {epoch}: {losses.item():.4f}')

练习 4：后处理优化
    任务: 实现实例掩码的后处理
    包含: NMS、掩码平滑、面积过滤

练习 4 答案：
    import numpy as np
    import cv2
    
    def post_process_masks(boxes, masks, scores, labels,
                           score_thresh=0.5, mask_thresh=0.5,
                           min_area=100):
        '''后处理实例分割结果'''
        results = []
        
        for i in range(len(boxes)):
            if scores[i] < score_thresh:
                continue
            
            # 1. 掩码二值化
            mask = (masks[i] > mask_thresh).astype(np.uint8)
            
            # 2. 形态学平滑
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 3. 面积过滤
            area = mask.sum()
            if area < min_area:
                continue
            
            # 4. 找轮廓平滑边界
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # 多边形近似
                epsilon = 0.01 * cv2.arcLength(contours[0], True)
                approx = cv2.approxPolyDP(contours[0], epsilon, True)
                
                mask_smooth = np.zeros_like(mask)
                cv2.fillPoly(mask_smooth, [approx], 1)
                mask = mask_smooth
            
            results.append({
                'box': boxes[i],
                'mask': mask,
                'score': scores[i],
                'label': labels[i]
            })
        
        return results

练习 5：单阶段方法对比
    任务: 对比 Mask R-CNN 和 YOLACT
    比较: 速度、精度、实现复杂度

练习 5 答案：
    # Mask R-CNN vs YOLACT 对比
    
    对比维度:
    ┌─────────────────────────────────────────────────────────┐
    │                 Mask R-CNN          YOLACT              │
    ├─────────────────────────────────────────────────────────┤
    │ 类型           两阶段               单阶段               │
    │ 速度           ~5 FPS              ~30 FPS              │
    │ 精度 (AP)      ~37%                ~30%                 │
    │ 复杂度         高                   中                   │
    │ 掩码质量       高                   中                   │
    │ 实时性         否                   是                   │
    └─────────────────────────────────────────────────────────┘
    
    YOLACT 核心思想:
    - 生成原型掩码 (Prototype Masks)
    - 预测组合系数 (Mask Coefficients)
    - 掩码 = 系数 × 原型的线性组合
    
    适用场景:
    - Mask R-CNN: 需要高精度的应用
    - YOLACT: 需要实时性的应用

思考题 1：为什么 Mask Head 对每个类别单独预测掩码？
    而不是对所有类别预测一个统一的掩码

思考题 1 答案：
    原因分析：
    
    1. 解耦分类和分割
       - 分类决定是什么类别
       - 掩码只关注形状，不需要知道类别
       - 两个任务独立，训练更稳定
    
    2. 避免类别竞争
       - 如果所有类别共享掩码
       - 不同类别的像素会互相竞争
       - 在重叠区域会产生歧义
    
    3. 更好的泛化
       - 每个类别学习自己的形状特征
       - 猫和狗的轮廓不同
       - 分开预测更灵活
    
    4. 推理时效率不变
       - 只使用分类结果对应的掩码
       - 其他类别的掩码不需要计算损失

思考题 2：RoI Align 为什么对分割任务很重要？
    对检测任务影响大吗？

思考题 2 答案：
    对分割任务的重要性：
    
    1. 像素级精度要求
       - 分割需要精确的空间对应
       - RoI Pooling 的量化会丢失位置信息
       - 掩码边界会变得模糊
    
    2. 量化误差累积
       - RoI 坐标量化一次
       - 划分网格时再量化一次
       - 误差累积导致掩码偏移
    
    对检测任务的影响：
    - 影响较小
    - 检测只需要边界框级别的定位
    - 几个像素的偏差不影响 mAP
    
    实验数据 (Mask R-CNN 论文):
    - 使用 RoI Align vs RoI Pool
    - 检测 AP 提升 ~0.5%
    - 分割 AP 提升 ~3%

思考题 3：实例分割和全景分割的区别是什么？
    如何从实例分割扩展到全景分割？

思考题 3 答案：
    区别：
    
    实例分割:
    - 只分割"things"(可数物体: 人、车、动物)
    - 背景不分割
    - 输出: 每个实例一个掩码
    
    全景分割:
    - 分割所有像素
    - "things" + "stuff"(不可数: 天空、道路、草地)
    - 输出: 每个像素一个 (类别, 实例ID) 标签
    
    扩展方法:
    
    1. Panoptic FPN
       - 共享 backbone
       - 实例分支: Mask R-CNN
       - 语义分支: FCN
       - 融合两个分支的输出
    
    2. 融合规则:
       - Things: 使用实例分割结果
       - Stuff: 使用语义分割结果
       - 处理重叠: Things 优先
    
    3. 端到端方法 (Panoptic SegFormer):
       - 使用 Transformer
       - 统一处理 things 和 stuff
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    mask_rcnn()
    roi_align()
    pretrained_model()
    custom_training()
    visualization()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 07-pose-estimation.py: 姿态估计

关键要点回顾：
    ✓ 实例分割 = 检测 + 分割，区分不同实例
    ✓ Mask R-CNN = Faster R-CNN + Mask Head
    ✓ RoI Align 使用双线性插值避免量化误差
    ✓ Mask Head 对每个类别预测独立的二值掩码
    ✓ 掩码输出形状为 [N, 1, H, W]
    """)


if __name__ == "__main__":
    main()
