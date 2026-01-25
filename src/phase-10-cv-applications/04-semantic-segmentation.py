"""
语义分割 (Semantic Segmentation)
=================================

学习目标：
    1. 理解语义分割与其他分割任务的区别
    2. 掌握 FCN 和编码器-解码器架构
    3. 了解常用的分割损失函数
    4. 使用预训练模型进行语义分割

核心概念：
    - 语义分割: 对每个像素进行分类
    - 编码器-解码器: 先下采样后上采样
    - 跳跃连接: 保留空间细节
    - Dice Loss: 处理类别不平衡

前置知识：
    - Phase 5: CNN 卷积神经网络
    - 01-03: 目标检测基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：分割任务概述 ====================


def introduction():
    """分割任务概述"""
    print("=" * 60)
    print("第一部分：分割任务概述")
    print("=" * 60)

    print("""
图像分割任务分类：

    ┌─────────────────────────────────────────────────────────┐
    │  语义分割 (Semantic Segmentation)                        │
    │  - 对每个像素分类                                        │
    │  - 不区分同类物体的不同实例                               │
    │  - 输出: H×W×C (C=类别数)                                │
    │                                                         │
    │  ┌───────────────────┐                                  │
    │  │ 🟦🟦🟦🟩🟩🟩🟩  │  蓝色=天空                         │
    │  │ 🟩🐕🐕🐕🐕🟩🟩  │  🐕=狗 (不区分个体)                │
    │  │ 🟩🟩🐕🐕🟩🟩🟩  │                                    │
    │  └───────────────────┘                                  │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │  实例分割 (Instance Segmentation)                        │
    │  - 检测 + 分割                                           │
    │  - 区分同类物体的不同实例                                 │
    │                                                         │
    │  ┌───────────────────┐                                  │
    │  │ 🟦🟦🟦🟩🟩🟩🟩  │                                    │
    │  │ 🟩🔴🔴🔴🟡🟩🟩  │  🔴=狗1, 🟡=狗2                    │
    │  │ 🟩🟩🔴🔴🟡🟡🟩  │  (区分不同个体)                    │
    │  └───────────────────┘                                  │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │  全景分割 (Panoptic Segmentation)                        │
    │  - 语义分割 + 实例分割                                   │
    │  - 对所有像素分类，并区分可数物体的实例                    │
    └─────────────────────────────────────────────────────────┘

语义分割应用：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 自动驾驶: 道路、车辆、行人分割                         │
    │ 2. 医学影像: 器官、病变区域分割                           │
    │ 3. 卫星图像: 土地利用分类                                │
    │ 4. 图像编辑: 背景移除、物体选择                           │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：FCN 架构 ====================


def fcn_architecture():
    """FCN 架构"""
    print("\n" + "=" * 60)
    print("第二部分：FCN (Fully Convolutional Network)")
    print("=" * 60)

    print("""
FCN 核心思想：

    将全连接层替换为卷积层，实现密集预测

    传统 CNN (用于分类):
    ┌──────┐    ┌──────┐    ┌──────┐
    │ Conv │ → │ Conv │ → │  FC  │ → 类别
    └──────┘    └──────┘    └──────┘
                              ↑
                           展平 (丧失空间信息)

    FCN (用于分割):
    ┌──────┐    ┌──────┐    ┌───────┐    ┌──────┐
    │ Conv │ → │ Conv │ → │ Conv  │ → │上采样│ → 分割图
    └──────┘    └──────┘    └───────┘    └──────┘
                           (1×1 卷积)
                           保留空间信息

FCN 变体：

    FCN-32s: 直接 32 倍上采样 (粗糙)
    FCN-16s: 融合 pool4 特征，16 倍上采样
    FCN-8s:  融合 pool3+pool4 特征，8 倍上采样 (最精细)

    ┌─────────────────────────────────────────────────────────┐
    │  FCN-8s 结构:                                           │
    │                                                         │
    │  输入 → Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Pool3   │
    │                                              ↓   ────→  │
    │  Conv4 → Pool4 → Conv5 → Pool5 → FC6 → FC7 → 融合      │
    │           ↓────────────────────────────→     ↓          │
    │                                           8倍上采样     │
    │                                              ↓          │
    │                                           输出分割图     │
    └─────────────────────────────────────────────────────────┘
    """)

    # FCN 简化实现
    print("示例: 简化的 FCN 实现\n")

    class SimpleFCN(nn.Module):
        """简化的 FCN 用于理解概念"""

        def __init__(self, num_classes=21):
            super().__init__()

            # 编码器 (使用 VGG 风格)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            # 1×1 卷积替代 FC
            self.classifier = nn.Conv2d(256, num_classes, 1)

            # 上采样
            self.upsample = nn.Upsample(
                scale_factor=8, mode="bilinear", align_corners=True
            )

        def forward(self, x):
            x = self.conv1(x)  # 1/2
            x = self.conv2(x)  # 1/4
            x = self.conv3(x)  # 1/8
            x = self.classifier(x)  # 分类
            x = self.upsample(x)  # 恢复分辨率
            return x

    model = SimpleFCN(num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}  (每个像素 21 类)")


# ==================== 第三部分：编码器-解码器架构 ====================


def encoder_decoder():
    """编码器-解码器架构"""
    print("\n" + "=" * 60)
    print("第三部分：编码器-解码器架构")
    print("=" * 60)

    print("""
编码器-解码器 (Encoder-Decoder)：

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  输入                                              输出  │
    │  H×W                                               H×W  │
    │    │                                                ↑   │
    │    ▼                                                │   │
    │  ┌────┐                                          ┌────┐ │
    │  │    │    编码器                      解码器    │    │ │
    │  │    │   (下采样)                   (上采样)    │    │ │
    │  └────┘                                          └────┘ │
    │    │                                                ↑   │
    │    ▼                                                │   │
    │  ┌────┐                                          ┌────┐ │
    │  │    │                                          │    │ │
    │  └────┘                                          └────┘ │
    │    │                                                ↑   │
    │    ▼                                                │   │
    │  ┌────┐ ─────────────────────────────────────────→ ┌────┐ │
    │  │    │              瓶颈层                       │    │ │
    │  └────┘                                          └────┘ │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

编码器作用:
    - 逐步降低空间分辨率
    - 增加通道数
    - 提取高级语义特征

解码器作用:
    - 逐步恢复空间分辨率
    - 减少通道数
    - 生成精细的分割图

上采样方法：

    1. 转置卷积 (Transposed Convolution)
       - 可学习的上采样
       - 可能产生棋盘效应

    2. 双线性插值 + 卷积
       - 先插值放大，再卷积
       - 更平滑

    3. 反池化 (Unpooling)
       - 记录池化位置，在相应位置恢复
    """)

    # 转置卷积示例
    print("示例: 转置卷积上采样\n")

    upsample = nn.ConvTranspose2d(
        in_channels=256,
        out_channels=128,
        kernel_size=2,
        stride=2,  # 2倍上采样
    )

    x = torch.randn(1, 256, 16, 16)
    out = upsample(x)
    print(f"输入: {x.shape}")
    print(f"转置卷积后: {out.shape}")


# ==================== 第四部分：分割损失函数 ====================


def segmentation_losses():
    """分割损失函数"""
    print("\n" + "=" * 60)
    print("第四部分：分割损失函数")
    print("=" * 60)

    print("""
常用分割损失函数：

    1. 交叉熵损失 (Cross Entropy Loss)
       - 逐像素分类
       - 问题: 类别不平衡时效果差

    2. Dice Loss
       - 基于 Dice 系数
       - 对类别不平衡更鲁棒
       - 公式: Dice = 2|A∩B| / (|A|+|B|)

    3. 组合损失
       - CE Loss + Dice Loss
       - 兼顾像素级准确和区域重叠
    """)

    # Dice Loss 实现
    print("示例: Dice Loss 实现\n")

    class DiceLoss(nn.Module):
        """Dice Loss for segmentation"""

        def __init__(self, smooth=1e-5):
            super().__init__()
            self.smooth = smooth

        def forward(self, pred, target):
            """
            Args:
                pred: [B, C, H, W] 预测 (softmax 后)
                target: [B, C, H, W] 标签 (one-hot)
            """
            # 展平
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)

            # 计算 Dice
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()

            dice = (2 * intersection + self.smooth) / (union + self.smooth)

            return 1 - dice

    class FocalLoss(nn.Module):
        """Focal Loss - 处理类别不平衡"""

        def __init__(self, alpha=0.25, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, pred, target):
            ce_loss = F.cross_entropy(pred, target, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()

    # 演示
    pred = torch.softmax(torch.randn(2, 3, 64, 64), dim=1)
    target = torch.zeros(2, 3, 64, 64)
    target[:, 0, :32, :32] = 1
    target[:, 1, :32, 32:] = 1
    target[:, 2, 32:, :] = 1

    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")


# ==================== 第五部分：预训练模型使用 ====================


def pretrained_models():
    """预训练模型使用"""
    print("\n" + "=" * 60)
    print("第五部分：使用预训练分割模型")
    print("=" * 60)

    print("示例: DeepLabV3\n")

    try:
        from torchvision.models.segmentation import deeplabv3_resnet50

        # 加载预训练模型
        model = deeplabv3_resnet50(pretrained=True)
        model.eval()

        print("DeepLabV3-ResNet50 加载成功!")

        # 推理
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(x)

        print(f"\n输入: {x.shape}")
        print(f"输出键: {output.keys()}")
        print(f"分割图: {output['out'].shape}")
        print("  → 21 类 (PASCAL VOC)")

        # 获取预测类别
        pred = output["out"].argmax(dim=1)
        print(f"预测类别图: {pred.shape}")

    except Exception as e:
        print(f"加载模型失败: {e}")

    print("""
常用预训练模型：

    ┌──────────────────────────────────────────────────────────┐
    │  模型           Backbone      mIoU    特点              │
    ├──────────────────────────────────────────────────────────┤
    │  FCN            VGG/ResNet    ~65%    经典基准          │
    │  DeepLabV3      ResNet        ~78%    空洞卷积+ASPP     │
    │  DeepLabV3+     Xception      ~82%    编码器-解码器     │
    │  PSPNet         ResNet        ~80%    金字塔池化        │
    │  UNet           -             ~76%    医学图像常用       │
    └──────────────────────────────────────────────────────────┘
    """)


# ==================== 第六部分：评估指标 ====================


def evaluation_metrics():
    """评估指标"""
    print("\n" + "=" * 60)
    print("第六部分：分割评估指标")
    print("=" * 60)

    print("""
常用评估指标：

    1. Pixel Accuracy (像素准确率)
       PA = 正确分类的像素数 / 总像素数

    2. Mean IoU (平均交并比)
       mIoU = (1/C) * Σ (TP_i / (TP_i + FP_i + FN_i))

    3. Dice Score
       Dice = 2 * |A∩B| / (|A| + |B|)
    """)

    def compute_miou(pred, target, num_classes):
        """计算 mIoU"""
        ious = []
        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection

            if union > 0:
                iou = intersection / union
                ious.append(iou.item())

        return np.mean(ious) if ious else 0

    # 示例
    pred = torch.randint(0, 5, (1, 256, 256))
    target = torch.randint(0, 5, (1, 256, 256))
    miou = compute_miou(pred, target, num_classes=5)
    print(f"mIoU: {miou:.4f}")


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：FCN 实现
    任务: 实现完整的 FCN-8s
    包含: pool3, pool4 的跳跃连接

练习 1 答案：
    class FCN8s(nn.Module):
        def __init__(self, num_classes=21):
            super().__init__()
            # 使用 VGG16 作为 backbone
            vgg = torchvision.models.vgg16(pretrained=True)
            features = list(vgg.features.children())
            
            # 分割成不同阶段
            self.pool3 = nn.Sequential(*features[:17])  # 到 pool3
            self.pool4 = nn.Sequential(*features[17:24])  # 到 pool4
            self.pool5 = nn.Sequential(*features[24:])  # 到 pool5
            
            # FC 层转为卷积
            self.fc6 = nn.Conv2d(512, 4096, 7, padding=3)
            self.fc7 = nn.Conv2d(4096, 4096, 1)
            self.score_fr = nn.Conv2d(4096, num_classes, 1)
            
            # 跳跃连接
            self.score_pool4 = nn.Conv2d(512, num_classes, 1)
            self.score_pool3 = nn.Conv2d(256, num_classes, 1)
            
            # 上采样
            self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1)
            self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1)
            self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4)
        
        def forward(self, x):
            pool3 = self.pool3(x)          # 1/8
            pool4 = self.pool4(pool3)      # 1/16
            pool5 = self.pool5(pool4)      # 1/32
            
            fc = F.relu(self.fc6(pool5))
            fc = F.relu(self.fc7(fc))
            score = self.score_fr(fc)
            
            # 融合 pool4
            upscore2 = self.upscore2(score)
            score_pool4 = self.score_pool4(pool4)
            fuse1 = upscore2 + score_pool4
            
            # 融合 pool3
            upscore4 = self.upscore4(fuse1)
            score_pool3 = self.score_pool3(pool3)
            fuse2 = upscore4 + score_pool3
            
            # 8倍上采样到原图
            out = self.upscore8(fuse2)
            return out

练习 2：分割可视化
    任务: 使用 DeepLabV3 分割图像
    要求: 将分割结果彩色可视化

练习 2 答案：
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    from torchvision.models.segmentation import deeplabv3_resnet50
    
    # VOC 调色板
    def get_voc_palette(num_classes=21):
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for i in range(num_classes):
            r = g = b = 0
            c = i
            for j in range(8):
                r |= (((c >> 0) & 1) << (7 - j))
                g |= (((c >> 1) & 1) << (7 - j))
                b |= (((c >> 2) & 1) << (7 - j))
                c >>= 3
            palette[i] = [r, g, b]
        return palette
    
    # 加载模型
    model = deeplabv3_resnet50(pretrained=True).eval()
    
    # 预处理
    image = Image.open('image.jpg')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)['out']
    pred = output.argmax(1).squeeze().numpy()
    
    # 彩色可视化
    palette = get_voc_palette()
    colored = palette[pred]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(colored)
    plt.title('Segmentation')
    plt.savefig('segmentation_result.png')

练习 3：自定义数据集
    任务: 制作一个简单的分割数据集
    使用: 标注工具如 labelme

练习 3 答案：
    # 1. 使用 labelme 标注
    # pip install labelme
    # labelme  # 启动 GUI
    
    # 2. 将 JSON 转换为掩码
    import json
    import numpy as np
    from PIL import Image, ImageDraw
    
    def json_to_mask(json_path, output_path):
        with open(json_path) as f:
            data = json.load(f)
        
        h, w = data['imageHeight'], data['imageWidth']
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            
            # 创建多边形掩码
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon([tuple(p) for p in points],
                                         outline=1, fill=1)
            mask = np.maximum(mask, np.array(img) * label_to_id[label])
        
        Image.fromarray(mask).save(output_path)
    
    # 3. 创建数据集类
    class SegmentationDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir, mask_dir, transform=None):
            self.images = sorted(glob.glob(f'{image_dir}/*.jpg'))
            self.masks = sorted(glob.glob(f'{mask_dir}/*.png'))
            self.transform = transform
        
        def __getitem__(self, idx):
            image = Image.open(self.images[idx]).convert('RGB')
            mask = Image.open(self.masks[idx])
            
            if self.transform:
                image = self.transform(image)
            mask = torch.from_numpy(np.array(mask)).long()
            
            return image, mask

练习 4：损失函数对比
    任务: 对比 CE Loss 和 Dice Loss
    场景: 类别极度不平衡的数据

练习 4 答案：
    import torch
    import torch.nn.functional as F
    
    # 模拟不平衡数据 (前景只占 5%)
    pred = torch.randn(1, 2, 256, 256)  # 2 类
    target = torch.zeros(1, 256, 256).long()
    target[0, 100:120, 100:120] = 1  # 小区域前景
    
    # CE Loss
    ce_loss = F.cross_entropy(pred, target)
    
    # Dice Loss
    def dice_loss(pred, target, smooth=1e-5):
        pred = F.softmax(pred, dim=1)[:, 1]  # 前景概率
        target_float = target.float()
        
        intersection = (pred * target_float).sum()
        union = pred.sum() + target_float.sum()
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    dice = dice_loss(pred, target)
    
    # 组合损失
    combined = 0.5 * ce_loss + 0.5 * dice
    
    print(f'CE Loss: {ce_loss:.4f}')
    print(f'Dice Loss: {dice:.4f}')
    
    # Dice Loss 对不平衡数据更敏感
    # 因为它直接优化区域重叠

练习 5：语义分割应用
    任务: 实现背景替换功能
    流程: 分割前景 → 提取 → 合成新背景

练习 5 答案：
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from torchvision.models.segmentation import deeplabv3_resnet50
    
    def replace_background(image_path, bg_path, output_path):
        # 加载模型
        model = deeplabv3_resnet50(pretrained=True).eval()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        background = Image.open(bg_path).convert('RGB')
        background = background.resize(image.size)
        
        # 预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 分割
        with torch.no_grad():
            output = model(transform(image).unsqueeze(0))['out']
        mask = output.argmax(1).squeeze().numpy()
        
        # 人物类别 = 15 (PASCAL VOC)
        person_mask = (mask == 15).astype(np.float32)
        
        # 平滑边缘 (可选)
        from scipy.ndimage import gaussian_filter
        person_mask = gaussian_filter(person_mask, sigma=2)
        
        # 合成
        image_np = np.array(image) / 255.0
        bg_np = np.array(background) / 255.0
        
        mask_3d = np.stack([person_mask] * 3, axis=-1)
        result = image_np * mask_3d + bg_np * (1 - mask_3d)
        
        result = (result * 255).astype(np.uint8)
        Image.fromarray(result).save(output_path)

思考题 1：为什么需要跳跃连接？
    没有跳跃连接会怎样？

思考题 1 答案：
    为什么需要跳跃连接：
    - 下采样过程中丢失空间细节信息
    - 深层特征有语义信息但分辨率低
    - 浅层特征有细节信息但语义弱
    
    没有跳跃连接的问题：
    - 边界模糊不清晰
    - 小物体容易丢失
    - 空间位置不精确
    
    跳跃连接的作用：
    - 将浅层的高分辨率特征传递给解码器
    - 帮助恢复精确的边界
    - 融合多尺度信息

思考题 2：空洞卷积 (Dilated Conv) 的作用是什么？
    DeepLab 为什么使用空洞卷积？

思考题 2 答案：
    空洞卷积的作用：
    - 在不增加参数的情况下扩大感受野
    - 保持空间分辨率不变
    
    普通卷积 vs 空洞卷积：
    - 3x3 普通卷积: 感受野 3x3
    - 3x3 空洞卷积 (rate=2): 感受野 5x5
    - 3x3 空洞卷积 (rate=3): 感受野 7x7
    
    DeepLab 使用空洞卷积的原因：
    1. 减少下采样次数，保持高分辨率
    2. 不需要大量上采样恢复分辨率
    3. ASPP 使用不同 rate 捕获多尺度上下文
    
    ASPP (Atrous Spatial Pyramid Pooling):
    - 并行使用不同 rate 的空洞卷积
    - 融合多尺度特征

思考题 3：语义分割和实例分割的区别是什么？
    如何从语义分割扩展到实例分割？

思考题 3 答案：
    语义分割 vs 实例分割：
    
    语义分割：
    - 只分类每个像素属于哪个类别
    - 不区分同类的不同实例
    - 输出: H×W×C (one-hot) 或 H×W (类别索引)
    
    实例分割：
    - 需要区分每个独立的实例
    - 同类的不同物体有不同标签
    - 输出: 每个实例一个掩码
    
    从语义分割扩展到实例分割：
    1. Mask R-CNN 方法
       - 先检测每个实例的边界框
       - 再对每个 RoI 进行分割
    
    2. 自底向上方法
       - 语义分割 + 实例嵌入
       - 聚类相近的像素形成实例
    
    3. 全景分割
       - 结合语义分割和实例分割
       - 对"stuff"用语义分割
       - 对"things"用实例分割
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    fcn_architecture()
    encoder_decoder()
    segmentation_losses()
    pretrained_models()
    evaluation_metrics()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 05-unet.py: U-Net 架构详解

关键要点回顾：
    ✓ 语义分割对每个像素进行分类
    ✓ FCN 用卷积替代全连接，保留空间信息
    ✓ 编码器-解码器架构先下采样后上采样
    ✓ Dice Loss 对类别不平衡更鲁棒
    ✓ mIoU 是主要评估指标
    """)


if __name__ == "__main__":
    main()
