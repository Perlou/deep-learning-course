"""
姿态估计 (Pose Estimation)
===========================

学习目标：
    1. 理解姿态估计任务和应用
    2. 掌握热力图方法的原理
    3. 了解 Top-Down 和 Bottom-Up 方法
    4. 使用预训练模型进行人体姿态估计

核心概念：
    - 关键点 (Keypoints): 人体的关节点
    - 热力图 (Heatmap): 关键点位置的概率分布
    - 骨架 (Skeleton): 关键点之间的连接
    - PAF (Part Affinity Fields): 关联关键点到实例

前置知识：
    - Phase 5: CNN
    - 目标检测基础
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：姿态估计概述 ====================


def introduction():
    """姿态估计概述"""
    print("=" * 60)
    print("第一部分：姿态估计概述")
    print("=" * 60)

    print("""
姿态估计 (Pose Estimation)：

    检测人体关键点的位置，重建人体姿态

    常见格式 - COCO 17 个关键点：

              ●  鼻子 (0)
             /|\\
        肩(5)─●─●(6)肩
            │ │
        肘(7)● │ ●(8)肘
            │ │
        腕(9)● │ ●(10)腕
              │
        髋(11)●─●(12)髋
             / \\
       膝(13)●   ●(14)膝
            |   |
       踝(15)●   ●(16)踝

    关键点列表:
    0: nose, 1: left_eye, 2: right_eye
    3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder
    7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle

应用场景：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 运动分析: 体育动作分析、健身指导                        │
    │ 2. 动作捕捉: 电影特效、游戏开发                           │
    │ 3. 人机交互: 手势识别、动作控制                           │
    │ 4. 安全监控: 跌倒检测、异常行为识别                        │
    │ 5. 虚拟试衣: AR 换装、服装展示                            │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：方法分类 ====================


def methods_overview():
    """方法分类"""
    print("\n" + "=" * 60)
    print("第二部分：姿态估计方法分类")
    print("=" * 60)

    print("""
姿态估计方法分类：

    ┌──────────────────────────────────────────────────────────┐
    │  Top-Down (自顶向下)        Bottom-Up (自底向上)          │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  1. 先检测人                 1. 先检测所有关键点           │
    │       ↓                          ↓                       │
    │  2. 裁剪每个人               2. 关联关键点到人             │
    │       ↓                          ↓                       │
    │  3. 估计每人姿态             3. 组装成完整骨架             │
    │                                                          │
    │  代表: HRNet, SimpleBaseline 代表: OpenPose, HigherHRNet  │
    │                                                          │
    │  优点: 精度高                优点: 速度与人数无关          │
    │  缺点: 速度与人数相关        缺点: 关联较复杂              │
    └──────────────────────────────────────────────────────────┘

流程对比图：

    Top-Down:
    ┌────────────────────────────────────────────────────────┐
    │  图像 → 人体检测 → 裁剪 → 姿态估计 → 结果               │
    │              ↓                                        │
    │         [人1, 人2, ...]                               │
    │              ↓                                        │
    │         分别估计姿态                                   │
    └────────────────────────────────────────────────────────┘

    Bottom-Up:
    ┌────────────────────────────────────────────────────────┐
    │  图像 → 检测所有关键点 + PAF → 关联分组 → 结果          │
    │              ↓                                        │
    │         [所有鼻子, 所有肩膀, ...]                      │
    │              ↓                                        │
    │         根据 PAF 分组到不同人                          │
    └────────────────────────────────────────────────────────┘
    """)


# ==================== 第三部分：热力图方法 ====================


def heatmap_method():
    """热力图方法"""
    print("\n" + "=" * 60)
    print("第三部分：热力图 (Heatmap) 方法")
    print("=" * 60)

    print("""
热力图方法原理：

    网络预测每个关键点的热力图 (概率分布)

    输入图像 → CNN → K 个热力图 (K=关键点数)

    热力图示例 (右手腕):
    ┌───────────────────────┐
    │ 0.0  0.0  0.0  0.0    │
    │ 0.0  0.2  0.4  0.1    │
    │ 0.0  0.5  ●1.0 0.3    │  ← 最大值位置 = 关键点位置
    │ 0.0  0.1  0.3  0.1    │
    │ 0.0  0.0  0.0  0.0    │
    └───────────────────────┘

    训练时: 生成高斯热力图作为标签
    推理时: 找热力图最大值位置

高斯热力图生成：
    """)

    def generate_heatmap(keypoint, heatmap_size, sigma=2):
        """
        生成关键点的高斯热力图

        Args:
            keypoint: (x, y) 关键点坐标
            heatmap_size: (H, W) 热力图大小
            sigma: 高斯分布标准差
        """
        H, W = heatmap_size
        x, y = keypoint

        # 创建坐标网格
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))

        # 计算高斯分布
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

        return heatmap

    # 示例
    print("示例: 生成高斯热力图\n")

    keypoint = (32, 24)  # 关键点在 (32, 24)
    heatmap = generate_heatmap(keypoint, (64, 64), sigma=3)

    print(f"关键点位置: {keypoint}")
    print(f"热力图大小: {heatmap.shape}")
    print(f"热力图最大值位置: {np.unravel_index(heatmap.argmax(), heatmap.shape)}")
    print(f"热力图最大值: {heatmap.max():.4f}")

    # 可视化
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()
    plt.scatter([keypoint[0]], [keypoint[1]], c="blue", s=100, marker="x")
    plt.title("Gaussian Heatmap")
    plt.savefig("pose_heatmap.png", dpi=100, bbox_inches="tight")
    print("\n✓ 热力图已保存到 pose_heatmap.png")


# ==================== 第四部分：网络架构 ====================


def network_architecture():
    """网络架构"""
    print("\n" + "=" * 60)
    print("第四部分：常用网络架构")
    print("=" * 60)

    print("""
代表性网络架构：

    1. SimpleBaseline (2018)
       - 简单但有效
       - ResNet + 反卷积上采样

       ResNet → 反卷积×3 → 1×1 Conv → 热力图

    2. HRNet (High-Resolution Net, 2019)
       - 多分辨率并行
       - 保持高分辨率特征

       ┌────────────────────────────────────────────────┐
       │  高分辨率 ─────────────────────────────── ──→   │
       │       ↓                                   ↑    │
       │  中分辨率 ─────────────────────────── ────┘    │
       │       ↓                             ↑          │
       │  低分辨率 ───────────────────── ────┘          │
       │       ↓                   ↑                    │
       │  最低分辨率 ──────── ────┘                      │
       └────────────────────────────────────────────────┘
       多尺度特征融合，保持高分辨率

    3. OpenPose (Bottom-Up)
       - 两分支: 热力图 + PAF
       - 多阶段迭代优化
    """)

    # SimpleBaseline 简化实现
    print("示例: SimpleBaseline 简化架构\n")

    class SimpleBaseline(nn.Module):
        """简化的 SimpleBaseline"""

        def __init__(self, num_keypoints=17):
            super().__init__()

            # 使用 ResNet 作为 backbone
            import torchvision.models as models

            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

            # 反卷积上采样
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(2048, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            # 输出热力图
            self.final_layer = nn.Conv2d(256, num_keypoints, 1)

        def forward(self, x):
            x = self.backbone(x)
            x = self.deconv_layers(x)
            heatmaps = self.final_layer(x)
            return heatmaps

    # 测试
    model = SimpleBaseline(num_keypoints=17)
    x = torch.randn(1, 3, 256, 192)
    with torch.no_grad():
        heatmaps = model(x)
    print(f"输入: {x.shape}")
    print(f"输出热力图: {heatmaps.shape}")
    print(f"  → 17 个关键点，分辨率 64×48")


# ==================== 第五部分：预训练模型 ====================


def pretrained_model():
    """使用预训练模型"""
    print("\n" + "=" * 60)
    print("第五部分：使用预训练模型")
    print("=" * 60)

    print("使用 torchvision 的 Keypoint R-CNN:\n")

    try:
        from torchvision.models.detection import keypointrcnn_resnet50_fpn

        # 加载模型
        model = keypointrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        print("Keypoint R-CNN 加载成功!")

        # 推理
        dummy_image = torch.randn(3, 480, 640)

        with torch.no_grad():
            predictions = model([dummy_image])

        pred = predictions[0]
        print(f"\n输出键: {pred.keys()}")
        print(f"  - boxes: 人体边界框")
        print(f"  - keypoints: 关键点 [N, K, 3] (x, y, visibility)")
        print(f"  - keypoints_scores: 关键点置信度")

        print("""
关键点格式:
    keypoints shape: [N, 17, 3]
    - N: 检测到的人数
    - 17: COCO 关键点数
    - 3: (x, y, visibility)
      - visibility > 0: 关键点可见
        """)

    except Exception as e:
        print(f"加载失败: {e}")


# ==================== 第六部分：可视化骨架 ====================


def visualization():
    """可视化骨架"""
    print("\n" + "=" * 60)
    print("第六部分：骨架可视化")
    print("=" * 60)

    # COCO 骨架连接
    COCO_SKELETON = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # 头部
        (0, 5),
        (0, 6),  # 肩膀
        (5, 7),
        (7, 9),  # 左臂
        (6, 8),
        (8, 10),  # 右臂
        (5, 11),
        (6, 12),  # 躯干
        (11, 12),  # 髋部
        (11, 13),
        (13, 15),  # 左腿
        (12, 14),
        (14, 16),  # 右腿
    ]

    def draw_skeleton(ax, keypoints, skeleton=COCO_SKELETON):
        """绘制骨架"""
        for i, kp in enumerate(keypoints):
            x, y, v = kp
            if v > 0:
                ax.scatter([x], [y], c="red", s=50)

        for start_idx, end_idx in skeleton:
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            if start[2] > 0 and end[2] > 0:  # 两点都可见
                ax.plot([start[0], end[0]], [start[1], end[1]], c="blue", linewidth=2)

    print("骨架连接定义完成!")
    print("COCO 格式: 17 个关键点，16 条骨架连接")

    # 生成示例关键点
    print("\n示例: 绘制骨架")

    # 模拟关键点 (x, y, visibility)
    keypoints = np.array(
        [
            [100, 50, 1],  # nose
            [90, 40, 1],  # left_eye
            [110, 40, 1],  # right_eye
            [80, 45, 1],  # left_ear
            [120, 45, 1],  # right_ear
            [70, 100, 1],  # left_shoulder
            [130, 100, 1],  # right_shoulder
            [50, 150, 1],  # left_elbow
            [150, 150, 1],  # right_elbow
            [40, 200, 1],  # left_wrist
            [160, 200, 1],  # right_wrist
            [80, 200, 1],  # left_hip
            [120, 200, 1],  # right_hip
            [75, 280, 1],  # left_knee
            [125, 280, 1],  # right_knee
            [70, 350, 1],  # left_ankle
            [130, 350, 1],  # right_ankle
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 8))
    draw_skeleton(ax, keypoints)
    ax.set_xlim(0, 200)
    ax.set_ylim(400, 0)  # 反转 y 轴
    ax.set_aspect("equal")
    ax.set_title("Pose Skeleton")
    plt.savefig("pose_skeleton.png", dpi=100, bbox_inches="tight")
    print("✓ 骨架图已保存到 pose_skeleton.png")


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：使用 Keypoint R-CNN
    任务: 加载预训练模型，检测真实图像中的人体姿态
    要求: 绘制骨架连接

练习 1 答案：
    import torch
    import cv2
    import matplotlib.pyplot as plt
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision import transforms
    
    # 加载模型
    model = keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # COCO 骨架连接
    SKELETON = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
                (6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    
    # 加载图片
    image = cv2.imread('person.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 推理
    transform = transforms.ToTensor()
    img_tensor = transform(image_rgb)
    
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # 可视化
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_rgb)
    
    for i in range(len(predictions[0]['keypoints'])):
        if predictions[0]['scores'][i] < 0.7:
            continue
        
        kpts = predictions[0]['keypoints'][i].numpy()
        
        # 绘制关键点
        for x, y, v in kpts:
            if v > 0:
                ax.scatter([x], [y], c='red', s=50)
        
        # 绘制骨架
        for start, end in SKELETON:
            if kpts[start][2] > 0 and kpts[end][2] > 0:
                ax.plot([kpts[start][0], kpts[end][0]],
                        [kpts[start][1], kpts[end][1]], 'b-', linewidth=2)
    
    plt.savefig('pose_result.png')

练习 2：实现热力图生成
    任务: 实现批量生成多关键点热力图
    包含: 处理关键点不可见的情况

练习 2 答案：
    import numpy as np
    import torch
    
    def generate_heatmaps(keypoints, heatmap_size, sigma=2):
        '''
        生成多关键点热力图
        
        Args:
            keypoints: [K, 3] (x, y, visibility)
            heatmap_size: (H, W)
            sigma: 高斯标准差
        
        Returns:
            heatmaps: [K, H, W]
        '''
        K = len(keypoints)
        H, W = heatmap_size
        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        
        # 创建坐标网格
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        
        for k, (x, y, v) in enumerate(keypoints):
            if v == 0:  # 不可见
                continue
            
            # 高斯热力图
            heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            heatmaps[k] = heatmap
        
        return heatmaps
    
    # 使用示例
    keypoints = np.array([
        [100, 50, 1],   # nose, visible
        [90, 40, 1],    # left_eye, visible
        [110, 40, 0],   # right_eye, occluded
        # ... 17个关键点
    ])
    
    heatmaps = generate_heatmaps(keypoints, (256, 256), sigma=3)
    print(f'热力图形状: {heatmaps.shape}')  # (17, 256, 256)

练习 3：姿态估计训练
    任务: 在 COCO 数据集上训练简单的姿态估计模型
    建议: 使用 SimpleBaseline 架构

练习 3 答案：
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    # 1. SimpleBaseline 模型 (见第四部分)
    model = SimpleBaseline(num_keypoints=17)
    
    # 2. 损失函数 - MSE on heatmaps
    criterion = nn.MSELoss()
    
    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. 训练循环
    for epoch in range(100):
        model.train()
        for images, target_heatmaps in dataloader:
            # 前向传播
            pred_heatmaps = model(images)
            
            # 计算损失 (热力图回归)
            loss = criterion(pred_heatmaps, target_heatmaps)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证 (使用 PCK 或 OKS 指标)
        model.eval()
        # ... 计算验证指标

练习 4：动作识别
    任务: 基于姿态估计结果进行简单动作识别
    提示: 可以用关键点的相对位置作为特征

练习 4 答案：
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    def extract_pose_features(keypoints):
        '''
        从关键点提取特征用于动作识别
        
        Args:
            keypoints: [17, 3] (x, y, visibility)
        '''
        features = []
        
        # 1. 归一化坐标 (相对于髋部中心)
        hip_center = (keypoints[11, :2] + keypoints[12, :2]) / 2
        normalized = keypoints[:, :2] - hip_center
        features.extend(normalized.flatten())
        
        # 2. 骨骼长度
        bones = [
            (5, 7), (7, 9),    # 左臂
            (6, 8), (8, 10),   # 右臂
            (11, 13), (13, 15), # 左腿
            (12, 14), (14, 16)  # 右腿
        ]
        for start, end in bones:
            length = np.linalg.norm(keypoints[start, :2] - keypoints[end, :2])
            features.append(length)
        
        # 3. 关节角度
        def angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return np.arccos(np.clip(cos, -1, 1))
        
        # 左肘角度
        features.append(angle(keypoints[5, :2], keypoints[7, :2], keypoints[9, :2]))
        
        return np.array(features)
    
    # 训练简单分类器
    X_train = [extract_pose_features(kpts) for kpts in train_keypoints]
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

练习 5：实时姿态估计
    任务: 使用摄像头实时检测人体姿态
    要求: 显示 FPS，绘制骨架

练习 5 答案：
    import cv2
    import time
    import torch
    from torchvision import transforms
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    
    SKELETON = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
                (6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    
    model = keypointrcnn_resnet50_fpn(pretrained=True).eval()
    transform = transforms.ToTensor()
    
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            preds = model([img_tensor])
        
        # 绘制骨架
        for i in range(len(preds[0]['keypoints'])):
            if preds[0]['scores'][i] < 0.7:
                continue
            kpts = preds[0]['keypoints'][i].numpy().astype(int)
            
            for x, y, v in kpts:
                if v > 0:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            for start, end in SKELETON:
                if kpts[start][2] > 0 and kpts[end][2] > 0:
                    cv2.line(frame, tuple(kpts[start][:2]), 
                             tuple(kpts[end][:2]), (255, 0, 0), 2)
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

思考题 1：Top-Down 和 Bottom-Up 各自的适用场景？
    考虑人数、速度、精度等因素

思考题 1 答案：
    Top-Down (先检测人再估计姿态):
    适用场景:
    - 人数较少 (1-5人)
    - 需要高精度
    - 对单人姿态质量要求高
    - 可接受稍慢速度
    
    特点:
    - 速度与人数成正比
    - 精度高，特别是遮挡情况
    - 依赖人体检测器质量
    
    Bottom-Up (先检测所有关键点再分组):
    适用场景:
    - 人数较多 (密集人群)
    - 需要实时性
    - 人数变化大
    - 可接受稍低精度
    
    特点:
    - 速度与人数基本无关
    - 实时性好
    - 遮挡严重时分组可能出错

思考题 2：热力图的分辨率如何影响精度？
    为什么不直接输出原图分辨率的热力图？

思考题 2 答案：
    分辨率影响:
    - 高分辨率: 定位更精确，但计算量大
    - 低分辨率: 计算快，但定位粗糙
    
    常见配置:
    - 输入 256×192，热力图 64×48 (1/4)
    - 输入 384×288，热力图 96×72 (1/4)
    
    不输出原图分辨率的原因:
    
    1. 计算量过大
       - 原图 1920×1080 × 17关键点
       - 内存和计算成本极高
    
    2. 语义与分辨率的平衡
       - 太高分辨率信息冗余
       - 热力图只需要大致位置
       - 后处理可以亚像素精度定位
    
    3. 训练稳定性
       - 高分辨率损失更难优化
       - 低分辨率收敛更快
    
    4. 后处理恢复精度
       - 热力图取 argmax
       - 加权平均获得亚像素精度

思考题 3：如何处理遮挡和自遮挡问题？
    被遮挡的关键点应该如何处理？

思考题 3 答案：
    遮挡类型:
    - 自遮挡: 身体部位互相遮挡 (如手背在身后)
    - 他遮挡: 被其他物体/人遮挡
    
    处理策略:
    
    1. 数据标注
       - 标记可见性标签 (v=0/1/2)
       - 训练时对不可见点降低权重
    
    2. 网络设计
       - 使用更深的网络捕获上下文
       - 根据可见点推断不可见点
       - 使用时序信息 (视频)
    
    3. 多阶段预测
       - 先预测初步结果
       - 根据人体结构约束优化
       - 迭代细化
    
    4. 图神经网络
       - 建模关键点之间的关系
       - 通过相邻可见点推断遮挡点
    
    5. 后处理
       - 使用骨骼长度约束
       - 时序平滑 (视频)
       - 物理合理性检查
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    methods_overview()
    heatmap_method()
    network_architecture()
    pretrained_model()
    visualization()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 08-face-recognition.py: 人脸识别

关键要点回顾：
    ✓ 姿态估计检测人体关键点位置
    ✓ 热力图方法预测关键点的概率分布
    ✓ Top-Down 先检测人再估计姿态
    ✓ Bottom-Up 先检测所有关键点再分组
    ✓ COCO 格式包含 17 个关键点
    """)


if __name__ == "__main__":
    main()
