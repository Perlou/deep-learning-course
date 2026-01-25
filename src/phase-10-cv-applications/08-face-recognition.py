"""
人脸识别 (Face Recognition)
============================

学习目标：
    1. 理解人脸识别系统的完整流程
    2. 掌握人脸检测、对齐、特征提取
    3. 了解常用的人脸识别损失函数
    4. 使用预训练模型进行人脸识别

核心概念：
    - 人脸检测: 定位图像中的人脸
    - 人脸对齐: 标准化人脸姿态
    - 特征提取: 将人脸映射为特征向量
    - 特征匹配: 比较两张人脸的相似度

前置知识：
    - Phase 5: CNN
    - 目标检测基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== 第一部分：人脸识别概述 ====================


def introduction():
    """人脸识别概述"""
    print("=" * 60)
    print("第一部分：人脸识别概述")
    print("=" * 60)

    print("""
人脸识别系统流程：

    ┌─────────────────────────────────────────────────────────┐
    │  输入图像 → 人脸检测 → 人脸对齐 → 特征提取 → 特征匹配     │
    │      ↓          ↓          ↓          ↓          ↓     │
    │    📷        🔲         🔄        📊        ✓/✗    │
    │    图像    定位人脸    标准化    512维向量   对比决策   │
    └─────────────────────────────────────────────────────────┘

任务分类：

    1. 人脸验证 (Face Verification) - 1:1
       问题: 这两张是同一个人吗？
       应用: 手机解锁、门禁系统

    2. 人脸识别 (Face Identification) - 1:N
       问题: 这个人是谁？
       应用: 考勤系统、嫌疑人搜索

    3. 人脸聚类 (Face Clustering)
       问题: 这些人脸可以分成几组？
       应用: 相册整理

主要挑战：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 光照变化: 不同光线条件下外观差异大                      │
    │ 2. 姿态变化: 正面、侧面差异大                             │
    │ 3. 表情变化: 微笑、哭泣等表情影响                         │
    │ 4. 遮挡: 眼镜、口罩、头发遮挡                             │
    │ 5. 年龄变化: 同一人不同年龄差异大                         │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：人脸检测 ====================


def face_detection():
    """人脸检测"""
    print("\n" + "=" * 60)
    print("第二部分：人脸检测")
    print("=" * 60)

    print("""
常用人脸检测方法：

    ┌──────────────────────────────────────────────────────────┐
    │  方法              特点                    速度/精度       │
    ├──────────────────────────────────────────────────────────┤
    │  Haar Cascade     传统方法，CPU快          ★★★/★★      │
    │  HOG + SVM        传统方法                 ★★★/★★★     │
    │  MTCNN            级联CNN，检测+对齐       ★★/★★★★      │
    │  RetinaFace       单阶段，精度高           ★★/★★★★★     │
    │  YOLOv8-face      快速，适合实时           ★★★★/★★★★   │
    └──────────────────────────────────────────────────────────┘

MTCNN 流程：

    三阶段级联网络:

    输入图像
        ↓
    ┌───────────────┐
    │   P-Net       │ → 快速筛选候选框
    │  (Proposal)   │
    └───────────────┘
        ↓
    ┌───────────────┐
    │   R-Net       │ → 精细筛选
    │  (Refine)     │
    └───────────────┘
        ↓
    ┌───────────────┐
    │   O-Net       │ → 输出边界框 + 5个关键点
    │  (Output)     │
    └───────────────┘

    5个关键点: 左眼、右眼、鼻子、左嘴角、右嘴角
    """)

    print("示例: 使用 MTCNN 检测人脸\n")
    print("""
# 安装: pip install facenet-pytorch

from facenet_pytorch import MTCNN

# 创建检测器
mtcnn = MTCNN(
    image_size=160,      # 输出人脸大小
    margin=0,            # 边缘扩展
    keep_all=True,       # 检测所有人脸
    device='cuda'
)

# 检测
boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

# boxes: [N, 4] 边界框
# probs: [N] 置信度
# landmarks: [N, 5, 2] 5个关键点坐标
    """)


# ==================== 第三部分：人脸对齐 ====================


def face_alignment():
    """人脸对齐"""
    print("\n" + "=" * 60)
    print("第三部分：人脸对齐")
    print("=" * 60)

    print("""
人脸对齐的作用：

    将不同姿态的人脸标准化到统一姿态

    ┌─────────────────────────────────────────────────────────┐
    │  对齐前:                    对齐后:                      │
    │                                                         │
    │   😊  😊  😊              😊  😊  😊               │
    │  (倾斜)(侧面)(正面)    →   (正面)(正面)(正面)             │
    │                                                         │
    │  使用关键点进行仿射变换                                   │
    └─────────────────────────────────────────────────────────┘

对齐步骤：

    1. 检测 5 个关键点
       - 左眼中心、右眼中心、鼻尖、左嘴角、右嘴角

    2. 计算目标位置
       - 标准化的关键点模板

    3. 仿射变换
       - 根据源和目标关键点计算变换矩阵
       - 对图像进行变换
    """)

    import cv2

    def align_face(image, landmarks, target_size=(112, 112)):
        """
        人脸对齐

        Args:
            image: 输入图像
            landmarks: 5个关键点 [(x1,y1), (x2,y2), ...]
            target_size: 输出大小
        """
        # 标准化的目标关键点位置 (112x112 图像)
        target_landmarks = np.float32(
            [
                [38.2946, 51.6963],  # 左眼
                [73.5318, 51.5014],  # 右眼
                [56.0252, 71.7366],  # 鼻子
                [41.5493, 92.3655],  # 左嘴角
                [70.7299, 92.2041],  # 右嘴角
            ]
        )

        # 源关键点
        src_landmarks = np.float32(landmarks)

        # 计算仿射变换矩阵
        M = cv2.estimateAffinePartial2D(src_landmarks, target_landmarks)[0]

        # 应用变换
        aligned = cv2.warpAffine(image, M, target_size)

        return aligned

    print("人脸对齐函数定义完成!")
    print("输入: 图像 + 5个关键点")
    print("输出: 对齐后的 112×112 人脸图像")


# ==================== 第四部分：特征提取 ====================


def feature_extraction():
    """特征提取"""
    print("\n" + "=" * 60)
    print("第四部分：特征提取")
    print("=" * 60)

    print("""
特征提取网络：

    将人脸图像映射为固定维度的特征向量 (embedding)

    输入: 对齐后的人脸图像 (112×112×3)
    输出: 特征向量 (512维)

    ┌─────────────────────────────────────────────────────────┐
    │  常用网络:                                               │
    │                                                         │
    │  FaceNet (2015)                                         │
    │  - Inception-ResNet-v1                                  │
    │  - Triplet Loss 训练                                    │
    │                                                         │
    │  ArcFace (2019)                                         │
    │  - ResNet 变体                                          │
    │  - ArcFace Loss 训练                                    │
    │  - 目前最佳性能之一                                      │
    │                                                         │
    │  CosFace (2018)                                         │
    │  - 大间隔余弦损失                                        │
    └─────────────────────────────────────────────────────────┘

特征向量特点：
    - 归一化到单位球面上
    - 同一人的特征向量相似
    - 不同人的特征向量分离
    """)

    print("示例: 使用预训练模型提取特征\n")
    print("""
from facenet_pytorch import InceptionResnetV1

# 加载预训练模型 (vggface2 或 casia-webface)
model = InceptionResnetV1(pretrained='vggface2').eval()

# 提取特征
# face: [B, 3, 160, 160] 对齐后的人脸
embedding = model(face)  # [B, 512]

# L2 归一化
embedding = F.normalize(embedding, p=2, dim=1)
    """)


# ==================== 第五部分：特征匹配 ====================


def feature_matching():
    """特征匹配"""
    print("\n" + "=" * 60)
    print("第五部分：特征匹配")
    print("=" * 60)

    print("""
特征匹配方法：

    1. 余弦相似度 (Cosine Similarity)
       sim = (a · b) / (||a|| × ||b||)
       范围: [-1, 1]，1 表示完全相同

    2. 欧氏距离 (Euclidean Distance)
       dist = ||a - b||₂
       距离越小越相似

匹配阈值：
    - 验证: sim > threshold → 同一人
    - 典型阈值: 0.5-0.7 (根据应用调整)
    """)

    def compare_faces(emb1, emb2, threshold=0.6):
        """比较两张人脸"""
        # 余弦相似度
        similarity = F.cosine_similarity(emb1, emb2, dim=1)

        # 欧氏距离
        distance = torch.dist(emb1, emb2, p=2)

        is_same = similarity > threshold

        return {
            "similarity": similarity.item(),
            "distance": distance.item(),
            "is_same_person": is_same.item(),
        }

    def search_face(query_emb, database_embs, top_k=5):
        """
        在数据库中搜索最相似的人脸

        Args:
            query_emb: [1, D] 查询特征
            database_embs: [N, D] 数据库特征
            top_k: 返回前 k 个结果
        """
        # 计算与所有人脸的相似度
        similarities = F.cosine_similarity(query_emb, database_embs)

        # 获取 top-k
        top_scores, top_indices = similarities.topk(top_k)

        return top_indices, top_scores

    # 示例
    print("示例: 人脸匹配\n")

    emb1 = F.normalize(torch.randn(1, 512), dim=1)
    emb2 = F.normalize(torch.randn(1, 512), dim=1)

    result = compare_faces(emb1, emb2)
    print(f"余弦相似度: {result['similarity']:.4f}")
    print(f"欧氏距离: {result['distance']:.4f}")
    print(f"是否同一人: {result['is_same_person']}")


# ==================== 第六部分：损失函数 ====================


def loss_functions():
    """损失函数"""
    print("\n" + "=" * 60)
    print("第六部分：人脸识别损失函数")
    print("=" * 60)

    print("""
主要损失函数：

    1. Softmax Loss (交叉熵)
       - 简单分类损失
       - 问题: 特征分离度不够

    2. Triplet Loss
       - 三元组: (Anchor, Positive, Negative)
       - 目标: d(A,P) + margin < d(A,N)

       ┌─────────────────────────────────────────────────────┐
       │         Anchor (锚点)                               │
       │           /     \\                                  │
       │          /       \\                                 │
       │    Positive      Negative                          │
       │     (同人)        (不同人)                          │
       │                                                    │
       │  让同一人更近，不同人更远                            │
       └─────────────────────────────────────────────────────┘

    3. ArcFace Loss (SOTA)
       - 在角度空间添加 additive margin
       - 增强类间可分性
    """)

    # Triplet Loss 实现
    print("示例: Triplet Loss\n")

    class TripletLoss(nn.Module):
        """Triplet Loss"""

        def __init__(self, margin=0.2):
            super().__init__()
            self.margin = margin

        def forward(self, anchor, positive, negative):
            # 距离计算
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)

            # Triplet Loss
            loss = F.relu(pos_dist - neg_dist + self.margin)

            return loss.mean()

    # ArcFace Loss 实现
    class ArcFace(nn.Module):
        """ArcFace Loss"""

        def __init__(self, in_features, out_features, s=30.0, m=0.50):
            super().__init__()
            self.s = s  # 缩放因子
            self.m = m  # margin
            self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
            nn.init.xavier_uniform_(self.weight)

        def forward(self, features, labels):
            # 归一化
            features = F.normalize(features, dim=1)
            weight = F.normalize(self.weight, dim=1)

            # 计算余弦相似度
            cosine = F.linear(features, weight)

            # 转换为角度
            theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))

            # 添加 margin
            target_logits = torch.cos(theta + self.m)

            # 替换目标类的 logits
            one_hot = F.one_hot(labels, cosine.size(1)).float()
            output = cosine * (1 - one_hot) + target_logits * one_hot

            # 缩放
            output *= self.s

            return output

    criterion = TripletLoss(margin=0.2)
    anchor = torch.randn(4, 512)
    positive = torch.randn(4, 512)
    negative = torch.randn(4, 512)
    loss = criterion(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item():.4f}")


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：人脸检测
    任务: 使用 MTCNN 检测图像中的人脸
    要求: 绘制边界框和 5 个关键点

练习 1 答案：
    # pip install facenet-pytorch
    from facenet_pytorch import MTCNN
    import cv2
    import matplotlib.pyplot as plt
    
    # 创建检测器
    mtcnn = MTCNN(keep_all=True, device='cuda')
    
    # 加载图片
    image = cv2.imread('group_photo.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测
    boxes, probs, landmarks = mtcnn.detect(image_rgb, landmarks=True)
    
    # 可视化
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    if boxes is not None:
        for box, prob, landmark in zip(boxes, probs, landmarks):
            if prob < 0.9:
                continue
            
            # 边界框
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                        fill=False, color='green', linewidth=2))
            ax.text(x1, y1-5, f'{prob:.2f}', color='green')
            
            # 5 个关键点
            colors = ['red', 'red', 'blue', 'green', 'green']
            for i, (x, y) in enumerate(landmark):
                ax.scatter([x], [y], c=colors[i], s=30)
    
    plt.savefig('mtcnn_result.png')

练习 2：人脸对齐
    任务: 实现基于关键点的人脸对齐
    测试: 对齐不同姿态的人脸

练习 2 答案：
    import cv2
    import numpy as np
    
    def align_face(image, landmarks, target_size=(112, 112)):
        '''
        基于 5 点的人脸对齐
        
        Args:
            image: 输入图像
            landmarks: 5 个关键点 [[x1,y1], [x2,y2], ...]
            target_size: 输出大小
        '''
        # 标准人脸的关键点位置 (基于 112×112)
        target_landmarks = np.float32([
            [38.2946, 51.6963],  # 左眼
            [73.5318, 51.5014],  # 右眼  
            [56.0252, 71.7366],  # 鼻子
            [41.5493, 92.3655],  # 左嘴角
            [70.7299, 92.2041]   # 右嘴角
        ])
        
        src = np.float32(landmarks)
        
        # 计算仿射变换矩阵
        M = cv2.estimateAffinePartial2D(src, target_landmarks)[0]
        
        # 应用变换
        aligned = cv2.warpAffine(image, M, target_size)
        
        return aligned
    
    # 使用示例
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False)
    
    image = cv2.imread('face.jpg')
    _, _, landmarks = mtcnn.detect(image, landmarks=True)
    
    if landmarks is not None:
        aligned = align_face(image, landmarks[0])
        cv2.imwrite('aligned_face.jpg', aligned)

练习 3：人脸验证系统
    任务: 构建 1:1 人脸验证系统
    流程: 检测 → 对齐 → 特征提取 → 匹配

练习 3 答案：
    import torch
    import torch.nn.functional as F
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import cv2
    import numpy as np
    
    class FaceVerifier:
        def __init__(self, threshold=0.6):
            self.mtcnn = MTCNN(keep_all=False, image_size=160)
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.threshold = threshold
        
        def get_embedding(self, image):
            '''提取人脸特征'''
            # 检测并对齐
            face = self.mtcnn(image)
            if face is None:
                return None
            
            # 提取特征
            with torch.no_grad():
                embedding = self.model(face.unsqueeze(0))
            
            # 归一化
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        
        def verify(self, image1, image2):
            '''验证两张图片是否为同一人'''
            emb1 = self.get_embedding(image1)
            emb2 = self.get_embedding(image2)
            
            if emb1 is None or emb2 is None:
                return None, 'Face not detected'
            
            # 计算相似度
            similarity = F.cosine_similarity(emb1, emb2).item()
            is_same = similarity > self.threshold
            
            return is_same, similarity
    
    # 使用
    verifier = FaceVerifier(threshold=0.6)
    is_same, score = verifier.verify(image1, image2)
    print(f'同一人: {is_same}, 相似度: {score:.4f}')

练习 4：人脸搜索系统
    任务: 构建 1:N 人脸搜索系统
    包含: 人脸库构建、特征索引、相似度搜索

练习 4 答案：
    import numpy as np
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    
    class FaceDatabase:
        def __init__(self):
            self.embeddings = []
            self.identities = []
        
        def add_face(self, embedding, identity):
            '''添加人脸到数据库'''
            self.embeddings.append(embedding)
            self.identities.append(identity)
        
        def build_index(self):
            '''构建索引'''
            self.db_tensor = torch.cat(self.embeddings, dim=0)
        
        def search(self, query_emb, top_k=5):
            '''搜索最相似的人脸'''
            # 计算与所有人脸的相似度
            similarities = F.cosine_similarity(
                query_emb, self.db_tensor
            )
            
            # 获取 top-k
            scores, indices = similarities.topk(top_k)
            
            results = []
            for score, idx in zip(scores.tolist(), indices.tolist()):
                results.append({
                    'identity': self.identities[idx],
                    'score': score
                })
            
            return results
    
    # 使用
    db = FaceDatabase()
    
    # 注册人脸
    for name, image in registered_faces:
        emb = get_embedding(image)
        db.add_face(emb, name)
    
    db.build_index()
    
    # 搜索
    query_emb = get_embedding(query_image)
    results = db.search(query_emb, top_k=3)

练习 5：Triplet Mining
    任务: 实现 Hard/Semi-Hard Triplet Mining
    比较: 不同挖掘策略对训练的影响

练习 5 答案：
    import torch
    import torch.nn.functional as F
    
    def batch_hard_triplet_mining(embeddings, labels, margin=0.2):
        '''
        Batch Hard Triplet Mining
        选择最难的正例和负例
        '''
        device = embeddings.device
        n = embeddings.size(0)
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 创建标签掩码
        labels = labels.unsqueeze(0)
        mask_pos = (labels == labels.T).float()  # 同类
        mask_neg = (labels != labels.T).float()  # 不同类
        
        # 对角线置零 (排除自己)
        mask_pos = mask_pos - torch.eye(n, device=device)
        
        # Hard Positive: 同类中最远的
        hardest_pos = (dist_matrix * mask_pos).max(dim=1)[0]
        
        # Hard Negative: 不同类中最近的
        # 将同类距离设为很大
        dist_neg = dist_matrix + mask_pos * 1e9
        hardest_neg = dist_neg.min(dim=1)[0]
        
        # Triplet Loss
        loss = F.relu(hardest_pos - hardest_neg + margin)
        
        return loss.mean()
    
    def semi_hard_triplet_mining(embeddings, labels, margin=0.2):
        '''
        Semi-Hard Triplet Mining
        选择比正例远但在 margin 内的负例
        '''
        # ... 类似实现
        # 条件: d(a,p) < d(a,n) < d(a,p) + margin
        pass

思考题 1：为什么需要人脸对齐？
    不对齐会有什么问题？

思考题 1 答案：
    为什么需要对齐:
    
    1. 减少姿态变化
       - 侧脸vs正脸差异大
       - 对齐后统一为正脸
       - 网络更容易学习
    
    2. 标准化输入
       - 眼睛、嘴巴位置固定
       - 网络只需关注身份特征
       - 不需要学习位置不变性
    
    不对齐的问题:
    - 同一人不同姿态特征差异大
    - 需要更多数据覆盖各种姿态
    - 网络需要学习姿态不变性
    - 识别准确率下降

思考题 2：ArcFace 为什么比 Softmax 效果好？
    角度 margin 的作用是什么？

思考题 2 答案：
    Softmax 的问题:
    - 只要求正确分类
    - 不强制类间分离
    - 特征分布可能紧密
    
    ArcFace 的改进:
    
    1. 角度空间
       - 将特征归一化到超球面
       - 距离变成角度
       - 更符合人脸分布
    
    2. Additive Angular Margin
       - 公式: cos(θ + m)
       - 强制同类更近 (角度更小)
       - 强制异类更远 (角度更大)
    
    3. 几何解释
       - 决策边界更严格
       - 类间需要更大间隔
       - 泛化能力更强
    
    效果对比:
    - Softmax: ~95% (LFW)
    - ArcFace: ~99.8% (LFW)

思考题 3：如何处理大规模人脸库的快速搜索？
    提示: 考虑 ANN (近似最近邻) 算法

思考题 3 答案：
    大规模人脸搜索挑战:
    - 百万/亿级人脸库
    - 暴力搜索太慢
    - 需要近似搜索
    
    ANN (Approximate Nearest Neighbor) 算法:
    
    1. Faiss (Facebook)
       - IVF: 倒排索引
       - PQ: 乘积量化
       - GPU 加速
       
       import faiss
       index = faiss.IndexFlatIP(512)  # 余弦相似度
       index.add(db_embeddings)
       D, I = index.search(query, k=5)
    
    2. Annoy (Spotify)
       - 基于树的结构
       - 内存映射，支持大数据
    
    3. HNSW
       - 分层可导航小世界
       - 高召回率
    
    4. 聚类分层
       - 先按聚类筛选
       - 再在候选中精确搜索
    
    典型性能:
    - 百万级人脸: ~10ms
    - 亿级人脸: ~100ms
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    face_detection()
    face_alignment()
    feature_extraction()
    feature_matching()
    loss_functions()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
Phase 10 学习完成！

关键要点回顾：
    ✓ 人脸识别流程: 检测 → 对齐 → 特征提取 → 匹配
    ✓ MTCNN: 多任务级联 CNN 人脸检测
    ✓ 人脸对齐: 使用关键点进行仿射变换
    ✓ 特征提取: 将人脸映射为 512 维向量
    ✓ ArcFace: 当前最佳人脸识别损失函数

恭喜你完成了计算机视觉应用阶段！
下一阶段: Phase 11 - 自然语言处理应用
    """)


if __name__ == "__main__":
    main()
