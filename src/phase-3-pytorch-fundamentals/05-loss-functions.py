"""
05-loss-functions.py
Phase 3: PyTorch 核心技能

损失函数 - 模型优化的目标

学习目标：
1. 理解常用损失函数的原理
2. 掌握分类和回归任务的损失选择
3. 了解自定义损失函数的方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("PyTorch 核心技能 - 损失函数")
print("=" * 60)

# =============================================================================
# 1. 损失函数概述
# =============================================================================
print("\n【1. 损失函数概述】")

print("""
损失函数（Loss Function）衡量预测值与真实值的差距：
- 回归任务: MSE, MAE, Huber
- 二分类: BCELoss, BCEWithLogitsLoss
- 多分类: CrossEntropyLoss, NLLLoss
- 特殊: 对比损失, Focal Loss, Dice Loss

使用方式:
1. 类形式: criterion = nn.MSELoss(); loss = criterion(pred, target)
2. 函数形式: loss = F.mse_loss(pred, target)
""")

# =============================================================================
# 2. 回归损失
# =============================================================================
print("\n" + "=" * 60)
print("【2. 回归损失】")

# 预测值和真实值
y_pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])

print(f"预测值: {y_pred}")
print(f"真实值: {y_true}")

# 2.1 MSE Loss (均方误差)
print("\n2.1 MSE Loss (L2 Loss):")
print("    公式: MSE = (1/n) Σ(y_pred - y_true)²")
mse = nn.MSELoss()
loss = mse(y_pred, y_true)
print(f"    MSE = {loss.item():.4f}")

# 手动计算验证
manual_mse = ((y_pred - y_true) ** 2).mean()
print(f"    手动计算: {manual_mse.item():.4f}")

# 2.2 MAE Loss (平均绝对误差)
print("\n2.2 MAE Loss (L1 Loss):")
print("    公式: MAE = (1/n) Σ|y_pred - y_true|")
mae = nn.L1Loss()
loss = mae(y_pred, y_true)
print(f"    MAE = {loss.item():.4f}")

# 2.3 Huber Loss (平滑 L1)
print("\n2.3 Huber Loss (Smooth L1):")
print("    在 |error| < δ 时用 L2，否则用 L1")
huber = nn.HuberLoss(delta=1.0)
loss = huber(y_pred, y_true)
print(f"    Huber = {loss.item():.4f}")

# =============================================================================
# 3. 二分类损失
# =============================================================================
print("\n" + "=" * 60)
print("【3. 二分类损失】")

# 3.1 BCELoss (Binary Cross Entropy)
print("\n3.1 BCELoss:")
print("    公式: BCE = -[y×log(p) + (1-y)×log(1-p)]")
print("    ⚠️ 输入必须是概率 (0-1)，需要先过 Sigmoid")

y_pred_logits = torch.tensor([1.0, -1.0, 0.5, 2.0])
y_true = torch.tensor([1.0, 0.0, 1.0, 1.0])

# 先 sigmoid，再 BCELoss
y_pred_prob = torch.sigmoid(y_pred_logits)
bce = nn.BCELoss()
loss = bce(y_pred_prob, y_true)
print(f"    预测 logits: {y_pred_logits}")
print(f"    预测概率: {y_pred_prob}")
print(f"    BCE Loss: {loss.item():.4f}")

# 3.2 BCEWithLogitsLoss (推荐)
print("\n3.2 BCEWithLogitsLoss (推荐):")
print("    = Sigmoid + BCELoss，数值更稳定")
bce_logits = nn.BCEWithLogitsLoss()
loss = bce_logits(y_pred_logits, y_true)
print(f"    BCE with Logits: {loss.item():.4f}")

# =============================================================================
# 4. 多分类损失
# =============================================================================
print("\n" + "=" * 60)
print("【4. 多分类损失】")

# 4.1 CrossEntropyLoss
print("\n4.1 CrossEntropyLoss:")
print("    = LogSoftmax + NLLLoss")
print("    输入: logits (未经 softmax)")
print("    标签: 类别索引 (不是 one-hot)")

# 3 个样本，5 个类别
logits = torch.tensor([
    [2.0, 1.0, 0.1, 0.5, 0.3],   # 样本 0
    [0.1, 3.0, 0.2, 0.4, 0.1],   # 样本 1
    [0.5, 0.3, 2.5, 0.1, 0.2]    # 样本 2
])
labels = torch.tensor([0, 1, 2])  # 正确类别

ce = nn.CrossEntropyLoss()
loss = ce(logits, labels)
print(f"    Logits 形状: {logits.shape}")
print(f"    Labels: {labels}")
print(f"    CrossEntropy Loss: {loss.item():.4f}")

# 手动验证
print("\n    手动计算:")
probs = F.softmax(logits, dim=1)
print(f"    Softmax 概率:\n{probs}")
log_probs = torch.log(probs)
manual_ce = -log_probs[range(3), labels].mean()
print(f"    手动 CE: {manual_ce.item():.4f}")

# 4.2 带权重的交叉熵
print("\n4.2 带权重的 CrossEntropyLoss (处理类别不平衡):")
weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0])  # 类别 1, 2 权重更高
ce_weighted = nn.CrossEntropyLoss(weight=weights)
loss = ce_weighted(logits, labels)
print(f"    权重: {weights}")
print(f"    加权 CE Loss: {loss.item():.4f}")

# 4.3 Label Smoothing
print("\n4.3 Label Smoothing:")
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = ce_smooth(logits, labels)
print(f"    Label Smoothing=0.1 CE Loss: {loss.item():.4f}")

# =============================================================================
# 5. NLLLoss
# =============================================================================
print("\n" + "=" * 60)
print("【5. NLLLoss】")

print("""
NLLLoss (Negative Log Likelihood):
- 输入: log 概率 (需要先经过 LogSoftmax)
- CrossEntropyLoss = LogSoftmax + NLLLoss
""")

log_probs = F.log_softmax(logits, dim=1)
nll = nn.NLLLoss()
loss = nll(log_probs, labels)
print(f"NLLLoss: {loss.item():.4f}")

# =============================================================================
# 6. 其他常用损失
# =============================================================================
print("\n" + "=" * 60)
print("【6. 其他常用损失】")

# 6.1 KL 散度
print("\n6.1 KL Divergence Loss (知识蒸馏):")
p = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # 目标分布
q = torch.tensor([[0.2, 0.8], [0.7, 0.3]])  # 预测分布

kl = nn.KLDivLoss(reduction='batchmean')
# 注意: 输入是 log 概率
loss = kl(torch.log(q), p)
print(f"    KL(p || q) = {loss.item():.4f}")

# 6.2 Cosine Embedding Loss
print("\n6.2 Cosine Embedding Loss (相似度学习):")
x1 = torch.randn(3, 10)
x2 = torch.randn(3, 10)
y = torch.tensor([1, -1, 1])  # 1: 相似, -1: 不相似
cosine = nn.CosineEmbeddingLoss()
loss = cosine(x1, x2, y)
print(f"    Cosine Embedding Loss: {loss.item():.4f}")

# 6.3 Triplet Margin Loss
print("\n6.3 Triplet Margin Loss (对比学习):")
anchor = torch.randn(3, 10)
positive = torch.randn(3, 10)  # 与 anchor 同类
negative = torch.randn(3, 10)  # 与 anchor 不同类
triplet = nn.TripletMarginLoss(margin=1.0)
loss = triplet(anchor, positive, negative)
print(f"    Triplet Loss: {loss.item():.4f}")

# =============================================================================
# 7. 自定义损失函数
# =============================================================================
print("\n" + "=" * 60)
print("【7. 自定义损失函数】")

# 方法 1: 普通函数
def focal_loss(logits, labels, gamma=2.0, alpha=1.0):
    """Focal Loss: 解决类别不平衡"""
    ce = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce)  # 预测正确类别的概率
    focal = alpha * (1 - pt) ** gamma * ce
    return focal.mean()

loss = focal_loss(logits, labels)
print(f"Focal Loss: {loss.item():.4f}")

# 方法 2: 继承 nn.Module
class DiceLoss(nn.Module):
    """Dice Loss: 常用于图像分割"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (N, C, H, W), target: (N, H, W)
        pred = torch.softmax(pred, dim=1)
        # 简化为二分类演示
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

# =============================================================================
# 8. reduction 参数
# =============================================================================
print("\n" + "=" * 60)
print("【8. reduction 参数】")

print("""
reduction 控制如何聚合各样本的损失:
- 'mean': 返回平均值 (默认)
- 'sum': 返回总和
- 'none': 返回每个样本的损失
""")

y_pred = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([1.5, 2.5, 3.5])

print(f"reduction='none': {F.mse_loss(y_pred, y_true, reduction='none')}")
print(f"reduction='mean': {F.mse_loss(y_pred, y_true, reduction='mean')}")
print(f"reduction='sum': {F.mse_loss(y_pred, y_true, reduction='sum')}")

# =============================================================================
# 9. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 给定预测 [0.9, 0.1] 和真实标签 [1, 0]，计算交叉熵

2. 实现带 L2 正则化的损失函数

3. 实现 Label Smoothing 的交叉熵

4. 什么时候用 BCELoss，什么时候用 BCEWithLogitsLoss？

5. 解释 Focal Loss 如何解决类别不平衡问题
""")

# === 练习答案 ===
# 1
# pred = torch.tensor([[0.9, 0.1]])
# label = torch.tensor([0])
# loss = F.cross_entropy(pred, label)
# print(f"CE: {loss.item():.4f}")

# 2
# def loss_with_l2(pred, target, model, lambda_l2=0.01):
#     ce_loss = F.cross_entropy(pred, target)
#     l2_reg = sum(p.pow(2).sum() for p in model.parameters())
#     return ce_loss + lambda_l2 * l2_reg

# 3
# def cross_entropy_with_smoothing(logits, labels, smoothing=0.1):
#     n_classes = logits.size(-1)
#     one_hot = F.one_hot(labels, n_classes).float()
#     smooth_labels = one_hot * (1 - smoothing) + smoothing / n_classes
#     log_probs = F.log_softmax(logits, dim=-1)
#     return -(smooth_labels * log_probs).sum(dim=-1).mean()

# 4
# 答案: BCELoss 需要输入已经是概率 (0-1)，需要先过 Sigmoid
#       BCEWithLogitsLoss 内部包含 Sigmoid，数值更稳定，推荐使用

# 5
# 答案: Focal Loss 通过 (1-pt)^γ 因子降低简单样本的权重
#       γ 越大，对难分类样本的关注越多
#       适用于正负样本极度不平衡的情况（如目标检测）

print("\n✅ 损失函数完成！")
print("下一步：06-optimizers.py - 优化器")
