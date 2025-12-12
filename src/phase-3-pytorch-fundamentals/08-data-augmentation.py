"""
08-data-augmentation.py
Phase 3: PyTorch 核心技能

数据增强 - 提升模型泛化能力

学习目标：
1. 理解数据增强的作用
2. 掌握图像数据增强技术
3. 了解文本和时间序列的增强方法
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("PyTorch 核心技能 - 数据增强")
print("=" * 60)

# =============================================================================
# 1. 数据增强概述
# =============================================================================
print("\n【1. 数据增强概述】")

print("""
数据增强的作用:
1. 增加数据多样性
2. 防止过拟合
3. 提升模型泛化能力
4. 模拟真实世界的变化

常见增强方式:
- 图像: 翻转、旋转、裁剪、颜色变换
- 文本: 同义词替换、回译、随机删除
- 时间序列: 加噪、时间变形
""")

# =============================================================================
# 2. torchvision.transforms
# =============================================================================
print("\n" + "=" * 60)
print("【2. torchvision.transforms (图像增强)】")

print("""
常用变换:
from torchvision import transforms

# 基础变换
transforms.ToTensor()           # PIL/ndarray → Tensor
transforms.Normalize(mean, std) # 标准化
transforms.Resize(size)         # 调整大小
transforms.CenterCrop(size)     # 中心裁剪

# 数据增强
transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转
transforms.RandomVerticalFlip(p=0.5)    # 随机垂直翻转
transforms.RandomRotation(degrees)      # 随机旋转
transforms.RandomCrop(size)             # 随机裁剪
transforms.RandomResizedCrop(size)      # 随机缩放裁剪
transforms.ColorJitter(...)             # 颜色抖动

# 组合变换
transforms.Compose([...])
""")

# 训练时的增强示例 (代码展示，不实际运行)
train_transform_code = """
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
"""
print("训练集增强配置:")
print(train_transform_code)

# 验证时不增强
val_transform_code = """
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
"""
print("验证集配置 (不增强):")
print(val_transform_code)

# =============================================================================
# 3. 高级图像增强
# =============================================================================
print("\n" + "=" * 60)
print("【3. 高级图像增强】")

print("""
高级增强技术:

1. AutoAugment / RandAugment:
   - 自动学习增强策略
   transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
   transforms.RandAugment(num_ops=2, magnitude=9)

2. MixUp:
   - 混合两张图片及其标签
   x_mix = λ * x1 + (1-λ) * x2
   y_mix = λ * y1 + (1-λ) * y2

3. CutOut:
   - 随机遮挡图片的一部分
   
4. CutMix:
   - 剪切一张图片的一部分贴到另一张上

5. Mosaic:
   - 四张图片拼接 (YOLO)
""")

# MixUp 实现
def mixup_data(x, y, alpha=0.2):
    """MixUp 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print("MixUp 实现已定义")

# CutOut 实现
class CutOut:
    """CutOut 数据增强"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0
        
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

print("CutOut 实现已定义")

# =============================================================================
# 4. 文本数据增强
# =============================================================================
print("\n" + "=" * 60)
print("【4. 文本数据增强】")

print("""
文本增强技术:

1. 同义词替换 (Synonym Replacement)
2. 随机插入 (Random Insertion)
3. 随机交换 (Random Swap)
4. 随机删除 (Random Deletion)
5. 回译 (Back Translation)
6. EDA (Easy Data Augmentation): 结合以上方法
""")

def random_deletion(words, p=0.1):
    """随机删除词"""
    if len(words) == 1:
        return words
    
    new_words = [w for w in words if np.random.random() > p]
    
    if len(new_words) == 0:
        return [np.random.choice(words)]
    
    return new_words

def random_swap(words, n=1):
    """随机交换词位置"""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) >= 2:
            idx1, idx2 = np.random.choice(len(new_words), 2, replace=False)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

# 示例
text = "PyTorch is a great deep learning framework".split()
print(f"\n原文: {' '.join(text)}")
print(f"随机删除: {' '.join(random_deletion(text, p=0.2))}")
print(f"随机交换: {' '.join(random_swap(text, n=2))}")

# =============================================================================
# 5. 时间序列数据增强
# =============================================================================
print("\n" + "=" * 60)
print("【5. 时间序列数据增强】")

print("""
时间序列增强:

1. 加噪声 (Jittering)
2. 缩放 (Scaling)
3. 时间变形 (Time Warping)
4. 窗口切片 (Window Slicing)
5. 置换 (Permutation)
""")

def jittering(x, sigma=0.03):
    """加高斯噪声"""
    return x + np.random.normal(0, sigma, x.shape)

def scaling(x, sigma=0.1):
    """缩放"""
    factor = np.random.normal(1, sigma, (1, x.shape[1]))
    return x * factor

# 示例
ts = np.sin(np.linspace(0, 4*np.pi, 100)).reshape(1, -1)
ts_jittered = jittering(ts)
ts_scaled = scaling(ts)
print(f"原始序列范围: [{ts.min():.2f}, {ts.max():.2f}]")
print(f"加噪后范围: [{ts_jittered.min():.2f}, {ts_jittered.max():.2f}]")
print(f"缩放后范围: [{ts_scaled.min():.2f}, {ts_scaled.max():.2f}]")

# =============================================================================
# 6. 在训练循环中使用增强
# =============================================================================
print("\n" + "=" * 60)
print("【6. 在训练循环中使用增强】")

training_code = """
# 方式 1: 在 Dataset 的 transform 中
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=train_transform  # 包含增强
)

# 方式 2: 在训练循环中 (MixUp/CutMix)
for x, y in train_loader:
    # 应用 MixUp
    x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    
    # 前向传播
    outputs = model(x)
    
    # 计算混合损失
    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
print(training_code)

# =============================================================================
# 7. 增强策略建议
# =============================================================================
print("\n" + "=" * 60)
print("【7. 增强策略建议】")

print("""
╔════════════════════════════════════════════════════════════╗
║                    增强策略建议                            ║
╠════════════════╦═══════════════════════════════════════════╣
║  任务          ║  推荐增强                                 ║
╠════════════════╬═══════════════════════════════════════════╣
║  图像分类      ║  RandAugment + MixUp/CutMix              ║
║  目标检测      ║  Mosaic + 随机缩放 + 颜色变换            ║
║  语义分割      ║  随机裁剪 + 翻转 + 颜色变换              ║
║  文本分类      ║  EDA + 回译                              ║
║  时间序列      ║  Jittering + Scaling + Window Slicing    ║
╚════════════════╩═══════════════════════════════════════════╝

注意事项:
1. 验证集和测试集不要使用增强
2. 增强强度要适度，过强会损害性能
3. 不同任务需要不同的增强策略
4. 可以使用 AutoAugment 自动搜索策略
""")

# =============================================================================
# 8. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 为 CIFAR-10 设计一个合适的训练增强 pipeline

2. 实现 CutMix 数据增强

3. 为文本数据实现 EDA (至少包含 2 种操作)

4. 解释为什么验证集不应该使用数据增强

5. 比较 MixUp 和 CutMix 的优缺点
""")

# === 练习答案 ===
# 1
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandAugment(num_ops=2, magnitude=9),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                         std=[0.2023, 0.1994, 0.2010])
# ])

# 2
# def cutmix(x, y, alpha=1.0):
#     lam = np.random.beta(alpha, alpha)
#     rand_index = torch.randperm(x.size(0))
#     y_a, y_b = y, y[rand_index]
#     
#     bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
#     x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
#     
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
#     return x, y_a, y_b, lam

# 3
# def eda(text, p=0.1):
#     words = text.split()
#     words = random_deletion(words, p)
#     words = random_swap(words, n=1)
#     return ' '.join(words)

# 4
# 答案: 验证集用于评估模型泛化能力，需要与测试集条件一致
#       数据增强会改变数据分布，导致验证指标不准确

# 5
# MixUp: 全图混合，更平滑，可能模糊边界
# CutMix: 局部替换，保持局部特征，更适合目标检测

print("\n✅ 数据增强完成！")
print("下一步：09-training-loop.py - 完整训练流程")
