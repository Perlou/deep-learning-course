"""
07-dataset-dataloader.py
Phase 3: PyTorch 核心技能

数据加载 - Dataset 和 DataLoader

学习目标：
1. 理解 Dataset 和 DataLoader 的作用
2. 掌握自定义 Dataset 的方法
3. 了解数据加载的最佳实践
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split, Subset
import numpy as np

print("=" * 60)
print("PyTorch 核心技能 - Dataset 和 DataLoader")
print("=" * 60)

# =============================================================================
# 1. Dataset 和 DataLoader 简介
# =============================================================================
print("\n【1. Dataset 和 DataLoader 简介】")

print("""
Dataset: 数据集的抽象
- __len__(): 返回数据集大小
- __getitem__(idx): 返回第 idx 个样本

DataLoader: 数据加载器
- 批量加载
- 打乱顺序
- 多进程加载
- 自动拼接
""")

# =============================================================================
# 2. 使用 TensorDataset
# =============================================================================
print("\n【2. 使用 TensorDataset】")

# 创建数据
X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))

# 创建 TensorDataset
dataset = TensorDataset(X, y)
print(f"数据集大小: {len(dataset)}")
print(f"第一个样本: X={dataset[0][0].shape}, y={dataset[0][1]}")

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0  # 主进程加载
)

print(f"\n遍历 DataLoader:")
for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    if batch_idx < 3:
        print(f"  Batch {batch_idx}: X={batch_x.shape}, y={batch_y.shape}")

# =============================================================================
# 3. 自定义 Dataset
# =============================================================================
print("\n" + "=" * 60)
print("【3. 自定义 Dataset】")

class CustomDataset(Dataset):
    def __init__(self, num_samples=100, num_features=10):
        """初始化：加载或创建数据"""
        self.X = torch.randn(num_samples, num_features)
        self.y = torch.randint(0, 3, (num_samples,))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """返回第 idx 个样本"""
        return self.X[idx], self.y[idx]

# 使用自定义数据集
dataset = CustomDataset(num_samples=100)
print(f"自定义数据集大小: {len(dataset)}")
print(f"样本示例: {dataset[0]}")

# =============================================================================
# 4. 更复杂的自定义 Dataset
# =============================================================================
print("\n" + "=" * 60)
print("【4. 更复杂的自定义 Dataset】")

class TextDataset(Dataset):
    """文本数据集示例"""
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 简单的 tokenize
        tokens = text.lower().split()
        
        # 转换为 ID
        ids = [self.vocab.get(t, 0) for t in tokens]  # 0 是 UNK
        
        # 填充或截断
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))  # 填充
        else:
            ids = ids[:self.max_len]  # 截断
        
        return torch.tensor(ids), torch.tensor(label)

# 示例
texts = ["hello world", "this is a test", "pytorch is great"]
labels = [0, 1, 0]
vocab = {"hello": 1, "world": 2, "this": 3, "is": 4, "a": 5, "test": 6, "pytorch": 7, "great": 8}

text_dataset = TextDataset(texts, labels, vocab, max_len=10)
print(f"文本数据集大小: {len(text_dataset)}")
ids, label = text_dataset[0]
print(f"样本: ids={ids}, label={label}")

# =============================================================================
# 5. DataLoader 参数详解
# =============================================================================
print("\n" + "=" * 60)
print("【5. DataLoader 参数详解】")

print("""
重要参数:
- batch_size: 批大小
- shuffle: 是否打乱 (训练集 True，验证集 False)
- num_workers: 加载数据的进程数
- drop_last: 是否丢弃最后不足一批的数据
- pin_memory: 是否锁页内存 (GPU 训练时设为 True)
- collate_fn: 自定义批次合并函数
""")

dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 3, (100,)))

# 训练用 DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,       # 训练时打乱
    num_workers=0,      # 多进程 (Mac 上可能有问题)
    drop_last=True,     # 丢弃不足一批的
    pin_memory=False    # GPU 时设为 True
)

# 验证用 DataLoader
val_loader = DataLoader(
    dataset,
    batch_size=64,      # 可以更大
    shuffle=False,      # 不打乱
    num_workers=0
)

print(f"训练批数 (drop_last=True): {len(train_loader)}")
print(f"验证批数: {len(val_loader)}")

# =============================================================================
# 6. collate_fn 自定义批次处理
# =============================================================================
print("\n" + "=" * 60)
print("【6. collate_fn 自定义批次处理】")

def custom_collate(batch):
    """自定义批次合并"""
    # batch 是 [(x1, y1), (x2, y2), ...] 的列表
    xs, ys = zip(*batch)
    
    # 转换为张量
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    
    # 可以在这里做更多处理
    return xs, ys, xs.shape[0]  # 返回额外信息

loader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate)
x, y, batch_size = next(iter(loader))
print(f"使用 collate_fn: x={x.shape}, y={y.shape}, batch_size={batch_size}")

# 变长序列的 collate_fn
def pad_collate(batch):
    """处理变长序列"""
    seqs, labels = zip(*batch)
    
    # 找最长序列
    max_len = max(len(s) for s in seqs)
    
    # 填充
    padded = torch.zeros(len(seqs), max_len)
    lengths = torch.tensor([len(s) for s in seqs])
    
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    
    return padded, torch.tensor(labels), lengths

print("变长序列 collate_fn 示例已定义")

# =============================================================================
# 7. 数据集划分
# =============================================================================
print("\n" + "=" * 60)
print("【7. 数据集划分】")

# 方法 1: random_split
dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 3, (100,)))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"random_split:")
print(f"  训练集: {len(train_dataset)}")
print(f"  验证集: {len(val_dataset)}")

# 方法 2: Subset (按索引划分)
indices = list(range(len(dataset)))
np.random.shuffle(indices)
train_indices = indices[:80]
val_indices = indices[80:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print(f"\nSubset:")
print(f"  训练集: {len(train_dataset)}")
print(f"  验证集: {len(val_dataset)}")

# =============================================================================
# 8. 使用 torchvision 数据集
# =============================================================================
print("\n" + "=" * 60)
print("【8. 使用 torchvision 数据集】")

print("""
torchvision.datasets 提供常用数据集:
- MNIST, FashionMNIST, CIFAR10, CIFAR100
- ImageNet, COCO
- 自定义: ImageFolder

示例:
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
""")

# =============================================================================
# 9. 最佳实践
# =============================================================================
print("\n" + "=" * 60)
print("【9. 最佳实践】")

print("""
1. num_workers 设置:
   - CPU 核心数的 2-4 倍
   - Windows 上可能需要 0
   - 过大会增加内存和 CPU 开销

2. pin_memory:
   - GPU 训练时设为 True
   - 加速 CPU → GPU 数据传输

3. prefetch_factor (PyTorch 1.7+):
   - 每个 worker 预取的批次数
   - 默认 2

4. persistent_workers (PyTorch 1.7+):
   - 保持 worker 进程存活
   - 减少启动开销

5. 内存优化:
   - 使用生成器/迭代器
   - 按需加载数据
   - 使用 memmap 处理大文件
""")

# =============================================================================
# 10. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 创建一个自定义 Dataset，从列表加载数据

2. 创建 DataLoader，设置 batch_size=32, shuffle=True

3. 实现一个处理变长序列的 collate_fn

4. 将数据集按 8:2 划分为训练集和验证集

5. 解释 num_workers > 0 时的注意事项
""")

# === 练习答案 ===
# 1
# class ListDataset(Dataset):
#     def __init__(self, data_list):
#         self.data = data_list
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         return self.data[idx]

# 2
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3
# def var_len_collate(batch):
#     seqs = [item[0] for item in batch]
#     labels = torch.tensor([item[1] for item in batch])
#     lengths = torch.tensor([len(s) for s in seqs])
#     padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
#     return padded, labels, lengths

# 4
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_ds, val_ds = random_split(dataset, [train_size, val_size])

# 5
# 注意事项:
# - Windows 需要在 if __name__ == '__main__' 中使用
# - Dataset 必须是可序列化的 (不能包含 lambda)
# - 调试时先用 num_workers=0
# - 内存会翻倍 (每个 worker 复制数据)

print("\n✅ Dataset 和 DataLoader 完成！")
print("下一步：08-data-augmentation.py - 数据增强")
