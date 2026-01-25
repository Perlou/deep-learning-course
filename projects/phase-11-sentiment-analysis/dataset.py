"""
数据集加载
==========

加载和预处理情感分析数据集。
"""

import os
import re
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, DATASET_CONFIG


# ===========================================
# 数据生成 (演示用)
# ===========================================

POSITIVE_TEMPLATES = [
    "I love this movie, it's {adj}!",
    "What a {adj} film, highly recommended!",
    "This is one of the best movies I've ever seen, {adj}!",
    "Absolutely {adj}! Great acting and storyline.",
    "A {adj} masterpiece, must watch!",
    "{adj} movie with great performances.",
    "Really enjoyed this {adj} film!",
    "The movie was {adj}, loved every minute.",
    "Such a {adj} experience, wonderful movie!",
    "Best movie of the year, truly {adj}!",
]

NEGATIVE_TEMPLATES = [
    "This movie was {adj}, don't waste your time.",
    "What a {adj} film, very disappointed.",
    "I regret watching this {adj} movie.",
    "Absolutely {adj}! Terrible acting and plot.",
    "A {adj} disaster, avoid at all costs!",
    "{adj} movie with poor performances.",
    "Really hated this {adj} film!",
    "The movie was {adj}, waste of time.",
    "Such a {adj} experience, awful movie!",
    "Worst movie of the year, truly {adj}!",
]

POSITIVE_ADJECTIVES = [
    "amazing",
    "wonderful",
    "fantastic",
    "excellent",
    "brilliant",
    "outstanding",
    "superb",
    "magnificent",
    "incredible",
    "awesome",
    "great",
    "perfect",
    "beautiful",
    "touching",
    "inspiring",
]

NEGATIVE_ADJECTIVES = [
    "terrible",
    "awful",
    "horrible",
    "dreadful",
    "boring",
    "disappointing",
    "bad",
    "poor",
    "mediocre",
    "pathetic",
    "waste",
    "stupid",
    "annoying",
    "painful",
    "unbearable",
]


def generate_sample_dataset(num_samples=1000):
    """生成示例情感数据集"""
    data = []

    for _ in range(num_samples // 2):
        # 正面样本
        template = random.choice(POSITIVE_TEMPLATES)
        adj = random.choice(POSITIVE_ADJECTIVES)
        text = template.format(adj=adj)
        data.append((text, 1))

        # 负面样本
        template = random.choice(NEGATIVE_TEMPLATES)
        adj = random.choice(NEGATIVE_ADJECTIVES)
        text = template.format(adj=adj)
        data.append((text, 0))

    random.shuffle(data)
    return data


def prepare_dataset(data_dir=None, num_samples=2000):
    """
    准备数据集

    Args:
        data_dir: 数据目录
        num_samples: 样本数量

    Returns:
        train_data, val_data, vocab
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"生成 {num_samples} 个情感分析样本...")
    data = generate_sample_dataset(num_samples)

    # 划分数据集
    split_idx = int(len(data) * DATASET_CONFIG["train_split"])
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # 构建词表
    vocab = build_vocab([text for text, _ in train_data])

    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    print(f"词表大小: {len(vocab)}")

    return train_data, val_data, vocab


# ===========================================
# 词表构建
# ===========================================


def tokenize(text):
    """简单分词"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_vocab(texts, max_size=None):
    """
    构建词表

    Args:
        texts: 文本列表
        max_size: 词表最大大小

    Returns:
        vocab: {word: index} 字典
    """
    word_counts = Counter()
    for text in texts:
        tokens = tokenize(text)
        word_counts.update(tokens)

    # 特殊 token
    vocab = {"<PAD>": 0, "<UNK>": 1}

    # 按频率排序
    max_size = max_size or DATASET_CONFIG["vocab_size"]
    for word, _ in word_counts.most_common(max_size - 2):
        vocab[word] = len(vocab)

    return vocab


def text_to_indices(text, vocab, max_length=None):
    """
    文本转索引

    Args:
        text: 输入文本
        vocab: 词表
        max_length: 最大长度

    Returns:
        indices: 索引列表
    """
    max_length = max_length or DATASET_CONFIG["max_length"]
    tokens = tokenize(text)

    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    # 截断或填充
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [vocab["<PAD>"]] * (max_length - len(indices))

    return indices


# ===========================================
# 数据集类
# ===========================================


class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(self, data, vocab, max_length=None):
        """
        Args:
            data: [(text, label), ...] 列表
            vocab: 词表字典
            max_length: 最大序列长度
        """
        self.data = data
        self.vocab = vocab
        self.max_length = max_length or DATASET_CONFIG["max_length"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        indices = text_to_indices(text, self.vocab, self.max_length)

        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


class BertSentimentDataset(Dataset):
    """BERT 情感分析数据集"""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ===========================================
# 数据加载器
# ===========================================


def get_dataloaders(
    data_dir=None,
    batch_size=32,
    num_workers=4,
    model_type="textcnn",
    tokenizer=None,
):
    """
    获取数据加载器

    Args:
        data_dir: 数据目录
        batch_size: 批量大小
        num_workers: 数据加载线程数
        model_type: 模型类型 ("textcnn", "lstm", "bert")
        tokenizer: BERT tokenizer (仅 bert 模型需要)

    Returns:
        train_loader, val_loader, vocab_size
    """
    # 准备数据
    train_data, val_data, vocab = prepare_dataset(data_dir)

    if model_type in ["textcnn", "lstm"]:
        train_dataset = SentimentDataset(train_data, vocab)
        val_dataset = SentimentDataset(val_data, vocab)
        vocab_size = len(vocab)

    elif model_type == "bert":
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            except ImportError:
                raise ImportError("请安装 transformers: pip install transformers")

        train_dataset = BertSentimentDataset(train_data, tokenizer)
        val_dataset = BertSentimentDataset(val_data, tokenizer)
        vocab_size = tokenizer.vocab_size
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, vocab_size, vocab


# ===========================================
# 测试
# ===========================================

if __name__ == "__main__":
    print("=" * 50)
    print("测试数据集加载")
    print("=" * 50)

    train_loader, val_loader, vocab_size, vocab = get_dataloaders(
        batch_size=4, num_workers=0, model_type="textcnn"
    )

    print(f"\n词表大小: {vocab_size}")
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")

    # 测试一批数据
    texts, labels = next(iter(train_loader))
    print(f"\n文本形状: {texts.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签值: {labels.tolist()}")
