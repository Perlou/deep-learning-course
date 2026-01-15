"""
BERT中文情感分类
使用HuggingFace Transformers实现
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np


# ==================== 示例数据 ====================


def create_sample_data():
    """创建示例情感分析数据"""
    # 正面评论
    positive_reviews = [
        "这部电影太精彩了，我非常喜欢！",
        "演员的表演很出色，剧情也很吸引人。",
        "非常值得一看的好电影，强烈推荐！",
        "导演的功力深厚，每个镜头都很用心。",
        "看完之后心情很好，是一部温暖的作品。",
    ]

    # 负面评论
    negative_reviews = [
        "剧情拖沓无聊，浪费时间。",
        "演技尴尬，完全看不下去。",
        "太糟糕了，不建议观看。",
        "逻辑混乱，不知道在讲什么。",
        "特效廉价，制作粗糙。",
    ]

    # 合并数据
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1=正面, 0=负面

    return texts, labels


# ==================== BERT分类器 ====================


class BERTClassifier(nn.Module):
    """BERT情感分类器"""

    def __init__(self, bert_model_name="bert-base-chinese", num_classes=2, dropout=0.1):
        super().__init__()

        # 加载预训练BERT
        self.bert = BertModel.from_pretrained(bert_model_name)

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 使用[CLS] token的输出
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)

        # Dropout + 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)

        return logits


# ==================== 数据预处理 ====================


def prepare_data(texts, labels, tokenizer, max_length=128):
    """准备训练数据"""
    encodings = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    labels = torch.LongTensor(labels)

    return encodings, labels


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("BERT中文情感分类")
    print("=" * 60)

    # 创建示例数据
    print("\n加载示例数据...")
    texts, labels = create_sample_data()
    print(
        f"共 {len(texts)} 个样本 (正面: {sum(labels)}, 负面: {len(labels) - sum(labels)})"
    )

    print("\n示例评论:")
    for i in range(2):
        label_str = "正面" if labels[i] == 1 else "负面"
        print(f"  [{label_str}] {texts[i]}")

    # 加载tokenizer
    print("\n加载BERT tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        print("✓ Tokenizer加载成功")
    except Exception as e:
        print(f"✗ 无法加载bert-base-chinese: {e}")
        print("  请先下载模型或使用HuggingFace镜像")
        print("  提示: export HF_ENDPOINT=https://hf-mirror.com")
        import sys

        sys.exit(1)

    # 测试tokenization
    sample_text = texts[0]
    tokens = tokenizer.tokenize(sample_text)
    print(f"\n分词示例:")
    print(f"  原文: {sample_text}")
    print(f"  分词: {' / '.join(tokens[:10])}...")

    # 准备数据
    encodings, label_tensor = prepare_data(texts, labels, tokenizer)
    print(f"\n数据编码:")
    print(f"  input_ids: {encodings['input_ids'].shape}")
    print(f"  attention_mask: {encodings['attention_mask'].shape}")
    print(f"  labels: {label_tensor.shape}")

    # 创建模型
    print("\n创建BERT分类器...")
    try:
        model = BERTClassifier(num_classes=2)
        print("✓ 模型创建成功")

        # 模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数: {total_params / 1e6:.1f}M")
        print(f"  可训练参数: {trainable_params / 1e6:.1f}M")

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import sys

        sys.exit(1)

    # 测试前向传播
    print("\n测试前向传播...")
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"]
        )

    print(f"  输出logits: {logits.shape}")

    # 预测
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    print(f"\n随机初始化的预测（未训练）:")
    for i in range(3):
        label = "正面" if labels[i] == 1 else "负面"
        pred = "正面" if preds[i] == 1 else "负面"
        conf = probs[i][preds[i]].item()
        print(f"  真实: {label} | 预测: {pred} (置信度: {conf:.2f})")

    print("\n✓ 模型测试成功!")
    print("请运行 train.py 开始训练")
