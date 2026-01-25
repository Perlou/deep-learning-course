"""
情感分析模型
============

实现 TextCNN、BiLSTM 和 BERT 分类器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================
# TextCNN 模型
# ===========================================


class TextCNN(nn.Module):
    """
    TextCNN 文本分类模型 (Kim, 2014)

    使用多尺度卷积核捕捉不同长度的 n-gram 特征
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_classes=2,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        dropout=0.5,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # 多尺度卷积
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size)
                for kernel_size in kernel_sizes
            ]
        )

        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) 词索引
        Returns:
            logits: (batch_size, num_classes)
        """
        # 词嵌入: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # 转置: (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # 多尺度卷积 + 最大池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # 拼接: (batch, num_filters * len(kernel_sizes))
        cat_out = torch.cat(conv_outputs, dim=1)

        # Dropout + 分类
        out = self.dropout(cat_out)
        logits = self.fc(out)

        return logits


# ===========================================
# BiLSTM 模型
# ===========================================


class BiLSTMClassifier(nn.Module):
    """
    双向 LSTM 文本分类模型
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=2,
        num_layers=2,
        bidirectional=True,
        dropout=0.5,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # 双向时输出维度翻倍
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """
        # 词嵌入
        embedded = self.embedding(x)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 使用平均池化
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        pooled = lstm_out.mean(dim=1)

        out = self.dropout(pooled)
        logits = self.fc(out)

        return logits


# ===========================================
# BERT 分类器
# ===========================================


class BertClassifier(nn.Module):
    """
    基于 HuggingFace Transformers 的 BERT 分类器
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_classes=2,
        freeze_bert=False,
    ):
        super().__init__()

        try:
            from transformers import AutoModel

            self.bert = AutoModel.from_pretrained(model_name)
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        # 冻结 BERT 参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 分类头
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # 使用 [CLS] token 的输出
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        return logits


# ===========================================
# 模型工厂
# ===========================================


def create_model(model_name, vocab_size=None, **kwargs):
    """
    创建模型

    Args:
        model_name: "textcnn", "lstm", "bert"
        vocab_size: 词表大小 (textcnn/lstm 需要)
        **kwargs: 模型参数

    Returns:
        model: PyTorch 模型
    """
    if model_name == "textcnn":
        if vocab_size is None:
            raise ValueError("TextCNN 需要 vocab_size 参数")
        return TextCNN(vocab_size=vocab_size, **kwargs)

    elif model_name == "lstm":
        if vocab_size is None:
            raise ValueError("LSTM 需要 vocab_size 参数")
        return BiLSTMClassifier(vocab_size=vocab_size, **kwargs)

    elif model_name == "bert":
        return BertClassifier(**kwargs)

    else:
        raise ValueError(f"未知模型: {model_name}")


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================
# 测试
# ===========================================

if __name__ == "__main__":
    print("=" * 50)
    print("测试 TextCNN")
    print("=" * 50)

    model = TextCNN(vocab_size=10000)
    x = torch.randint(0, 10000, (4, 128))
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print(f"参数量: {count_parameters(model):,}")

    print("\n" + "=" * 50)
    print("测试 BiLSTM")
    print("=" * 50)

    model = BiLSTMClassifier(vocab_size=10000)
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print(f"参数量: {count_parameters(model):,}")

    print("\n测试通过!")
