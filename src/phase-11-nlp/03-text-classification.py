"""
文本分类
========

学习目标：
    1. 理解文本分类任务和应用场景
    2. 掌握 TextCNN 和 LSTM 文本分类模型
    3. 实现情感分析任务
    4. 了解文本分类的评估方法

核心概念：
    - 文本分类：为文本分配类别标签
    - TextCNN：使用卷积神经网络处理文本
    - LSTM 分类：使用循环神经网络捕捉序列信息
    - 情感分析：判断文本的情感倾向

前置知识：
    - Phase 5: CNN 基础
    - Phase 6: RNN/LSTM 基础
    - 01-word2vec.py: 词向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== 第一部分：文本分类概述 ====================


def introduction():
    """文本分类介绍"""
    print("=" * 60)
    print("第一部分：文本分类概述")
    print("=" * 60)

    print("""
文本分类任务：

    输入：一段文本
    输出：类别标签

常见应用：
    - 情感分析：正面/负面/中性
    - 垃圾邮件检测：垃圾/正常
    - 新闻分类：体育/财经/科技/...
    - 意图识别：查询/购买/投诉/...

处理流程：
    文本 → 分词 → 词向量 → 编码器 → 分类器 → 类别
    
    编码器选择：
    - CNN：捕捉局部 n-gram 特征
    - RNN/LSTM：捕捉序列信息
    - Transformer：现代最佳选择
    """)


# ==================== 第二部分：TextCNN 模型 ====================


class TextCNN(nn.Module):
    """TextCNN 文本分类模型 (Kim, 2014)"""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        dropout=0.5,
    ):
        """
        Args:
            vocab_size: 词表大小
            embed_dim: 词向量维度
            num_classes: 类别数
            kernel_sizes: 卷积核大小列表
            num_filters: 每种卷积核的数量
            dropout: Dropout 比例
        """
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 多种尺寸的卷积层
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size)
                for kernel_size in kernel_sizes
            ]
        )

        # 全连接分类层
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 输入文本索引 (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """
        # 词嵌入: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # 转置用于卷积: (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # 多尺度卷积 + ReLU + 最大池化
        conv_outputs = []
        for conv in self.convs:
            # 卷积: (batch, num_filters, seq_len - kernel_size + 1)
            conv_out = F.relu(conv(embedded))
            # 最大池化: (batch, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # 拼接所有卷积输出: (batch, num_filters * len(kernel_sizes))
        cat_out = torch.cat(conv_outputs, dim=1)

        # Dropout + 全连接
        out = self.dropout(cat_out)
        logits = self.fc(out)

        return logits


def textcnn_demo():
    """TextCNN 演示"""
    print("\n" + "=" * 60)
    print("第二部分：TextCNN 模型")
    print("=" * 60)

    print("""
TextCNN 核心思想 (Kim, 2014):
    使用不同大小的卷积核捕捉不同长度的 n-gram 特征
    
    架构：
    词嵌入 → [卷积核3, 卷积核4, 卷积核5] → 最大池化 → 拼接 → 全连接
    """)

    # 创建模型
    vocab_size = 10000
    embed_dim = 100
    num_classes = 2  # 二分类

    model = TextCNN(vocab_size, embed_dim, num_classes)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    batch_size = 4
    seq_len = 50
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")


# ==================== 第三部分：LSTM 文本分类 ====================


class LSTMClassifier(nn.Module):
    """LSTM 文本分类模型"""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        num_layers=2,
        bidirectional=True,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # 双向 LSTM 输出维度翻倍
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden_dim * 2)

        # 取最后一个时间步的输出
        # 或者取平均：lstm_out.mean(dim=1)
        last_output = lstm_out[:, -1, :]

        # 分类
        out = self.dropout(last_output)
        logits = self.fc(out)

        return logits


def lstm_demo():
    """LSTM 分类演示"""
    print("\n" + "=" * 60)
    print("第三部分：LSTM 文本分类")
    print("=" * 60)

    print("""
LSTM 文本分类：
    - 使用 LSTM 编码文本序列
    - 取最后一个隐状态或平均池化
    - 通过全连接层分类
    
优势：
    - 捕捉长距离依赖
    - 考虑词序信息
    """)

    # 创建模型
    vocab_size = 10000
    embed_dim = 100
    hidden_dim = 128
    num_classes = 2

    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    batch_size = 4
    seq_len = 50
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")


# ==================== 第四部分：训练流程 ====================


def training_example():
    """训练示例"""
    print("\n" + "=" * 60)
    print("第四部分：训练流程")
    print("=" * 60)

    print("""
完整训练流程示例：

    # 1. 准备数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 2. 创建模型
    model = TextCNN(vocab_size, embed_dim, num_classes)
    
    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. 训练循环
    for epoch in range(num_epochs):
        model.train()
        for texts, labels in train_loader:
            # 前向传播
            logits = model(texts)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for texts, labels in val_loader:
                logits = model(texts)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            accuracy = correct / total
        
        print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")
    """)


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：实现情感分析
    任务：在 IMDB 数据集上训练 TextCNN
    
练习 1 答案：
    from torchtext.datasets import IMDB
    from torchtext.data.utils import get_tokenizer
    
    # 加载数据
    tokenizer = get_tokenizer("basic_english")
    train_iter, test_iter = IMDB()
    
    # 构建词表和数据集
    # ... (数据预处理代码)
    
    # 训练
    model = TextCNN(vocab_size, 100, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        for texts, labels in train_loader:
            logits = model(texts)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

练习 2：对比 TextCNN 和 LSTM
    任务：在相同数据上训练两个模型，比较效果

练习 2 答案：
    模型        准确率   训练时间   推理速度
    TextCNN     87%      快        快
    LSTM        88%      较慢      较慢
    
    结论：
    - TextCNN 训练更快，适合长文本
    - LSTM 精度略高，捕捉序列信息更好

思考题 1：TextCNN 的卷积核大小有什么含义？
答案：
    - kernel_size=3: 捕捉三元组特征
    - kernel_size=4: 捕捉四元组特征
    - 多种尺寸组合捕捉不同粒度的模式

思考题 2：如何处理变长文本？
答案：
    - 填充(Padding)到固定长度
    - 使用 pack_padded_sequence
    - 注意力机制忽略 padding
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    textcnn_demo()
    lstm_demo()
    training_example()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！下一步：04-ner.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
