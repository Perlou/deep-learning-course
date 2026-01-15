"""
BERT 架构详解 (BERT Architecture)
================================

学习目标：
    1. 理解BERT的核心思想：Masked Language Modeling
    2. 掌握BERT的双向编码机制
    3. 理解BERT的预训练任务
    4. 了解BERT的应用场景

核心概念：
    - 双向Transformer编码器
    - Masked Language Modeling (MLM)
    - Next Sentence Prediction (NSP)
    - [CLS] 和 [SEP] 特殊token
    - Fine-tuning范式

前置知识：
    - 04-transformer-encoder.py
    - 06-transformer-full.py
"""

import torch
import torch.nn as nn
import math


class BERTEmbedding(nn.Module):
    """BERT的输入嵌入层"""

    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()

        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Segment Embedding (用于区分句子A和B)
        self.segment_emb = nn.Embedding(2, d_model)

        # Position Embedding (可学习，不是正弦)
        self.position_emb = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, token_ids, segment_ids=None):
        """
        Args:
            token_ids: (batch, seq_len)
            segment_ids: (batch, seq_len) - 0或1，表示句子A或B
        """
        batch_size, seq_len = token_ids.size()

        # Token嵌入
        token_embedding = self.token_emb(token_ids)

        # Position嵌入
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        position_embedding = self.position_emb(positions)

        # Segment嵌入
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        segment_embedding = self.segment_emb(segment_ids)

        # 三者相加
        embedding = token_embedding + position_embedding + segment_embedding
        return self.dropout(embedding)


class BERTModel(nn.Module):
    """简化的BERT模型"""

    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()

        # Embedding层
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # MLM头（预测被mask的token）
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        # NSP头（预测句子关系）
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2),  # 是/否下一句
        )

    def forward(self, token_ids, segment_ids=None, attention_mask=None):
        """
        Args:
            token_ids: (batch, seq_len)
            segment_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        """
        # 1. Embedding
        x = self.embedding(token_ids, segment_ids)

        # 2. Transformer Encoder
        # 转换attention_mask格式
        if attention_mask is not None:
            attention_mask = attention_mask == 0  # True表示mask

        encoded = self.encoder(x, src_key_padding_mask=attention_mask)

        # 3. 输出
        # MLM预测
        mlm_logits = self.mlm_head(encoded)

        # NSP预测（使用[CLS] token，即第一个位置）
        cls_output = encoded[:, 0, :]
        nsp_logits = self.nsp_head(cls_output)

        return mlm_logits, nsp_logits

    def get_embeddings(self, token_ids, segment_ids=None, attention_mask=None):
        """获取序列表示（用于下游任务）"""
        x = self.embedding(token_ids, segment_ids)

        if attention_mask is not None:
            attention_mask = attention_mask == 0

        encoded = self.encoder(x, src_key_padding_mask=attention_mask)
        return encoded


def main():
    print("=" * 60)
    print("BERT 架构详解")
    print("=" * 60)

    print("""
BERT核心思想：

1. 双向编码：
   - 不像GPT只看左边，BERT同时看左右两边
   - 使用Transformer Encoder（不是Decoder）

2. 预训练任务：
   
   a) Masked Language Modeling (MLM):
      - 随机mask 15%的token
      - 预测被mask的token
      - 例如："我 [MASK] 深度 学习" → 预测 "爱"
   
   b) Next Sentence Prediction (NSP):
      - 判断句子B是否是句子A的下一句
      - 帮助理解句子间关系

3. 特殊Token：
   - [CLS]: 句子开始，用于分类任务
   - [SEP]: 句子分隔符
   - [MASK]: 被遮蔽的token

4. 三种嵌入：
   - Token Embedding: 词嵌入
   - Segment Embedding: 句子A/B标识
   - Position Embedding: 位置信息
    """)

    # 创建BERT模型
    vocab_size = 30000
    model = BERTModel(vocab_size=vocab_size, d_model=768, num_heads=12, num_layers=12)

    print(f"\nBERT-base配置:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  隐藏层维度: 768")
    print(f"  注意力头数: 12")
    print(f"  编码器层数: 12")
    print(f"  Feed-Forward维度: 3072")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  总参数量: {total_params / 1e6:.1f}M")

    # 测试
    batch_size = 2
    seq_len = 128

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    segment_ids = torch.cat(
        [torch.zeros(batch_size, seq_len // 2), torch.ones(batch_size, seq_len // 2)],
        dim=1,
    ).long()

    mlm_logits, nsp_logits = model(token_ids, segment_ids)

    print(f"\n前向传播:")
    print(f"  输入: {token_ids.shape}")
    print(f"  MLM输出: {mlm_logits.shape}")
    print(f"  NSP输出: {nsp_logits.shape}")

    print("""
BERT vs GPT:

特性          | BERT                    | GPT
-------------|-------------------------|------------------
架构          | Encoder-only            | Decoder-only
方向性        | 双向                     | 单向（从左到右）
预训练        | MLM + NSP               | Next Token Prediction
适用场景      | 理解任务（分类、NER等）    | 生成任务
代表模型      | BERT, RoBERTa, ALBERT   | GPT-2, GPT-3

使用场景：
    BERT擅长：
    ✓ 文本分类
    ✓ 命名实体识别
    ✓ 问答系统
    ✓ 语义相似度
    
    GPT擅长：
    ✓ 文本生成
    ✓ 对话系统
    ✓ 代码补全
    ✓ Few-shot学习
    """)


if __name__ == "__main__":
    main()
