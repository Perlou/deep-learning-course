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
        x = self.embedding(token_ids, segment_ids)

        if attention_mask is not None:
            attention_mask = attention_mask == 0

        encoded = self.encoder(x, src_key_padding_mask=attention_mask)

        mlm_logits = self.mlm_head(encoded)

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


if __name__ == "__main__":
    main()
