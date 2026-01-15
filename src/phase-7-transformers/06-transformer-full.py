"""
完整 Transformer 模型 (Full Transformer)
=======================================

学习目标：
    1. 整合编码器和解码器
    2. 实现完整的Transformer架构
    3. 理解Transformer的输入输出
    4. 实现简单的序列到序列任务

核心概念：
    - Encoder-Decoder架构
    - Token Embedding + Positional Encoding
    - 输出投影层
    - 完整的前向传播流程

前置知识：
    - 04-transformer-encoder.py
    - 05-transformer-decoder.py
"""

import torch
import torch.nn as nn
import math


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(
        self,
        src_vocab_size,  # 源语言词汇表大小
        tgt_vocab_size,  # 目标语言词汇表大小
        d_model=512,  # 模型维度
        num_heads=8,  # 注意力头数
        num_encoder_layers=6,  # 编码器层数
        num_decoder_layers=6,  # 解码器层数
        d_ff=2048,  # Feed-Forward维度
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = self._create_positional_encoding(d_model, max_len)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _init_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """编码源序列"""
        # Embedding + Positional Encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoding[:, : src.size(1), :]
        src_emb = self.dropout(src_emb)

        # Encoder
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """解码目标序列"""
        # Embedding + Positional Encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoding[:, : tgt.size(1), :]
        tgt_emb = self.dropout(tgt_emb)

        # Decoder
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: 源序列token ids (batch, src_len)
            tgt: 目标序列token ids (batch, tgt_len)
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # Encode
        memory = self.encode(src, src_mask)

        # Decode
        output = self.decode(tgt, memory, tgt_mask)

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits

    @torch.no_grad()
    def generate(self, src, max_len=50, start_token=1, end_token=2):
        """贪心生成（用于推理）"""
        self.eval()

        # Encode source
        memory = self.encode(src)

        batch_size = src.size(0)
        # 初始化目标序列（只有start token）
        tgt = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )

        for _ in range(max_len - 1):
            # Decode
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
                src.device
            )
            output = self.decode(tgt, memory, tgt_mask=tgt_mask)

            # 获取最后一个位置的预测
            logits = self.output_projection(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            # 拼接到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)

            # 检查是否所有序列都生成了end token
            if (next_token == end_token).all():
                break

        return tgt


def main():
    print("=" * 60)
    print("完整 Transformer 模型")
    print("=" * 60)

    # 配置
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 512
    num_heads = 8
    num_layers = 6

    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
    )

    print(f"\n模型配置:")
    print(f"  源词汇表: {src_vocab_size}")
    print(f"  目标词汇表: {tgt_vocab_size}")
    print(f"  模型维度: {d_model}")
    print(f"  编码器层数: {num_layers}")
    print(f"  解码器层数: {num_layers}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 测试前向传播
    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # 创建掩码
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)

    logits = model(src, tgt, tgt_mask=tgt_mask)

    print(f"\n前向传播:")
    print(f"  输入src: {src.shape}")
    print(f"  输入tgt: {tgt.shape}")
    print(f"  输出logits: {logits.shape}")

    # 测试生成
    generated = model.generate(src, max_len=20)
    print(f"\n生成序列: {generated.shape}")

    print("""
Transformer架构总结：

输入流程：
    源序列 → Embedding → Encoder → Memory
    目标序列 → Embedding → Decoder → Output

关键组件：
    1. Multi-Head Self-Attention
    2. Cross-Attention (Decoder)
    3. Feed-Forward Networks
    4. Positional Encoding
    5. Residual Connections
    6. Layer Normalization

典型应用：
    - 机器翻译
    - 文本摘要
    - 对话系统
    """)


if __name__ == "__main__":
    main()
