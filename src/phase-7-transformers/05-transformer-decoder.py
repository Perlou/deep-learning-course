"""
Transformer 解码器 (Transformer Decoder)
======================================

学习目标：
    1. 理解编码器-解码器架构
    2. 掌握Masked Self-Attention
    3. 实现交叉注意力（Cross-Attention）
    4. 理解解码器的因果性质

核心概念：
    - Masked Attention：只能看到之前的位置
    - Cross Attention：查询来自解码器，键值来自编码器
    - 自回归生成：逐步生成输出序列

前置知识：
    - 04-transformer-encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            x: 解码器输入 (batch, tgt_len, d_model)
            memory: 编码器输出 (batch, src_len, d_model)
            tgt_mask: 目标序列掩码（因果掩码）
            memory_mask: 源序列掩码
        """
        # 1. Masked Self-Attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-Attention
        attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 3. Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


def create_causal_mask(seq_len):
    """创建因果掩码（下三角矩阵）"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


def main():
    print("=" * 60)
    print("Transformer 解码器")
    print("=" * 60)

    batch_size = 2
    src_len = 10  # 源序列长度
    tgt_len = 8  # 目标序列长度
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # 编码器输出（memory）
    memory = torch.randn(batch_size, src_len, d_model)

    # 解码器输入
    tgt = torch.randn(batch_size, tgt_len, d_model)

    # 因果掩码
    tgt_mask = create_causal_mask(tgt_len)

    print("\n因果掩码（防止看到未来）:")
    print(tgt_mask.int()[:5, :5])

    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
    output = decoder_layer(tgt, memory, tgt_mask=tgt_mask)

    print(f"\n解码器输出: {output.shape}")

    print("""
关键区别：
    编码器：只有Self-Attention
    解码器：Masked Self-Attention + Cross-Attention
    
    Cross-Attention作用：
    - Q来自解码器（我要什么）
    - K,V来自编码器（源序列有什么）
    - 实现编码器-解码器的信息融合
    """)


if __name__ == "__main__":
    main()
