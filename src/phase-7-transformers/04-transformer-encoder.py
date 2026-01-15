"""
Transformer 编码器 (Transformer Encoder)
======================================

学习目标：
    1. 理解Transformer编码器的完整结构
    2. 实现编码器层和编码器栈
    3. 掌握残差连接和Layer Normalization
    4. 理解Feed-Forward网络的作用

核心概念：
    - 编码器层：Multi-Head Attention + Feed-Forward
    - 残差连接：Add & Norm
    - Layer Normalization：稳定训练
    - 堆叠：多层编码器

前置知识：
    - 01-self-attention.py
    - 02-multi-head-attention.py
    - 03-positional-encoding.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 第一部分：编码器层 ====================


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-Forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask
        """
        # 1. Multi-Head Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器（多层堆叠）"""

    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()

        # 位置编码
        self.pos_encoding = self._create_positional_encoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # 编码器层堆叠
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入嵌入 (batch_size, seq_len, d_model)
        """
        # 添加位置编码
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ==================== 第二部分：示例 ====================


def main():
    print("=" * 60)
    print("Transformer 编码器")
    print("=" * 60)

    print("\n示例 1: 单层编码器\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    output = encoder_layer(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    print("\n示例 2: 完整编码器（6层）\n")

    num_layers = 6
    encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff)

    output = encoder(x)
    print(f"6层编码器输出: {output.shape}")

    # 参数量
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")

    print("""
关键要点：
    ✓ 每层包含：Self-Attention + Feed-Forward
    ✓ 残差连接帮助梯度流动
    ✓ Layer Norm稳定训练
    ✓ 典型配置：6-12层
    """)


if __name__ == "__main__":
    main()
