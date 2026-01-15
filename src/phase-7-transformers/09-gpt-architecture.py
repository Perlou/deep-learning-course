"""
GPT 架构详解 (GPT Architecture)
==============================

学习目标：
    1. 理解GPT的自回归语言建模
    2. 掌握Decoder-only架构
    3. 理解GPT与BERT的区别
    4. 了解GPT的演进（GPT → GPT-2 → GPT-3）

核心概念：
    - Causal Language Modeling
    - Decoder-only Transformer
    - 自回归生成
    - Zero-shot / Few-shot Learning

前置知识：
    - 05-transformer-decoder.py
    - 07-bert-architecture.py
"""

import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    """GPT的Transformer Block"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Masked Self-Attention
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT使用GELU而非ReLU
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer Norms (Pre-LN: 在子层之前)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN + Self-Attention + Residual
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + self.dropout(attn_out)

        # Pre-LN + FF + Residual
        x = x + self.dropout(self.ff(self.ln2(x)))

        return x


class GPTModel(nn.Module):
    """简化的GPT模型"""

    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()

        # Token + Position Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output head (Language Modeling Head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定（与input embedding共享权重）
        self.lm_head.weight = self.token_emb.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()

        # Position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(positions)
        x = self.dropout(token_emb + pos_emb)

        # Causal mask (下三角)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)

        # Through Transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        自回归生成

        Args:
            input_ids: (batch, seq_len) 初始序列
            max_new_tokens: 生成的最大token数
            temperature: 温度参数
            top_k: Top-K采样
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 只取最后的上下文（如果超过最大长度）
            idx_cond = input_ids if input_ids.size(1) <= 1024 else input_ids[:, -1024:]

            # 前向传播
            logits = self(idx_cond)

            # 只取最后一个位置的logits
            logits = logits[:, -1, :] / temperature

            # Top-K采样
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Softmax并采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 追加到序列
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def main():
    print("=" * 60)
    print("GPT 架构详解")
    print("=" * 60)

    print("""
GPT核心特点：

1. Decoder-only架构：
   - 只使用Transformer Decoder（去掉Cross-Attention）
   - 单向注意力（Causal/Masked Attention）
   - 只能看到左边的token

2. 预训练任务：
   - Causal Language Modeling (CLM)
   - 预测下一个token
   - 例如："我 爱 深度" → 预测 "学习"

3. 生成能力：
   - 天然适合文本生成
   - 自回归：逐个生成token
   - 可以进行zero-shot / few-shot学习

4. 模型演进：
   - GPT (2018): 117M参数
   - GPT-2 (2019): 1.5B参数
   - GPT-3 (2020): 175B参数
   - GPT-4 (2023): 未公开参数量
    """)

    # 创建GPT模型
    vocab_size = 50000
    model = GPTModel(vocab_size=vocab_size, d_model=768, num_heads=12, num_layers=12)

    print(f"\nGPT-small配置:")
    print(f"  词汇表: {vocab_size}")
    print(f"  维度: 768")
    print(f"  头数: 12")
    print(f"  层数: 12")

    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params / 1e6:.1f}M")

    # 测试
    input_ids = torch.randint(0, vocab_size, (2, 20))
    logits = model(input_ids)

    print(f"\n前向传播:")
    print(f"  输入: {input_ids.shape}")
    print(f"  输出: {logits.shape}")

    # 生成测试
    generated = model.generate(input_ids[:1], max_new_tokens=10)
    print(f"\n生成: {generated.shape}")

    print("""
GPT vs BERT对比：

特性       | GPT                  | BERT
-----------|---------------------|--------------------
方向       | 单向（从左到右）       | 双向
架构       | Decoder-only        | Encoder-only
预训练     | Next Token          | MLM + NSP
擅长       | 生成任务              | 理解任务
使用       | 提示学习(Prompting)   | 微调(Fine-tuning)

关键区别：
    - GPT是生成式模型，BERT是判别式模型
    - GPT更适合few-shot learning
    - BERT在分类任务上通常更好
    - GPT-3展示了强大的zero-shot能力
    """)


if __name__ == "__main__":
    main()
