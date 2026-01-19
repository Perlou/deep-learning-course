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


if __name__ == "__main__":
    main()
