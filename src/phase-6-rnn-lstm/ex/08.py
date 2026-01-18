import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第8节: 基础注意力机制")
print("=" * 60)


# =============================================================================
# 3. 手动实现注意力
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 手动实现注意力")
print("-" * 60)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        energy = torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs))

        scores = self.v(energy).squeeze(-1)
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


# 测试
attention = BahdanauAttention(hidden_dim=256)
decoder_h = torch.randn(4, 256)
encoder_out = torch.randn(4, 20, 256)

context, weights = attention(decoder_h, encoder_out)

print(f"Bahdanau 注意力:")
print(f"  解码器状态: {decoder_h.shape}")
print(f"  编码器输出: {encoder_out.shape}")
print(f"  上下文向量: {context.shape}")
print(f"  注意力权重: {weights.shape}")
print(f"  权重和 (应为1): {weights.sum(dim=-1).mean().item():.4f}")


# =============================================================================
# 4. Luong (Multiplicative) 注意力
# =============================================================================
print("\n" + "=" * 60)
print("📌 4. Luong (Multiplicative) 注意力")
print("-" * 60)


class LuongAttention(nn.Module):
    """Luong (Multiplicative) 注意力 - 更简单高效"""

    def __init__(self, hidden_dim, method="dot"):
        super().__init__()
        self.method = method

        if method == "general":
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == "concat":
            self.W = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: [batch, hidden_dim]
            encoder_outputs: [batch, src_len, hidden_dim]
        """
        if self.method == "dot":
            # 直接点积: sᵀ · h
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(-1)).squeeze(
                -1
            )

        elif self.method == "general":
            # 带权重的点积: sᵀ · W · h
            scores = torch.bmm(
                encoder_outputs, self.W(decoder_hidden).unsqueeze(-1)
            ).squeeze(-1)

        elif self.method == "concat":
            # 拼接: vᵀ · tanh(W · [s; h])
            src_len = encoder_outputs.size(1)
            decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            concat = torch.cat([decoder_hidden, encoder_outputs], dim=-1)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(-1)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


print("Luong 注意力变体:")
for method in ["dot", "general", "concat"]:
    attn = LuongAttention(256, method=method)
    ctx, wts = attn(decoder_h, encoder_out)
    params = sum(p.numel() for p in attn.parameters())
    print(f"  {method:8s}: 上下文 {ctx.shape}, 参数量 {params:,}")
