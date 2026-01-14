"""
08-attention-basic.py - 基础注意力机制

本节学习:
1. 注意力机制的直觉
2. 注意力计算公式
3. 手动实现注意力
4. Seq2Seq + Attention
"""

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
# 1. 注意力机制的直觉
# =============================================================================
print("""
📚 注意力机制 (Attention Mechanism)

核心思想: 解码时动态关注输入的不同部分
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  翻译 "I love you" → "我爱你"                                │
│                                                              │
│  生成 "我" 时:  关注 "I"      (权重高)                       │
│  生成 "爱" 时:  关注 "love"   (权重高)                       │
│  生成 "你" 时:  关注 "you"    (权重高)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

注意力 vs 基础 Seq2Seq:
  
  基础 Seq2Seq:
    所有信息 → 压缩成固定向量 → 解码 (信息瓶颈)
  
  带注意力:
    每个解码步骤 → 查看所有编码输出 → 加权求和 (动态关注)
""")


# =============================================================================
# 2. 注意力计算公式
# =============================================================================
print("\n" + "=" * 60)
print("📌 2. 注意力计算公式")
print("-" * 60)

print("""
标准注意力计算 (Bahdanau Attention):
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  1. 计算注意力分数 (energy):                                 │
│     eₜᵢ = attention_score(sₜ₋₁, hᵢ)                         │
│     通常: eₜᵢ = vᵀ · tanh(W₁·sₜ₋₁ + W₂·hᵢ)                  │
│                                                              │
│  2. 计算注意力权重 (softmax):                                │
│     αₜᵢ = softmax(eₜᵢ) = exp(eₜᵢ) / Σⱼ exp(eₜⱼ)             │
│                                                              │
│  3. 计算上下文向量 (加权求和):                               │
│     cₜ = Σᵢ αₜᵢ · hᵢ                                         │
│                                                              │
│  符号:                                                        │
│    sₜ₋₁: 解码器上一时刻隐藏状态 (Query)                      │
│    hᵢ:   编码器第 i 时刻输出 (Key & Value)                   │
│    αₜᵢ:  注意力权重 (关注程度)                               │
│    cₜ:   上下文向量 (加权后的输入信息)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 3. 手动实现注意力
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 手动实现注意力")
print("-" * 60)


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) 注意力"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 处理解码器状态
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 处理编码器输出
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # 输出分数

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: [batch, hidden_dim] 解码器当前状态
            encoder_outputs: [batch, src_len, hidden_dim] 编码器所有输出
        Returns:
            context: [batch, hidden_dim] 上下文向量
            attention_weights: [batch, src_len] 注意力权重
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # 扩展解码器状态以匹配编码器输出的维度
        decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        # 计算注意力分数
        # energy: [batch, src_len, hidden_dim]
        energy = torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs))

        # 分数: [batch, src_len]
        scores = self.v(energy).squeeze(-1)

        # 注意力权重: [batch, src_len]
        attention_weights = F.softmax(scores, dim=-1)

        # 上下文向量: [batch, hidden_dim]
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


# =============================================================================
# 5. 可视化注意力权重
# =============================================================================
print("\n" + "=" * 60)
print("📌 5. 可视化注意力权重")
print("-" * 60)

# 模拟一个注意力矩阵
np.random.seed(42)
src_words = ["I", "love", "machine", "learning", "<EOS>"]
trg_words = ["<SOS>", "我", "爱", "机器学习", "<EOS>"]

# 创建一个有意义的注意力矩阵
attention_matrix = np.array(
    [
        [0.9, 0.05, 0.02, 0.02, 0.01],  # "我" 关注 "I"
        [0.05, 0.85, 0.05, 0.03, 0.02],  # "爱" 关注 "love"
        [0.02, 0.03, 0.45, 0.48, 0.02],  # "机器学习" 关注 "machine" + "learning"
        [0.01, 0.02, 0.02, 0.05, 0.90],  # "<EOS>" 关注 "<EOS>"
    ]
)

plt.figure(figsize=(8, 6))
plt.imshow(attention_matrix, cmap="Blues", aspect="auto")
plt.colorbar(label="注意力权重")
plt.xticks(range(len(src_words)), src_words, fontsize=12)
plt.yticks(range(len(trg_words) - 1), trg_words[1:], fontsize=12)
plt.xlabel("源序列 (英文)")
plt.ylabel("目标序列 (中文)")
plt.title("注意力权重热力图")

# 添加数值标注
for i in range(attention_matrix.shape[0]):
    for j in range(attention_matrix.shape[1]):
        plt.text(
            j,
            i,
            f"{attention_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if attention_matrix[i, j] > 0.5 else "black",
        )

plt.tight_layout()
plt.savefig("outputs/attention_weights.png", dpi=100)
plt.close()
print("注意力权重热力图已保存: outputs/attention_weights.png")


# =============================================================================
# 6. Seq2Seq + Attention
# =============================================================================
print("\n" + "=" * 60)
print("📌 6. Seq2Seq with Attention")
print("-" * 60)


class AttentionDecoder(nn.Module):
    """带注意力的解码器"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        # LSTM 输入 = embedding + context
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs):
        """
        Args:
            input_token: [batch, 1]
            hidden: (h, c)
            encoder_outputs: [batch, src_len, hidden_dim]
        """
        # 嵌入: [batch, 1, embed_dim]
        embedded = self.dropout(self.embedding(input_token))

        # 注意力: [batch, hidden_dim]
        h = hidden[0].squeeze(0)  # [batch, hidden_dim]
        context, attention_weights = self.attention(h, encoder_outputs)

        # 拼接嵌入和上下文作为 LSTM 输入
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)

        # LSTM
        output, hidden = self.lstm(lstm_input, hidden)

        # 预测: 拼接 LSTM 输出和上下文
        output = torch.cat([output.squeeze(1), context], dim=-1)
        prediction = self.fc(output)

        return prediction, hidden, attention_weights


# 测试
attn_decoder = AttentionDecoder(vocab_size=3000, embed_dim=256, hidden_dim=512)
input_token = torch.randint(1, 3000, (4, 1))
h = torch.randn(1, 4, 512)
c = torch.randn(1, 4, 512)
encoder_outputs = torch.randn(4, 20, 512)

pred, new_hidden, attn_weights = attn_decoder(input_token, (h, c), encoder_outputs)

print(f"带注意力的解码器:")
print(f"  预测输出: {pred.shape}")
print(f"  注意力权重: {attn_weights.shape}")
print(f"  参数量: {sum(p.numel() for p in attn_decoder.parameters()):,}")


# =============================================================================
# 7. 注意力的好处
# =============================================================================
print("\n" + "=" * 60)
print("📌 7. 注意力的好处")
print("-" * 60)

print("""
注意力机制的优势:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  1. 解决信息瓶颈                                             │
│     不再压缩到固定向量，可访问所有编码输出                   │
│                                                              │
│  2. 缓解梯度问题                                             │
│     注意力提供了更短的梯度路径                               │
│                                                              │
│  3. 可解释性                                                 │
│     注意力权重可视化，理解模型关注什么                       │
│                                                              │
│  4. 处理长序列                                               │
│     每个输出步骤可以直接关注任意输入位置                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

从注意力到 Transformer:
  • Self-Attention: Query, Key, Value 都来自同一序列
  • Multi-Head Attention: 多个注意力头并行
  • Transformer: 完全基于注意力，没有 RNN
""")


# =============================================================================
# 8. 练习
# =============================================================================
print("\n" + "=" * 60)
print("📝 练习题")
print("-" * 60)

print("""
1. 注意力权重 α 代表什么？
   答：解码当前词时，对每个输入词的关注程度

2. Bahdanau 和 Luong 注意力的主要区别？
   答：Bahdanau 是加法注意力，Luong 是乘法注意力；
       Bahdanau 计算更复杂但更灵活

3. 上下文向量 c 是如何计算的？
   答：编码器输出的加权求和，权重是注意力分数

4. 为什么注意力能缓解梯度消失？
   答：注意力提供了从输出到输入的直接连接，缩短梯度路径
""")

print("\n✅ 第8节完成！")
print("下一节：09-time-series.py - 时间序列预测")
