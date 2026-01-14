"""
07-seq2seq.py - 序列到序列模型 (Seq2Seq)

本节学习:
1. Seq2Seq 架构
2. 编码器-解码器结构
3. 手动实现 Seq2Seq
4. Teacher Forcing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("第7节: 序列到序列模型 (Seq2Seq)")
print("=" * 60)

# =============================================================================
# 1. Seq2Seq 核心思想
# =============================================================================
print("""
📚 Seq2Seq (Sequence to Sequence)

核心架构: 编码器-解码器
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  输入序列 → [编码器] → 上下文向量 → [解码器] → 输出序列       │
│                                                              │
│  "How are you" → [Encoder] → context → [Decoder] → "你好吗"  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

详细结构:
          编码器 (Encoder)              解码器 (Decoder)
  ┌──────────────────────┐       ┌──────────────────────┐
  │  How   are   you     │       │  <SOS>  你   好   吗  │
  │   ↓     ↓     ↓      │       │    ↓    ↓    ↓    ↓  │
  │  LSTM→LSTM→LSTM      │  →→   │  LSTM→LSTM→LSTM→LSTM │
  │         ↓            │  h,c  │    ↓    ↓    ↓    ↓  │
  │     (h, c)           │       │   你   好   吗  <EOS> │
  └──────────────────────┘       └──────────────────────┘
         编码隐藏状态         初始化解码器       预测输出
""")


# =============================================================================
# 2. 编码器实现
# =============================================================================
print("\n" + "=" * 60)
print("📌 2. 编码器实现")
print("-" * 60)


class Encoder(nn.Module):
    """Seq2Seq 编码器"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: [batch, src_len] 源序列
        Returns:
            outputs: [batch, src_len, hidden_dim]
            (h, c): 最终隐藏状态
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (h, c) = self.lstm(embedded)
        return outputs, (h, c)


# 测试编码器
encoder = Encoder(vocab_size=5000, embed_dim=256, hidden_dim=512)
src = torch.randint(1, 5000, (4, 20))  # 4个样本，长度20
enc_outputs, (h, c) = encoder(src)

print(f"编码器:")
print(f"  输入 (源序列): {src.shape}")
print(f"  编码输出: {enc_outputs.shape}")
print(f"  最终隐藏状态: h={h.shape}, c={c.shape}")


# =============================================================================
# 3. 解码器实现
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 解码器实现")
print("-" * 60)


class Decoder(nn.Module):
    """Seq2Seq 解码器"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden):
        """
        单步解码
        Args:
            trg: [batch, 1] 当前输入 token
            hidden: (h, c) 来自编码器或上一步
        Returns:
            output: [batch, vocab_size] 预测分布
            hidden: 更新后的隐藏状态
        """
        embedded = self.dropout(self.embedding(trg))
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden


# 测试解码器
decoder = Decoder(vocab_size=3000, embed_dim=256, hidden_dim=512)
trg_token = torch.randint(1, 3000, (4, 1))  # 当前 token
pred, new_hidden = decoder(trg_token, (h, c))

print(f"\n解码器:")
print(f"  输入 (当前 token): {trg_token.shape}")
print(f"  输出 (词汇表分布): {pred.shape}")


# =============================================================================
# 4. 完整 Seq2Seq 模型
# =============================================================================
print("\n" + "=" * 60)
print("📌 4. 完整 Seq2Seq 模型")
print("-" * 60)


class Seq2Seq(nn.Module):
    """完整的 Seq2Seq 模型"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch, src_len] 源序列
            trg: [batch, trg_len] 目标序列
            teacher_forcing_ratio: 使用真实标签的概率
        Returns:
            outputs: [batch, trg_len, vocab_size]
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        # 存储输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 编码
        _, hidden = self.encoder(src)

        # 解码器的第一个输入是 <SOS> token (假设是 trg 的第一个)
        input_token = trg[:, 0:1]

        for t in range(1, trg_len):
            # 解码一步
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output

            # Teacher Forcing: 随机决定使用真实标签还是预测结果
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t : t + 1] if teacher_force else top1

        return outputs


# 创建完整模型
device = torch.device("cpu")
encoder = Encoder(vocab_size=5000, embed_dim=256, hidden_dim=512).to(device)
decoder = Decoder(vocab_size=3000, embed_dim=256, hidden_dim=512).to(device)
model = Seq2Seq(encoder, decoder, device)

# 测试
src = torch.randint(1, 5000, (4, 20)).to(device)
trg = torch.randint(1, 3000, (4, 15)).to(device)
outputs = model(src, trg)

print(f"\nSeq2Seq 模型:")
print(f"  源序列: {src.shape}")
print(f"  目标序列: {trg.shape}")
print(f"  输出: {outputs.shape}")
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")


# =============================================================================
# 5. Teacher Forcing
# =============================================================================
print("\n" + "=" * 60)
print("📌 5. Teacher Forcing")
print("-" * 60)

print("""
Teacher Forcing 策略:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  训练时决定解码器的下一个输入:                               │
│                                                              │
│  Teacher Forcing (使用真实标签):                             │
│    优点: 训练更稳定，收敛更快                                │
│    缺点: 推理时可能遇到从未见过的错误                        │
│                                                              │
│  Free Running (使用自己的预测):                              │
│    优点: 更接近真实推理场景                                  │
│    缺点: 训练初期可能不稳定                                  │
│                                                              │
│  解决方案: Scheduled Sampling                                │
│    开始时高 teacher_forcing_ratio (如 1.0)                   │
│    逐渐降低到较低值 (如 0.5)                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 6. 推理 (生成)
# =============================================================================
print("\n" + "=" * 60)
print("📌 6. 推理 (生成)")
print("-" * 60)


def greedy_decode(model, src, max_len, sos_idx, eos_idx):
    """贪婪解码"""
    model.eval()
    with torch.no_grad():
        # 编码
        _, hidden = model.encoder(src)

        # 初始输入是 <SOS>
        input_token = torch.tensor([[sos_idx]]).to(src.device)

        outputs = [sos_idx]

        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            top1 = output.argmax(1).item()
            outputs.append(top1)

            if top1 == eos_idx:
                break

            input_token = torch.tensor([[top1]]).to(src.device)

        return outputs


# 模拟推理
print("贪婪解码示例:")
src_single = torch.randint(1, 5000, (1, 10)).to(device)
result = greedy_decode(model, src_single, max_len=20, sos_idx=1, eos_idx=2)
print(f"  输入长度: {src_single.size(1)}")
print(f"  输出长度: {len(result)}")
print(f"  输出序列: {result[:10]}...")


# =============================================================================
# 7. Seq2Seq 的局限性
# =============================================================================
print("\n" + "=" * 60)
print("📌 7. Seq2Seq 的局限性")
print("-" * 60)

print("""
基础 Seq2Seq 的问题:

1. 信息瓶颈
   所有输入信息压缩到固定大小的向量
   长序列信息丢失严重

2. 梯度问题
   虽然 LSTM 缓解了，但超长序列仍有问题

3. 对齐问题
   不知道输出的哪个词对应输入的哪个词

解决方案: 注意力机制 (下一节)
  • 不依赖单一上下文向量
  • 每个输出步骤关注不同的输入位置
  • 可视化对齐关系
""")


# =============================================================================
# 8. 练习
# =============================================================================
print("\n" + "=" * 60)
print("📝 练习题")
print("-" * 60)

print("""
1. Seq2Seq 中编码器的最终输出是什么？
   答：最终的隐藏状态 (h, c)，用于初始化解码器

2. Teacher Forcing 的作用是什么？
   答：用真实标签作为下一步输入，加速训练

3. 为什么需要 <SOS> 和 <EOS> 标记？
   答：<SOS> 标记解码开始，<EOS> 标记解码结束

4. 基础 Seq2Seq 的信息瓶颈问题如何解决？
   答：使用注意力机制，让解码器访问所有编码输出
""")

print("\n✅ 第7节完成！")
print("下一节：08-attention-basic.py - 基础注意力机制")
