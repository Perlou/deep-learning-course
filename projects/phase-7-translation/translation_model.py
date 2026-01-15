"""
中英机器翻译 - Transformer模型
基于完整的Encoder-Decoder架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import jieba
from collections import Counter


# ==================== 数据处理 ====================


class Vocabulary:
    """词表类"""

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.word_count = Counter()

    def add_sentence(self, sentence):
        """添加句子到词表"""
        for word in sentence:
            self.word_count[word] += 1

    def build_vocab(self, min_freq=1):
        """构建词表"""
        idx = len(self.word2idx)
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, sentence, add_sos_eos=True):
        """将句子编码为索引"""
        indices = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence]
        if add_sos_eos:
            indices = [self.word2idx["<sos>"]] + indices + [self.word2idx["<eos>"]]
        return indices

    def decode(self, indices):
        """将索引解码为句子"""
        return [self.idx2word[idx] for idx in indices if idx not in [0, 1, 2]]

    def __len__(self):
        return len(self.word2idx)


def tokenize_chinese(text):
    """中文分词"""
    return list(jieba.cut(text.strip()))


def tokenize_english(text):
    """英文分词（简单空格分词）"""
    return text.strip().lower().split()


def create_sample_data():
    """创建示例训练数据（中英句对）"""
    # 简化的示例数据
    zh_sentences = [
        "我 爱 学习 深度 学习",
        "今天 天气 很 好",
        "机器 翻译 很 有趣",
        "神经 网络 很 强大",
        "我 喜欢 编程",
        "这 是 一个 例子",
        "人工智能 正在 改变 世界",
        "自然 语言 处理 很 重要",
        "Transformer 模型 很 流行",
        "我们 正在 学习 PyTorch",
    ]

    en_sentences = [
        "i love learning deep learning",
        "the weather is nice today",
        "machine translation is interesting",
        "neural networks are powerful",
        "i like programming",
        "this is an example",
        "artificial intelligence is changing the world",
        "natural language processing is important",
        "transformer models are popular",
        "we are learning pytorch",
    ]

    # 分词
    zh_data = [sent.split() for sent in zh_sentences]
    en_data = [sent.split() for sent in en_sentences]

    return zh_data, en_data


# ==================== Transformer模型 ====================


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerTranslator(nn.Module):
    """完整的Transformer翻译模型"""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_len=100,
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        """
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
        """
        # Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        # Output projection
        return self.output_proj(output)

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    @torch.no_grad()
    def translate(self, src, src_vocab, tgt_vocab, max_len=50, device="cpu"):
        """贪心翻译"""
        self.eval()

        # Encode source
        src = torch.LongTensor(src).unsqueeze(0).to(device)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))

        memory = self.transformer.encoder(src_emb)

        # Initialize target with <sos>
        tgt_indices = [tgt_vocab.word2idx["<sos>"]]

        for _ in range(max_len):
            tgt = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            tgt_mask = self.generate_square_subsequent_mask(len(tgt_indices)).to(device)

            tgt_emb = self.pos_encoder(
                self.tgt_embedding(tgt) * math.sqrt(self.d_model)
            )
            output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            logits = self.output_proj(output[0, -1, :])
            next_word = logits.argmax().item()

            tgt_indices.append(next_word)

            if next_word == tgt_vocab.word2idx["<eos>"]:
                break

        return tgt_vocab.decode(tgt_indices)


# ==================== 工具函数 ====================


def create_padding_mask(seq, pad_idx=0):
    """创建padding mask"""
    return seq == pad_idx


def collate_fn(batch, src_vocab, tgt_vocab):
    """批处理函数"""
    src_batch, tgt_batch = [], []

    for src, tgt in batch:
        src_batch.append(src_vocab.encode(src))
        tgt_batch.append(tgt_vocab.encode(tgt))

    # Padding
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)

    src_padded = []
    tgt_padded = []

    for src in src_batch:
        src_padded.append(src + [0] * (src_max_len - len(src)))

    for tgt in tgt_batch:
        tgt_padded.append(tgt + [0] * (tgt_max_len - len(tgt)))

    return torch.LongTensor(src_padded), torch.LongTensor(tgt_padded)


if __name__ == "__main__":
    print("=" * 60)
    print("中英机器翻译 - Transformer模型")
    print("=" * 60)

    # 创建示例数据
    zh_data, en_data = create_sample_data()

    print(f"\n加载了 {len(zh_data)} 个训练样本")
    print("\n示例：")
    for i in range(3):
        print(f"  中文: {' '.join(zh_data[i])}")
        print(f"  英文: {' '.join(en_data[i])}")
        print()

    # 构建词表
    zh_vocab = Vocabulary()
    en_vocab = Vocabulary()

    for sent in zh_data:
        zh_vocab.add_sentence(sent)
    for sent in en_data:
        en_vocab.add_sentence(sent)

    zh_vocab.build_vocab()
    en_vocab.build_vocab()

    print(f"中文词表大小: {len(zh_vocab)}")
    print(f"英文词表大小: {len(en_vocab)}")

    # 创建模型
    model = TransformerTranslator(
        src_vocab_size=len(zh_vocab),
        tgt_vocab_size=len(en_vocab),
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 测试前向传播
    src_sample = zh_vocab.encode(zh_data[0])
    tgt_sample = en_vocab.encode(en_data[0])

    src = torch.LongTensor([src_sample])
    tgt_input = torch.LongTensor([tgt_sample[:-1]])  # 去掉<eos>用于输入

    tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1))

    output = model(src, tgt_input, tgt_mask=tgt_mask)
    print(f"\n前向传播测试:")
    print(f"  输入src: {src.shape}")
    print(f"  输入tgt: {tgt_input.shape}")
    print(f"  输出: {output.shape}")

    print("\n✓ 模型创建成功！")
    print("请运行 train.py 开始训练")
