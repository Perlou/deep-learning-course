"""
Word2Vec 词向量训练
==================

学习目标：
    1. 理解词向量的动机和核心思想
    2. 掌握 Skip-gram 和 CBOW 两种架构
    3. 理解负采样训练技巧
    4. 使用 Gensim 训练和使用词向量

核心概念：
    - 分布式假设：相似上下文的词有相似含义
    - Skip-gram：中心词预测上下文
    - CBOW：上下文预测中心词
    - 负采样：高效训练的关键技巧

前置知识：
    - Phase 3: PyTorch 基础
    - Phase 4: 神经网络基础
    - 基本的概率论知识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random


# ==================== 第一部分：词向量概述 ====================


def introduction():
    """词向量介绍"""
    print("=" * 60)
    print("第一部分：词向量概述")
    print("=" * 60)

    print("""
词向量的动机：

传统文本表示的问题：
┌─────────────────────────────────────────────────────────┐
│  One-Hot 编码                                            │
│                                                          │
│  "猫"  →  [1, 0, 0, 0, ...]  (词表大小维度)              │
│  "狗"  →  [0, 1, 0, 0, ...]                              │
│  "汽车" →  [0, 0, 1, 0, ...]                              │
│                                                          │
│  问题：                                                   │
│  - 维度灾难：向量维度 = 词表大小（可能是百万级）           │
│  - 稀疏表示：大量的 0，信息效率低                         │
│  - 无法表达语义相似性：cos("猫", "狗") = 0               │
└─────────────────────────────────────────────────────────┘

词向量的解决方案：
┌─────────────────────────────────────────────────────────┐
│  密集向量表示                                            │
│                                                          │
│  "猫"  →  [0.2, -0.4, 0.7, 0.1, ...]  (通常 100-300 维)  │
│  "狗"  →  [0.3, -0.3, 0.6, 0.2, ...]  ← 与猫相似！       │
│  "汽车" →  [-0.1, 0.5, -0.2, 0.8, ...]                   │
│                                                          │
│  优势：                                                   │
│  - 低维稠密：固定维度，信息紧凑                           │
│  - 语义相似性：相似词的向量接近                           │
│  - 可迁移：预训练词向量可用于下游任务                     │
└─────────────────────────────────────────────────────────┘

核心思想：分布式假设 (Distributional Hypothesis)
    "You shall know a word by the company it keeps."
                                        — J.R. Firth, 1957
    
    一个词的含义由它的上下文决定。
    经常出现在相似上下文中的词，语义是相似的。
    """)


# ==================== 第二部分：Word2Vec 原理 ====================


def word2vec_theory():
    """Word2Vec 理论讲解"""
    print("\n" + "=" * 60)
    print("第二部分：Word2Vec 原理")
    print("=" * 60)

    print("""
Word2Vec (2013, Mikolov et al.) 提出了两种架构：

┌─────────────────────────────────────────────────────────────┐
│                    Skip-gram vs CBOW                         │
├─────────────────────────────┬───────────────────────────────┤
│       Skip-gram             │           CBOW                │
│    (中心词 → 上下文)         │     (上下文 → 中心词)          │
│                             │                               │
│    context: [The, cat, on]  │    context: [The, cat, on]    │
│              ↑              │              ↓                │
│         [sat] → 预测        │         [_] ← 预测             │
│              ↓              │              ↑                │
│    "sat" 预测周围的词        │    周围的词预测 "sat"          │
│                             │                               │
│    适合：稀有词，小数据集    │    适合：频繁词，大数据集      │
└─────────────────────────────┴───────────────────────────────┘

Skip-gram 模型详解：

    输入：中心词 wt (one-hot 或 index)
    输出：上下文词概率 P(wt+j | wt), j ∈ [-c, c], j ≠ 0
    
    网络结构：
    ┌─────────────────────────────────────────────────────┐
    │  输入层        隐藏层 (词向量)      输出层          │
    │  one-hot   →   W (V × d)     →   W' (d × V)   → softmax  │
    │  (1 × V)       (1 × d)            (1 × V)               │
    │                                                          │
    │  中心词 "sat"   词向量           上下文词概率            │
    └─────────────────────────────────────────────────────┘
    
    目标函数（最大化对数似然）：
    J(θ) = (1/T) Σ Σ log P(w_{t+j} | w_t)
    
    softmax 概率：
    P(w_o | w_i) = exp(v'_o · v_i) / Σ exp(v'_w · v_i)
    
    问题：词表 V 很大时，softmax 计算代价高！
    →  解决方案：负采样 (Negative Sampling)

负采样 (Negative Sampling)：
    
    核心思想：不计算全词表 softmax，只采样少量负样本
    
    原目标：最大化 P(context | center)
    新目标：最大化正样本概率，最小化负样本概率
    
    log σ(v'_o · v_i) + Σ log σ(-v'_k · v_i)
         正样本            k 个负样本
    
    负样本采样概率：P(w) ∝ count(w)^(3/4)
    - 3/4 次方使低频词采样概率提高
    """)


# ==================== 第三部分：从零实现 Skip-gram ====================


class SkipGram(nn.Module):
    """Skip-gram 模型实现"""

    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
        """
        super().__init__()
        # 中心词嵌入矩阵
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文词嵌入矩阵
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 初始化
        self._init_embeddings()

    def _init_embeddings(self):
        """初始化嵌入矩阵"""
        init_range = 0.5 / self.center_embeddings.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_words, context_words, negative_words):
        """
        Args:
            center_words: 中心词索引 (batch_size,)
            context_words: 正样本上下文词索引 (batch_size,)
            negative_words: 负样本词索引 (batch_size, num_neg)

        Returns:
            loss: 负采样损失
        """
        # 获取词向量
        center_embeds = self.center_embeddings(center_words)  # (batch, dim)
        context_embeds = self.context_embeddings(context_words)  # (batch, dim)
        neg_embeds = self.context_embeddings(negative_words)  # (batch, num_neg, dim)

        # 正样本得分：中心词与上下文词的点积
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # (batch,)
        pos_loss = F.logsigmoid(pos_score)

        # 负样本得分
        # center_embeds: (batch, dim) -> (batch, dim, 1)
        # neg_embeds: (batch, num_neg, dim)
        # bmm 结果: (batch, num_neg, 1) -> (batch, num_neg)
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        # 总损失（取负号因为要最大化）
        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        """获取训练好的词向量"""
        return self.center_embeddings.weight.data


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第三部分：基础实现")
    print("=" * 60)

    print("\n示例 1: Skip-gram 模型结构\n")

    # 模型参数
    vocab_size = 10000
    embedding_dim = 100

    model = SkipGram(vocab_size, embedding_dim)
    print(f"词表大小: {vocab_size}")
    print(f"词向量维度: {embedding_dim}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟训练数据
    batch_size = 32
    num_neg = 5  # 负采样数量

    center_words = torch.randint(0, vocab_size, (batch_size,))
    context_words = torch.randint(0, vocab_size, (batch_size,))
    negative_words = torch.randint(0, vocab_size, (batch_size, num_neg))

    # 前向传播
    loss = model(center_words, context_words, negative_words)
    print(f"\n批次损失: {loss.item():.4f}")

    print("\n示例 2: 准备训练数据\n")

    # 示例文本
    text = (
        "我 爱 自然 语言 处理 自然 语言 处理 很 有趣 深度 学习 改变 了 自然 语言 处理"
    )
    tokens = text.split()
    print(f"原始文本: {text}")
    print(f"分词结果: {tokens}")

    # 构建词表
    word2idx = {word: idx for idx, word in enumerate(set(tokens))}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(word2idx)
    print(f"\n词表大小: {vocab_size}")
    print(f"词表: {word2idx}")

    # 生成训练样本 (Skip-gram)
    window_size = 2
    training_data = []

    for i, center_word in enumerate(tokens):
        center_idx = word2idx[center_word]
        # 上下文窗口
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                context_idx = word2idx[tokens[j]]
                training_data.append((center_idx, context_idx))

    print(f"\n训练样本数量: {len(training_data)}")
    print("部分训练样本 (中心词, 上下文词):")
    for center, context in training_data[:5]:
        print(f"  {idx2word[center]} → {idx2word[context]}")


# ==================== 第四部分：使用 Gensim 训练词向量 ====================


def gensim_examples():
    """使用 Gensim 训练词向量"""
    print("\n" + "=" * 60)
    print("第四部分：使用 Gensim 训练词向量")
    print("=" * 60)

    print("""
Gensim 是一个成熟的 NLP 库，提供了高效的 Word2Vec 实现。

安装: pip install gensim

基本使用示例:
""")

    code_example = """
from gensim.models import Word2Vec

# 准备训练数据（分词后的句子列表）
sentences = [
    ["我", "爱", "自然", "语言", "处理"],
    ["深度", "学习", "改变", "了", "NLP"],
    ["词向量", "是", "NLP", "的", "基础"],
    # ... 更多句子
]

# 训练模型
model = Word2Vec(
    sentences,
    vector_size=100,  # 词向量维度
    window=5,         # 上下文窗口大小
    min_count=1,      # 最小词频
    workers=4,        # 并行训练线程数
    sg=1,             # 1=Skip-gram, 0=CBOW
    epochs=10         # 训练轮数
)

# 获取词向量
vector = model.wv["自然"]
print(f"'自然' 的词向量形状: {vector.shape}")

# 查找相似词
similar_words = model.wv.most_similar("自然", topn=5)
print("与 '自然' 最相似的词:")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# 词向量运算（类比推理）
# king - man + woman ≈ queen
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
print(f"king - man + woman ≈ {result[0][0]}")

# 保存和加载模型
model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")

# 只保存词向量（更小的文件）
model.wv.save("word2vec.wordvectors")
"""
    print(code_example)

    print("\n使用预训练词向量:\n")

    pretrained_example = """
import gensim.downloader as api

# 下载预训练词向量
# 中文: 可以使用 Tencent AI Lab 或其他预训练向量
# 英文: Google News, GloVe 等

# 加载预训练模型
model = api.load("glove-wiki-gigaword-100")  # 英文 GloVe

# 或者从文件加载
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", 
    binary=True
)
"""
    print(pretrained_example)


# ==================== 第五部分：词向量可视化 ====================


def visualization_examples():
    """词向量可视化"""
    print("\n" + "=" * 60)
    print("第五部分：词向量可视化")
    print("=" * 60)

    print("\n使用 t-SNE 降维可视化词向量:\n")

    # 模拟一些词向量数据
    words = [
        "king",
        "queen",
        "man",
        "woman",
        "prince",
        "princess",
        "dog",
        "cat",
        "puppy",
        "kitten",
        "car",
        "bus",
        "bike",
    ]

    # 模拟词向量（实际应该是训练得到的）
    np.random.seed(42)
    # 让相关词的向量相近
    embeddings = {
        "king": np.array([1.0, 0.5, 0.2]),
        "queen": np.array([1.1, 0.6, 0.3]),
        "man": np.array([0.8, 0.3, 0.1]),
        "woman": np.array([0.9, 0.4, 0.2]),
        "prince": np.array([1.0, 0.55, 0.15]),
        "princess": np.array([1.05, 0.65, 0.25]),
        "dog": np.array([-0.5, 0.8, 0.1]),
        "cat": np.array([-0.4, 0.9, 0.2]),
        "puppy": np.array([-0.5, 0.85, 0.05]),
        "kitten": np.array([-0.35, 0.95, 0.15]),
        "car": np.array([0.1, -0.7, 0.9]),
        "bus": np.array([0.2, -0.6, 0.95]),
        "bike": np.array([0.15, -0.65, 0.85]),
    }

    # 转换为数组
    vectors = np.array([embeddings[w] for w in words])

    print("词向量示例 (3维，便于演示):")
    for word in words[:5]:
        print(f"  {word}: {embeddings[word]}")

    print("\nt-SNE 可视化代码示例:\n")

    tsne_code = """
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设 vectors 是 (n_words, embedding_dim) 的词向量矩阵
# words 是对应的词列表

# t-SNE 降维到 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1))
vectors_2d = tsne.fit_transform(vectors)

# 可视化
plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], marker='o')
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                 xytext=(5, 2), textcoords='offset points',
                 fontsize=12)

plt.title("Word Embeddings Visualization (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.savefig("word_embeddings_tsne.png", dpi=300, bbox_inches='tight')
plt.show()
"""
    print(tsne_code)

    print("""
观察要点：
- 语义相似的词在空间中聚集
- 可以发现词的类比关系
- 不同类别的词形成不同簇
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：实现 CBOW 模型
    任务：参考 Skip-gram 实现，编写 CBOW 模型
    提示：CBOW 是用上下文词预测中心词

练习 1 答案：
    class CBOW(nn.Module):
        '''Continuous Bag of Words 模型'''
        
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, vocab_size)
        
        def forward(self, context_words):
            '''
            context_words: (batch_size, context_size)
            '''
            # 获取上下文词向量
            embeds = self.embeddings(context_words)  # (batch, ctx, dim)
            
            # 平均上下文向量
            avg_embeds = embeds.mean(dim=1)  # (batch, dim)
            
            # 预测中心词
            logits = self.linear(avg_embeds)  # (batch, vocab)
            
            return logits
    
    # 训练时使用 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, center_words)

练习 2：负采样实现
    任务：实现负采样器，按词频的 3/4 次方采样
    提示：使用 numpy 的多项式分布采样

练习 2 答案：
    import numpy as np
    
    class NegativeSampler:
        def __init__(self, word_counts, power=0.75):
            '''
            word_counts: 词频字典 {word_idx: count}
            power: 次方参数，默认 0.75
            '''
            # 计算采样概率
            freqs = np.array(list(word_counts.values()))
            freqs = freqs ** power
            self.probs = freqs / freqs.sum()
            self.vocab_size = len(word_counts)
        
        def sample(self, num_samples):
            '''采样负样本'''
            return np.random.choice(
                self.vocab_size,
                size=num_samples,
                p=self.probs
            )
    
    # 使用
    word_counts = {0: 100, 1: 50, 2: 30, 3: 20, 4: 10}
    sampler = NegativeSampler(word_counts)
    neg_samples = sampler.sample(5)

练习 3：训练自己的词向量
    任务：使用 Gensim 在中文语料上训练词向量
    步骤：
    1. 准备中文文本（可用新闻、wiki）
    2. 分词（使用 jieba）
    3. 训练 Word2Vec
    4. 测试相似词、类比推理

练习 3 答案：
    import jieba
    from gensim.models import Word2Vec
    
    # 1. 准备文本
    texts = [
        "深度学习是机器学习的一个分支",
        "自然语言处理是人工智能的重要领域",
        "词向量是文本表示的基础方法",
        # ... 更多文本
    ]
    
    # 2. 分词
    sentences = [list(jieba.cut(text)) for text in texts]
    
    # 3. 训练
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=100
    )
    
    # 4. 测试
    print(model.wv.most_similar("深度"))
    print(model.wv.similarity("学习", "机器"))

思考题 1：Skip-gram vs CBOW
    为什么 Skip-gram 更适合小数据集和稀有词？
    CBOW 在什么情况下更好？

思考题 1 答案：
    Skip-gram 优势：
    - 每个中心词产生多个训练样本（一个中心词对应多个上下文词）
    - 稀有词作为中心词时也能充分训练
    - 对稀有词的表示更准确
    
    CBOW 优势：
    - 训练速度更快（多个上下文词合并为一个样本）
    - 对高频词的表示更稳定
    - 大数据集时效果更好
    
    选择建议：
    - 小数据集 / 关注稀有词：Skip-gram
    - 大数据集 / 追求速度：CBOW

思考题 2：词向量的局限性
    Word2Vec 有什么局限性？
    提示：考虑多义词、上下文

思考题 2 答案：
    1. 静态表示
       - 每个词只有一个向量
       - "银行" 在 "去银行取钱" 和 "河的银行" 中向量相同
       - 无法处理一词多义
    
    2. 无法处理 OOV
       - 未登录词（不在训练词表中的词）无法表示
       - 需要子词嵌入（如 FastText）解决
    
    3. 忽略词序
       - "狗咬人" 和 "人咬狗" 中 "咬" 的向量相同
       - 需要上下文嵌入（如 ELMo, BERT）解决
    
    4. 窗口大小限制
       - 只能捕捉局部上下文
       - 长距离依赖难以建模

思考题 3：为什么使用两个嵌入矩阵？
    Skip-gram 中为什么要用两个嵌入矩阵（中心词和上下文词）？
    能不能只用一个？

思考题 3 答案：
    使用两个矩阵的原因：
    
    1. 角色区分
       - 中心词和上下文词扮演不同角色
       - 分开建模更灵活
    
    2. 避免自相关
       - 如果只用一个矩阵，词与自己的点积最大
       - 两个矩阵可以避免这个问题
    
    3. 实践效果
       - 实验表明两个矩阵效果更好
       - 训练完成后通常只使用中心词矩阵，或取平均
    
    可以只用一个吗？
       - 可以，但需要特殊处理（如排除自身）
       - 效果可能略差
       - GloVe 就是使用对称的方法
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    word2vec_theory()
    basic_implementation()
    gensim_examples()
    visualization_examples()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 02-embeddings-advanced.py: 词嵌入进阶分析
    
关键要点回顾：
    ✓ 词向量将词映射到低维稠密空间
    ✓ Skip-gram：中心词预测上下文
    ✓ CBOW：上下文预测中心词
    ✓ 负采样大幅提升训练效率
    ✓ Gensim 提供便捷的 Word2Vec 实现
    ✓ t-SNE 可用于词向量可视化
    """)


if __name__ == "__main__":
    main()
