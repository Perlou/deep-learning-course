"""
词嵌入进阶分析
==============

学习目标：
    1. 掌握预训练词向量的加载和使用
    2. 理解词向量的类比推理能力
    3. 了解 GloVe 和 FastText 的特点

核心概念：
    - 预训练词向量：在大规模语料上训练好的向量
    - GloVe：基于共现矩阵的词向量
    - FastText：基于子词的词向量
    - 类比推理：king - man + woman = queen

前置知识：
    - 01-word2vec.py: Word2Vec 基础
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：预训练词向量 ====================


def introduction():
    """预训练词向量介绍"""
    print("=" * 60)
    print("第一部分：预训练词向量")
    print("=" * 60)

    print("""
为什么使用预训练词向量？
    1. 节省训练时间（大规模语料需要几天到几周）
    2. 更好的泛化能力（海量数据覆盖更多词汇）
    3. 迁移学习（将通用知识迁移到特定任务）

常见的预训练词向量：
    - Word2Vec (Google): 300维，1000亿词
    - GloVe (Stanford): 50-300维，60亿词
    - FastText (Facebook): 300维，支持子词
    - 腾讯词向量: 200维，中文最佳之一
    """)


# ==================== 第二部分：GloVe 词向量 ====================


def glove_introduction():
    """GloVe 词向量介绍"""
    print("\n" + "=" * 60)
    print("第二部分：GloVe 词向量")
    print("=" * 60)

    print("""
GloVe (Global Vectors for Word Representation)

核心思想：结合全局统计信息（共现矩阵）和局部上下文

Word2Vec vs GloVe:
    - Word2Vec: 滑动窗口，隐式利用共现
    - GloVe: 共现矩阵，显式利用全局信息

目标函数：J = Σᵢⱼ f(Xᵢⱼ)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²
    - Xᵢⱼ: 词 i 和词 j 的共现次数
    - 词向量点积应与共现次数对数成正比

加载 GloVe 代码：
    def load_glove(filepath):
        word2vec = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                word2vec[word] = vector
        return word2vec
    """)


# ==================== 第三部分：FastText 词向量 ====================


def fasttext_introduction():
    """FastText 词向量介绍"""
    print("\n" + "=" * 60)
    print("第三部分：FastText 词向量")
    print("=" * 60)

    print("""
FastText (Facebook, 2016)

核心创新：将词表示为字符 n-gram 的向量之和

Word2Vec vs FastText:
    - Word2Vec: 每个词一个向量，无法处理 OOV
    - FastText: 词 = Σ(子词向量)，可处理 OOV

字符 n-gram 示例 (n=3):
    "where" → "<wh", "whe", "her", "ere", "re>"

优势：
    - 处理未登录词（OOV）
    - 捕捉词缀信息（形态学）
    - 对拼写错误更鲁棒

使用方法：
    from gensim.models import FastText
    model = FastText(sentences, vector_size=100)
    vector = model.wv["未登录词"]  # 可处理 OOV
    """)


# ==================== 第四部分：类比推理 ====================


def analogy_reasoning():
    """词向量类比推理"""
    print("\n" + "=" * 60)
    print("第四部分：类比推理")
    print("=" * 60)

    print("""
词向量的神奇特性：向量运算可以表达语义关系！

经典案例：king - man + woman ≈ queen

语义关系的向量表示：
    - "性别"方向: woman - man ≈ queen - king
    - "首都"方向: Paris - France ≈ Tokyo - Japan
    - "时态"方向: walked - walk ≈ swam - swim
    """)

    # 模拟演示
    print("\n模拟类比推理演示:\n")

    mock_vectors = {
        "man": np.array([1.0, 0.0, 0.5]),
        "woman": np.array([1.0, 1.0, 0.5]),
        "king": np.array([0.5, 0.0, 1.0]),
        "queen": np.array([0.5, 1.0, 1.0]),
    }

    target = mock_vectors["king"] - mock_vectors["man"] + mock_vectors["woman"]
    print(f"king - man + woman = {target}")
    print(f"queen 向量 = {mock_vectors['queen']}")

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"与 queen 的相似度: {cosine_sim(target, mock_vectors['queen']):.4f}")


# ==================== 第五部分：词向量可视化 ====================


def visualization():
    """词向量可视化"""
    print("\n" + "=" * 60)
    print("第五部分：词向量可视化")
    print("=" * 60)

    print("""
可视化方法：
    - t-SNE: 非线性降维，保留局部结构
    - PCA: 线性降维，计算更快
    - UMAP: 比 t-SNE 更快，保留全局结构

t-SNE 可视化代码：
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
    plt.savefig("word_embeddings.png")
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：加载 GloVe 并找相似词
    任务：加载 GloVe，找与 "computer" 最相似的 10 个词

练习 1 答案：
    def most_similar(word, word2vec, topn=10):
        target = word2vec[word]
        similarities = []
        for w, vec in word2vec.items():
            if w != word:
                sim = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
                similarities.append((w, sim))
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]

练习 2：实现类比推理
    任务：实现 analogy(a, b, c) 函数，返回 d

练习 2 答案：
    def analogy(word_a, word_b, word_c, vectors):
        target = vectors[word_c] - vectors[word_a] + vectors[word_b]
        best_word, best_sim = None, -1
        for word, vec in vectors.items():
            if word in [word_a, word_b, word_c]:
                continue
            sim = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
            if sim > best_sim:
                best_sim, best_word = sim, word
        return best_word

思考题 1：GloVe vs Word2Vec 各自优缺点？
答案：
    - Word2Vec: 增量训练方便，但无全局信息
    - GloVe: 利用全局共现，但需预建矩阵

思考题 2：词向量的偏见问题？
答案：
    - 词向量会学习语料中的偏见（如性别偏见）
    - 需要去偏算法或使用平衡语料

思考题 3：静态词向量的根本限制？
答案：
    - 一词一向量，无法处理多义词
    - 解决方案：上下文词向量（ELMo, BERT）
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    glove_introduction()
    fasttext_introduction()
    analogy_reasoning()
    visualization()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！下一步：03-text-classification.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
