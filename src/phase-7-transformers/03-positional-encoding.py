"""
位置编码 (Positional Encoding)
==============================

学习目标：
    1. 理解Transformer为什么需要位置编码
    2. 掌握正弦位置编码的原理
    3. 实现可学习位置编码
    4. 理解绝对位置编码vs相对位置编码

核心概念：
    - 位置不变性：Self-Attention本身对位置不敏感
    - 正弦编码：使用sin/cos函数生成位置信息
    - 可学习编码：将位置编码作为参数学习
    - 位置叠加：将位置编码加到词嵌入上

前置知识：
    - 01-self-attention.py
    - 三角函数基础
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：位置编码概述 ====================


def introduction():
    """位置编码介绍"""
    print("=" * 60)
    print("第一部分：位置编码概述")
    print("=" * 60)

    print("""
为什么需要位置编码？

问题：Self-Attention是位置不变的（Permutation Invariant）
- 如果打乱输入序列的顺序，Self-Attention的输出是一样的
- 例如："我 爱 你" 和 "你 爱 我" 会得到相同的表示（如果没有位置信息）

解决方案：位置编码（Positional Encoding）
- 为序列中的每个位置添加唯一的标识
- 让模型能够区分不同位置的词

正弦位置编码公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中：
    - pos: 位置索引 (0, 1, 2, ...)
    - i: 维度索引 (0, 1, 2, ..., d_model//2)
    - 偶数维度用sin，奇数维度用cos

特点：
    ✓ 不需要训练（确定性函数）
    ✓ 可以泛化到任意长度序列
    ✓ 每个位置的编码是唯一的
    ✓ 相对位置信息可以通过三角恒等式计算
    """)


# ==================== 第二部分：正弦位置编码实现 ====================


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比例
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母：10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # 应用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 添加batch维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 注册为buffer（不作为参数训练）
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: 输入嵌入 (batch_size, seq_len, d_model)
        Returns:
            加上位置编码的嵌入
        """
        # 取出对应长度的位置编码
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第二部分：基础实现")
    print("=" * 60)

    print("\n示例 1: 生成位置编码\n")

    d_model = 512
    max_len = 100

    pe_layer = PositionalEncoding(d_model, max_len)

    # 查看位置编码矩阵
    pe_matrix = pe_layer.pe[0].numpy()  # (max_len, d_model)

    print(f"位置编码矩阵形状: {pe_matrix.shape}")
    print(f"第1个位置的编码前10维: {pe_matrix[0, :10]}")
    print(f"第2个位置的编码前10维: {pe_matrix[1, :10]}")

    print("\n示例 2: 可视化位置编码\n")

    # 可视化位置编码
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 热力图
    im = ax1.imshow(pe_matrix[:50, :].T, cmap="RdBu", aspect="auto")
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Dimension")
    ax1.set_title("Positional Encoding Heatmap")
    plt.colorbar(im, ax=ax1)

    # 波形图：显示前几个维度随位置的变化
    for i in range(8):
        ax2.plot(pe_matrix[:100, i], label=f"dim {i}")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Value")
    ax2.set_title("Positional Encoding Waveforms")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=300, bbox_inches="tight")
    print("✓ 位置编码可视化已保存到 'positional_encoding.png'")

    print("\n观察：")
    print("- 不同维度有不同的频率")
    print("- 低维度变化快（高频），高维度变化慢（低频）")
    print("- 类似于傅里叶基函数")

    print("\n示例 3: 使用位置编码\n")

    batch_size = 2
    seq_len = 20
    d_model = 128

    # 创建词嵌入
    word_embeddings = torch.randn(batch_size, seq_len, d_model)
    print(f"词嵌入形状: {word_embeddings.shape}")

    # 添加位置编码
    pe_layer = PositionalEncoding(d_model)
    embeddings_with_pos = pe_layer(word_embeddings)

    print(f"加入位置编码后形状: {embeddings_with_pos.shape}")
    print("位置编码直接加到词嵌入上")


# ==================== 第三部分：其他位置编码方式 ====================


class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 位置编码作为可学习参数
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第三部分：进阶应用")
    print("=" * 60)

    print("\n示例 1: 正弦编码 vs 可学习编码\n")

    d_model = 256
    max_len = 100

    # 正弦编码
    sinusoidal_pe = PositionalEncoding(d_model, max_len, dropout=0)
    # 可学习编码
    learned_pe = LearnedPositionalEncoding(d_model, max_len, dropout=0)

    print("正弦位置编码:")
    print("  优点: 不需要训练，可泛化到任意长度")
    print("  缺点: 固定模式，可能不是最优")
    print(f"  参数量: 0")

    print("\n可学习位置编码:")
    print("  优点: 可以学习任务特定的位置模式")
    print("  缺点: 受max_len限制，需要训练")
    learned_params = sum(p.numel() for p in learned_pe.parameters())
    print(f"  参数量: {learned_params:,}")

    print("\n示例 2: 相对位置编码思想\n")

    print("""
相对位置编码 (Relative Positional Encoding):
- 不编码绝对位置，而是编码相对位置
- 在注意力计算中加入相对位置偏置
- 优势：更好的长度泛化能力

公式（简化版）:
    Attention(Q, K, V) = softmax((QK^T + R)/√d_k) V
    
    其中 R[i,j] 表示位置i到位置j的相对位置偏置
    
应用：
    - Transformer-XL
    - T5
    - DeBERTa
    """)

    print("\n示例 3: 位置编码的影响分析\n")

    # 创建简单测试
    seq_len = 10
    d_model = 64

    # 无位置编码
    x_no_pos = torch.randn(1, seq_len, d_model)

    # 有位置编码
    pe = PositionalEncoding(d_model, dropout=0)
    x_with_pos = pe(x_no_pos.clone())

    # 计算差异
    diff = (x_with_pos - x_no_pos).norm(dim=-1)[0]

    print(f"每个位置添加的位置信息强度:")
    for i in range(seq_len):
        print(f"  位置 {i}: {diff[i].item():.4f}")

    print("\n观察：位置编码为每个位置添加了独特的信号")


# ==================== 第四部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：手动计算
    给定 d_model=4, pos=0,1,2
    手动计算正弦位置编码的值

练习 2：实现2D位置编码
    对于图像，需要2D位置编码
    任务：实现 PositionalEncoding2D 类
    提示：分别对x和y坐标进行编码

练习 3：位置编码可视化
    任务：
    1. 计算任意两个位置编码的余弦相似度
    2. 可视化位置相似度矩阵
    3. 分析：距离近的位置是否更相似？

思考题 1：为什么用sin和cos？
    - 为什么不用其他函数？
    - sin和cos的组合有什么数学性质？
    - 提示：考虑三角恒等式

思考题 2：位置编码的缩放
    在原始Transformer论文中，位置编码直接加到词嵌入上
    这要求两者的scale相似
    如何确保？有更好的方法吗？

思考题 3：长度泛化
    正弦位置编码可以处理训练时未见过的长度
    但实际效果如何？
    为什么有些模型（如BERT）使用可学习位置编码？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    basic_implementation()
    advanced_examples()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 04-transformer-encoder.py: Transformer编码器
    
关键要点回顾：
    ✓ 位置编码为Self-Attention添加位置信息
    ✓ 正弦编码使用sin/cos函数生成确定性位置特征
    ✓ 可学习编码将位置作为参数训练
    ✓ 相对位置编码关注相对距离而非绝对位置
    ✓ 位置编码通过加法融入词嵌入
    """)


if __name__ == "__main__":
    main()
