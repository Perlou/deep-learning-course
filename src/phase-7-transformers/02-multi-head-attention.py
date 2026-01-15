"""
多头注意力 (Multi-Head Attention)
=================================

学习目标：
    1. 理解多头注意力的动机和优势
    2. 掌握多头注意力的实现细节
    3. 理解为什么要使用多个注意力头
    4. 实现完整的多头注意力层

核心概念：
    - 多头注意力：并行多个注意力机制
    - 不同子空间：每个头关注不同的特征子空间
    - Concat + Linear：合并多个头的输出
    - 参数效率：通过降维保持参数量

前置知识：
    - 01-self-attention.py: 自注意力机制
    - 矩阵运算和张量操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ======================第一部分：多头注意力概述 ====================


def introduction():
    """多头注意力机制介绍"""
    print("=" * 60)
    print("第一部分：多头注意力概述")
    print("=" * 60)

    print("""
为什么需要多头注意力？

单头注意力的局限：
- 只能从一个角度关注序列
- 可能错过某些重要的关联模式
- 表达能力有限

多头注意力的优势：
- 多个"视角"：每个头可以关注不同类型的关系
  * 头1：可能关注语法关系
  * 头2：可能关注语义关系
  * 头3：可能关注位置关系
- 更丰富的表示：捕捉多种模式
- 类似于CNN中的多个卷积核

数学形式：
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    
    其中 head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
    
    关键参数：
    - h: 头数（通常为 8 或 16）
    - d_k = d_model / h: 每个头的维度
    """)


# ==================== 第二部分：多头注意力实现 ====================


class MultiHeadAttention(nn.Module):
    """多头注意力层"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性变换层（为所有头一次性计算）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, X):
        """
        将最后一维分割成 (num_heads, d_k)

        Args:
            X: (batch, seq_len, d_model)
        Returns:
            X: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = X.size()

        # 重塑: (batch, seq_len, num_heads, d_k)
        X = X.view(batch_size, seq_len, self.num_heads, self.d_k)

        # 转置: (batch, num_heads, seq_len, d_k)
        return X.transpose(1, 2)

    def combine_heads(self, X):
        """
        将多头合并回单个张量

        Args:
            X: (batch, num_heads, seq_len, d_k)
        Returns:
            X: (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = X.size()

        # 转置: (batch, seq_len, num_heads, d_k)
        X = X.transpose(1, 2)

        # 重塑: (batch, seq_len, d_model)
        return X.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)

        # 1. 线性变换
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)

        # 2. 分割成多个头
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算缩放点积注意力
        # Q @ K^T: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 4. 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 5. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 6. 加权求和
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        attn_output = torch.matmul(attention_weights, V)

        # 7. 合并多个头
        attn_output = self.combine_heads(attn_output)  # (batch, seq_len, d_model)

        # 8. 最终线性变换
        output = self.W_o(attn_output)

        return output, attention_weights


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第二部分：基础实现")
    print("=" * 60)

    print("\n示例 1: 多头注意力的形状变换\n")

    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # 创建模型
    mha = MultiHeadAttention(d_model, num_heads)

    # 创建输入
    X = torch.randn(batch_size, seq_len, d_model)

    print(f"输入形状: {X.shape}")
    print(f"模型维度 d_model: {d_model}")
    print(f"注意力头数: {num_heads}")
    print(f"每个头维度 d_k: {d_model // num_heads}")

    # 前向传播
    output, attention_weights = mha(X, X, X)

    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"  -> (batch_size, num_heads, seq_len, seq_len)")

    print("\n示例 2: 可视化不同头的注意力模式\n")

    # 创建一个简单的例子
    seq_len = 8
    d_model = 64
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)
    X = torch.randn(1, seq_len, d_model)

    output, attention_weights = mha(X, X, X)

    # 可视化每个头的注意力
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        attn = attention_weights[0, head_idx].detach().numpy()

        im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        # 添加颜色条
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("multi_head_attention_patterns.png", dpi=300, bbox_inches="tight")
    print("✓ 多头注意力模式已保存到 'multi_head_attention_patterns.png'")
    print("注意：不同的头可能会学习到不同的注意力模式")


# ==================== 第三部分：进阶应用 ====================


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第三部分：进阶应用")
    print("=" * 60)

    print("\n示例 1: 交叉注意力（Cross-Attention）\n")

    # 在 Transformer 解码器中，Q 来自解码器，K 和 V 来自编码器
    batch_size = 1
    encoder_seq_len = 10  # 编码器序列长度
    decoder_seq_len = 8  # 解码器序列长度
    d_model = 256
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)

    # 解码器的查询
    decoder_output = torch.randn(batch_size, decoder_seq_len, d_model)
    # 编码器的键和值
    encoder_output = torch.randn(batch_size, encoder_seq_len, d_model)

    # 交叉注意力：Q 来自解码器，K 和 V 来自编码器
    output, cross_attn_weights = mha(
        Q=decoder_output, K=encoder_output, V=encoder_output
    )

    print(f"解码器序列长度: {decoder_seq_len}")
    print(f"编码器序列长度: {encoder_seq_len}")
    print(f"交叉注意力权重形状: {cross_attn_weights.shape}")
    print(f"  -> (batch, num_heads, decoder_len, encoder_len)")

    print("\n示例 2: 参数量分析\n")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    configs = [
        (512, 8),  # 标准配置
        (512, 16),  # 更多头
        (768, 12),  # BERT-base
        (1024, 16),  # BERT-large
    ]

    print("d_model | num_heads | 参数量")
    print("-" * 40)
    for d_model, num_heads in configs:
        mha = MultiHeadAttention(d_model, num_heads)
        params = count_parameters(mha)
        print(f"{d_model:7d} | {num_heads:9d} | {params:9,d}")

    print("""
观察：
    - 参数量主要由 d_model 决定
    - 增加头数不会显著增加参数量
    - 4 个线性层：W_q, W_k, W_v, W_o，每个都是 d_model × d_model
    """)

    print("\n示例 3: 多头注意力的计算效率\n")

    import time

    d_model = 512
    seq_len = 100
    batch_size = 32

    # 比较不同头数的运行时间
    head_nums = [1, 4, 8, 16]
    times = []

    X = torch.randn(batch_size, seq_len, d_model)

    for num_heads in head_nums:
        mha = MultiHeadAttention(d_model, num_heads)
        mha.eval()

        # 预热
        with torch.no_grad():
            _ = mha(X, X, X)

        # 计时
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = mha(X, X, X)
        elapsed = (time.time() - start) / 10
        times.append(elapsed * 1000)  # 转换为毫秒

    print("num_heads | 时间 (ms)")
    print("-" * 30)
    for heads, t in zip(head_nums, times):
        print(f"{heads:9d} | {t:10.2f}")

    print("""
观察：
    - 多头并行计算，时间差异不大
    - GPU 上并行效率高
    - 主要瓶颈是矩阵乘法，而非头数
    """)


# ==================== 第四部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：单头 vs 多头
    实现两个版本的注意力层：
    1. 单头注意力（d_model = 512）
    2. 8头注意力（每头 d_k = 64）
    
    比较：
    - 参数量
    - 表达能力
    - 计算复杂度

练习 2：可视化注意力头
    加载预训练的 BERT 或 GPT-2 模型
    任务：
    1. 提取第一层的所有注意力头
    2. 可视化每个头关注的模式
    3. 分析不同头是否学习到了不同的模式

练习 3：实现分组注意力
    在某些高效 Transformer 中，使用分组注意力（Grouped Attention）
    不同于多头注意力，分组注意力对 Q、K、V 也进行分组
    
    任务：实现 GroupedAttention 类

思考题 1：为什么要降维？
    在多头注意力中，为什么要将 d_model 分成 num_heads 份？
    为什么不直接用 num_heads 个完整的 d_model 维度的头？

思考题 2：多头的独立性
    理论上，多个头应该学习不同的模式
    实践中，如何确保不同的头真的学习到了不同的东西？
    提示：考虑初始化、正则化等技巧

思考题 3：最优头数
    头数是否越多越好？
    如何确定最优的头数？
    考虑：
    - 模型容量
    - 计算效率
    - 表达能力
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    basic_implementation()
    advanced_examples()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 03-positional-encoding.py: 位置编码
    - 04-transformer-encoder.py: Transformer 编码器
    
关键要点回顾：
    ✓ 多头注意力允许模型从多个角度关注序列
    ✓ 每个头在不同的子空间中运作 (d_k = d_model / num_heads)
    ✓ 多个头的输出通过拼接和线性变换合并
    ✓ 多头注意力不会显著增加参数量
    ✓ 典型配置：8 或 16 个头
    """)


if __name__ == "__main__":
    main()
