"""
自注意力机制 (Self-Attention)
==============================

学习目标：
    1. 理解注意力机制的核心思想
    2. 掌握 Q、K、V 三个矩阵的含义
    3. 从零实现 Scaled Dot-Product Attention
    4. 理解 softmax 在注意力中的作用

核心概念：
    - 注意力机制：动态权重，关注重要信息
    - Query, Key, Value：查询、键、值三元组
    - Scaled Dot-Product：缩放点积注意力
    - 注意力权重：表示不同位置之间的相关性

前置知识：
    - Phase 3: PyTorch 基础
    - Phase 4: 神经网络基础
    - 矩阵乘法和 softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：注意力机制概述 ====================


def introduction():
    """
    注意力机制介绍

    注意力机制的核心思想：
    - 在处理序列数据时，不同位置的信息重要程度不同
    - 模型应该"注意"重要的部分，忽略不重要的部分
    - 通过计算注意力权重来实现这一目标
    """
    print("=" * 60)
    print("第一部分：注意力机制概述")
    print("=" * 60)

    print("""
注意力机制的直觉理解：

假设你在读一段文字："猫坐在垫子上"
- 当理解"坐"这个词时，我们会自动关注"猫"和"垫子"
- 而不会平等地关注所有词
- 注意力机制就是模拟这种"选择性关注"的能力

数学形式：
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    其中：
    - Q (Query): 查询向量，表示"我在找什么"
    - K (Key): 键向量，表示"我是什么"
    - V (Value): 值向量，表示"我的内容是什么"
    - d_k: Key 的维度，用于缩放
    """)


# ==================== 第二部分：从零实现自注意力 ====================


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Args:
        Q: Query矩阵 (batch_size, seq_len, d_k)
        K: Key矩阵   (batch_size, seq_len, d_k)
        V: Value矩阵 (batch_size, seq_len, d_v)
        mask: 掩码，用于屏蔽某些位置 (batch_size, seq_len, seq_len)

    Returns:
        output: 注意力输出 (batch_size, seq_len, d_v)
        attention_weights: 注意力权重 (batch_size, seq_len, seq_len)
    """
    # 1. 计算点积：Q @ K^T
    # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 2. 缩放：除以 √d_k (防止点积过大导致梯度消失)
    scores = scores / np.sqrt(d_k)

    # 3. 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 4. Softmax：将分数转换为概率分布
    attention_weights = F.softmax(scores, dim=-1)

    # 5. 加权求和：attention_weights @ V
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def basic_implementation():
    """基础实现示例"""
    print("\n" + "=" * 60)
    print("第二部分：基础实现")
    print("=" * 60)

    print("\n示例 1: 简单的自注意力计算\n")

    # 设置参数
    batch_size = 1
    seq_len = 4  # 序列长度（例如4个词）
    d_model = 8  # 特征维度

    # 创建输入序列（假设是4个词的嵌入）
    X = torch.randn(batch_size, seq_len, d_model)
    print(f"输入序列形状: {X.shape}")
    print(f"输入序列:\n{X[0, :2, :4]}...\n")  # 只打印前2个词的前4个维度

    # 在自注意力中，Q、K、V 都来自同一个输入 X
    Q = X
    K = X
    V = X

    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}\n")

    print("注意力权重矩阵:")
    print(attention_weights[0].detach().numpy())
    print("""
解释：注意力权重 [i, j] 表示位置 i 对位置 j 的关注程度
- 每一行的和为 1（因为经过 softmax）
- 对角线值通常较大（每个位置最关注自己）
    """)

    print("\n示例 2: 可视化注意力权重\n")

    # 创建一个有意义的示例
    # 假设输入是："我 爱 深度 学习"
    words = ["我", "爱", "深度", "学习"]

    # 创建简单的输入（手工设计，便于理解）
    # 让"深度"和"学习"更相关
    X = torch.tensor(
        [
            [1.0, 0.0],  # "我"
            [0.0, 1.0],  # "爱"
            [0.5, 0.5],  # "深度"
            [0.6, 0.4],  # "学习"
        ]
    ).unsqueeze(0)  # 添加 batch 维度

    Q = K = V = X
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights[0].detach().numpy(), cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Attention Weight")
    plt.xticks(range(len(words)), words)
    plt.yticks(range(len(words)), words)
    plt.xlabel("Key (被关注的词)")
    plt.ylabel("Query (当前词)")
    plt.title("自注意力权重矩阵")

    # 在每个格子中显示数值
    for i in range(len(words)):
        for j in range(len(words)):
            weight = attention_weights[0, i, j].item()
            plt.text(
                j,
                i,
                f"{weight:.2f}",
                ha="center",
                va="center",
                color="white" if weight > 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig("self_attention_weights.png", dpi=300, bbox_inches="tight")
    print("✓ 注意力权重可视化已保存到 'self_attention_weights.png'")


# ==================== 第三部分：自注意力层实现 ====================


class SelfAttention(nn.Module):
    """自注意力层"""

    def __init__(self, d_model):
        """
        Args:
            d_model: 模型维度
        """
        super().__init__()
        self.d_model = d_model

        # 线性变换层：将输入映射为 Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性层
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):
        """
        Args:
            X: 输入张量 (batch_size, seq_len, d_model)
            mask: 掩码 (batch_size, seq_len, seq_len)

        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
            attention_weights: 注意力权重
        """
        # 1. 生成 Q, K, V
        Q = self.W_q(X)  # (batch, seq_len, d_model)
        K = self.W_k(X)
        V = self.W_v(X)

        # 2. 计算缩放点积注意力
        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 3. 输出线性变换
        output = self.W_o(attn_output)

        return output, attention_weights


def advanced_examples():
    """进阶应用示例"""
    print("\n" + "=" * 60)
    print("第三部分：进阶应用")
    print("=" * 60)

    print("\n示例 1: 使用自注意力层\n")

    # 创建模型
    d_model = 64
    seq_len = 10
    batch_size = 2

    self_attn = SelfAttention(d_model)

    # 创建输入
    X = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output, attention_weights = self_attn(X)

    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")

    print("\n示例 2: 使用掩码（Masked Attention）\n")

    # 创建一个因果掩码（用于自回归生成）
    # 只允许关注当前位置及之前的位置
    seq_len = 5
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = ~mask  # 反转：True表示可以关注，False表示不能关注
    mask = mask.unsqueeze(0)  # 添加batch维度

    print("因果掩码（下三角矩阵）:")
    print(mask[0].int().numpy())
    print("(1 = 可以关注, 0 = 不能关注)")

    # 应用掩码
    X = torch.randn(1, seq_len, d_model)
    output, attention_weights = self_attn(X, mask=mask)

    print("\n带掩码的注意力权重:")
    print(attention_weights[0].detach().numpy())
    print("注意：上三角部分的权重为0（被掩蔽）")

    print("\n示例 3: 注意力的计算复杂度\n")

    seq_lens = [10, 50, 100, 200, 500]
    d_model = 512

    print(f"模型维度 d_model = {d_model}")
    print("\n序列长度 | 注意力矩阵大小 | FLOPs (相对)")
    print("-" * 50)

    for seq_len in seq_lens:
        matrix_size = seq_len * seq_len
        flops_relative = seq_len * seq_len * d_model / 1e6
        print(f"{seq_len:8d} | {matrix_size:14,d} | {flops_relative:8.2f}M")

    print("""
观察：
    - 注意力的计算复杂度是 O(n²d)，其中 n 是序列长度
    - 序列长度翻倍，计算量增加 4 倍
    - 这是 Transformer 处理长序列时的主要瓶颈
    """)


# ==================== 第四部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：手动计算注意力
    给定：
    Q = [[1, 0], [0, 1]]
    K = [[1, 0], [0, 1]]
    V = [[2, 3], [4, 5]]
    
    任务：手动计算 Attention(Q, K, V) 的结果

练习 2：实现加性注意力
    除了点积注意力，还有加性注意力：
    score(Q, K) = v^T tanh(W_q Q + W_k K)
    
    任务：实现一个 AdditiveAttention 类

练习 3：注意力可视化
    任务：
    1. 加载一个预训练的 BERT 模型
    2. 提取第一层的注意力权重
    3. 可视化不同注意力头对输入句子的关注模式

思考题 1：为什么需要缩放？
    在 Scaled Dot-Product Attention 中，为什么要除以 √d_k？
    提示：考虑点积的方差和 softmax 的饱和问题

思考题 2：自注意力 vs 循环神经网络
    与 RNN/LSTM 相比，自注意力有什么优势和劣势？
    
    优势：
    - ?
    
    劣势：
    - ?

思考题 3：位置信息
    自注意力本身是位置不变的（permutation invariant）
    如果交换输入序列的顺序，输出会怎样？
    这说明了什么问题？
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
    - 02-multi-head-attention.py: 多头注意力机制
    - 03-positional-encoding.py: 位置编码
    
关键要点回顾：
    ✓ 注意力机制允许模型动态关注重要信息
    ✓ Q、K、V 分别代表查询、键、值
    ✓ 缩放点积注意力是最常用的注意力形式
    ✓ 注意力权重通过 softmax(QK^T/√d_k) 计算
    ✓ 自注意力的计算复杂度是 O(n²d)
    """)


if __name__ == "__main__":
    main()
