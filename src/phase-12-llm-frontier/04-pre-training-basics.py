"""
预训练基础
=========

学习目标：
    1. 理解语言模型预训练的目标
    2. 了解预训练数据处理流程
    3. 掌握分布式训练基本概念

核心概念：
    - CLM: 因果语言建模
    - 分布式训练: DP, TP, PP, ZeRO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 第一部分：预训练目标 ====================


def introduction():
    """预训练目标介绍"""
    print("=" * 60)
    print("第一部分：预训练目标")
    print("=" * 60)

    print("""
大语言模型预训练的核心目标：

因果语言建模 (Causal Language Modeling):
    目标：预测下一个token
    
    输入: [我, 爱, 学习, 深度]
    目标: [爱, 学习, 深度, 学习]
    
    损失: L = -1/T × Σ log P(xₜ | x₁, ..., xₜ₋₁)

为什么这么简单的目标能产生智能？
1. 预测下一个词需要理解上下文
2. 长距离依赖需要逻辑推理  
3. 海量数据涵盖各种知识
4. Scaling Laws: 更大模型+更多数据=更强能力
    """)


# ==================== 第二部分：训练示例 ====================


def training_demo():
    """训练演示"""
    print("\n" + "=" * 60)
    print("第二部分：语言模型训练")
    print("=" * 60)

    # 模拟数据
    batch_size, seq_len, vocab_size = 2, 8, 100
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # 准备标签（向右移动一位）
    labels = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :]

    # 计算损失
    loss = F.cross_entropy(logits_shifted.reshape(-1, vocab_size), labels.reshape(-1))

    print(f"输入形状: {input_ids.shape}")
    print(f"损失: {loss.item():.4f}")
    print(f"困惑度: {math.exp(loss.item()):.2f}")


# ==================== 第三部分：分布式训练 ====================


def distributed_training():
    """分布式训练"""
    print("\n" + "=" * 60)
    print("第三部分：分布式训练")
    print("=" * 60)

    print("""
7B模型显存需求（FP16）：
    模型参数:    14 GB
    梯度:        14 GB
    优化器状态:   56 GB
    激活值:      ~10 GB
    总计:        ~94 GB

分布式训练策略：

1. 数据并行 (DP): 每GPU完整模型，数据分片
2. 张量并行 (TP): 矩阵切分到多GPU
3. 流水线并行 (PP): 模型按层切分
4. ZeRO: 消除冗余存储

3D并行: TP + PP + DP 组合使用
    """)


# ==================== 第四部分：训练稳定性 ====================


def training_stability():
    """训练稳定性"""
    print("\n" + "=" * 60)
    print("第四部分：训练稳定性")
    print("=" * 60)

    print("""
关键技术：

1. 梯度裁剪: max_norm = 1.0
2. 学习率调度: Warmup + Cosine Decay
3. 混合精度: BF16前向/反向 + FP32主参数
    """)

    # 梯度裁剪示例
    params = [torch.randn(10, requires_grad=True) for _ in range(3)]
    for p in params:
        p.grad = torch.randn(10) * 10

    norm_before = torch.sqrt(sum(p.grad.norm() ** 2 for p in params))
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    norm_after = torch.sqrt(sum(p.grad.norm() ** 2 for p in params))

    print(f"\n梯度裁剪: {norm_before.item():.2f} → {norm_after.item():.2f}")


# ==================== 第五部分：代码示例 ====================


def code_examples():
    """代码示例"""
    print("\n" + "=" * 60)
    print("第五部分：HuggingFace训练示例")
    print("=" * 60)

    print("""
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=1000,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
    """)


def main():
    introduction()
    training_demo()
    distributed_training()
    training_stability()
    code_examples()

    print("\n" + "=" * 60)
    print("课程完成！下一步: 05-instruction-tuning.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
