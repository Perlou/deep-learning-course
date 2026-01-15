"""
GPT 文本生成 (GPT Text Generation)
=================================

学习目标：
    1. 掌握不同的文本生成策略
    2. 实现贪心解码、采样、Top-K和Top-P
    3. 理解温度参数的作用
    4. 学习Prompt Engineering

核心概念：
    - 解码策略
    - 温度采样
    - Top-K和Top-P（nucleus）采样
    - Beam Search
    - Prompt Engineering

前置知识：
    - 09-gpt-architecture.py
"""

import torch
import torch.nn.functional as F


def greedy_decode(logits):
    """贪心解码：选择概率最高的token"""
    return logits.argmax(dim=-1)


def sample_with_temperature(logits, temperature=1.0):
    """温度采样"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_k_sampling(logits, k=50, temperature=1.0):
    """Top-K采样：只从概率最高的K个token中采样"""
    logits = logits / temperature

    # 获取top-k的值和索引
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # 将其他位置设为负无穷
    logits_filtered = torch.full_like(logits, float("-inf"))
    logits_filtered.scatter_(-1, top_k_indices, top_k_logits)

    # 采样
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_p_sampling(logits, p=0.9, temperature=1.0):
    """Top-P (Nucleus) 采样：从累积概率达到p的最小集合中采样"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # 排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到累积概率超过p的位置
    sorted_indices_to_remove = cumulative_probs > p
    # 保留第一个超过p的token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 移除这些token
    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # 采样
    return torch.multinomial(sorted_probs, num_samples=1)


def main():
    print("=" * 60)
    print("GPT 文本生成策略")
    print("=" * 60)

    print("""
文本生成解码策略：

1. 贪心解码 (Greedy Decoding):
   - 每步选择概率最高的token
   - 优点：简单、确定性
   - 缺点：容易陷入重复、缺乏多样性

2. 温度采样 (Temperature Sampling):
   - temperature < 1: 使分布更集中（更确定）
   - temperature > 1: 使分布更平坦（更随机）
   - temperature = 1: 标准softmax
   
   公式: p_i = exp(logit_i / T) / Σ exp(logit_j / T)

3. Top-K 采样:
   - 只从概率最高的K个token中采样
   - 好处：避免采样到低概率的噪声token
   - 常用值：K=50

4. Top-P (Nucleus) 采样:
   - 从累积概率达到P的最小集合中采样
   - 动态调整候选集大小
   - 更灵活，适应不同情况
   - 常用值：P=0.9

5. Beam Search:
   - 维护K个最可能的序列
   - 在翻译等任务中很有效
   - 但可能导致generic输出
    """)

    # 示例：比较不同解码策略
    vocab_size = 1000
    logits = torch.randn(1, vocab_size)

    print("\n示例：相同logits，不同解码策略\n")

    # 贪心
    greedy_token = greedy_decode(logits)
    print(f"贪心解码: token {greedy_token.item()}")

    # 不同温度
    for temp in [0.5, 1.0, 1.5]:
        token = sample_with_temperature(logits, temperature=temp)
        print(f"温度={temp}: token {token.item()}")

    # Top-K
    token = top_k_sampling(logits, k=50)
    print(f"Top-K (k=50): token {token.item()}")

    # Top-P
    token = top_p_sampling(logits, p=0.9)
    print(f"Top-P (p=0.9): token {token.item()}")

    print("""
Prompt Engineering 技巧：

1. 清晰的指令:
   "将下面的文本翻译成英文："

2. Few-shot示例:
   "Q: 什么是机器学习？
    A: 机器学习是...
    
    Q: 什么是深度学习？
    A:"

3. 思维链 (Chain-of-Thought):
   "让我们一步步思考："

4. 角色设定:
   "你是一位经验丰富的Python程序员..."

最佳实践：
    ✓ 明确任务描述
    ✓ 提供清晰的输入格式
    ✓ 使用示例引导
    ✓ 迭代优化prompt
    ✓ 控制生成长度
    """)


if __name__ == "__main__":
    main()
