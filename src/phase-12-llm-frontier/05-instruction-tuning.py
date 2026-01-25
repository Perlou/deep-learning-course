"""
指令微调
========

学习目标：
    1. 理解指令微调的目的和方法
    2. 掌握LoRA等高效微调技术
    3. 学会准备指令数据和应用对话模板
"""

import torch
import torch.nn as nn


# ==================== 第一部分：指令微调概述 ====================


def introduction():
    """指令微调介绍"""
    print("=" * 60)
    print("第一部分：指令微调概述")
    print("=" * 60)

    print("""
预训练 vs 指令微调：

预训练模型：
    输入: "写一篇关于AI的文章"
    输出: "写一篇关于AI的发展历史..."  (续写)

指令微调后：
    输入: "写一篇关于AI的文章"
    输出: "人工智能是计算机科学的分支..."  (回答)

指令微调教会模型：遵循指令、对话交互
    """)


# ==================== 第二部分：指令数据格式 ====================


def instruction_data():
    """指令数据格式"""
    print("\n" + "=" * 60)
    print("第二部分：指令数据格式")
    print("=" * 60)

    print("""
标准对话格式 (OpenAI):
{
  "messages": [
    {"role": "system", "content": "你是AI助手"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是..."}
  ]
}

Alpaca格式:
{
  "instruction": "解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支..."
}

对话模板 (ChatML):
<|im_start|>system
你是AI助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
    """)


# ==================== 第三部分：LoRA ====================


class LoRALayer(nn.Module):
    """Low-Rank Adaptation层"""

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 低秩分解: W + ΔW = W + BA
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x, original_output):
        # ΔW × x = B × A × x
        lora_output = x @ self.lora_A @ self.lora_B * self.scaling
        return original_output + lora_output


def lora_examples():
    """LoRA示例"""
    print("\n" + "=" * 60)
    print("第三部分：LoRA高效微调")
    print("=" * 60)

    print("""
LoRA核心思想：
    原始权重W固定，只训练低秩增量 ΔW = BA
    
    W: (d × d) 矩阵，参数量 d²
    ΔW = B × A
    A: (d × r), B: (r × d), 参数量 2dr
    
    当r << d时，参数量大幅减少
    """)

    # 参数量对比
    d, r = 4096, 8
    original = d * d
    lora = 2 * d * r
    print(f"\n参数量对比 (d={d}, r={r}):")
    print(f"  原始: {original:,} ({original / 1e6:.1f}M)")
    print(f"  LoRA: {lora:,} ({lora / 1e3:.1f}K)")
    print(f"  比例: {lora / original * 100:.2f}%")


# ==================== 第四部分：微调代码 ====================


def finetuning_code():
    """微调代码示例"""
    print("\n" + "=" * 60)
    print("第四部分：微调代码示例")
    print("=" * 60)

    print("""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16
)

# 配置LoRA
lora_config = LoraConfig(
    r=8,                    # 秩
    lora_alpha=16,          # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用到哪些层
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable: 4M / total: 7B = 0.06%

# 训练...
    """)


def main():
    introduction()
    instruction_data()
    lora_examples()
    finetuning_code()

    print("\n" + "=" * 60)
    print("课程完成！下一步: 06-rlhf-basics.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
