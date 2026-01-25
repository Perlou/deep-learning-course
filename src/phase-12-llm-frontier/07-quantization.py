"""
模型量化
========

学习目标：
    1. 理解量化的原理和目的
    2. 掌握常用量化方法
    3. 学会使用量化工具
"""

import torch
import torch.nn as nn


# ==================== 第一部分：量化概述 ====================


def introduction():
    """量化介绍"""
    print("=" * 60)
    print("第一部分：量化概述")
    print("=" * 60)

    print("""
为什么需要量化？

7B模型显存需求：
    FP32: 28 GB
    FP16: 14 GB
    INT8:  7 GB
    INT4:  3.5 GB

量化优势：
    1. 减少显存占用
    2. 加速推理（整数运算更快）
    3. 可以在消费级GPU上运行大模型

量化公式：
    量化:   q = round(x / scale) + zero_point
    反量化: x ≈ (q - zero_point) × scale
    """)


# ==================== 第二部分：量化方法 ====================


def quantization_methods():
    """量化方法"""
    print("\n" + "=" * 60)
    print("第二部分：量化方法")
    print("=" * 60)

    print("""
常用量化方法：

1. Absmax量化 (对称量化)
   scale = max(|x|) / 127
   q = round(x / scale)
   
2. Zero-point量化 (非对称量化)
   scale = (max - min) / 255
   zero_point = round(-min / scale)
   q = round(x / scale) + zero_point
   
3. GPTQ (权重量化)
   - 基于Hessian信息的最优量化
   - 需要校准数据
   - INT4效果好
   
4. AWQ (激活感知量化)
   - 保护重要权重
   - 基于激活分布
   - 效果最佳
    """)

    # 演示absmax量化
    print("\nAbsmax量化演示：")
    x = torch.tensor([-2.5, -1.0, 0.0, 1.5, 3.0])

    # 量化
    scale = x.abs().max() / 127
    q = torch.round(x / scale).to(torch.int8)

    # 反量化
    x_dequant = q.float() * scale

    print(f"原始: {x.tolist()}")
    print(f"量化: {q.tolist()}")
    print(f"反量化: {x_dequant.tolist()}")
    print(f"误差: {(x - x_dequant).abs().max().item():.4f}")


# ==================== 第三部分：量化工具 ====================


def quantization_tools():
    """量化工具"""
    print("\n" + "=" * 60)
    print("第三部分：量化工具使用")
    print("=" * 60)

    print("""
1. BitsAndBytes (简单易用)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# INT8量化
model_8bit = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_8bit=True
)

# INT4量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"  # NormalFloat4
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config
)

2. AutoGPTQ

from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0"
)

3. llama.cpp (GGUF格式，CPU友好)

# 命令行量化
./quantize model.gguf model-q4.gguf Q4_K_M
    """)


# ==================== 第四部分：量化效果 ====================


def quantization_comparison():
    """量化效果对比"""
    print("\n" + "=" * 60)
    print("第四部分：量化效果对比")
    print("=" * 60)

    print("""
7B模型量化对比：

┌──────────────────────────────────────────────────────────┐
│  精度      显存      推理速度    质量损失                │
├──────────────────────────────────────────────────────────┤
│  FP16      14GB      1.0x        0%                     │
│  INT8      7GB       1.2x        <1%                    │
│  INT4-GPTQ 3.5GB     1.5x        1-3%                   │
│  INT4-AWQ  3.5GB     1.5x        <1%                    │
└──────────────────────────────────────────────────────────┘

选择建议：
    - 显存充足: FP16/BF16
    - 一般推理: INT8
    - 极限压缩: INT4 (AWQ > GPTQ)
    - CPU推理: GGUF格式
    """)


def main():
    introduction()
    quantization_methods()
    quantization_tools()
    quantization_comparison()

    print("\n" + "=" * 60)
    print("课程完成！下一步: 08-inference-optimization.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
