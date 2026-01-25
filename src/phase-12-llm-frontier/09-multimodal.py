"""
多模态模型
==========

学习目标：
    1. 理解多模态LLM架构
    2. 了解视觉编码器和投影方式
    3. 掌握多模态模型使用
"""

import torch
import torch.nn as nn


# ==================== 第一部分：多模态架构 ====================


def introduction():
    """多模态介绍"""
    print("=" * 60)
    print("第一部分：多模态LLM架构")
    print("=" * 60)

    print("""
多模态LLM的核心思想：
    将非文本模态（图像、视频、音频）编码后
    投影到LLM的文本嵌入空间

典型架构：
┌─────────────────────────────────────────────────────────┐
│                                                          │
│   图像 → [Vision Encoder] → [Projector] → LLM → 回复    │
│          (ViT/CLIP)        (MLP/Q-Former)               │
│                                                          │
│   输入: <image> 描述这张图片                             │
│   输出: 这是一张展示...的图片                            │
└─────────────────────────────────────────────────────────┘

代表模型：
    - LLaVA: CLIP + MLP + LLaMA
    - Qwen-VL: ViT + Resampler + Qwen
    - GPT-4V: 原生多模态
    - Gemini: 原生多模态
    """)


# ==================== 第二部分：视觉编码 ====================


class SimpleVisionProjector(nn.Module):
    """简化的视觉投影器"""

    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_features):
        return self.projector(vision_features)


def vision_encoding():
    """视觉编码"""
    print("\n" + "=" * 60)
    print("第二部分：视觉编码")
    print("=" * 60)

    print("""
视觉编码流程：

1. Vision Encoder (视觉编码器)
   - 通常使用预训练的ViT或CLIP
   - 将图像切分为patch，提取特征
   - 输出: (num_patches, vision_dim)

2. Projector (投影器)
   - 将视觉特征映射到LLM的嵌入空间
   - 类型:
     * MLP: 简单线性变换 (LLaVA)
     * Q-Former: 交叉注意力 (BLIP-2)
     * Resampler: 可学习查询 (Qwen-VL)

3. 融合到LLM
   - 视觉token插入到文本序列中
   - <image>占位符替换为视觉特征
    """)

    # 演示
    vision_dim, llm_dim = 1024, 4096
    num_patches = 256  # 16x16

    projector = SimpleVisionProjector(vision_dim, llm_dim)
    vision_features = torch.randn(1, num_patches, vision_dim)
    projected = projector(vision_features)

    print(f"\n视觉特征: {vision_features.shape}")
    print(f"投影后: {projected.shape}")
    print(f"这些特征将作为'视觉token'输入LLM")


# ==================== 第三部分：使用示例 ====================


def usage_examples():
    """使用示例"""
    print("\n" + "=" * 60)
    print("第三部分：多模态模型使用")
    print("=" * 60)

    print("""
1. Qwen-VL使用:

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "example.jpg"},
        {"type": "text", "text": "描述这张图片"}
    ]
}]

inputs = processor(messages, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0]))

2. LLaVA使用:

from llava.model import LlavaLlamaForCausalLM

model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

3. 多图理解:

messages = [{
    "role": "user", 
    "content": [
        {"type": "image", "image": "cat.jpg"},
        {"type": "image", "image": "dog.jpg"},
        {"type": "text", "text": "比较这两张图片的区别"}
    ]
}]
    """)


# ==================== 第四部分：训练流程 ====================


def training_pipeline():
    """训练流程"""
    print("\n" + "=" * 60)
    print("第四部分：多模态模型训练")
    print("=" * 60)

    print("""
两阶段训练：

阶段1: 预训练 (图文对齐)
    - 冻结Vision Encoder和LLM
    - 只训练Projector
    - 使用大规模图文对数据
    - 目标: 让视觉特征与文本空间对齐

阶段2: 指令微调
    - 解冻LLM (或使用LoRA)
    - 使用多模态对话数据
    - 目标: 学会理解和回答多模态问题

常用数据集：
    - CC3M: 图文对
    - COCO: 图像描述
    - LLaVA-Instruct: 多模态对话
    - ShareGPT4V: 高质量描述
    """)


def main():
    introduction()
    vision_encoding()
    usage_examples()
    training_pipeline()

    print("\n" + "=" * 60)
    print("课程完成！下一步: 10-agents-tools.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
