"""
评估脚本
========

评估微调后的模型性能。

学习要点：
    1. 困惑度 (Perplexity) 计算
    2. 生成质量评估
    3. 模型对比分析
"""

import os
import sys
from typing import List, Dict, Optional

import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from model import create_model, TinyLLM
from dataset import SimpleTokenizer, SAMPLE_DATA, apply_chatml_template
from utils import get_device, load_lora_weights


def compute_perplexity(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 256,
) -> float:
    """计算困惑度

    困惑度 = exp(平均交叉熵损失)
    困惑度越低，模型越好

    Args:
        model: 模型
        tokenizer: 分词器
        texts: 文本列表
        device: 设备
        max_length: 最大长度

    Returns:
        困惑度值
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            # 编码
            input_ids = tokenizer.encode(text)[:max_length]
            input_tensor = torch.tensor([input_ids], device=device)
            labels = input_tensor.clone()

            # 前向传播
            outputs = model(input_tensor, labels=labels)

            # 累计损失
            num_tokens = len(input_ids) - 1  # 减1因为是预测下一个token
            total_loss += outputs["loss"].item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def generate_samples(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> List[str]:
    """生成样本文本

    Args:
        model: 模型
        tokenizer: 分词器
        prompts: 提示文本列表
        device: 设备
        max_new_tokens: 最大生成token数
        temperature: 温度参数

    Returns:
        生成的文本列表
    """
    model.eval()
    generated = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device)

        # 自回归生成
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs["logits"][:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_tensor = torch.cat([input_tensor, next_token], dim=-1)

            # 检查是否生成了结束符
            if next_token.item() == tokenizer.eos_token_id:
                break

        # 解码
        output_ids = input_tensor[0].tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        generated.append(text)

    return generated


def evaluate_model(config: Optional[Config] = None, model_path: Optional[str] = None):
    """评估模型

    Args:
        config: 配置
        model_path: LoRA权重路径
    """
    if config is None:
        config = get_config()

    device = get_device()
    print(f"使用设备: {device}")

    # 创建分词器
    tokenizer = SimpleTokenizer(config.model.vocab_size)

    # 创建模型
    model = create_model(config.model, use_lora=True, lora_config=config.lora)

    # 加载LoRA权重
    if model_path is None:
        model_path = os.path.join(config.model_dir, "best_lora.pt")

    if os.path.exists(model_path):
        load_lora_weights(model, model_path)
    else:
        print(f"警告: 未找到权重文件 {model_path}，使用随机初始化的模型")

    model = model.to(device)

    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    # 1. 计算困惑度
    print("\n1. 困惑度评估")
    print("-" * 40)

    eval_texts = []
    for sample in SAMPLE_DATA[:4]:
        text = apply_chatml_template(sample["messages"], config.data)
        eval_texts.append(text)

    ppl = compute_perplexity(model, tokenizer, eval_texts, device)
    print(f"困惑度 (Perplexity): {ppl:.2f}")

    # 2. 生成样本
    print("\n2. 生成质量评估")
    print("-" * 40)

    test_prompts = [
        f"{config.data.im_start}system\n你是一个有帮助的AI助手。{config.data.im_end}\n"
        f"{config.data.im_start}user\n什么是Python？{config.data.im_end}\n"
        f"{config.data.im_start}assistant\n",
        f"{config.data.im_start}system\n你是一个有帮助的AI助手。{config.data.im_end}\n"
        f"{config.data.im_start}user\n解释深度学习。{config.data.im_end}\n"
        f"{config.data.im_start}assistant\n",
    ]

    generations = generate_samples(
        model, tokenizer, test_prompts, device, max_new_tokens=30, temperature=0.7
    )

    for i, (prompt, gen) in enumerate(zip(test_prompts, generations)):
        print(f"\n样本 {i + 1}:")
        print(f"输入: {prompt.split('user')[-1][:50]}...")
        print(f"输出: {gen[-50:]}")

    # 3. 参数统计
    print("\n3. 模型统计")
    print("-" * 40)

    trainable, total = model.get_trainable_parameters()
    print(f"总参数: {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"可训练比例: {100 * trainable / total:.2f}%")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--model", type=str, default=None, help="LoRA权重路径")
    args = parser.parse_args()

    evaluate_model(model_path=args.model)
