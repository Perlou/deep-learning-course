"""
推理脚本
========

实现模型推理和交互式对话。

学习要点：
    1. 自回归文本生成
    2. Top-k / Top-p 采样策略
    3. KV Cache 加速推理
"""

import os
import sys
from typing import Optional, List

import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from model import create_model, TinyLLM
from dataset import SimpleTokenizer
from utils import get_device, load_lora_weights


def top_k_top_p_filtering(
    logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0
) -> torch.Tensor:
    """应用 Top-k 和 Top-p (nucleus) 采样过滤

    Args:
        logits: 模型输出的logits (batch_size, vocab_size)
        top_k: 保留概率最高的k个token
        top_p: 保留累积概率达到p的token
        temperature: 温度参数，控制随机性

    Returns:
        过滤后的logits
    """
    # 应用温度
    logits = logits / temperature

    # Top-k 过滤
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) 过滤
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    return logits


def generate(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> str:
    """生成文本

    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_k: Top-k采样参数
        top_p: Top-p采样参数
        device: 计算设备
        use_cache: 是否使用KV Cache

    Returns:
        生成的完整文本
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # 编码输入
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)

    # KV Cache
    past_key_values = None

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 使用cache时只传入最后一个token
            if use_cache and past_key_values is not None:
                curr_input = input_tensor[:, -1:]
            else:
                curr_input = input_tensor

            # 前向传播
            outputs = model(
                curr_input, use_cache=use_cache, past_key_values=past_key_values
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs.get("past_key_values")

            # 采样
            filtered_logits = top_k_top_p_filtering(
                logits, top_k=top_k, top_p=top_p, temperature=temperature
            )
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 检查结束符
            if next_token.item() == tokenizer.eos_token_id:
                break

            # 更新输入
            generated_tokens.append(next_token.item())
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)

    # 解码
    all_tokens = input_ids + generated_tokens
    generated_text = tokenizer.decode(all_tokens, skip_special_tokens=False)

    return generated_text


def chat(
    model: TinyLLM, tokenizer: SimpleTokenizer, config: Config, device: torch.device
):
    """交互式对话

    Args:
        model: 模型
        tokenizer: 分词器
        config: 配置
        device: 设备
    """
    print("\n" + "=" * 60)
    print("交互式对话 (输入 'quit' 退出)")
    print("=" * 60)

    system_prompt = config.data.default_system_prompt
    history = []

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == "clear":
            history = []
            print("对话历史已清除")
            continue

        # 构建对话
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        # 应用模板
        prompt = ""
        for msg in messages:
            prompt += f"{config.data.im_start}{msg['role']}\n{msg['content']}{config.data.im_end}\n"
        prompt += f"{config.data.im_start}assistant\n"

        # 生成回复
        response = generate(
            model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, device=device
        )

        # 提取助手回复
        assistant_marker = f"{config.data.im_start}assistant\n"
        if assistant_marker in response:
            reply = response.split(assistant_marker)[-1]
            if config.data.im_end in reply:
                reply = reply.split(config.data.im_end)[0]
        else:
            reply = response[len(prompt) :]

        reply = reply.strip()
        print(f"AI: {reply}")

        # 更新历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

        # 限制历史长度
        if len(history) > 10:
            history = history[-10:]


def main(model_path: Optional[str] = None):
    """推理主函数"""
    config = get_config()
    device = get_device()
    print(f"使用设备: {device}")

    # 创建分词器
    tokenizer = SimpleTokenizer(config.model.vocab_size)

    # 创建模型
    model = create_model(config.model, use_lora=True, lora_config=config.lora)

    # 加载权重
    if model_path is None:
        model_path = os.path.join(config.model_dir, "best_lora.pt")

    if os.path.exists(model_path):
        load_lora_weights(model, model_path)
    else:
        print(f"警告: 未找到权重文件 {model_path}，使用随机初始化的模型")

    model = model.to(device)

    # 启动对话
    chat(model, tokenizer, config, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument("--model", type=str, default=None, help="LoRA权重路径")
    args = parser.parse_args()

    main(model_path=args.model)
