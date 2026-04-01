"""
chat.py — CLI 交互式对话 (HF 版)
==================================

对应 from-scratch 版的 scripts/chat.py，
使用 model.generate() 替代手写生成循环。

用法:
  python scripts/chat.py --model outputs/sft
  python scripts/chat.py --model outputs/dpo --max_tokens 300
"""

import os
import sys
import argparse
from pathlib import Path

# 将 src/ 加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from inference.chat import chat_loop


def main():
    parser = argparse.ArgumentParser(description="ClearMind CLI 对话 (HF 版)")
    parser.add_argument("--model", type=str, help="HF 格式模型目录")
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizer")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_history", type=int, default=5)
    args = parser.parse_args()

    # 自动查找模型
    if not args.model:
        for path in ["outputs/dpo", "outputs/sft", "outputs/pretrain"]:
            if os.path.exists(path):
                args.model = path
                break
        else:
            print("未找到任何模型目录，请使用 --model 指定或先完成训练")
            sys.exit(1)

    print(f"加载模型: {args.model}")
    model = ClearMindForCausalLM.from_pretrained(args.model)
    model.eval()

    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = ClearMindTokenizer.load(args.tokenizer)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count / 1e6:.1f}M")

    chat_loop(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_history=args.max_history,
    )


if __name__ == "__main__":
    main()
