"""
06_chat.py — 对话入口脚本
=========================

加载已训练的模型，启动交互式对话界面。

使用方法:
  # 使用 DPO 模型 (最佳)
  python scripts/06_chat.py

  # 使用 SFT 模型
  python scripts/06_chat.py --model outputs/sft/final.pth

  # 使用预训练模型 (只会续写, 不会对话)
  python scripts/06_chat.py --model outputs/pretrain/final.pth
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer
from src.inference.chat import chat_loop
from src.training.trainer_utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="ClearMind 对话")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/dpo/final.pth",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=300)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])

    # 检查文件
    if not os.path.exists(args.tokenizer):
        print(f"❌ 分词器不存在: {args.tokenizer}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}")
        # 自动查找可用模型
        for path in [
            "outputs/dpo/final.pth",
            "outputs/sft/final.pth",
            "outputs/pretrain/final.pth",
        ]:
            if os.path.exists(path):
                print(f"   找到: {path}")
                args.model = path
                break
        else:
            print("   请先完成训练流程")
            sys.exit(1)

    # 加载分词器
    tokenizer = ClearMindTokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 加载模型
    device = get_device()
    model = GPT(model_config).to(device)
    load_checkpoint(model, args.model, device=device)

    print(f"\n📋 模型信息:")
    print(f"  模型: {args.model}")
    print(f"  参数量: {model.count_parameters()['total_millions']:.1f}M")
    print(f"  设备: {device}")

    # 启动对话
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
