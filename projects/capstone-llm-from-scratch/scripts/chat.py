"""
chat.py — 对话入口脚本
========================

加载已训练的模型，基于 minimind chat_template 启动多轮对话。

使用方法:

  # 默认加载 main.yaml + outputs/dpo/final.pth
  python scripts/chat.py

  # 指定模型与配置
  python scripts/chat.py --config configs/plus.yaml --model outputs/sft/final.pth

  # 启用 open_thinking（让模型先在 <think> 中推理）
  python scripts/chat.py --thinking

  # 自定义 system prompt
  python scripts/chat.py --system "你是一个严谨的数学助手"
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

# 复用 train.py 的 tokenizer 加载分支（hf / sentencepiece 自动选择）
from scripts.train import load_tokenizer


def _candidate_checkpoints() -> list[str]:
    """按 DPO > SFT > Pretrain 顺序的 fallback 候选"""
    return [
        "outputs/dpo/final.pth",
        "outputs/sft/final.pth",
        "outputs/pretrain/final.pth",
    ]


def main():
    parser = argparse.ArgumentParser(description="ClearMind 对话")
    parser.add_argument(
        "--config", type=str, default="configs/main.yaml", help="模型配置文件"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型 checkpoint 路径（默认按 dpo→sft→pretrain 顺序自动查找）",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="分词器路径（默认读 yaml.tokenizer.path）",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_history", type=int, default=8)
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="可选 system prompt（None 表示不注入）",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="启用 open_thinking 模式（模型先 <think>...</think> 推理后再答）",
    )
    args = parser.parse_args()

    # ========== 加载配置 + tokenizer + 模型 ==========
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 延迟导入，避免在 --help 时强行依赖 torch
    import torch
    from src.model.config import ModelConfig
    from src.model.gpt import GPT
    from src.inference.chat import chat_loop
    from src.training.trainer_utils import get_device, load_checkpoint

    model_config = ModelConfig(**config["model"])

    tokenizer = load_tokenizer(config, args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        print(
            f"   ⚠️  tokenizer.vocab_size={tokenizer.vocab_size} 与 "
            f"model.vocab_size={model_config.vocab_size} 不一致，自动对齐到 tokenizer"
        )
        model_config.vocab_size = tokenizer.vocab_size

    # 解析模型路径
    model_path = args.model
    if model_path is None:
        for candidate in _candidate_checkpoints():
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            print("❌ 没找到任何已训练模型。请先运行：")
            print("     python scripts/train.py --stage pretrain --config", args.config)
            print("     python scripts/train.py --stage sft      --config", args.config)
            sys.exit(1)
    elif not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    # ========== 加载模型 ==========
    device = get_device()
    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)

    print("\n📋 模型信息:")
    print(f"  Config:    {args.config}")
    print(f"  Checkpoint: {model_path}")
    print(f"  参数量:    {model.count_parameters()['total_millions']:.1f}M")
    print(f"  Device:    {device}")
    print(f"  Tokenizer: {tokenizer}")
    if args.system:
        print(f"  System:    {args.system!r}")
    if args.thinking:
        print("  Thinking:  ON")

    # ========== 启动对话 ==========
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_history=args.max_history,
        system_prompt=args.system,
        open_thinking=args.thinking,
    )


if __name__ == "__main__":
    main()
