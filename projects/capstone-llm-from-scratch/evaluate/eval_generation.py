"""
eval_generation.py — 生成质量评估
==================================

不依赖 reference 答案，从模型多次采样生成结果，统计：
  - **Distinct-N**：n-gram 多样性，越接近 1.0 越好（避免输出重复）
  - **重复率**：生成中 trigram 重复占比，越低越好
  - **平均长度**：生成 token 数（截到 EOS 前）
  - **EOS 命中率**：生成是否在 max_new_tokens 内自然停止

测试 prompt 集合可指定为 jsonl 文件（每行 {"prompt":...} 或 {"conversations":[...]}），
默认用一组通用中文 prompts。

用法:
  python evaluate/eval_generation.py --model outputs/sft/final.pth
  python evaluate/eval_generation.py --model outputs/dpo/final.pth \\
      --prompts data/sft_t2t_mini.jsonl --num_prompts 50 --num_samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.training.trainer_utils import get_device, load_checkpoint
from src.inference.generate import generate_chat
from scripts.train import load_tokenizer


# ============================================================
# 多样性 / 重复率指标
# ============================================================


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(text: str, n: int = 2) -> float:
    """Distinct-N：唯一 n-gram 数 / 总 n-gram 数（字符级）"""
    tokens = list(text)
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def trigram_repetition_rate(text: str) -> float:
    """重复 trigram 占比（>1 出现的 trigram 占总数比例）"""
    tokens = list(text)
    grams = _ngrams(tokens, 3)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


# ============================================================
# Prompt 加载
# ============================================================


_DEFAULT_PROMPTS: list[str] = [
    "请简单介绍一下你自己。",
    "用一段话解释什么是机器学习。",
    "如何理解 Transformer 架构？",
    "推荐三本中文小说。",
    "写一首关于秋天的短诗。",
    "1+1 等于几？为什么？",
    "如果今天下雨，应该带什么出门？",
    "请用 Python 写一个判断质数的函数。",
    "比较一下猫和狗作为宠物的优缺点。",
    "解释什么是 Direct Preference Optimization。",
]


def _load_prompts(path: str | None, num: int, seed: int) -> list[str]:
    """从 jsonl 中加载 prompts，或返回默认列表"""
    if path is None or not os.path.exists(path):
        return _DEFAULT_PROMPTS[: max(1, min(num, len(_DEFAULT_PROMPTS)))]

    rng = random.Random(seed)
    samples: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "conversations" in obj and obj["conversations"]:
                for m in obj["conversations"]:
                    role = m.get("role") or m.get("from")
                    if role in ("user", "human"):
                        c = m.get("content") or m.get("value") or ""
                        if c:
                            samples.append(c)
                            break
            elif "prompt" in obj:
                samples.append(obj["prompt"])
            elif "instruction" in obj:
                samples.append(obj["instruction"])
    if not samples:
        return _DEFAULT_PROMPTS[:num]
    rng.shuffle(samples)
    return samples[:num]


# ============================================================
# 主流程
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearMind 生成质量评估")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--model", default=None, help="模型 checkpoint")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompts", default=None, help="prompts jsonl 文件路径")
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument(
        "--num_samples", type=int, default=2, help="每个 prompt 采样几次"
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show_samples", type=int, default=3, help="打印前 N 条生成样本"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = ModelConfig(**config["model"])

    tokenizer = load_tokenizer(config, args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 自动找模型
    model_path = args.model
    if model_path is None:
        for cand in (
            "outputs/dpo/final.pth",
            "outputs/sft/final.pth",
            "outputs/pretrain/final.pth",
        ):
            if os.path.exists(cand):
                model_path = cand
                break
    if model_path is None or not os.path.exists(model_path):
        print(f"❌ 找不到 checkpoint: {model_path}")
        return 1

    device = get_device()
    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    print("=" * 60)
    print("ClearMind 生成质量评估")
    print("=" * 60)
    print(f"📦 Model:           {model_path}")
    print(f"🎯 Prompts:         {args.num_prompts}")
    print(f"🎲 Samples/prompt:  {args.num_samples}")
    print(f"🌡️  Temperature:     {args.temperature}")
    print(f"📏 Max new tokens:  {args.max_new_tokens}")
    print()

    prompts = _load_prompts(args.prompts, args.num_prompts, args.seed)
    torch.manual_seed(args.seed)

    distinct_1_all: list[float] = []
    distinct_2_all: list[float] = []
    rep_rate_all: list[float] = []
    lengths: list[int] = []
    eos_hits = 0
    total_gen = 0
    sampled: list[tuple[str, str]] = []

    for prompt in prompts:
        for _ in range(args.num_samples):
            messages = [{"role": "user", "content": prompt}]
            reply = generate_chat(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            reply = (reply or "").strip()
            total_gen += 1

            distinct_1_all.append(distinct_n(reply, 1))
            distinct_2_all.append(distinct_n(reply, 2))
            rep_rate_all.append(trigram_repetition_rate(reply))
            lengths.append(len(reply))
            # EOS 命中估算（生成的字符数 < max_new_tokens × 2 可粗略认为遇到 EOS）
            if len(reply) < args.max_new_tokens * 2:
                eos_hits += 1

            sampled.append((prompt, reply))

    # ========== 汇总 ==========
    def avg(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    print("=" * 60)
    print("📊 汇总指标")
    print("=" * 60)
    print(f"  Distinct-1:        {avg(distinct_1_all):.4f}  (字符级唯一率，越高越好)")
    print(f"  Distinct-2:        {avg(distinct_2_all):.4f}  (二元 n-gram 唯一率)")
    print(f"  Trigram 重复率:     {avg(rep_rate_all):.4f}  (越低越好)")
    print(f"  平均生成长度:       {avg(lengths):.1f} 字符")
    print(
        f"  EOS 命中率:        {eos_hits}/{total_gen} = "
        f"{eos_hits / max(total_gen, 1) * 100:.1f}%"
    )
    print()

    if args.show_samples > 0:
        print("=" * 60)
        print(f"📝 生成样例（前 {min(args.show_samples, len(sampled))} 条）")
        print("=" * 60)
        for i, (p, r) in enumerate(sampled[: args.show_samples]):
            print(f"\n[#{i+1}] Q: {p}")
            print(f"     A: {r[:200]}{'...' if len(r) > 200 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
