"""
push_to_hub.py — 推送模型和数据集到 HuggingFace Hub
=====================================================

将训练好的模型、Tokenizer、数据集推送到 HuggingFace Hub。

from-scratch 对比:
  - from-scratch: 无此功能，需要手动上传
  - HF 版: model.push_to_hub() / dataset.push_to_hub() 一行完成

前置条件:
  1. pip install huggingface_hub
  2. huggingface-cli login (需要 HF token)

用法:
  # 推送模型
  python scripts/push_to_hub.py --model outputs/sft --repo your-username/clearmind-sft

  # 推送模型 + 数据集
  python scripts/push_to_hub.py --model outputs/sft --repo your-username/clearmind-sft --push_data

  # 仅生成 Model Card (不推送)
  python scripts/push_to_hub.py --model outputs/sft --repo your-username/clearmind-sft --dry_run
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer


# ============================================================
# Model Card 模板
# ============================================================

MODEL_CARD_TEMPLATE = """---
language:
  - zh
  - en
tags:
  - text-generation
  - causal-lm
  - from-scratch
  - educational
license: apache-2.0
---

# ClearMind — HuggingFace 生态大语言模型

> 使用 HF 全家桶从零训练的教育项目

## 模型描述

ClearMind 是一个教育项目，使用 HuggingFace 生态从零实现 Tokenizer → Pre-training → SFT → DPO 全流程。

**架构特点:** RoPE + RMSNorm + SwiGLU + GQA + KV Cache

## 模型配置

| 参数 | 值 |
|------|-----|
| hidden_size | {hidden_size} |
| num_attention_heads | {num_attention_heads} |
| num_hidden_layers | {num_hidden_layers} |
| vocab_size | {vocab_size} |
| max_position_embeddings | {max_position_embeddings} |
| 总参数量 | {param_count} |

## 训练阶段

| 阶段 | 说明 |
|------|------|
| Pretrain | 在通用文本上学习语言建模 |
| SFT | 指令微调，学会按指令回答 |
| DPO | 偏好对齐，提升回复质量和安全性 |

## 使用方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("Human: 你好\\nAssistant: ", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 免责声明

ClearMind 是一个教育项目的小型模型，回复质量有限，仅用于学习和演示目的。
"""


def generate_model_card(model, repo_id: str) -> str:
    """生成 Model Card

    Args:
        model: ClearMindForCausalLM
        repo_id: HF Hub repo ID

    Returns:
        Model Card 内容 (Markdown)
    """
    config = model.config
    param_count = sum(p.numel() for p in model.parameters())

    if param_count >= 1e9:
        param_str = f"{param_count / 1e9:.1f}B"
    else:
        param_str = f"{param_count / 1e6:.1f}M"

    return MODEL_CARD_TEMPLATE.format(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        param_count=param_str,
        repo_id=repo_id,
    )


def push_model(model_path: str, tokenizer_path: str, repo_id: str, dry_run: bool = False):
    """推送模型和 Tokenizer 到 Hub

    Args:
        model_path:    HF 格式模型目录
        tokenizer_path: tokenizer 目录
        repo_id:       Hub repo ID (e.g. "username/clearmind-sft")
        dry_run:       仅生成 Model Card，不推送
    """
    print(f"加载模型: {model_path}")
    model = ClearMindForCausalLM.from_pretrained(model_path)

    print(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = ClearMindTokenizer.load(tokenizer_path)

    # 生成 Model Card
    card = generate_model_card(model, repo_id)

    if dry_run:
        print(f"\n--- Model Card ---\n{card}\n--- End ---")
        print(f"\nDry run 模式，未推送到 Hub")
        return

    # 保存 Model Card 到模型目录
    card_path = Path(model_path) / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Model Card 已保存: {card_path}")

    # 推送
    print(f"\n推送模型到: {repo_id}")
    model.push_to_hub(repo_id)

    print(f"推送 tokenizer 到: {repo_id}")
    tokenizer.push_to_hub(repo_id)

    print(f"\n模型已推送到: https://huggingface.co/{repo_id}")


def push_dataset(data_dir: str, repo_id: str):
    """推送数据集到 Hub

    Args:
        data_dir: 数据目录 (含 pretrain/sft/dpo 子目录)
        repo_id:  Hub repo ID
    """
    from datasets import load_dataset, DatasetDict

    datasets = {}

    # 加载各阶段数据
    for stage in ["pretrain", "sft", "dpo"]:
        data_path = Path(data_dir) / stage / f"{stage}_data.jsonl"
        if data_path.exists():
            ds = load_dataset("json", data_files=str(data_path), split="train")
            datasets[stage] = ds
            print(f"加载 {stage}: {len(ds)} 条")

    if not datasets:
        print("未找到数据文件")
        return

    dataset_dict = DatasetDict(datasets)

    print(f"\n推送数据集到: {repo_id}")
    dataset_dict.push_to_hub(repo_id)
    print(f"数据集已推送到: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="推送到 HuggingFace Hub")
    parser.add_argument("--model", type=str, help="HF 格式模型目录")
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizer")
    parser.add_argument("--repo", type=str, required=True, help="Hub repo ID")
    parser.add_argument("--data_dir", type=str, default="data/", help="数据目录")
    parser.add_argument("--push_data", action="store_true", help="同时推送数据集")
    parser.add_argument("--data_repo", type=str, help="数据集 repo ID (默认: {repo}-data)")
    parser.add_argument("--dry_run", action="store_true", help="仅生成 Model Card，不推送")
    args = parser.parse_args()

    # 自动查找模型
    if not args.model:
        for path in ["outputs/dpo", "outputs/sft", "outputs/pretrain"]:
            if os.path.exists(path):
                args.model = path
                break
        else:
            print("未找到模型目录，请使用 --model 指定")
            sys.exit(1)

    # 推送模型
    push_model(args.model, args.tokenizer, args.repo, dry_run=args.dry_run)

    # 推送数据集
    if args.push_data and not args.dry_run:
        data_repo = args.data_repo or f"{args.repo}-data"
        push_dataset(args.data_dir, data_repo)


if __name__ == "__main__":
    main()
