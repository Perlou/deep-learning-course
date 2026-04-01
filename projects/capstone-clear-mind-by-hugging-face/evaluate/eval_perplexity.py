"""
eval_perplexity.py — 困惑度评估 (HF 版)
=========================================

使用 HuggingFace from_pretrained + datasets 计算困惑度 (Perplexity)。

from-scratch 对比:
  - from-scratch: GPT + load_checkpoint + PretrainDataset + DataLoader → 手写 loop
  - HF 版: from_pretrained + datasets.map + DataCollator → HF Trainer.evaluate()

困惑度 (PPL) 是评估语言模型质量的标准指标:
  PPL = exp(average negative log-likelihood)
  PPL 越低 → 模型越好 (对文本的"困惑"越少)

典型值参考:
  - 随机模型 (vocab=2000): PPL ≈ 2000
  - 预训练后: PPL ≈ 50-200
  - SFT 后: PPL ≈ 10-50

使用方法:
  python evaluate/eval_perplexity.py --model outputs/pretrain
  python evaluate/eval_perplexity.py --compare
"""

import os
import sys
import argparse
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from model import ClearMindConfig, ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from data.data_utils import load_pretrain_dataset


def evaluate_perplexity(
    model,
    eval_dataset,
    tokenizer,
    batch_size: int = 8,
    max_batches: int | None = None,
) -> float:
    """计算模型在数据集上的困惑度

    from-scratch 对比:
      - from-scratch: 手写 DataLoader + forward loop + loss 累加
      - HF 版: 同样手写 loop，但使用 DataCollatorForLanguageModeling 自动处理

    Args:
        model:        ClearMindForCausalLM 实例
        eval_dataset: tokenized 的评估数据集 (datasets.Dataset)
        tokenizer:    tokenizer 实例
        batch_size:   批大小
        max_batches:  最大评估批数 (None = 全部)

    Returns:
        困惑度 (PPL)
    """
    model.eval()
    device = next(model.parameters()).device

    # DataCollator: 自动生成 labels (对比 from-scratch 手写 labels = input_ids)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        drop_last=False,
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # HF 模型 forward: 返回 CausalLMOutput (含 loss)
            # 对比 from-scratch: logits, loss, _ = model(input_ids, labels)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # 统计有效 token 数 (labels != -100)
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def load_model(model_path: str) -> tuple:
    """加载 HF 格式模型

    from-scratch 对比:
      - from-scratch: GPT(config) + load_checkpoint(model, path) 两步
      - HF 版: from_pretrained(path) 一步完成

    Args:
        model_path: HF 格式模型目录

    Returns:
        (model, config) 元组
    """
    model = ClearMindForCausalLM.from_pretrained(model_path)
    model.eval()
    return model, model.config


def main():
    parser = argparse.ArgumentParser(description="ClearMind 困惑度评估 (HF 版)")
    parser.add_argument("--model", type=str, help="HF 格式模型目录路径")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml",
                        help="YAML 配置文件 (用于新建模型时)")
    parser.add_argument("--data", type=str, default="data/pretrain/pretrain_data.jsonl")
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizer")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--compare", action="store_true",
                        help="对比 pretrain/sft/dpo 三阶段的困惑度")
    args = parser.parse_args()

    print("=" * 60)
    print("ClearMind 困惑度评估 (HF 版)")
    print("=" * 60)

    # 加载 tokenizer
    tokenizer = ClearMindTokenizer.load(args.tokenizer)

    if args.compare:
        # === 多阶段对比模式 ===
        stages = {
            "pretrain": "outputs/pretrain",
            "sft": "outputs/sft",
            "dpo": "outputs/dpo",
        }

        results = {}
        for stage_name, model_path in stages.items():
            if not os.path.exists(model_path):
                print(f"\n  跳过 {stage_name}: {model_path} 不存在")
                continue

            print(f"\n  评估 {stage_name}...")
            model, config = load_model(model_path)

            # 加载评估数据
            max_length = config.max_position_embeddings
            datasets = load_pretrain_dataset(args.data, tokenizer, max_length=max_length)
            eval_dataset = datasets["validation"]

            ppl = evaluate_perplexity(
                model=model,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
            )

            results[stage_name] = ppl
            print(f"  {stage_name} PPL = {ppl:.2f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 打印对比表格
        if results:
            print(f"\n{'=' * 50}")
            print(f"  困惑度对比")
            print(f"{'=' * 50}")
            print(f"  {'阶段':^10} | {'PPL':^12} | {'说明':^15}")
            print(f"  {'-' * 45}")
            for stage, ppl in results.items():
                note = {"pretrain": "语言建模", "sft": "指令微调", "dpo": "偏好对齐"}.get(stage, "")
                print(f"  {stage:^10} | {ppl:^12.2f} | {note:^15}")
            print(f"  {'-' * 45}")
            print(f"  (PPL 越低越好)")
            print(f"{'=' * 50}")

    else:
        # === 单模型评估模式 ===
        if not args.model:
            for path in ["outputs/dpo", "outputs/sft", "outputs/pretrain"]:
                if os.path.exists(path):
                    args.model = path
                    break
            else:
                print("未找到任何模型目录，请使用 --model 指定或先完成训练")
                sys.exit(1)

        model, config = load_model(args.model)

        max_length = config.max_position_embeddings
        datasets = load_pretrain_dataset(args.data, tokenizer, max_length=max_length)
        eval_dataset = datasets["validation"]

        print(f"\n  评估中...")
        ppl = evaluate_perplexity(
            model=model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )

        print(f"\n  结果:")
        print(f"  模型: {args.model}")
        print(f"  困惑度 (PPL): {ppl:.2f}")
        print(f"  (PPL 越低越好, 随机模型 ≈ {config.vocab_size})")
        print("=" * 60)


if __name__ == "__main__":
    main()
