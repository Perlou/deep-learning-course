"""
eval_perplexity.py — 困惑度评估
================================

计算模型在验证集上的困惑度 (Perplexity)。

困惑度 (PPL) 是评估语言模型质量的标准指标:
  PPL = exp(average negative log-likelihood)
  PPL 越低 → 模型越好 (对文本的"困惑"越少)

典型值参考:
  - 随机模型 (vocab=8000): PPL ≈ 8000
  - 未训练: PPL ≈ 8000
  - 预训练后: PPL ≈ 50-200 (取决于数据和模型大小)
  - SFT 后 (在任务数据上): PPL ≈ 10-50

使用方法:
  python evaluate/eval_perplexity.py --model outputs/pretrain/final.pth
  python evaluate/eval_perplexity.py --model outputs/sft/final.pth
"""

import os
import sys
import argparse
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

import yaml
from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer
from src.data.pretrain_dataset import PretrainDataset
from src.training.trainer_utils import get_device, load_checkpoint


def evaluate_perplexity(
    model,
    dataset,
    batch_size: int = 8,
    device: torch.device = None,
    max_batches: int = None,
) -> float:
    """计算模型困惑度

    Args:
        model:       模型
        dataset:     数据集
        batch_size:  批大小
        device:      设备
        max_batches: 最大评估批数 (None = 全部)

    Returns:
        困惑度 (PPL)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, labels)

            # 统计有效 token 数
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def main():
    parser = argparse.ArgumentParser(description="ClearMind 困惑度评估")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--model", type=str, help="模型 checkpoint 路径")
    parser.add_argument("--data", type=str, default="data/pretrain/pretrain_data.jsonl")
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument(
        "--compare", action="store_true", help="对比 pretrain/sft/dpo 三阶段的困惑度"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])

    print("=" * 60)
    print("ClearMind 困惑度评估")
    print("=" * 60)

    # 加载分词器
    tokenizer = ClearMindTokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 加载数据
    dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
    )

    device = get_device()

    if args.compare:
        # === 多阶段对比模式 ===
        stages = {
            "pretrain": "outputs/pretrain/final.pth",
            "sft": "outputs/sft/final.pth",
            "dpo": "outputs/dpo/final.pth",
        }

        results = {}
        for stage_name, model_path in stages.items():
            if not os.path.exists(model_path):
                print(f"\n⚠️  跳过 {stage_name}: {model_path} 不存在")
                continue

            print(f"\n🔍 评估 {stage_name}...")
            model = GPT(model_config).to(device)
            load_checkpoint(model, model_path, device=device)

            ppl = evaluate_perplexity(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
                device=device,
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
            print(f"📊 困惑度对比")
            print(f"{'=' * 50}")
            print(f"  {'阶段':^10} │ {'PPL':^12} │ {'说明':^15}")
            print(f"  {'─' * 45}")
            print(
                f"  {'随机基线':^10} │ {model_config.vocab_size:^12} │ {'未训练':^15}"
            )
            for stage, ppl in results.items():
                note = {
                    "pretrain": "语言建模",
                    "sft": "指令微调",
                    "dpo": "偏好对齐",
                }.get(stage, "")
                print(f"  {stage:^10} │ {ppl:^12.2f} │ {note:^15}")
            print(f"  {'─' * 45}")
            print(f"  (PPL 越低越好)")
            print(f"{'=' * 50}")

    else:
        # === 单模型评估模式 ===
        if not args.model:
            for path in [
                "outputs/dpo/final.pth",
                "outputs/sft/final.pth",
                "outputs/pretrain/final.pth",
            ]:
                if os.path.exists(path):
                    args.model = path
                    break
            else:
                print("❌ 未找到任何 checkpoint，请使用 --model 指定或先完成训练")
                sys.exit(1)

        # 加载模型
        model = GPT(model_config).to(device)
        load_checkpoint(model, args.model, device=device)

        # 评估
        print(f"\n🔍 评估中...")
        ppl = evaluate_perplexity(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            max_batches=args.max_batches,
        )

        print(f"\n📊 结果:")
        print(f"  模型: {args.model}")
        print(f"  困惑度 (PPL): {ppl:.2f}")
        print(f"  (PPL 越低越好, 随机模型 ≈ {model_config.vocab_size})")
        print("=" * 60)


if __name__ == "__main__":
    main()
