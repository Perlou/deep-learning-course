"""
eval_perplexity.py — 困惑度评估
================================

计算模型在验证集上的困惑度 (PPL)。

PPL = exp(mean(cross_entropy_per_token))
PPL 越低 → 模型越好；随机模型 PPL ≈ vocab_size。

典型值参考（minimind tokenizer，vocab=6400）：
  - 未训练 / 随机基线: PPL ≈ 6400
  - Pretrain 后:        PPL ≈ 30-200
  - SFT 后:             PPL ≈ 5-30
  - DPO 后:             与 SFT 接近（DPO 不强行追求降 PPL）

用法:
  # 单模型
  python evaluate/eval_perplexity.py --model outputs/sft/final.pth

  # 三阶段对比
  python evaluate/eval_perplexity.py --compare

  # 自定义验证集与 batch
  python evaluate/eval_perplexity.py --model outputs/dpo/final.pth \\
      --data data/pretrain_t2t_mini.jsonl --batch_size 16 --max_batches 200
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.pretrain_dataset import PretrainDataset
from src.training.trainer_utils import get_device, load_checkpoint

# 复用 train.py 的 tokenizer 加载
from scripts.train import load_tokenizer


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 8,
    device: torch.device = None,
    max_batches: int | None = None,
) -> float:
    """计算模型困惑度

    Args:
        model:       模型（已加载权重）
        dataset:     PretrainDataset 实例
        batch_size:  评估 batch 大小
        device:      计算设备
        max_batches: 最多评估多少 batch（None = 全量）

    Returns:
        困惑度（float）
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss, _ = model(input_ids, labels)

        # 用有效 token 数加权平均（per_sample 模式下 padding 位置 labels=-100）
        n_tokens = (labels != -100).sum().item()
        if n_tokens > 0:
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def _load_model(model_path: str, model_config: ModelConfig, device: torch.device) -> GPT:
    """加载纯权重或 _resume.pth"""
    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearMind 困惑度评估")
    parser.add_argument(
        "--config", type=str, default="configs/main.yaml", help="模型配置（决定架构）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型 checkpoint 路径（默认按 dpo→sft→pretrain 顺序自动查）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/pretrain_t2t_mini.jsonl",
        help="验证数据路径",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="可选 tokenizer 路径（默认读 yaml.tokenizer.path）",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=["packed", "per_sample"],
        default="per_sample",
        help="数据集模式（与训练时保持一致更准）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比 pretrain/sft/dpo 三阶段 PPL",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = ModelConfig(**config["model"])

    print("=" * 60)
    print("ClearMind 困惑度评估")
    print("=" * 60)
    print(f"📄 Config:    {args.config}")
    print(f"📦 Data:      {args.data}")
    print(f"🎯 Batch:     {args.batch_size}, max_batches={args.max_batches}")

    if not os.path.exists(args.data):
        print(f"\n❌ 验证数据不存在: {args.data}")
        print("   请先 python scripts/download_data.py --profile zero")
        return 1

    tokenizer = load_tokenizer(config, args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 数据集（一次构建，三阶段共享）
    dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        mode=args.mode,
    )

    device = get_device()
    print(f"🖥️  Device:    {device}")

    if args.compare:
        # ===== 三阶段对比 =====
        stages = {
            "pretrain": "outputs/pretrain/final.pth",
            "sft": "outputs/sft/final.pth",
            "dpo": "outputs/dpo/final.pth",
        }
        results: dict[str, float] = {}
        for stage_name, model_path in stages.items():
            if not os.path.exists(model_path):
                print(f"\n⚠️  跳过 {stage_name}: {model_path} 不存在")
                continue
            print(f"\n🔍 评估 {stage_name} ...")
            model = _load_model(model_path, model_config, device)
            ppl = evaluate_perplexity(
                model, dataset, args.batch_size, device, args.max_batches
            )
            results[stage_name] = ppl
            print(f"  {stage_name} PPL = {ppl:.2f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if results:
            print(f"\n{'=' * 60}")
            print("📊 困惑度对比")
            print(f"{'=' * 60}")
            print(f"  {'阶段':^10} │ {'PPL':^12} │ {'说明':^15}")
            print("  " + "─" * 45)
            print(f"  {'随机基线':^10} │ {model_config.vocab_size:^12} │ {'未训练':^15}")
            for stage, ppl in results.items():
                note = {
                    "pretrain": "语言建模",
                    "sft": "指令微调",
                    "dpo": "偏好对齐",
                }.get(stage, "")
                print(f"  {stage:^10} │ {ppl:^12.2f} │ {note:^15}")
            print("  " + "─" * 45)
            print("  (PPL 越低越好)")
            print(f"{'=' * 60}")
        return 0

    # ===== 单模型 =====
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
        if model_path is None:
            print("❌ 未找到任何 checkpoint，请用 --model 指定或先训练")
            return 1
    elif not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return 1

    print(f"📦 Model:     {model_path}")
    model = _load_model(model_path, model_config, device)
    print("\n🔍 评估中 ...")
    ppl = evaluate_perplexity(model, dataset, args.batch_size, device, args.max_batches)

    print("\n📊 结果:")
    print(f"  Model: {model_path}")
    print(f"  PPL:   {ppl:.2f}")
    print(f"  (随机基线 ≈ {model_config.vocab_size}，越低越好)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
