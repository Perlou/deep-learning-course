"""
train.py — 统一训练入口 (HF Trainer 版)
========================================

对应 from-scratch 版的 scripts/train.py，
通过 --stage 参数分发到不同的训练阶段。

用法:
  python scripts/train.py --stage pretrain --config configs/tiny.yaml
  python scripts/train.py --stage sft --config configs/tiny.yaml
  python scripts/train.py --stage dpo --config configs/tiny.yaml

from-scratch 对比:
  - from-scratch: load_setup() + run_pretrain/sft/dpo()
  - HF 版: 直接调用各阶段的 run_xxx() 函数
"""

import argparse
import sys
from pathlib import Path

# 将 src/ 加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="ClearMind 统一训练入口")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pretrain", "sft", "dpo"],
        help="训练阶段: pretrain / sft / dpo",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="数据路径 (默认按 stage 自动选择)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="outputs/tokenizer",
        help="tokenizer 目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录 (默认: outputs/<stage>)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="覆盖 config 中的 max_steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="使用 LoRA 微调 (SFT/DPO 用)",
    )
    return parser.parse_args()


def get_default_data_path(stage: str) -> str:
    """根据 stage 返回默认数据路径"""
    defaults = {
        "pretrain": "data/pretrain/pretrain_data.jsonl",
        "sft": "data/sft/sft_data.jsonl",
        "dpo": "data/dpo/dpo_data.jsonl",
    }
    return defaults[stage]


def main():
    args = parse_args()

    # 默认值
    if args.data is None:
        args.data = get_default_data_path(args.stage)
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.stage}"

    print(f"\n{'=' * 60}")
    print(f"ClearMind 训练 — 阶段: {args.stage}")
    print(f"{'=' * 60}")
    print(f"  配置: {args.config}")
    print(f"  数据: {args.data}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  输出: {args.output_dir}")
    if args.max_steps is not None:
        print(f"  max_steps: {args.max_steps}")
    print()

    if args.stage == "pretrain":
        from training.pretrain import run_pretrain

        run_pretrain(
            config_path=args.config,
            data_path=args.data,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume,
        )
    elif args.stage == "sft":
        from training.sft import run_sft

        run_sft(
            config_path=args.config,
            data_path=args.data,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            pretrained_path=args.resume,
            max_steps=args.max_steps,
            use_lora=args.use_lora,
        )
    elif args.stage == "dpo":
        from training.dpo import run_dpo

        run_dpo(
            config_path=args.config,
            data_path=args.data,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            sft_model_path=args.resume,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
