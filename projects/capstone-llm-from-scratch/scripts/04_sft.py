"""
04_sft.py — SFT 指令微调入口脚本
=================================

在预训练模型上进行指令微调,让模型学会"按指令回答"。

使用方法:
  # 使用默认配置
  python scripts/04_sft.py

  # 指定预训练模型
  python scripts/04_sft.py --pretrained outputs/pretrain/final.pth

  # 快速验证
  python scripts/04_sft.py --epochs 1 --log_every 5

前置步骤:
  1. python scripts/01_prepare_data.py
  2. python scripts/02_train_tokenizer.py
  3. python scripts/03_pretrain.py
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
from src.data.sft_dataset import SFTDataset
from src.training.sft import SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="ClearMind SFT 指令微调")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--data", type=str, default="data/sft/sft_data.jsonl")
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="outputs/pretrain/final.pth",
        help="预训练模型路径",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])
    sft_config = config["sft"]
    if args.epochs:
        sft_config["epochs"] = args.epochs
    sft_config["log_every"] = args.log_every

    print("=" * 60)
    print("ClearMind SFT 指令微调")
    print("=" * 60)

    # 检查前置
    if not os.path.exists(args.tokenizer):
        print(f"\n❌ 分词器不存在: {args.tokenizer}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"\n❌ SFT 数据不存在: {args.data}")
        sys.exit(1)

    # 加载分词器
    tokenizer = ClearMindTokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 数据集
    train_dataset = SFTDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
    )

    # 模型
    model = GPT(model_config)
    params = model.count_parameters()
    print(f"🧠 参数量: {params['total_millions']:.1f}M")

    # 训练
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        config=sft_config,
        output_dir=args.output_dir,
    )

    trainer.train(pretrained_path=args.pretrained)

    print(f"\n💡 下一步: python scripts/05_dpo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
