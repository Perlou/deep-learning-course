"""
05_dpo.py — DPO 对齐训练入口脚本
=================================

在 SFT 模型上进行 DPO 偏好对齐训练。

使用方法:
  python scripts/05_dpo.py
  python scripts/05_dpo.py --sft_model outputs/sft/final.pth

前置步骤:
  1~3. 数据准备 + 分词器 + 预训练
  4.   python scripts/04_sft.py
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
from src.data.dpo_dataset import DPODataset
from src.training.dpo import DPOTrainer


def main():
    parser = argparse.ArgumentParser(description="ClearMind DPO 对齐训练")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--data", type=str, default="data/dpo/dpo_data.jsonl")
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument(
        "--sft_model", type=str, default="outputs/sft/final.pth", help="SFT 模型路径"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/dpo")
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])
    dpo_config = config["dpo"]
    dpo_config["log_every"] = args.log_every

    print("=" * 60)
    print("ClearMind DPO 对齐训练")
    print("=" * 60)

    # 检查前置
    if not os.path.exists(args.tokenizer):
        print(f"\n❌ 分词器不存在: {args.tokenizer}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"\n❌ DPO 数据不存在: {args.data}")
        sys.exit(1)

    # 加载分词器
    tokenizer = ClearMindTokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 数据集
    train_dataset = DPODataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
    )

    # 模型
    model = GPT(model_config)
    params = model.count_parameters()
    print(f"🧠 参数量: {params['total_millions']:.1f}M")

    # 训练
    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        config=dpo_config,
        output_dir=args.output_dir,
    )

    trainer.train(sft_path=args.sft_model)

    print(f"\n💡 下一步: python scripts/06_chat.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
