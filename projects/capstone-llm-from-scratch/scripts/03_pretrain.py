"""
03_pretrain.py — 预训练入口脚本
================================

启动 GPT 模型的预训练 (Next-token Prediction)。

使用方法:
  # 使用默认 Small 配置
  python scripts/03_pretrain.py

  # 指定配置和参数
  python scripts/03_pretrain.py --config configs/small.yaml --max_steps 1000

  # 断点续训
  python scripts/03_pretrain.py --resume outputs/pretrain/checkpoint_step1000.pth

  # 快速验证 (100 步)
  python scripts/03_pretrain.py --max_steps 100 --log_every 10

前置步骤:
  1. python scripts/01_prepare_data.py
  2. python scripts/02_train_tokenizer.py
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer
from src.data.pretrain_dataset import PretrainDataset
from src.training.pretrain import PreTrainer


def main():
    parser = argparse.ArgumentParser(description="ClearMind 预训练")
    parser.add_argument(
        "--config", type=str, default="configs/small.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/pretrain/pretrain_data.jsonl",
        help="预训练数据路径",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="outputs/tokenizer/tokenizer.model",
        help="分词器模型路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/pretrain", help="输出目录"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="最大训练步数 (覆盖配置)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (覆盖配置)"
    )
    parser.add_argument("--log_every", type=int, default=10, help="每多少步打印日志")
    parser.add_argument(
        "--resume", type=str, default=None, help="断点续训的 checkpoint 路径"
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])
    train_config = config["pretrain"]

    # 命令行参数覆盖
    if args.max_steps:
        train_config["max_steps"] = args.max_steps
    if args.batch_size:
        train_config["batch_size"] = args.batch_size
    train_config["log_every"] = args.log_every

    print("=" * 60)
    print("ClearMind 预训练")
    print("=" * 60)

    # 检查前置文件
    if not os.path.exists(args.tokenizer):
        print(f"\n❌ 分词器不存在: {args.tokenizer}")
        print("   请先运行:")
        print("   1. python scripts/01_prepare_data.py")
        print("   2. python scripts/02_train_tokenizer.py")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"\n❌ 训练数据不存在: {args.data}")
        print("   请先运行: python scripts/01_prepare_data.py")
        sys.exit(1)

    # 加载分词器
    print(f"\n📖 加载分词器: {args.tokenizer}")
    tokenizer = ClearMindTokenizer(args.tokenizer)
    print(f"   词表大小: {tokenizer.vocab_size}")

    # 更新 vocab_size (使用实际分词器的词表大小)
    if tokenizer.vocab_size != model_config.vocab_size:
        print(
            f"   ⚠️  更新 vocab_size: {model_config.vocab_size} → {tokenizer.vocab_size}"
        )
        model_config.vocab_size = tokenizer.vocab_size

    # 创建数据集
    print(f"\n📦 加载训练数据: {args.data}")
    train_dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
    )

    # 创建模型
    print(f"\n🧠 创建模型...")
    model = GPT(model_config)
    params = model.count_parameters()
    print(f"   参数量: {params['total_millions']:.1f}M")

    # 创建 Trainer 并开始训练
    trainer = PreTrainer(
        model=model,
        train_dataset=train_dataset,
        config=train_config,
        output_dir=args.output_dir,
    )

    trainer.train(resume_from=args.resume)

    print(f"\n💡 下一步: python scripts/04_sft.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
