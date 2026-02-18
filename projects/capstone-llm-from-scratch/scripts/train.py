"""
train.py — 统一训练入口
========================

通过 --stage 参数切换训练阶段:
  - pretrain: 预训练 (Next-token Prediction)
  - sft: 指令微调 (Supervised Fine-Tuning)
  - dpo: 偏好对齐 (Direct Preference Optimization)

使用方法:
  # 预训练
  python scripts/train.py --stage pretrain

  # 指令微调
  python scripts/train.py --stage sft

  # DPO 对齐
  python scripts/train.py --stage dpo

  # 指定配置 + 参数覆盖
  python scripts/train.py --stage pretrain --config configs/medium.yaml --max_steps 1000

  # 断点续训
  python scripts/train.py --stage pretrain --resume outputs/pretrain/checkpoint_step1000.pth

前置步骤:
  1. python scripts/prepare_data.py
  2. python scripts/train_tokenizer.py
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch  # noqa: F401
from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer


# ============================================================
# 通用初始化
# ============================================================

STAGE_DEFAULTS = {
    "pretrain": {
        "data": "data/pretrain/pretrain_data.jsonl",
        "output_dir": "outputs/pretrain",
        "description": "预训练",
    },
    "sft": {
        "data": "data/sft/sft_data.jsonl",
        "output_dir": "outputs/sft",
        "description": "SFT 指令微调",
    },
    "dpo": {
        "data": "data/dpo/dpo_data.jsonl",
        "output_dir": "outputs/dpo",
        "description": "DPO 对齐训练",
    },
}


def load_setup(args):
    """通用初始化: 加载配置、分词器、模型"""
    # 加载 YAML 配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])

    # 检查分词器
    if not os.path.exists(args.tokenizer):
        print(f"\n❌ 分词器不存在: {args.tokenizer}")
        print("   请先运行:")
        print("   1. python scripts/prepare_data.py")
        print("   2. python scripts/train_tokenizer.py")
        sys.exit(1)

    # 检查数据
    if not os.path.exists(args.data):
        print(f"\n❌ 训练数据不存在: {args.data}")
        print("   请先运行: python scripts/prepare_data.py")
        sys.exit(1)

    # 加载分词器
    tokenizer = ClearMindTokenizer(args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        print(
            f"   ⚠️  更新 vocab_size: {model_config.vocab_size} → {tokenizer.vocab_size}"
        )
        model_config.vocab_size = tokenizer.vocab_size

    # 创建模型
    model = GPT(model_config)
    params = model.count_parameters()
    print(f"🧠 模型参数量: {params['total_millions']:.1f}M")

    return config, model_config, tokenizer, model


# ============================================================
# 各阶段训练逻辑
# ============================================================


def run_pretrain(args, config, model_config, tokenizer, model):
    """预训练"""
    from src.data.pretrain_dataset import PretrainDataset
    from src.training.pretrain import PreTrainer

    train_config = config["pretrain"]
    if args.max_steps:
        train_config["max_steps"] = args.max_steps
    if args.batch_size:
        train_config["batch_size"] = args.batch_size
    train_config["log_every"] = args.log_every

    print(f"\n📦 加载训练数据: {args.data}")
    train_dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
    )

    trainer = PreTrainer(
        model=model,
        train_dataset=train_dataset,
        config=train_config,
        output_dir=args.output_dir,
    )
    trainer.train(resume_from=args.resume)

    print(f"\n💡 下一步: python scripts/train.py --stage sft")


def run_sft(args, config, model_config, tokenizer, model):
    """指令微调"""
    from src.data.sft_dataset import SFTDataset
    from src.training.sft import SFTTrainer

    sft_config = config["sft"]
    if args.epochs:
        sft_config["epochs"] = args.epochs
    if args.batch_size:
        sft_config["batch_size"] = args.batch_size
    sft_config["log_every"] = args.log_every

    # 预训练模型路径
    pretrained = args.resume or "outputs/pretrain/final.pth"
    if not os.path.exists(pretrained):
        print(f"\n❌ 预训练模型不存在: {pretrained}")
        print("   请先运行: python scripts/train.py --stage pretrain")
        sys.exit(1)

    print(f"📦 加载训练数据: {args.data}")
    train_dataset = SFTDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        config=sft_config,
        output_dir=args.output_dir,
    )
    trainer.train(pretrained_path=pretrained)

    print(f"\n💡 下一步: python scripts/train.py --stage dpo")


def run_dpo(args, config, model_config, tokenizer, model):
    """DPO 对齐训练"""
    from src.data.dpo_dataset import DPODataset
    from src.training.dpo import DPOTrainer

    dpo_config = config["dpo"]
    if args.batch_size:
        dpo_config["batch_size"] = args.batch_size
    dpo_config["log_every"] = args.log_every

    # SFT 模型路径
    sft_model = args.resume or "outputs/sft/final.pth"
    if not os.path.exists(sft_model):
        print(f"\n❌ SFT 模型不存在: {sft_model}")
        print("   请先运行: python scripts/train.py --stage sft")
        sys.exit(1)

    print(f"📦 加载训练数据: {args.data}")
    train_dataset = DPODataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
    )

    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        config=dpo_config,
        output_dir=args.output_dir,
    )
    trainer.train(sft_path=sft_model)

    print(f"\n💡 下一步: python scripts/chat.py")


# ============================================================
# 主入口
# ============================================================

STAGES = {
    "pretrain": run_pretrain,
    "sft": run_sft,
    "dpo": run_dpo,
}


def main():
    parser = argparse.ArgumentParser(description="ClearMind 统一训练入口")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["pretrain", "sft", "dpo"],
        help="训练阶段",
    )
    parser.add_argument(
        "--config", type=str, default="configs/small.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="训练数据路径 (默认按阶段自动选择)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="outputs/tokenizer/tokenizer.model",
        help="分词器模型路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="输出目录 (默认按阶段自动选择)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="最大训练步数 (覆盖配置)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (覆盖配置)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (SFT/DPO)")
    parser.add_argument("--log_every", type=int, default=10, help="日志间隔步数")
    parser.add_argument(
        "--resume", type=str, default=None, help="续训 checkpoint / 上阶段模型路径"
    )
    args = parser.parse_args()

    # 填充默认值
    defaults = STAGE_DEFAULTS[args.stage]
    if args.data is None:
        args.data = defaults["data"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]

    print("=" * 60)
    print(f"ClearMind {defaults['description']}")
    print("=" * 60)

    config, model_config, tokenizer, model = load_setup(args)
    STAGES[args.stage](args, config, model_config, tokenizer, model)

    print("=" * 60)


if __name__ == "__main__":
    main()
