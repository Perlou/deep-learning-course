"""
train.py — 统一训练入口
========================

通过 ``--stage`` 参数切换训练阶段:
  - ``pretrain``: 预训练 (Next-token Prediction)
  - ``sft``: 指令微调 (Supervised Fine-Tuning)
  - ``dpo``: 偏好对齐 (Direct Preference Optimization)

使用方法:

  # 预训练
  python scripts/train.py --stage pretrain

  # 指令微调
  python scripts/train.py --stage sft

  # DPO 对齐
  python scripts/train.py --stage dpo

  # 指定配置 + 参数覆盖
  python scripts/train.py --stage pretrain --config configs/main.yaml --max_steps 1000

  # 断点续训
  python scripts/train.py --stage pretrain --resume outputs/pretrain/checkpoint_step1000.pth

数据约定（参见 README）：
  data/pretrain_t2t_mini.jsonl     ← MiniMind 预训练数据（每行 {"text": ...}）
  data/sft_t2t_mini.jsonl          ← MiniMind SFT 数据（每行 {"conversations": [...]}）
  data/dpo.jsonl                   ← MiniMind DPO 数据（每行 {"chosen": [...], "rejected": [...]}）
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


# ============================================================
# 通用初始化
# ============================================================

STAGE_DEFAULTS = {
    "pretrain": {
        "data": "data/pretrain_t2t_mini.jsonl",
        "output_dir": "outputs/pretrain",
        "description": "预训练",
    },
    "sft": {
        "data": "data/sft_t2t_mini.jsonl",
        "output_dir": "outputs/sft",
        "description": "SFT 指令微调",
    },
    "dpo": {
        "data": "data/dpo.jsonl",
        "output_dir": "outputs/dpo",
        "description": "DPO 对齐训练",
    },
    "distillation": {
        "data": "data/sft_t2t_mini.jsonl",
        "output_dir": "outputs/distillation",
        "description": "白盒蒸馏（teacher → student）",
    },
    "grpo": {
        "data": "data/rlaif.jsonl",
        "output_dir": "outputs/grpo",
        "description": "GRPO + CISPO RL 训练",
    },
}


def load_tokenizer(config: dict, cli_tokenizer: str | None):
    """根据配置选择 HF tokenizer 或 sentencepiece tokenizer

    优先级：
      1. ``--tokenizer`` 显式传入 → 按文件后缀判断（.model 走 sentencepiece，目录走 HF）
      2. yaml ``tokenizer.type`` 字段：``"hf"`` 走 :class:`HFTokenizer`，
         ``"sentencepiece"`` 走 :class:`ClearMindTokenizer`
      3. 否则 fallback 到 ``tokenizer/minimind`` 目录的 HF tokenizer
    """
    if cli_tokenizer:
        path = cli_tokenizer
        # .model 是 sentencepiece，目录是 HF
        if path.endswith(".model"):
            from src.data.tokenizer import ClearMindTokenizer

            return ClearMindTokenizer(path)
        from src.data.hf_tokenizer import HFTokenizer

        return HFTokenizer(path)

    tk_cfg = config.get("tokenizer", {}) or {}
    tk_type = tk_cfg.get("type", "hf")
    if tk_type == "sentencepiece":
        from src.data.tokenizer import ClearMindTokenizer

        path = tk_cfg.get("path") or "outputs/tokenizer/tokenizer.model"
        return ClearMindTokenizer(path)

    # HF（默认）
    from src.data.hf_tokenizer import HFTokenizer

    path = tk_cfg.get("path") or "tokenizer/minimind"
    return HFTokenizer(path)


def load_setup(args):
    """通用初始化: 加载配置、分词器、模型"""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])

    # 加载分词器
    tokenizer = load_tokenizer(config, args.tokenizer)

    # 检查数据
    if not os.path.exists(args.data):
        print(f"\n❌ 训练数据不存在: {args.data}")
        print(
            "   请从 https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files "
            "下载对应文件并放到 data/ 目录下"
        )
        sys.exit(1)

    # vocab 一致性校验：tokenizer 实际 size 必须 ≥ config 声明
    if tokenizer.vocab_size != model_config.vocab_size:
        print(
            f"   ⚠️  tokenizer.vocab_size={tokenizer.vocab_size} 与 "
            f"model.vocab_size={model_config.vocab_size} 不一致，已自动对齐到 tokenizer"
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

    pretrain_mode = train_config.get("pretrain_mode", "per_sample")
    val_ratio = train_config.get("val_ratio", 0.05)

    print(f"\n📦 加载训练数据 ({pretrain_mode}): {args.data}")
    train_dataset, val_dataset = PretrainDataset.create_with_split(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        val_ratio=val_ratio,
        mode=pretrain_mode,
    )

    trainer = PreTrainer(
        model=model,
        train_dataset=train_dataset,
        config=train_config,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )
    trainer.train(resume_from=args.resume)

    print("\n💡 下一步: python scripts/train.py --stage sft")


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
    sft_config["pad_token_id"] = tokenizer.pad_id

    if args.lora:
        sft_config["lora"] = True
        sft_config["lora_rank"] = args.lora_rank
        sft_config["lora_alpha"] = args.lora_alpha

    pretrained = args.resume or "outputs/pretrain/final.pth"
    if not os.path.exists(pretrained):
        print(f"\n❌ 预训练模型不存在: {pretrained}")
        print("   请先运行: python scripts/train.py --stage pretrain")
        sys.exit(1)

    print(f"📦 加载训练数据: {args.data}")
    train_dataset, val_dataset = SFTDataset.create_with_split(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        val_ratio=sft_config.get("val_ratio", 0.05),
        system_prompt_ratio=sft_config.get("system_prompt_ratio", 0.2),
        empty_think_strip_ratio=sft_config.get("empty_think_strip_ratio", 0.8),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        config=sft_config,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )
    trainer.train(pretrained_path=pretrained)

    print("\n💡 下一步: python scripts/train.py --stage dpo")


def run_dpo(args, config, model_config, tokenizer, model):
    """DPO 对齐训练"""
    from src.data.dpo_dataset import DPODataset
    from src.training.dpo import DPOTrainer

    dpo_config = config["dpo"]
    if args.batch_size:
        dpo_config["batch_size"] = args.batch_size
    dpo_config["log_every"] = args.log_every
    dpo_config["pad_token_id"] = tokenizer.pad_id

    sft_model = args.resume or "outputs/sft/final.pth"
    if not os.path.exists(sft_model):
        print(f"\n❌ SFT 模型不存在: {sft_model}")
        print("   请先运行: python scripts/train.py --stage sft")
        sys.exit(1)

    print(f"📦 加载训练数据: {args.data}")
    train_dataset, val_dataset = DPODataset.create_with_split(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        val_ratio=dpo_config.get("val_ratio", 0.05),
    )

    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        config=dpo_config,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )
    trainer.train(sft_path=sft_model)

    print("\n💡 下一步: python scripts/chat.py")


def run_distillation(args, config, model_config, tokenizer, model):
    """白盒蒸馏训练（teacher → student）

    用法:
      python scripts/train.py --stage distillation \\
          --config configs/main.yaml \\
          --teacher outputs/dpo/final.pth \\
          --teacher_config configs/plus.yaml
    """
    from src.data.sft_dataset import SFTDataset
    from src.training.distillation import DistillationTrainer

    distill_config = dict(config.get("distillation", config.get("sft", {})))
    if args.epochs:
        distill_config["epochs"] = args.epochs
    if args.batch_size:
        distill_config["batch_size"] = args.batch_size
    distill_config["log_every"] = args.log_every
    distill_config["pad_token_id"] = tokenizer.pad_id
    if args.teacher:
        distill_config["teacher_path"] = args.teacher

    if not distill_config.get("teacher_path"):
        print("\n❌ 蒸馏需要 teacher checkpoint，请用 --teacher 指定")
        sys.exit(1)
    if not os.path.exists(distill_config["teacher_path"]):
        print(f"\n❌ Teacher 不存在: {distill_config['teacher_path']}")
        sys.exit(1)

    print(f"📦 加载训练数据: {args.data}")
    train_dataset, val_dataset = SFTDataset.create_with_split(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        val_ratio=distill_config.get("val_ratio", 0.05),
        system_prompt_ratio=distill_config.get("system_prompt_ratio", 0.0),
        empty_think_strip_ratio=distill_config.get("empty_think_strip_ratio", 0.0),
    )

    trainer = DistillationTrainer(
        model=model,
        train_dataset=train_dataset,
        config=distill_config,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )
    trainer.train(
        teacher_path=distill_config["teacher_path"],
        teacher_config=args.teacher_config,
        pretrained_path=args.resume,
    )

    print("\n💡 下一步: python scripts/chat.py")


# ============================================================
# 主入口
# ============================================================

def run_grpo(args, config, model_config, tokenizer, model):
    """GRPO + CISPO 训练（DeepSeek-R1 同款）"""
    from src.data.rl_dataset import RLAIFDataset
    from src.training.grpo import GRPOTrainer

    grpo_config = dict(config.get("grpo", {}))
    if args.epochs:
        grpo_config["epochs"] = args.epochs
    if args.batch_size:
        grpo_config["batch_size"] = args.batch_size
    grpo_config["log_every"] = args.log_every
    grpo_config["pad_token_id"] = tokenizer.pad_id

    sft_model = args.resume or "outputs/sft/final.pth"
    if not os.path.exists(sft_model):
        print(f"\n❌ SFT 模型不存在: {sft_model}")
        print("   GRPO 需要 SFT 权重作起点；先 python scripts/train.py --stage sft")
        sys.exit(1)

    print(f"📦 加载 RL 数据: {args.data}")
    dataset = RLAIFDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_prompt_len=model_config.max_seq_len // 2,
        thinking_ratio=grpo_config.get("thinking_ratio", 0.5),
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        config=grpo_config,
        val_dataset=None,
        output_dir=args.output_dir,
    )
    trainer.train(sft_path=sft_model)

    print("\n💡 下一步: python scripts/chat.py")


STAGES = {
    "pretrain": run_pretrain,
    "sft": run_sft,
    "dpo": run_dpo,
    "distillation": run_distillation,
    "grpo": run_grpo,
}


def main():
    parser = argparse.ArgumentParser(description="ClearMind 统一训练入口")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["pretrain", "sft", "dpo", "distillation", "grpo"],
        help="训练阶段",
    )
    parser.add_argument(
        "--config", type=str, default="configs/main.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="训练数据路径 (默认按阶段自动选择)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="分词器路径 (默认读 yaml.tokenizer.path; .model 后缀走 sentencepiece，目录走 HF)",
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
    parser.add_argument("--lora", action="store_true", help="启用 LoRA 微调")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA 秩 (默认 8)")
    parser.add_argument(
        "--lora_alpha", type=float, default=16.0, help="LoRA alpha (默认 16)"
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="蒸馏 teacher checkpoint 路径（仅 --stage distillation 用）",
    )
    parser.add_argument(
        "--teacher_config",
        type=str,
        default=None,
        help="teacher 模型配置 yaml（teacher 与 student 架构不同时必传，如 plus.yaml）",
    )
    args = parser.parse_args()

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
