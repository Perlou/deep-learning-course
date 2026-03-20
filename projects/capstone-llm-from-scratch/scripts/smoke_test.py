"""
smoke_test.py — 端到端最小流程冒烟测试
=====================================

执行最小训练链路:
  1. 准备样例数据
  2. 训练分词器
  3. 预训练 1~N 步

用法:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --max_steps 2 --work_dir outputs/smoke
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"\n▶ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    parser = argparse.ArgumentParser(description="ClearMind 端到端 smoke test")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
        help="模型配置文件",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="outputs/smoke",
        help="smoke test 输出目录",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1,
        help="预训练步数",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="执行前清理 work_dir",
    )
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max_steps 必须 > 0")

    if args.clean and os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    out_dir = work_dir / "outputs"
    tok_dir = out_dir / "tokenizer"
    pretrain_dir = out_dir / "pretrain"

    work_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # 先检查 tokenizer 依赖，避免运行到中途失败。
    try:
        import sentencepiece  # noqa: F401
    except ModuleNotFoundError:
        print("❌ 缺少依赖 sentencepiece，无法执行 smoke test")
        print("   请先运行: pip install -r requirements.txt")
        sys.exit(2)

    print("=" * 60)
    print("ClearMind Smoke Test")
    print("=" * 60)
    print(f"📄 Config: {args.config}")
    print(f"📁 Work Dir: {work_dir}")
    print(f"🎯 Max Steps: {args.max_steps}")

    run_cmd(
        [
            python,
            "scripts/prepare_data.py",
            "--scale",
            "small",
            "--data_dir",
            str(data_dir),
        ],
        PROJECT_ROOT,
    )

    run_cmd(
        [
            python,
            "scripts/train_tokenizer.py",
            "--config",
            args.config,
            "--corpus",
            str(data_dir / "tokenizer_corpus.txt"),
            "--output_dir",
            str(tok_dir),
        ],
        PROJECT_ROOT,
    )

    run_cmd(
        [
            python,
            "scripts/train.py",
            "--stage",
            "pretrain",
            "--config",
            args.config,
            "--data",
            str(data_dir / "pretrain" / "pretrain_data.jsonl"),
            "--tokenizer",
            str(tok_dir / "tokenizer.model"),
            "--output_dir",
            str(pretrain_dir),
            "--max_steps",
            str(args.max_steps),
            "--log_every",
            "1",
        ],
        PROJECT_ROOT,
    )

    final_ckpt = pretrain_dir / "final.pth"
    if not final_ckpt.exists():
        print(f"❌ smoke test 失败: 未找到 checkpoint {final_ckpt}")
        sys.exit(1)

    print(f"\n✅ 预训练阶段通过: {final_ckpt}")

    # ========== SFT 阶段 ==========
    sft_dir = out_dir / "sft"
    sft_data = data_dir / "sft" / "sft_data.jsonl"

    # 准备 SFT 数据
    run_cmd(
        [
            python,
            "scripts/prepare_data.py",
            "--stage",
            "sft",
            "--scale",
            "small",
            "--data_dir",
            str(data_dir),
        ],
        PROJECT_ROOT,
    )

    # SFT 训练 1 epoch
    run_cmd(
        [
            python,
            "scripts/train.py",
            "--stage",
            "sft",
            "--config",
            args.config,
            "--data",
            str(sft_data),
            "--tokenizer",
            str(tok_dir / "tokenizer.model"),
            "--output_dir",
            str(sft_dir),
            "--resume",
            str(final_ckpt),
            "--epochs",
            "1",
            "--log_every",
            "1",
        ],
        PROJECT_ROOT,
    )

    sft_ckpt = sft_dir / "final.pth"
    if not sft_ckpt.exists():
        print(f"❌ smoke test 失败: 未找到 SFT checkpoint {sft_ckpt}")
        sys.exit(1)

    print(f"\n✅ SFT 阶段通过: {sft_ckpt}")

    # ========== 推理验证 ==========
    print("\n▶ 推理生成验证")
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from src.model.config import ModelConfig
    from src.model.gpt import GPT
    from src.inference.generate import generate_text
    from src.data.tokenizer import ClearMindTokenizer
    import yaml

    with open(PROJECT_ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_config = ModelConfig(**cfg["model"])
    tokenizer = ClearMindTokenizer(str(tok_dir / "tokenizer.model"))
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    model = GPT(model_config)
    ckpt = torch.load(str(sft_ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="你好",
        max_new_tokens=16,
    )
    if not output or len(output.strip()) == 0:
        print("❌ smoke test 失败: 推理生成输出为空")
        sys.exit(1)
    print(f"  生成结果: {output[:100]}")
    print("✅ 推理验证通过")

    print("\n" + "=" * 60)
    print("✅ Smoke test 全部通过 (预训练 + SFT + 推理)")
    print(f"📦 产物: {sft_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
