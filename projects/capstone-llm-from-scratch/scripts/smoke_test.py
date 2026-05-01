"""
smoke_test.py — 端到端最小链路冒烟测试
==========================================

执行最小训练 + 推理链路，确保整条数据流在新 minimind 数据集上跑通：

  1. 检查 tokenizer/minimind/ 与 data/{pretrain_t2t_mini, sft_t2t_mini}.jsonl 存在
  2. 用 configs/tiny.yaml 跑 pretrain --max_steps 2
  3. 截一份 100 条 SFT 子集，跑 SFT --epochs 1
  4. 用训完的模型走一次 generate_chat 验证推理链路

预计 CPU/MPS 上 ~2 分钟完成。

用法:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --config configs/tiny.yaml --max_steps 2 --sft_lines 100
  python scripts/smoke_test.py --clean         # 清理上次产物再跑
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n▶ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _check_prerequisites() -> None:
    """检查 tokenizer 与最小数据是否就位"""
    missing: list[str] = []
    tk = PROJECT_ROOT / "tokenizer" / "minimind" / "tokenizer.json"
    if not tk.exists():
        missing.append(f"tokenizer 文件: {tk.relative_to(PROJECT_ROOT)}")
    pretrain = PROJECT_ROOT / "data" / "pretrain_t2t_mini.jsonl"
    if not pretrain.exists():
        missing.append(f"预训练数据: {pretrain.relative_to(PROJECT_ROOT)}")
    sft = PROJECT_ROOT / "data" / "sft_t2t_mini.jsonl"
    if not sft.exists():
        missing.append(f"SFT 数据: {sft.relative_to(PROJECT_ROOT)}")

    if missing:
        print("❌ 冒烟前置不满足，缺少：")
        for m in missing:
            print(f"     - {m}")
        print()
        print("快速下载（推荐 zero profile，~2.8 GB）：")
        print("    python scripts/download_data.py --profile zero")
        print()
        print("查看所有 profile：")
        print("    python scripts/download_data.py --list")
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClearMind 端到端冒烟测试")
    parser.add_argument("--config", default="configs/tiny.yaml", help="模型配置")
    parser.add_argument("--max_steps", type=int, default=2, help="预训练步数")
    parser.add_argument("--sft_lines", type=int, default=100, help="SFT 子集行数")
    parser.add_argument(
        "--work_dir", default="outputs/smoke", help="冒烟产物目录"
    )
    parser.add_argument("--clean", action="store_true", help="执行前清理 work_dir")
    args = parser.parse_args()

    _check_prerequisites()

    work = PROJECT_ROOT / args.work_dir
    if args.clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    pretrain_dir = work / "pretrain"
    sft_dir = work / "sft"
    sft_subset = work / "sft_subset.jsonl"

    print("=" * 60)
    print("ClearMind Smoke Test (minimind data + HF tokenizer)")
    print("=" * 60)
    print(f"📄 Config:     {args.config}")
    print(f"📁 Work Dir:   {work.relative_to(PROJECT_ROOT)}")
    print(f"🎯 Max Steps:  {args.max_steps}")
    print(f"📚 SFT Lines:  {args.sft_lines}")

    python = sys.executable

    # ---- Step 1: Pretrain 2 steps ----
    _run(
        [
            python,
            "scripts/train.py",
            "--stage", "pretrain",
            "--config", args.config,
            "--output_dir", str(pretrain_dir),
            "--max_steps", str(args.max_steps),
            "--log_every", "1",
        ]
    )
    pretrain_final = pretrain_dir / "final.pth"
    if not pretrain_final.exists():
        print(f"❌ Pretrain checkpoint 缺失: {pretrain_final}")
        sys.exit(1)
    print(f"\n✅ Pretrain 阶段通过: {pretrain_final.relative_to(PROJECT_ROOT)}")

    # ---- Step 2: 抽 SFT 子集 ----
    sft_full = PROJECT_ROOT / "data" / "sft_t2t_mini.jsonl"
    with sft_full.open("r", encoding="utf-8") as f_in, sft_subset.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i >= args.sft_lines:
                break
            f_out.write(line)
    print(f"✅ 抽取 SFT 子集: {sft_subset.relative_to(PROJECT_ROOT)} ({args.sft_lines} 行)")

    # ---- Step 3: SFT 1 epoch ----
    _run(
        [
            python,
            "scripts/train.py",
            "--stage", "sft",
            "--config", args.config,
            "--data", str(sft_subset),
            "--resume", str(pretrain_final),
            "--output_dir", str(sft_dir),
            "--epochs", "1",
            "--batch_size", "4",
            "--log_every", "4",
        ]
    )
    sft_final = sft_dir / "final.pth"
    if not sft_final.exists():
        print(f"❌ SFT checkpoint 缺失: {sft_final}")
        sys.exit(1)
    print(f"\n✅ SFT 阶段通过: {sft_final.relative_to(PROJECT_ROOT)}")

    # ---- Step 4: 推理验证（generate_chat） ----
    print("\n▶ 推理验证（generate_chat with chat_template）")
    sys.path.insert(0, str(PROJECT_ROOT))

    import torch
    import yaml

    from scripts.train import load_tokenizer
    from src.model.config import ModelConfig
    from src.model.gpt import GPT
    from src.inference.generate import generate_chat

    with (PROJECT_ROOT / args.config).open("r") as f:
        cfg = yaml.safe_load(f)
    model_config = ModelConfig(**cfg["model"])
    tokenizer = load_tokenizer(cfg, None)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    model = GPT(model_config)
    ckpt = torch.load(str(sft_final), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    reply = generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "你好"}],
        max_new_tokens=16,
        temperature=0.8,
    )
    if not reply or not reply.strip():
        print("❌ 推理验证失败：generate_chat 返回空字符串")
        sys.exit(1)
    print(f"  生成结果（前 100 字符）: {reply[:100]!r}")
    print("✅ 推理验证通过")

    print("\n" + "=" * 60)
    print("✅ Smoke test 全部通过 (Pretrain + SFT + Chat 推理)")
    print(f"📦 产物: {sft_final.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
