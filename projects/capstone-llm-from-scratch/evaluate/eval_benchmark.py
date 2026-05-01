"""
eval_benchmark.py — 综合评估报告
==================================

一键运行 PPL + 生成质量 + 指令跟随评估，输出 markdown 表格。
对比 Pretrain / SFT / DPO 三阶段的模型表现，作为发布到 HF/ModelScope 时
模型卡的评测数据来源。

用法:
  python evaluate/eval_benchmark.py --config configs/main.yaml
  python evaluate/eval_benchmark.py --config configs/plus.yaml --output report.md
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> tuple[int, str]:
    """运行子命令，返回 (returncode, stdout+stderr)"""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:
        return -1, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearMind 综合评估")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument(
        "--data",
        default="data/pretrain_t2t_mini.jsonl",
        help="PPL 评估用的验证数据",
    )
    parser.add_argument("--max_batches", type=int, default=200, help="PPL 评估批数")
    parser.add_argument("--num_prompts", type=int, default=20, help="生成质量 prompt 数")
    parser.add_argument("--num_samples", type=int, default=2, help="每 prompt 采样数")
    parser.add_argument(
        "--output", default=None, help="导出 markdown 报告路径（如 reports/eval.md）"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["ppl", "generation", "instruction"],
        help="跳过哪些评估",
    )
    args = parser.parse_args()

    python = sys.executable
    print("=" * 60)
    print("ClearMind 综合评估")
    print("=" * 60)
    print(f"📄 Config:  {args.config}")
    print()

    sections: list[tuple[str, str, str]] = []  # (title, command_str, output)

    # ===== 1. PPL 三阶段对比 =====
    if "ppl" not in args.skip:
        print("\n▶ Step 1/3: 困惑度（PPL）三阶段对比")
        print("-" * 60)
        cmd = [
            python,
            "evaluate/eval_perplexity.py",
            "--config", args.config,
            "--data", args.data,
            "--max_batches", str(args.max_batches),
            "--compare",
        ]
        rc, out = _run(cmd)
        print(out)
        sections.append(("📊 困惑度（PPL）", " ".join(cmd), out))
        if rc != 0:
            print(f"  ⚠️ 退出码: {rc}")

    # ===== 2. 生成质量 =====
    if "generation" not in args.skip:
        print("\n▶ Step 2/3: 生成质量")
        print("-" * 60)
        # 优先用 SFT 模型；否则 fallback 自动选择
        for stage_name, model_path in [("dpo", "outputs/dpo/final.pth"),
                                        ("sft", "outputs/sft/final.pth")]:
            if os.path.exists(PROJECT_ROOT / model_path):
                cmd = [
                    python,
                    "evaluate/eval_generation.py",
                    "--config", args.config,
                    "--model", model_path,
                    "--num_prompts", str(args.num_prompts),
                    "--num_samples", str(args.num_samples),
                    "--show_samples", "0",  # benchmark 不打印 samples
                ]
                rc, out = _run(cmd)
                print(out)
                sections.append((f"📝 生成质量（{stage_name} 模型）", " ".join(cmd), out))
                break
        else:
            print("  ⚠️ 没有 SFT/DPO checkpoint，跳过")

    # ===== 3. 指令跟随 =====
    if "instruction" not in args.skip:
        print("\n▶ Step 3/3: 指令跟随")
        print("-" * 60)
        for stage_name, model_path in [("dpo", "outputs/dpo/final.pth"),
                                        ("sft", "outputs/sft/final.pth")]:
            if os.path.exists(PROJECT_ROOT / model_path):
                cmd = [
                    python,
                    "evaluate/eval_instruction.py",
                    "--config", args.config,
                    "--model", model_path,
                ]
                rc, out = _run(cmd)
                print(out)
                sections.append((f"🎯 指令跟随（{stage_name} 模型）", " ".join(cmd), out))
                break
        else:
            print("  ⚠️ 没有 SFT/DPO checkpoint，跳过")

    # ===== 导出 markdown =====
    if args.output:
        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# ClearMind 综合评估报告\n\n")
            f.write(f"- Config: `{args.config}`\n")
            f.write(f"- 数据: `{args.data}`\n\n")
            for title, cmd, output in sections:
                f.write(f"## {title}\n\n")
                f.write(f"```\n{cmd}\n```\n\n")
                f.write("```\n")
                f.write(output)
                f.write("\n```\n\n")
        print(f"\n📝 报告已保存: {out_path.relative_to(PROJECT_ROOT)}")

    print()
    print("=" * 60)
    print(f"✅ 综合评估完成（共 {len(sections)} 项）")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
