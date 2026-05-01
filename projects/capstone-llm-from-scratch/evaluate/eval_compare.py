"""
eval_compare.py — 多模型同基准对照表
=====================================

把 ClearMind 的某个 checkpoint 与 minimind / Qwen3-0.5B / 任意 HF 模型
在**同一基准、同一硬件、同一 sampling 配置**下跑同一遍，输出 markdown 对照表。

这是 "在同规模反超 minimind" 的硬证据。

核心设计：
  - **本项目模型**：通过 (config_path, ckpt_path) 加载（走 ``_common.load_model_for_eval``）
  - **HF 模型**：通过 ``transformers.AutoModelForCausalLM.from_pretrained``，但**只跑
    生成类基准**（AlignBench）；MCQ 基准（C-Eval/CMMLU）需要本项目的 GPT 接口
    一致才能跑，所以跨 HF 模型时仅取已存在的 JSON 报告做合并
  - **报告合并**：如果对照模型已经有同基准的 JSON 报告（例如别人发的官方分），
    直接读 JSON 拿数；否则现场跑

用法:
  # 从已有 JSON 报告合并对照表
  python evaluate/eval_compare.py merge \\
      reports/clearmind_base_ceval.json reports/minimind3_ceval.json \\
      --output reports/compare_ceval.md

  # 在同一硬件上跑两个本项目 checkpoint 对照
  python evaluate/eval_compare.py run \\
      --benchmark ceval \\
      --models clearmind-base:configs/main.yaml:outputs/dpo/final.pth \\
                clearmind-plus:configs/plus.yaml:outputs/dpo/final.pth
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_summary_row(report: dict, model_label: str) -> dict:
    """从一个 eval JSON 报告里抽 summary 字段，准备渲染 markdown 表格"""
    bench = report["meta"]["benchmark"]
    results = report["results"]

    row: dict = {"模型": model_label, "基准": bench}

    if bench in ("ceval", "cmmlu"):
        s = results["summary"]
        row["Macro Acc"] = f"{s['macro_acc'] * 100:.2f}%"
        row["Micro Acc"] = f"{s['micro_acc'] * 100:.2f}%"
        row["学科数"] = s["n_subjects"]
        row["题数"] = s["n_total"]
        # 4 大类
        for cat, v in results.get("by_category", {}).items():
            row[cat] = f"{v['acc'] * 100:.1f}%"

    elif bench == "alignbench-zh":
        s = results["summary"]
        row["Overall"] = (
            f"{s['avg_overall']:.2f}" if s["avg_overall"] is not None else "—"
        )
        row["已评分"] = f"{s['n_judged']}/{s['n_total']}"
        for cat, v in results.get("by_category", {}).items():
            row[cat] = (
                f"{v['avg_overall']:.2f}" if v["avg_overall"] is not None else "—"
            )

    else:
        # 通用：只把 summary 平铺
        for k, v in results.get("summary", {}).items():
            row[k] = str(v)
    return row


def _render_markdown(rows: list[dict], title: str = "对照表") -> str:
    """从 list[dict] 渲染 markdown 表格（自动取并集列）"""
    if not rows:
        return f"# {title}\n\n（无数据）\n"

    cols: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    md = f"# {title}\n\n"
    md += "| " + " | ".join(cols) + " |\n"
    md += "|" + "|".join(["---"] * len(cols)) + "|\n"
    for r in rows:
        md += "| " + " | ".join(str(r.get(c, "—")) for c in cols) + " |\n"
    return md


def cmd_merge(args: argparse.Namespace) -> int:
    """合并多个 JSON 报告 → markdown"""
    rows: list[dict] = []
    for path in args.reports:
        if ":" in path and not Path(path).exists():
            label, p = path.split(":", 1)
        else:
            label = Path(path).stem
            p = path
        if not Path(p).exists():
            print(f"⚠️  跳过：{p} 不存在")
            continue
        rep = _load_report(p)
        rows.append(_format_summary_row(rep, label))

    if not rows:
        print("❌ 没有可用报告")
        return 1

    md = _render_markdown(rows, title=args.title or "ClearMind 对照评测")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"📝 对照表已保存: {args.output}")
    else:
        print(md)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """对每个本项目模型跑 benchmark，最后合并成 markdown"""
    bench = args.benchmark
    runner_map = {
        "ceval": "evaluate/benchmarks/ceval.py",
        "cmmlu": "evaluate/benchmarks/cmmlu.py",
        "alignbench": "evaluate/benchmarks/alignbench.py",
    }
    if bench not in runner_map:
        print(f"❌ 未知 benchmark: {bench}（可选: {list(runner_map)})")
        return 1
    runner = runner_map[bench]

    out_paths: list[str] = []
    rows: list[dict] = []
    for spec in args.models:
        # 格式：label:config:ckpt
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"❌ 模型格式错误（应为 label:config:ckpt）: {spec}")
            return 1
        label, cfg, ckpt = parts
        out_json = f"reports/{label}_{bench}.json"
        out_paths.append(out_json)

        cmd = [
            sys.executable, runner,
            "--config", cfg, "--model", ckpt,
            "--output", out_json,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n🚀 跑 {label} on {bench} ...")
        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
        if rc != 0:
            print(f"⚠️  {label} 评测失败 (rc={rc})；该模型不进对照表")
            continue
        rep = _load_report(out_json)
        rows.append(_format_summary_row(rep, label))

    md = _render_markdown(rows, title=f"ClearMind × {bench} 对照评测")
    out_md = args.output or f"reports/compare_{bench}.md"
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(md, encoding="utf-8")
    print(f"\n📝 对照表已保存: {out_md}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="多模型对照评测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_merge = sub.add_parser("merge", help="合并已有 JSON 报告")
    p_merge.add_argument("reports", nargs="+",
                         help="JSON 报告路径，支持 'label:path' 格式")
    p_merge.add_argument("--output", default=None)
    p_merge.add_argument("--title", default=None)
    p_merge.set_defaults(func=cmd_merge)

    p_run = sub.add_parser("run", help="跑多个本项目模型并合并")
    p_run.add_argument("--benchmark", required=True, choices=["ceval", "cmmlu", "alignbench"])
    p_run.add_argument("--models", nargs="+", required=True,
                       help="格式：label:config:ckpt，例如 base:configs/main.yaml:outputs/dpo/final.pth")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--output", default=None)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
