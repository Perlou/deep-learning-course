"""
download_data.py — minimind 数据集按需下载
==========================================

按训练目标选择数据 profile，从 ModelScope 或 HuggingFace 下载到 ``data/``。

数据 profile 对应表（与 ClearMind 训练规格匹配）：

  zero    pretrain_t2t_mini.jsonl + sft_t2t_mini.jsonl                      (~2.8 GB)
            适合：tiny/small/base 快速复现 minimind-zero 对话模型
  base    + dpo.jsonl                                                       (+53 MB → ~2.85 GB)
            适合：ClearMind-Base 完整 Pretrain → SFT → DPO 流程
  plus    pretrain_t2t.jsonl + sft_t2t.jsonl + dpo.jsonl                    (~24 GB)
            适合：ClearMind-Plus 旗舰版完整复刻 minimind-3 主线
  rl      在 base 基础上加 rlaif.jsonl + agent_rl.jsonl + agent_rl_math.jsonl
            适合：未来 PPO/GRPO/CISPO/Agentic RL 训练
  all     全部 8 个文件                                                      (~28 GB)

使用方法:

  # 默认 zero profile + 优先 ModelScope
  python scripts/download_data.py

  # 指定 profile + source
  python scripts/download_data.py --profile plus --source hf

  # 自定义文件列表
  python scripts/download_data.py --files dpo.jsonl rlaif.jsonl

  # 强制覆盖已存在文件
  python scripts/download_data.py --profile base --force

  # 列出所有 profile 与对应文件
  python scripts/download_data.py --list

依赖：``modelscope`` 与 ``huggingface_hub`` 已写入 ``requirements.txt``。
数据来源（任选）：
  - ModelScope: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
  - HuggingFace: https://huggingface.co/datasets/jingyaogong/minimind_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# 数据集元信息（来自 minimind/README.md §Ⅴ MiniMind 训练数据集）
# ============================================================

DATASET_FILES: dict[str, dict] = {
    "pretrain_t2t_mini.jsonl": {
        "size_gb": 1.2,
        "purpose": "轻量预训练（minimind-zero 快速复现）",
        "recommended": True,
    },
    "pretrain_t2t.jsonl": {
        "size_gb": 10.0,
        "purpose": "主线预训练（minimind-3 完整复刻）",
    },
    "sft_t2t_mini.jsonl": {
        "size_gb": 1.6,
        "purpose": "轻量 SFT（含 Tool Call 样本）",
        "recommended": True,
    },
    "sft_t2t.jsonl": {
        "size_gb": 14.0,
        "purpose": "主线 SFT（完整版，含 Tool Call）",
    },
    "dpo.jsonl": {
        "size_gb": 0.053,
        "purpose": "DPO 偏好对齐数据",
    },
    "rlaif.jsonl": {
        "size_gb": 0.024,
        "purpose": "RLAIF (PPO/GRPO/CISPO) RL 训练",
        "recommended": True,
    },
    "agent_rl.jsonl": {
        "size_gb": 0.086,
        "purpose": "Agentic RL 主线（Tool-Use）",
    },
    "agent_rl_math.jsonl": {
        "size_gb": 0.018,
        "purpose": "Agentic RL 数学补充（RLVR）",
    },
}


PROFILES: dict[str, list[str]] = {
    "zero": [
        "pretrain_t2t_mini.jsonl",
        "sft_t2t_mini.jsonl",
    ],
    "base": [
        "pretrain_t2t_mini.jsonl",
        "sft_t2t_mini.jsonl",
        "dpo.jsonl",
    ],
    "plus": [
        "pretrain_t2t.jsonl",
        "sft_t2t.jsonl",
        "dpo.jsonl",
    ],
    "rl": [
        "pretrain_t2t_mini.jsonl",
        "sft_t2t_mini.jsonl",
        "dpo.jsonl",
        "rlaif.jsonl",
        "agent_rl.jsonl",
        "agent_rl_math.jsonl",
    ],
    "all": list(DATASET_FILES.keys()),
}


PROFILE_DESCRIPTIONS: dict[str, str] = {
    "zero": "tiny/small/base 快速复现 minimind-zero 对话模型",
    "base": "ClearMind-Base 完整 Pretrain → SFT → DPO 流程",
    "plus": "ClearMind-Plus 旗舰版完整复刻 minimind-3 主线",
    "rl": "PPO/GRPO/CISPO/Agentic RL 全栈训练",
    "all": "全部 8 个文件",
}


HF_REPO_ID = "jingyaogong/minimind_dataset"
MODELSCOPE_REPO_ID = "gongjy/minimind_dataset"


# ============================================================
# 下载后端
# ============================================================


def _human_size(gb: float) -> str:
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


def _format_size(path: Path) -> str:
    n = path.stat().st_size
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def _download_modelscope(filename: str, dest: Path) -> bool:
    """通过 modelscope 库下载，失败返回 False"""
    try:
        from modelscope.hub.file_download import dataset_file_download
    except ImportError:
        print(
            "  ⚠️  未安装 modelscope。requirements.txt 中已声明，请运行：\n"
            "       pip install -r requirements.txt"
        )
        return False
    try:
        path = dataset_file_download(
            dataset_id=MODELSCOPE_REPO_ID,
            file_path=filename,
            local_dir=str(DATA_DIR),
        )
        path = Path(path)
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)
        return True
    except Exception as e:
        print(f"  ❌ ModelScope 下载失败: {e}")
        return False


def _download_huggingface(filename: str, dest: Path) -> bool:
    """通过 huggingface_hub 下载"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "  ⚠️  未安装 huggingface_hub。requirements.txt 中已声明，请运行：\n"
            "       pip install -r requirements.txt"
        )
        return False
    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(DATA_DIR),
        )
        path = Path(path)
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)
        return True
    except Exception as e:
        print(f"  ❌ HuggingFace 下载失败: {e}")
        return False


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按需下载 minimind 数据集到 data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="zero",
        help="数据 profile（默认 zero）",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="自定义文件列表（覆盖 --profile）",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf", "auto"],
        default="auto",
        help="下载源；auto 先 modelscope 后 hf",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使本地已存在）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有 profile 与对应文件，然后退出",
    )
    return parser.parse_args()


def list_profiles() -> None:
    print("\n📚 可用数据 profile:")
    for prof, files in PROFILES.items():
        total = sum(DATASET_FILES[f]["size_gb"] for f in files)
        print(f"\n  {prof:6s}  ({_human_size(total)})  {PROFILE_DESCRIPTIONS[prof]}")
        for f in files:
            meta = DATASET_FILES[f]
            star = "✨" if meta.get("recommended") else "  "
            print(f"    {star}  {f:32s}  {_human_size(meta['size_gb']):>7s}  {meta['purpose']}")
    print("\n详细文件信息：见 minimind README.md §Ⅴ\n")


def main() -> int:
    args = parse_args()

    if args.list:
        list_profiles()
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.files:
        files = args.files
        unknown = [f for f in files if f not in DATASET_FILES]
        if unknown:
            print(f"❌ 未知文件: {unknown}")
            print(f"   有效文件: {list(DATASET_FILES.keys())}")
            return 1
    else:
        files = PROFILES[args.profile]

    total_gb = sum(DATASET_FILES[f]["size_gb"] for f in files)

    print("=" * 60)
    print("  ClearMind × MiniMind 数据下载")
    print("=" * 60)
    if args.files:
        print(f"  自定义文件:     {len(files)} 个")
    else:
        print(f"  Profile:        {args.profile} ({PROFILE_DESCRIPTIONS[args.profile]})")
    print(f"  文件数:         {len(files)}")
    print(f"  预估总大小:     {_human_size(total_gb)}")
    print(f"  目标目录:       {DATA_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  下载源:         {args.source}")
    print("=" * 60)

    print("\n待下载:")
    for f in files:
        dest = DATA_DIR / f
        status = "已存在" if dest.exists() and not args.force else "待下载"
        size_remote = _human_size(DATASET_FILES[f]["size_gb"])
        size_local = _format_size(dest) if dest.exists() else "—"
        print(f"  [{status}]  {f:32s}  remote≈{size_remote}  local={size_local}")

    if total_gb > 5:
        print(f"\n⚠️  总下载量 {_human_size(total_gb)}，下载需要较长时间和磁盘空间")
        try:
            ans = input("继续吗？[y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("已取消")
            return 0

    print()
    successes, skipped, failures = 0, 0, 0
    for f in files:
        dest = DATA_DIR / f
        if dest.exists() and not args.force:
            print(f"⏭️  跳过 {f}（已存在）")
            skipped += 1
            continue

        print(f"⬇️  下载 {f} ...")
        ok = False
        if args.source in ("modelscope", "auto"):
            ok = _download_modelscope(f, dest)
        if not ok and args.source in ("hf", "auto"):
            ok = _download_huggingface(f, dest)

        if ok and dest.exists():
            print(f"✅ {f}  ({_format_size(dest)})")
            successes += 1
        else:
            print(f"❌ {f} 下载失败")
            failures += 1

    print()
    print("=" * 60)
    print(f"  下载完成：成功 {successes} / 跳过 {skipped} / 失败 {failures}")
    print("=" * 60)

    if failures > 0:
        print("\n失败的文件可手动下载：")
        print(f"  - https://www.modelscope.cn/datasets/{MODELSCOPE_REPO_ID}/files")
        print(f"  - https://huggingface.co/datasets/{HF_REPO_ID}/tree/main")
        return 1

    print("\n下一步:")
    print("  1) 端到端冒烟（CPU/MPS）：")
    print("       python scripts/smoke_test.py --clean")
    print("  2) 训练（A100/A800）：")
    print("       bash scripts/autodl_train.sh base   # 或 plus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
