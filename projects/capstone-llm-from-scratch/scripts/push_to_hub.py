"""
push_to_hub.py — 一键推送 ClearMind 模型到 HuggingFace Hub
==========================================================

把 ``release/<model_name>/`` 目录（``convert_to_qwen3.py`` 的产物）推送到
HuggingFace Hub 上的指定仓库。

前置：
  1. ``pip install huggingface_hub``
  2. ``huggingface-cli login``（或设环境变量 ``HF_TOKEN`` / 传 ``--token``）
  3. 已运行 ``scripts/convert_to_qwen3.py`` 生成 release 目录

用法:

  # 公开仓库
  python scripts/push_to_hub.py \\
      --model_dir release/clearmind-base \\
      --repo <your_username>/ClearMind-Base

  # 私有仓库
  python scripts/push_to_hub.py \\
      --model_dir release/clearmind-base \\
      --repo <your_username>/ClearMind-Base \\
      --private

  # 自定义提交信息
  python scripts/push_to_hub.py \\
      --model_dir release/clearmind-plus \\
      --repo <your_username>/ClearMind-Plus \\
      --commit "Initial release v1.0"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]
RECOMMENDED_FILES = [
    "generation_config.json",
    "README.md",
]


def _check_model_dir(model_dir: Path) -> bool:
    """检查 release 目录是否包含必要文件"""
    missing_required: list[str] = []
    missing_recommended: list[str] = []

    for f in REQUIRED_FILES:
        if not (model_dir / f).exists():
            missing_required.append(f)
    for f in RECOMMENDED_FILES:
        if not (model_dir / f).exists():
            missing_recommended.append(f)

    print(f"📂 检查 model_dir: {model_dir}")
    if missing_required:
        print(f"❌ 缺少必要文件: {missing_required}")
        print("   请先运行: python scripts/convert_to_qwen3.py ...")
        return False
    print(f"✅ 必要文件齐全（{len(REQUIRED_FILES)} 个）")
    if missing_recommended:
        print(f"⚠️  建议补全: {missing_recommended}")
    print()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="推送 ClearMind 模型到 HuggingFace Hub"
    )
    parser.add_argument(
        "--model_dir", "-m", required=True, help="release 目录（HF 仓库格式）"
    )
    parser.add_argument(
        "--repo", "-r", required=True, help="HF 仓库 id（如 username/ClearMind-Base）"
    )
    parser.add_argument(
        "--token", default=None, help="HF 访问 token（默认读 HF_TOKEN 环境变量或登录态）"
    )
    parser.add_argument(
        "--private", action="store_true", help="创建私有仓库（默认公开）"
    )
    parser.add_argument(
        "--commit", "-c", default="Upload model", help="commit message"
    )
    parser.add_argument(
        "--allow_patterns",
        nargs="+",
        default=None,
        help="只上传匹配这些 glob 的文件（默认全部）",
    )
    parser.add_argument(
        "--ignore_patterns",
        nargs="+",
        default=None,
        help="不上传匹配这些 glob 的文件",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只检查、不真上传",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        print("❌ 缺少 huggingface_hub。请运行：pip install huggingface_hub")
        return 1

    model_dir = Path(args.model_dir)
    if not os.path.isabs(args.model_dir):
        model_dir = PROJECT_ROOT / model_dir
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"❌ 模型目录不存在: {model_dir}")
        return 1

    if not _check_model_dir(model_dir):
        return 1

    # 总大小估算
    total_bytes = sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
    total_mb = total_bytes / 1024**2
    file_count = sum(1 for f in model_dir.iterdir() if f.is_file())
    print(f"📦 待上传：{file_count} 个文件，约 {total_mb:.1f} MB")

    print()
    print("=" * 60)
    print("  HuggingFace Hub 推送")
    print("=" * 60)
    print(f"  Repo:       {args.repo}")
    print(f"  Private:    {args.private}")
    print(f"  Commit:     {args.commit!r}")
    print(f"  Source:     {model_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] 跳过实际上传")
        return 0

    token = args.token or os.environ.get("HF_TOKEN")

    # ---- 创建仓库（如不存在）----
    api = HfApi(token=token)
    try:
        api.repo_info(args.repo, repo_type="model")
        print(f"📥 仓库已存在，将更新: {args.repo}")
    except RepositoryNotFoundError:
        print(f"🆕 创建新仓库: {args.repo}")
        try:
            create_repo(
                repo_id=args.repo,
                token=token,
                private=args.private,
                exist_ok=True,
                repo_type="model",
            )
        except HfHubHTTPError as e:
            print(f"❌ 创建仓库失败: {e}")
            return 1

    # ---- 上传 ----
    print(f"\n⬆️  开始上传 ({total_mb:.1f} MB)...")
    try:
        info = upload_folder(
            folder_path=str(model_dir),
            repo_id=args.repo,
            repo_type="model",
            commit_message=args.commit,
            token=token,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
        )
    except HfHubHTTPError as e:
        print(f"❌ 上传失败: {e}")
        return 1

    print()
    print("=" * 60)
    print("✅ 上传完成")
    print("=" * 60)
    print(f"  Repo URL:    https://huggingface.co/{args.repo}")
    print(f"  Commit:      {info}")
    print()
    print("加载示例：")
    print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.repo}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.repo}')")
    print()
    print("生态用法：")
    print(f"  - vllm: vllm serve {args.repo}")
    print(f"  - ollama: 先 ollama 转 GGUF：")
    print(f"      cd <llama.cpp>; python convert_hf_to_gguf.py {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
