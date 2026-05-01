"""
push_to_modelscope.py — 一键推送 ClearMind 模型到 ModelScope
=============================================================

把 ``release/<model_name>/`` 目录（``convert_to_qwen3.py`` 的产物）推送到
ModelScope 上的指定仓库。国内用户访问 ModelScope 比 HuggingFace 更稳定。

前置：
  1. ``pip install modelscope``
  2. 在 https://www.modelscope.cn 注册账号 + 获取 access token
  3. 设环境变量 ``MODELSCOPE_API_TOKEN`` 或传 ``--token``
  4. 已运行 ``scripts/convert_to_qwen3.py`` 生成 release 目录

用法:

  python scripts/push_to_modelscope.py \\
      --model_dir release/clearmind-base \\
      --repo <your_username>/ClearMind-Base

  # 私有仓库
  python scripts/push_to_modelscope.py \\
      --model_dir release/clearmind-plus \\
      --repo <your_username>/ClearMind-Plus \\
      --visibility private
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


def _check_model_dir(model_dir: Path) -> bool:
    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    print(f"📂 检查 model_dir: {model_dir}")
    if missing:
        print(f"❌ 缺少必要文件: {missing}")
        print("   请先运行: python scripts/convert_to_qwen3.py ...")
        return False
    print(f"✅ 必要文件齐全（{len(REQUIRED_FILES)} 个）")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="推送 ClearMind 模型到 ModelScope"
    )
    parser.add_argument("--model_dir", "-m", required=True, help="release 目录")
    parser.add_argument(
        "--repo", "-r", required=True, help="ModelScope 仓库 id（如 username/ClearMind-Base）"
    )
    parser.add_argument("--token", default=None, help="ModelScope access token")
    parser.add_argument(
        "--visibility",
        choices=["public", "private"],
        default="public",
        help="仓库可见性",
    )
    parser.add_argument(
        "--license",
        default="apache-2.0",
        help="模型 license（apache-2.0 / mit / cc-by-nc-4.0 等）",
    )
    parser.add_argument(
        "--commit", "-c", default="Upload model", help="commit message"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只检查、不真上传",
    )
    args = parser.parse_args()

    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.constants import Licenses, ModelVisibility
        from modelscope.hub.errors import NotExistError
    except ImportError:
        print("❌ 缺少 modelscope。请运行：pip install modelscope")
        return 1

    model_dir = Path(args.model_dir)
    if not os.path.isabs(args.model_dir):
        model_dir = PROJECT_ROOT / model_dir
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"❌ 模型目录不存在: {model_dir}")
        return 1

    if not _check_model_dir(model_dir):
        return 1

    total_bytes = sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
    total_mb = total_bytes / 1024**2
    file_count = sum(1 for f in model_dir.iterdir() if f.is_file())
    print(f"📦 待上传：{file_count} 个文件，约 {total_mb:.1f} MB")

    print()
    print("=" * 60)
    print("  ModelScope 推送")
    print("=" * 60)
    print(f"  Repo:       {args.repo}")
    print(f"  Visibility: {args.visibility}")
    print(f"  License:    {args.license}")
    print(f"  Commit:     {args.commit!r}")
    print(f"  Source:     {model_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] 跳过实际上传")
        return 0

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        print("\n❌ 未提供 token。请设环境变量 MODELSCOPE_API_TOKEN 或传 --token")
        print("   获取 token: https://www.modelscope.cn/my/myaccesstoken")
        return 1

    # ModelScope API 登录 + 创建/更新仓库
    api = HubApi()
    try:
        api.login(token)
    except Exception as e:
        print(f"❌ ModelScope 登录失败: {e}")
        return 1

    # ---- 创建/检查仓库 ----
    visibility_const = (
        ModelVisibility.PUBLIC if args.visibility == "public" else ModelVisibility.PRIVATE
    )
    license_const = getattr(Licenses, args.license.upper().replace("-", "_"), None)
    if license_const is None:
        # 未在 Licenses 枚举中找到，直接传字符串
        license_const = args.license

    try:
        api.get_model(model_id=args.repo)
        print(f"📥 仓库已存在，将更新: {args.repo}")
    except (NotExistError, Exception):
        print(f"🆕 创建新仓库: {args.repo}")
        try:
            api.create_model(
                model_id=args.repo,
                visibility=visibility_const,
                license=license_const,
                chinese_name=args.repo.split("/")[-1],
            )
        except Exception as e:
            print(f"⚠️  创建可能失败（如已存在则忽略）: {e}")

    # ---- 上传 ----
    print(f"\n⬆️  开始上传 ({total_mb:.1f} MB)...")
    try:
        api.push_model(
            model_id=args.repo,
            model_dir=str(model_dir),
            commit_message=args.commit,
        )
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return 1

    print()
    print("=" * 60)
    print("✅ 上传完成")
    print("=" * 60)
    print(f"  Repo URL:  https://www.modelscope.cn/models/{args.repo}")
    print()
    print("加载示例：")
    print("  from modelscope import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.repo}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.repo}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
