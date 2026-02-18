"""
download_dataset.py — 真实数据集下载
=====================================

从 HuggingFace 下载真实的中英文训练数据集，用于大规模预训练。

支持的数据集:
  预训练:
    - SkyPile (中文网络语料, ~200GB 全量)
    - Wikipedia (中英文维基百科)
    - TinyStories (英文简短故事, ~2GB)

  SFT:
    - Alpaca-GPT4 (5.2万条高质量英文指令数据)
    - BELLE (50万条中文指令数据)

  DPO:
    - HH-RLHF (Anthropic 偏好数据, 英文)

使用方法:
  # 下载小规模数据 (快速验证)
  python scripts/download_dataset.py --scale small

  # 下载中等规模数据
  python scripts/download_dataset.py --scale medium

  # 下载大规模数据 (A100 训练)
  python scripts/download_dataset.py --scale large

  # 只下载预训练数据
  python scripts/download_dataset.py --stage pretrain --scale medium
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ============================================================
# 数据集配置
# ============================================================

DATASET_CONFIG = {
    "small": {
        "pretrain": {
            "source": "sample",  # 使用内置样例数据
            "max_samples": 5000,
        },
        "sft": {
            "source": "sample",
            "max_samples": 2000,
        },
        "dpo": {
            "source": "sample",
            "max_samples": 500,
        },
    },
    "medium": {
        "pretrain": {
            "source": "huggingface",
            "datasets": [
                {
                    "name": "roneneldan/TinyStories",
                    "split": "train",
                    "text_field": "text",
                    "max_samples": 100000,
                    "description": "TinyStories (英文短故事)",
                },
                {
                    "name": "liwu/MNBVC",
                    "subset": "wikipedia",
                    "split": "train",
                    "text_field": "段落",
                    "max_samples": 50000,
                    "description": "中文维基百科",
                    "streaming": True,
                },
            ],
        },
        "sft": {
            "source": "huggingface",
            "datasets": [
                {
                    "name": "shibing624/alpaca-zh",
                    "split": "train",
                    "max_samples": 20000,
                    "description": "中文 Alpaca 指令数据",
                },
            ],
        },
        "dpo": {
            "source": "sample",
            "max_samples": 2000,
        },
    },
    "large": {
        "pretrain": {
            "source": "huggingface",
            "datasets": [
                {
                    "name": "roneneldan/TinyStories",
                    "split": "train",
                    "text_field": "text",
                    "max_samples": 500000,
                    "description": "TinyStories (全量)",
                },
                {
                    "name": "liwu/MNBVC",
                    "subset": "wikipedia",
                    "split": "train",
                    "text_field": "段落",
                    "max_samples": 200000,
                    "description": "中文维基百科",
                    "streaming": True,
                },
                {
                    "name": "wikipedia",
                    "subset": "20220301.en",
                    "split": "train",
                    "text_field": "text",
                    "max_samples": 200000,
                    "description": "英文维基百科",
                },
            ],
        },
        "sft": {
            "source": "huggingface",
            "datasets": [
                {
                    "name": "shibing624/alpaca-zh",
                    "split": "train",
                    "max_samples": 50000,
                    "description": "中文 Alpaca 指令数据",
                },
                {
                    "name": "tatsu-lab/alpaca",
                    "split": "train",
                    "max_samples": 52000,
                    "description": "英文 Alpaca 指令数据",
                },
            ],
        },
        "dpo": {
            "source": "huggingface",
            "datasets": [
                {
                    "name": "Anthropic/hh-rlhf",
                    "split": "train",
                    "max_samples": 20000,
                    "description": "Anthropic HH-RLHF 偏好数据",
                },
            ],
        },
    },
}


def download_pretrain_data(config: dict, output_dir: str):
    """下载预训练数据"""
    output_path = os.path.join(output_dir, "pretrain", "pretrain_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if config["source"] == "sample":
        # 使用 01_prepare_data.py 的内置样例
        from scripts import prepare_sample_data

        prepare_sample_data.create_sample_pretrain_data(
            output_path, config["max_samples"]
        )
        return

    if not HAS_DATASETS:
        print("❌ 需要安装 datasets 库: pip install datasets")
        sys.exit(1)

    total_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ds_config in config["datasets"]:
            name = ds_config["name"]
            desc = ds_config.get("description", name)
            max_samples = ds_config.get("max_samples", 100000)
            text_field = ds_config.get("text_field", "text")
            streaming = ds_config.get("streaming", False)

            print(f"\n  📥 下载: {desc}")
            print(f"     来源: {name}")
            print(f"     目标: {max_samples:,} 条")

            try:
                kwargs = {
                    "path": name,
                    "split": ds_config.get("split", "train"),
                    "streaming": streaming,
                }
                if "subset" in ds_config:
                    kwargs["name"] = ds_config["subset"]

                dataset = load_dataset(**kwargs)

                count = 0
                for item in dataset:
                    text = item.get(text_field, "")
                    if not text or len(text.strip()) < 50:
                        continue

                    # 截断过长文本
                    if len(text) > 5000:
                        text = text[:5000]

                    f.write(
                        json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n"
                    )
                    count += 1
                    total_count += 1

                    if count >= max_samples:
                        break

                    if count % 10000 == 0:
                        print(f"     已下载: {count:,}/{max_samples:,}")

                print(f"     ✅ 完成: {count:,} 条")

            except Exception as e:
                print(f"     ⚠️  下载失败: {e}")
                print(f"     跳过此数据集, 继续...")
                continue

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  📊 预训练数据总计: {total_count:,} 条, {size_mb:.1f} MB")


def download_sft_data(config: dict, output_dir: str):
    """下载 SFT 数据"""
    output_path = os.path.join(output_dir, "sft", "sft_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if config["source"] == "sample":
        from scripts import prepare_sample_data

        prepare_sample_data.create_sample_sft_data(output_path, config["max_samples"])
        return

    if not HAS_DATASETS:
        print("❌ 需要安装 datasets 库: pip install datasets")
        sys.exit(1)

    total_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ds_config in config["datasets"]:
            name = ds_config["name"]
            desc = ds_config.get("description", name)
            max_samples = ds_config.get("max_samples", 50000)

            print(f"\n  📥 下载: {desc}")

            try:
                dataset = load_dataset(
                    name,
                    split=ds_config.get("split", "train"),
                )

                count = 0
                for item in dataset:
                    # 统一格式为 Alpaca 格式
                    sample = {}
                    if "instruction" in item:
                        sample = {
                            "instruction": item.get("instruction", ""),
                            "input": item.get("input", ""),
                            "output": item.get("output", ""),
                        }
                    elif "conversations" in item:
                        sample = {"conversations": item["conversations"]}
                    else:
                        continue

                    if not sample.get("instruction") and not sample.get(
                        "conversations"
                    ):
                        continue

                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
                    total_count += 1

                    if count >= max_samples:
                        break

                print(f"     ✅ 完成: {count:,} 条")

            except Exception as e:
                print(f"     ⚠️  下载失败: {e}")
                continue

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  📊 SFT 数据总计: {total_count:,} 条, {size_mb:.1f} MB")


def download_dpo_data(config: dict, output_dir: str):
    """下载 DPO 数据"""
    output_path = os.path.join(output_dir, "dpo", "dpo_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if config["source"] == "sample":
        from scripts import prepare_sample_data

        prepare_sample_data.create_sample_dpo_data(output_path, config["max_samples"])
        return

    if not HAS_DATASETS:
        print("❌ 需要安装 datasets 库: pip install datasets")
        sys.exit(1)

    total_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ds_config in config["datasets"]:
            name = ds_config["name"]
            desc = ds_config.get("description", name)
            max_samples = ds_config.get("max_samples", 20000)

            print(f"\n  📥 下载: {desc}")

            try:
                dataset = load_dataset(name, split=ds_config.get("split", "train"))

                count = 0
                for item in dataset:
                    # HH-RLHF 格式: chosen/rejected 是完整对话
                    if "chosen" in item and "rejected" in item:
                        chosen = item["chosen"]
                        rejected = item["rejected"]

                        # 提取 prompt (两者共享)
                        # HH-RLHF 格式: "\n\nHuman: ...\n\nAssistant: ..."
                        prompt = ""
                        chosen_response = chosen
                        rejected_response = rejected

                        if "\n\nAssistant:" in chosen:
                            parts = chosen.rsplit("\n\nAssistant:", 1)
                            prompt = parts[0].replace("\n\nHuman:", "").strip()
                            chosen_response = (
                                parts[1].strip() if len(parts) > 1 else chosen
                            )

                        if "\n\nAssistant:" in rejected:
                            parts = rejected.rsplit("\n\nAssistant:", 1)
                            rejected_response = (
                                parts[1].strip() if len(parts) > 1 else rejected
                            )

                        sample = {
                            "prompt": prompt,
                            "chosen": chosen_response,
                            "rejected": rejected_response,
                        }
                    else:
                        continue

                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
                    total_count += 1

                    if count >= max_samples:
                        break

                print(f"     ✅ 完成: {count:,} 条")

            except Exception as e:
                print(f"     ⚠️  下载失败: {e}")
                continue

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  📊 DPO 数据总计: {total_count:,} 条, {size_mb:.1f} MB")


def create_tokenizer_corpus(data_dir: str):
    """从已下载的数据中创建分词器训练语料"""
    output_path = os.path.join(data_dir, "tokenizer_corpus.txt")

    texts = []

    pretrain_path = os.path.join(data_dir, "pretrain", "pretrain_data.jsonl")
    if os.path.exists(pretrain_path):
        with open(pretrain_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100000:  # 最多 10 万条用于训练分词器
                    break
                obj = json.loads(line.strip())
                texts.append(obj.get("text", ""))

    sft_path = os.path.join(data_dir, "sft", "sft_data.jsonl")
    if os.path.exists(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                texts.append(obj.get("instruction", ""))
                texts.append(obj.get("output", ""))

    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            if text and text.strip():
                f.write(text.strip() + "\n")

    print(f"✅ 分词器语料: {output_path} ({len(texts):,} 行)")


def main():
    parser = argparse.ArgumentParser(description="ClearMind 数据集下载")
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="数据规模",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "pretrain", "sft", "dpo"],
        help="只下载指定阶段的数据",
    )
    parser.add_argument("--data_dir", type=str, default="data", help="数据存储目录")
    args = parser.parse_args()

    config = DATASET_CONFIG[args.scale]

    print("=" * 60)
    print(f"ClearMind 数据下载 (规模: {args.scale})")
    print("=" * 60)

    if args.scale != "small" and not HAS_DATASETS:
        print("\n⚠️  medium/large 规模需要 datasets 库")
        print("   安装: pip install datasets")
        print("   或使用 small 规模: --scale small")
        sys.exit(1)

    # 下载数据
    if args.stage in ("all", "pretrain"):
        print(f"\n📦 Step 1: 预训练数据")
        download_pretrain_data(config["pretrain"], args.data_dir)

    if args.stage in ("all", "sft"):
        print(f"\n📦 Step 2: SFT 数据")
        download_sft_data(config["sft"], args.data_dir)

    if args.stage in ("all", "dpo"):
        print(f"\n📦 Step 3: DPO 数据")
        download_dpo_data(config["dpo"], args.data_dir)

    # 分词器语料
    if args.stage == "all":
        print(f"\n📦 Step 4: 分词器语料")
        create_tokenizer_corpus(args.data_dir)

    # 显示目录结构
    print(f"\n{'=' * 60}")
    print("✅ 数据下载完成!")
    print(f"\n📁 数据目录:")
    for root, dirs, files in os.walk(args.data_dir):
        level = root.replace(args.data_dir, "").count(os.sep)
        indent = "  " * level
        print(f"  {indent}{os.path.basename(root)}/")
        for file in files:
            filepath = os.path.join(root, file)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  {'  ' * (level + 1)}{file} ({size_mb:.1f} MB)")

    print(
        f"\n💡 下一步: python scripts/02_train_tokenizer.py --config configs/{args.scale}.yaml"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
