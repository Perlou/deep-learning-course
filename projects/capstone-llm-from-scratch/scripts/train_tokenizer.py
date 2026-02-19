"""
train_tokenizer.py — 分词器训练脚本
=====================================

使用 sentencepiece 训练 BPE 分词器。

使用方法:
  python scripts/train_tokenizer.py
  python scripts/train_tokenizer.py --vocab_size 8000 --config configs/small.yaml

前置步骤:
  python scripts/prepare_data.py
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.tokenizer import ClearMindTokenizer


def main():
    parser = argparse.ArgumentParser(description="ClearMind 分词器训练")
    parser.add_argument(
        "--config", type=str, default="configs/small.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--corpus", type=str, default="data/tokenizer_corpus.txt", help="训练语料路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/tokenizer", help="输出目录"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=None, help="词表大小 (覆盖配置文件)"
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    vocab_size = args.vocab_size or config["tokenizer"]["vocab_size"]
    character_coverage = config["tokenizer"].get("character_coverage", 0.9995)

    print("=" * 60)
    print("ClearMind 分词器训练")
    print("=" * 60)
    print(f"\n📄 训练语料: {args.corpus}")
    print(f"📊 词表大小: {vocab_size}")
    print(f"📊 字符覆盖率: {character_coverage}")
    print(f"📁 输出目录: {args.output_dir}")

    # 检查训练语料
    if not os.path.exists(args.corpus):
        print(f"\n❌ 训练语料不存在: {args.corpus}")
        print("   请先运行: python scripts/prepare_data.py")
        sys.exit(1)

    if os.path.getsize(args.corpus) == 0:
        print(f"\n❌ 训练语料为空: {args.corpus}")
        print(
            "   可能是数据下载失败, 请检查网络后重新运行: python scripts/prepare_data.py"
        )
        sys.exit(1)

    import re

    # 训练分词器 (语料不足时自动降低 vocab_size)
    print("\n🔄 开始训练...")
    model_prefix = os.path.join(args.output_dir, "tokenizer")

    try:
        ClearMindTokenizer.train(
            input_file=args.corpus,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
        )
    except RuntimeError as e:
        # 解析 sentencepiece 错误中的实际上限
        match = re.search(r"value <= (\d+)", str(e))
        if match:
            max_vocab = int(match.group(1))
            adjusted = max(256, int(max_vocab * 0.8))  # 留 20% 余量
            print(f"\n⚠️  语料不足以支撑 vocab_size={vocab_size}, 自动调整为 {adjusted}")
            print("   提示: 使用更大的语料 (--scale medium/large) 可支撑更大的词表")
            vocab_size = adjusted
            ClearMindTokenizer.train(
                input_file=args.corpus,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=character_coverage,
            )
        else:
            raise

    # 验证
    print("\n🔍 验证分词器...")
    model_path = f"{model_prefix}.model"
    tokenizer = ClearMindTokenizer(model_path)

    print(f"  实际词表大小: {tokenizer.vocab_size}")
    print(f"  BOS ID: {tokenizer.bos_id}")
    print(f"  EOS ID: {tokenizer.eos_id}")
    print(f"  UNK ID: {tokenizer.unk_id}")

    # 测试编解码
    test_texts = [
        "Hello, world!",
        "深度学习是人工智能的核心技术。",
        "The Transformer architecture is amazing! 注意力机制很重要。",
    ]

    print("\n📝 编解码测试:")
    for text in test_texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        pieces = tokenizer.tokenize(text)
        decoded = tokenizer.decode(ids)
        print(f"  原文:   {text}")
        print(f"  Tokens: {pieces[:10]}{'...' if len(pieces) > 10 else ''}")
        print(f"  长度:   {len(ids)} tokens")
        print(f"  解码:   {decoded}")
        print()

    print(f"{'=' * 60}")
    print("✅ 分词器训练完成!")
    print(f"\n💡 下一步: python scripts/train.py --stage pretrain")
    print("=" * 60)


if __name__ == "__main__":
    main()
