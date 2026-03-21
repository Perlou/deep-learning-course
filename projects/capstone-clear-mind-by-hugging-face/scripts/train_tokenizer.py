"""
train_tokenizer.py — 训练 ClearMind BPE Tokenizer
===================================================

使用 HuggingFace tokenizers 库训练 ByteLevel BPE tokenizer。
读取 YAML 配置中的 tokenizer.vocab_size 和 tokenizer.character_coverage。

使用方法:
  python scripts/train_tokenizer.py --config configs/tiny.yaml
  python scripts/train_tokenizer.py --config configs/tiny.yaml --corpus data/tokenizer_corpus.txt
"""

import sys
import argparse
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import ClearMindTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train ClearMind BPE Tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file path (e.g. configs/tiny.yaml)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/tokenizer_corpus.txt",
        help="Training corpus file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/tokenizer",
        help="Output directory for trained tokenizer",
    )
    args = parser.parse_args()

    # 读取配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tokenizer_config = config.get("tokenizer", {})
    vocab_size = tokenizer_config.get("vocab_size", 2000)
    character_coverage = tokenizer_config.get("character_coverage", 0.9995)

    # 检查语料文件
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus file not found: {corpus_path}")
        print("Run `python scripts/prepare_data.py` first to generate training data.")
        sys.exit(1)

    # 训练
    print("=" * 60)
    print("ClearMind Tokenizer Training")
    print("=" * 60)
    print(f"  Config: {config_path}")
    print(f"  Corpus: {corpus_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Output: {args.output}")
    print()

    tokenizer = ClearMindTokenizer.train(
        corpus_path=str(corpus_path),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
    )

    # 保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))

    # 验证
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Special tokens:")
    for name in ["unk_token", "bos_token", "eos_token", "pad_token"]:
        token = getattr(tokenizer, name)
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    {name}: '{token}' (id={token_id})")

    # 编码示例
    print(f"\nEncoding examples:")
    test_texts = [
        "Hello, world!",
        "深度学习是机器学习的一个子领域。",
        "Transformer architecture revolutionized NLP.",
    ]
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  '{text}'")
        print(f"    -> ids ({len(ids)}): {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"    -> decoded: '{decoded}'")

    # Chat template 示例
    print(f"\nChat template example:")
    messages = [
        {"role": "user", "content": "What is deep learning?"},
        {"role": "assistant", "content": "Deep learning is a subset of ML."},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"  {repr(formatted)}")

    print(f"\nSaved to: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
