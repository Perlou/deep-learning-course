"""
推理脚本
========

对新文本进行情感分析推理。
"""

import argparse
from pathlib import Path

import torch

from config import DEVICE, MODELS_DIR, LABELS, NUM_CLASSES
from model import create_model
from dataset import text_to_indices, tokenize


def load_model(model_type="textcnn", model_path=None, device=None):
    """加载训练好的模型"""
    device = device or DEVICE

    if model_path is None:
        model_path = MODELS_DIR / f"best_{model_type}.pth"
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"⚠ 模型不存在: {model_path}")
        return None, None, None

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    vocab = checkpoint.get("vocab")

    # 创建模型
    if model_type == "textcnn":
        from config import TEXTCNN_CONFIG

        model = create_model(
            model_type,
            vocab_size=len(vocab) if vocab else 20000,
            num_classes=NUM_CLASSES,
            **TEXTCNN_CONFIG,
        )
    elif model_type == "lstm":
        from config import LSTM_CONFIG

        model = create_model(
            model_type,
            vocab_size=len(vocab) if vocab else 20000,
            num_classes=NUM_CLASSES,
            **LSTM_CONFIG,
        )
    elif model_type == "bert":
        from config import BERT_CONFIG

        model = create_model(model_type, num_classes=NUM_CLASSES, **BERT_CONFIG)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = None
    if model_type == "bert":
        try:
            from transformers import AutoTokenizer
            from config import BERT_CONFIG

            tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["model_name"])
        except ImportError:
            pass

    return model, vocab, tokenizer


@torch.no_grad()
def predict(text, model, vocab=None, tokenizer=None, model_type="textcnn", device=None):
    """
    预测单条文本的情感

    Args:
        text: 输入文本
        model: 模型
        vocab: 词表 (textcnn/lstm)
        tokenizer: tokenizer (bert)
        model_type: 模型类型
        device: 设备

    Returns:
        label: 预测标签
        confidence: 置信度
    """
    device = device or DEVICE

    if model_type == "bert":
        if tokenizer is None:
            raise ValueError("BERT 模型需要 tokenizer")

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
    else:
        if vocab is None:
            raise ValueError("TextCNN/LSTM 模型需要 vocab")

        indices = text_to_indices(text, vocab)
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

        outputs = model(input_tensor)

    probs = torch.softmax(outputs, dim=1)
    confidence, pred = probs.max(dim=1)

    label = pred.item()
    confidence = confidence.item()

    return label, confidence


def predict_interactive(model_type="textcnn"):
    """交互式预测"""
    print("=" * 60)
    print("💬 情感分析交互模式")
    print("=" * 60)
    print(f"模型: {model_type}")
    print("输入 'quit' 或 'q' 退出")
    print("=" * 60)

    model, vocab, tokenizer = load_model(model_type)
    if model is None:
        return

    while True:
        try:
            text = input("\n请输入文本: ").strip()

            if text.lower() in ["quit", "q", "exit"]:
                print("再见!")
                break

            if not text:
                continue

            label, confidence = predict(text, model, vocab, tokenizer, model_type)

            sentiment = LABELS[label]
            print(f"情感: {sentiment}")
            print(f"置信度: {confidence:.2%}")

        except KeyboardInterrupt:
            print("\n再见!")
            break


def predict_batch(texts, model_type="textcnn"):
    """批量预测"""
    model, vocab, tokenizer = load_model(model_type)
    if model is None:
        return []

    results = []
    for text in texts:
        label, confidence = predict(text, model, vocab, tokenizer, model_type)
        results.append(
            {
                "text": text,
                "label": label,
                "sentiment": LABELS[label],
                "confidence": confidence,
            }
        )

    return results


def demo_inference():
    """推理演示"""
    print("=" * 60)
    print("💬 情感分析推理演示")
    print("=" * 60)

    # 检查模型
    model_path = MODELS_DIR / "best_textcnn.pth"
    if not model_path.exists():
        print("\n⚠ 模型不存在，请先训练:")
        print("   python main.py train --quick")
        return

    model, vocab, _ = load_model("textcnn")
    if model is None:
        return

    # 示例文本
    test_texts = [
        "I love this movie, it's amazing!",
        "This is a terrible film, very disappointing.",
        "Great acting and wonderful story!",
        "Boring and waste of time.",
    ]

    print("\n📝 示例预测:\n")
    for text in test_texts:
        label, confidence = predict(text, model, vocab, None, "textcnn")
        sentiment = LABELS[label]
        print(f"文本: {text}")
        print(f"情感: {sentiment} (置信度: {confidence:.2%})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情感分析推理")
    parser.add_argument(
        "--model",
        type=str,
        default="textcnn",
        choices=["textcnn", "lstm", "bert"],
        help="模型类型",
    )
    parser.add_argument("--text", type=str, default=None, help="输入文本")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--demo", action="store_true", help="演示模式")

    args = parser.parse_args()

    if args.demo:
        demo_inference()
    elif args.interactive:
        predict_interactive(model_type=args.model)
    elif args.text:
        model, vocab, tokenizer = load_model(args.model)
        if model is not None:
            label, confidence = predict(args.text, model, vocab, tokenizer, args.model)
            print(f"文本: {args.text}")
            print(f"情感: {LABELS[label]}")
            print(f"置信度: {confidence:.2%}")
    else:
        print("用法:")
        print("  python inference.py --demo")
        print("  python inference.py --text 'I love this movie!'")
        print("  python inference.py --interactive")
