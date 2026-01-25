"""
情感分析系统 - 主入口
====================

Phase 11 实战项目：多模型情感分析。
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="📝 情感分析系统 (TextCNN / BiLSTM / BERT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速演示
  python main.py demo

  # 快速训练 (TextCNN)
  python main.py train --model textcnn --quick

  # 训练 BiLSTM
  python main.py train --model lstm --epochs 10

  # 评估模型
  python main.py eval --model textcnn

  # 推理
  python main.py predict --text "I love this movie!"

  # 交互模式
  python main.py predict --interactive

  # 系统信息
  python main.py info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # demo
    demo_parser = subparsers.add_parser("demo", help="快速演示")

    # train
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--model",
        type=str,
        default="textcnn",
        choices=["textcnn", "lstm", "bert"],
        help="模型类型",
    )
    train_parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    train_parser.add_argument("--lr", type=float, default=None, help="学习率")
    train_parser.add_argument("--quick", action="store_true", help="快速训练")

    # eval
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument(
        "--model",
        type=str,
        default="textcnn",
        choices=["textcnn", "lstm", "bert"],
        help="模型类型",
    )
    eval_parser.add_argument("--model-path", type=str, default=None, help="模型路径")

    # predict
    predict_parser = subparsers.add_parser("predict", help="推理预测")
    predict_parser.add_argument(
        "--model",
        type=str,
        default="textcnn",
        choices=["textcnn", "lstm", "bert"],
        help="模型类型",
    )
    predict_parser.add_argument("--text", type=str, default=None, help="输入文本")
    predict_parser.add_argument(
        "--interactive", "-i", action="store_true", help="交互模式"
    )

    # info
    info_parser = subparsers.add_parser("info", help="显示系统信息")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()


def run_demo():
    """运行演示"""
    print("=" * 60)
    print("📝 情感分析系统演示")
    print("=" * 60)

    try:
        from config import DEVICE, LABELS
        from model import TextCNN, BiLSTMClassifier, count_parameters
        from dataset import prepare_dataset

        # 设备信息
        print("\n1️⃣  系统环境")
        print("-" * 40)
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"设备: {DEVICE}")

        # 数据集
        print("\n2️⃣  准备数据集")
        print("-" * 40)
        train_data, val_data, vocab = prepare_dataset(num_samples=500)

        # 模型信息
        print("\n3️⃣  模型信息")
        print("-" * 40)

        textcnn = TextCNN(vocab_size=len(vocab))
        lstm = BiLSTMClassifier(vocab_size=len(vocab))

        print(f"TextCNN 参数量: {count_parameters(textcnn):,}")
        print(f"BiLSTM 参数量: {count_parameters(lstm):,}")

        # 标签
        print("\n4️⃣  情感标签")
        print("-" * 40)
        for idx, label in LABELS.items():
            print(f"  {idx}: {label}")

        print("\n5️⃣  可用命令")
        print("-" * 40)
        print("""
  🚀 训练 TextCNN:
     python main.py train --model textcnn --quick

  🚀 训练 BiLSTM:
     python main.py train --model lstm --quick

  📊 评估模型:
     python main.py eval --model textcnn

  💬 情感预测:
     python main.py predict --text "I love this movie!"
        """)

        print("\n" + "=" * 60)
        print("✅ 演示完成!")
        print("=" * 60)

    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")


def run_train(args):
    """运行训练"""
    from train import train, quick_train

    if args.quick:
        quick_train(model_type=args.model)
    else:
        train(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )


def run_eval(args):
    """运行评估"""
    from evaluate import evaluate

    evaluate(model_type=args.model, model_path=args.model_path)


def run_predict(args):
    """运行推理"""
    from inference import predict_interactive, demo_inference, load_model, predict
    from config import LABELS

    if args.interactive:
        predict_interactive(model_type=args.model)
    elif args.text:
        model, vocab, tokenizer = load_model(args.model)
        if model is not None:
            label, confidence = predict(args.text, model, vocab, tokenizer, args.model)
            print(f"文本: {args.text}")
            print(f"情感: {LABELS[label]}")
            print(f"置信度: {confidence:.2%}")
    else:
        demo_inference()


def show_info():
    """显示系统信息"""
    import torch
    from config import MODELS_DIR

    print("=" * 60)
    print("📊 系统信息")
    print("=" * 60)

    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if hasattr(torch.backends, "mps"):
        print(f"MPS 可用: {torch.backends.mps.is_available()}")

    print("\n项目路径:")
    print(f"  {Path(__file__).parent}")

    print("\n已保存模型:")
    for model_type in ["textcnn", "lstm", "bert"]:
        model_path = MODELS_DIR / f"best_{model_type}.pth"
        if model_path.exists():
            print(f"  ✓ best_{model_type}.pth")

    if not any(
        (MODELS_DIR / f"best_{m}.pth").exists() for m in ["textcnn", "lstm", "bert"]
    ):
        print("  ⚠ 暂无模型，请先训练")


if __name__ == "__main__":
    main()
