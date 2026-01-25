"""
主入口
======

项目主入口，支持训练、评估和对话模式。

使用方法：
    python main.py train   # 训练模型
    python main.py eval    # 评估模型
    python main.py chat    # 交互对话
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="LLM 微调项目 - Phase 12 实战",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python main.py train --epochs 3           # 训练3轮
    python main.py eval                       # 评估模型
    python main.py chat                       # 交互对话
    python main.py demo                       # 运行演示
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="学习率")

    # 评估命令
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model", type=str, default=None, help="模型路径")

    # 对话命令
    chat_parser = subparsers.add_parser("chat", help="交互对话")
    chat_parser.add_argument("--model", type=str, default=None, help="模型路径")

    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="运行演示")

    args = parser.parse_args()

    if args.command == "train":
        from config import get_config
        from train import train

        config = get_config()
        config.training.num_epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.lr

        train(config)

    elif args.command == "eval":
        from evaluate import evaluate_model

        evaluate_model(model_path=args.model)

    elif args.command == "chat":
        from inference import main as inference_main

        inference_main(model_path=args.model)

    elif args.command == "demo":
        run_demo()

    else:
        # 默认运行演示
        parser.print_help()
        print("\n" + "=" * 60)
        print("运行演示...")
        print("=" * 60)
        run_demo()


def run_demo():
    """运行演示，展示项目各个模块"""
    print("\n" + "=" * 60)
    print("LLM 微调项目演示")
    print("=" * 60)

    # 1. 配置
    print("\n[1/5] 加载配置...")
    from config import get_config, print_config

    config = get_config()
    print_config(config)

    # 2. 分词器
    print("\n[2/5] 测试分词器...")
    from dataset import SimpleTokenizer, apply_chatml_template, SAMPLE_DATA

    tokenizer = SimpleTokenizer(config.model.vocab_size)
    sample_text = "你好，这是一个测试。Hello, this is a test."
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    print(f"  原文: {sample_text}")
    print(f"  Token数: {len(tokens)}")
    print(f"  解码: {decoded}")

    # 3. ChatML 模板
    print("\n[3/5] 测试 ChatML 模板...")
    sample = SAMPLE_DATA[0]
    formatted = apply_chatml_template(sample["messages"], config.data)
    print(f"  格式化结果:\n{formatted[:200]}...")

    # 4. 模型
    print("\n[4/5] 创建模型...")
    from model import create_model

    model = create_model(config.model, use_lora=True, lora_config=config.lora)
    model.print_trainable_parameters()

    # 5. 前向传播测试
    print("\n[5/5] 测试前向传播...")
    import torch

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出 logits 形状: {outputs['logits'].shape}")
    print(f"  损失: {outputs['loss'].item():.4f}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("""
下一步：
    1. 训练模型: python main.py train --epochs 3
    2. 评估模型: python main.py eval
    3. 交互对话: python main.py chat
    """)


if __name__ == "__main__":
    main()
