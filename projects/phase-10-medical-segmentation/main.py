"""
医学图像分割系统 - 主入口
==========================

Phase 10 实战项目：使用 U-Net 进行肺部分割。
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="🏥 医学图像分割系统 (肺部 X 光片)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速演示
  python main.py demo

  # 快速训练 (5 轮)
  python main.py train --quick

  # 完整训练
  python main.py train --epochs 50

  # 评估模型
  python main.py eval

  # 推理
  python main.py predict --source image.png

  # 系统信息
  python main.py info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ========== demo 子命令 ==========
    demo_parser = subparsers.add_parser("demo", help="快速演示")

    # ========== train 子命令 ==========
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data", type=str, default=None, help="数据目录")
    train_parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "attention_unet"],
        help="模型类型",
    )
    train_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=8, help="批量大小")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    train_parser.add_argument(
        "--quick", action="store_true", help="快速训练模式 (5 轮)"
    )

    # ========== eval 子命令 ==========
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    eval_parser.add_argument("--data", type=str, default=None, help="数据目录")
    eval_parser.add_argument(
        "--postprocess", action="store_true", help="评估后处理效果"
    )

    # ========== predict 子命令 ==========
    predict_parser = subparsers.add_parser("predict", help="推理")
    predict_parser.add_argument(
        "--source", type=str, required=True, help="图像路径或目录"
    )
    predict_parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    predict_parser.add_argument("--output", type=str, default=None, help="输出路径")
    predict_parser.add_argument("--show", action="store_true", help="显示结果")

    # ========== info 子命令 ==========
    info_parser = subparsers.add_parser("info", help="显示系统信息")

    # 解析参数
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
    print("🏥 医学图像分割系统演示")
    print("=" * 60)

    try:
        from config import DEVICE, MODELS_DIR
        from model import UNet, count_parameters
        from dataset import download_montgomery_dataset, get_dataloaders
        from utils import print_device_info

        # 设备信息
        print("\n1️⃣  系统环境")
        print("-" * 40)
        print_device_info()

        # 准备数据
        print("\n2️⃣  准备数据集")
        print("-" * 40)
        data_dir = download_montgomery_dataset()

        # 测试数据加载
        train_loader, val_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,
        )
        print(f"训练集批次: {len(train_loader)}")
        print(f"验证集批次: {len(val_loader)}")

        # 测试模型
        print("\n3️⃣  模型信息")
        print("-" * 40)
        model = UNet(n_channels=1, n_classes=1)
        print(f"模型: U-Net")
        print(f"参数量: {count_parameters(model):,}")

        # 测试前向传播
        import torch

        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            y = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {y.shape}")

        print("\n4️⃣  可用命令")
        print("-" * 40)
        print("""
  🚀 快速训练:
     python main.py train --quick

  📊 完整训练:
     python main.py train --epochs 50

  📈 评估模型:
     python main.py eval

  🔍 推理预测:
     python main.py predict --source <图像路径>
        """)

        print("\n" + "=" * 60)
        print("✅ 演示完成! 环境检测通过")
        print("=" * 60)

    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print(
            "请确保已安装所有依赖: pip install torch torchvision pillow matplotlib tqdm"
        )


def run_train(args):
    """运行训练"""
    from train import train, quick_train

    if args.quick:
        quick_train(data_dir=args.data, model_name=args.model)
    else:
        train(
            data_dir=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )


def run_eval(args):
    """运行评估"""
    from evaluate import evaluate, evaluate_with_postprocess

    if args.postprocess:
        evaluate_with_postprocess(model_path=args.model, data_dir=args.data)
    else:
        evaluate(model_path=args.model, data_dir=args.data)


def run_predict(args):
    """运行推理"""
    from inference import predict_and_visualize, predict_batch
    from pathlib import Path

    source = Path(args.source)
    if source.is_dir():
        predict_batch(source, args.output, args.model)
    else:
        predict_and_visualize(
            source,
            output_path=args.output,
            model_path=args.model,
            show=args.show,
        )


def show_info():
    """显示系统信息"""
    from utils import print_device_info

    print("=" * 60)
    print("📊 系统信息")
    print("=" * 60)

    print_device_info()

    print("\n项目路径:")
    print(f"  {Path(__file__).parent}")

    # 检查模型
    from config import MODELS_DIR

    best_model = MODELS_DIR / "best_model.pth"
    final_model = MODELS_DIR / "final_model.pth"

    print("\n已保存模型:")
    if best_model.exists():
        print(f"  ✓ {best_model.name}")
    if final_model.exists():
        print(f"  ✓ {final_model.name}")
    if not best_model.exists() and not final_model.exists():
        print("  ⚠ 暂无模型，请先训练")


if __name__ == "__main__":
    main()
