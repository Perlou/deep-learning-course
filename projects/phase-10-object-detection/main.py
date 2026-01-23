"""
目标检测系统 - 主入口
======================

Phase 10 实战项目：使用 YOLOv8 进行目标检测。
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 目标检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速推理演示
  python main.py demo

  # 图片检测
  python main.py predict --source image.jpg

  # 摄像头实时检测
  python main.py predict --source 0

  # 训练模型
  python main.py train --data coco128.yaml --epochs 100

  # 评估模型
  python main.py eval --model runs/train/weights/best.pt --data coco128.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ==========  demo 子命令 ==========
    demo_parser = subparsers.add_parser("demo", help="快速演示")

    # ==========  predict 子命令 ==========
    predict_parser = subparsers.add_parser("predict", help="目标检测推理")
    predict_parser.add_argument(
        "--source", type=str, required=True, help="图片/视频路径，或摄像头 ID"
    )
    predict_parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="模型权重路径"
    )
    predict_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    predict_parser.add_argument("--save", action="store_true", help="保存结果")
    predict_parser.add_argument("--show", action="store_true", help="显示结果")
    predict_parser.add_argument("--device", type=str, default="auto", help="设备")

    # ==========  train 子命令 ==========
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--data", type=str, default="coco128.yaml", help="数据集配置文件"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="模型大小",
    )
    train_parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    train_parser.add_argument("--batch", type=int, default=16, help="批量大小")
    train_parser.add_argument("--device", type=str, default="auto", help="设备")
    train_parser.add_argument("--quick", action="store_true", help="快速训练 (10 轮)")

    # ==========  eval 子命令 ==========
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    eval_parser.add_argument(
        "--data", type=str, default="coco128.yaml", help="数据集配置文件"
    )
    eval_parser.add_argument("--device", type=str, default="auto", help="设备")

    # ==========  info 子命令 ==========
    info_parser = subparsers.add_parser("info", help="显示系统信息")

    # 解析参数
    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()


def run_demo():
    """运行演示"""
    print("=" * 60)
    print("🚀 YOLOv8 目标检测系统演示")
    print("=" * 60)

    try:
        from model import ObjectDetector

        # 创建检测器
        print("\n1. 加载模型...")
        detector = ObjectDetector(model_size="n")
        detector.info()

        print("\n2. 演示功能说明:")
        print("""
可用功能:

  📷 图片检测:
     python main.py predict --source image.jpg --show

  📹 视频检测:
     python main.py predict --source video.mp4 --save

  🎥 实时摄像头:
     python main.py predict --source 0

  🏋️ 训练模型:
     python main.py train --data coco128.yaml --epochs 100

  📊 评估模型:
     python main.py eval --model best.pt --data coco128.yaml
        """)

        # 检测示例图片
        print("\n3. 测试检测功能...")
        try:
            import urllib.request
            import tempfile
            import os

            # 下载测试图片
            test_url = "https://ultralytics.com/images/bus.jpg"
            temp_dir = tempfile.gettempdir()
            test_image = os.path.join(temp_dir, "test_bus.jpg")

            print(f"   下载测试图片: {test_url}")
            urllib.request.urlretrieve(test_url, test_image)

            # 检测
            print(f"   执行检测...")
            detections = detector.predict(test_image, conf=0.5)

            for det in detections:
                print(f"\n   ✓ 检测到 {len(det['boxes'])} 个物体:")
                for name in det["class_names"][:5]:
                    print(f"     - {name}")
                if len(det["class_names"]) > 5:
                    print(f"     ... 等共 {len(det['class_names'])} 个")

            # 清理
            os.remove(test_image)

        except Exception as e:
            print(f"   测试跳过 (网络问题): {e}")

        print("\n" + "=" * 60)
        print("✓ 演示完成!")
        print("=" * 60)

    except ImportError as e:
        print(f"\n⚠ 错误: {e}")
        print("请安装 ultralytics: pip install ultralytics")


def run_predict(args):
    """运行推理"""
    from inference import predict_image, predict_video, predict_batch

    source = args.source

    if source.isdigit():
        # 摄像头
        predict_video(
            source=int(source),
            model_path=args.model,
            conf=args.conf,
            save=args.save,
            show=True,
            device=args.device,
        )
    elif Path(source).is_dir():
        # 批量图片
        predict_batch(
            source=source,
            model_path=args.model,
            conf=args.conf,
            save=args.save,
            device=args.device,
        )
    elif Path(source).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        # 视频
        predict_video(
            source=source,
            model_path=args.model,
            conf=args.conf,
            save=args.save,
            show=args.show,
            device=args.device,
        )
    else:
        # 单张图片
        predict_image(
            source=source,
            model_path=args.model,
            conf=args.conf,
            save=args.save,
            show=args.show,
            device=args.device,
        )


def run_train(args):
    """运行训练"""
    from train import train, quick_train

    if args.quick:
        quick_train(data_yaml=args.data, epochs=10, model_size=args.model)
    else:
        train(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
        )


def run_eval(args):
    """运行评估"""
    from evaluate import evaluate

    evaluate(model_path=args.model, data_yaml=args.data, device=args.device)


def show_info():
    """显示系统信息"""
    import torch

    print("=" * 60)
    print("系统信息")
    print("=" * 60)

    # PyTorch 信息
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        print(f"MPS 可用: {torch.backends.mps.is_available()}")

    # ultralytics
    try:
        import ultralytics

        print(f"\nUltralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("\n⚠ Ultralytics 未安装")
        print("  安装命令: pip install ultralytics")

    print("\n项目路径:")
    print(f"  {Path(__file__).parent}")


if __name__ == "__main__":
    main()
