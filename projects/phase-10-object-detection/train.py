"""
训练脚本
========

使用 YOLOv8 在自定义数据集上进行目标检测模型训练。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import TrainConfig, ModelConfig, DataConfig, MODEL_DIR


def train(
    data_yaml: str = "coco128.yaml",
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    pretrained: bool = True,
    device: str = "auto",
    project: Optional[str] = None,
    name: str = "train",
    resume: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    训练 YOLOv8 目标检测模型

    Args:
        data_yaml: 数据集配置文件路径 (YOLO 格式)
        model_size: 模型大小 (n/s/m/l/x)
        epochs: 训练轮数
        batch_size: 批量大小
        image_size: 输入图像尺寸
        pretrained: 是否使用预训练权重
        device: 训练设备 (auto/cuda/cpu/mps)
        project: 项目保存目录
        name: 实验名称
        resume: 是否从上次中断处继续训练
        **kwargs: 其他训练参数

    Returns:
        训练结果字典
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    print("=" * 60)
    print("YOLOv8 目标检测训练")
    print("=" * 60)

    # 设置项目目录
    if project is None:
        project = str(MODEL_DIR)

    # 加载模型
    if resume:
        # 从检查点恢复
        model_path = Path(project) / name / "weights" / "last.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到检查点: {model_path}")
        print(f"从检查点恢复训练: {model_path}")
        model = YOLO(str(model_path))
    else:
        # 加载预训练模型
        model_name = f"yolov8{model_size}.pt"
        print(f"加载模型: {model_name}")
        model = YOLO(model_name)

    # 打印训练配置
    print(f"\n训练配置:")
    print(f"  数据集: {data_yaml}")
    print(f"  模型: YOLOv8{model_size}")
    print(f"  轮数: {epochs}")
    print(f"  批量大小: {batch_size}")
    print(f"  图像尺寸: {image_size}")
    print(f"  设备: {device}")
    print(f"  保存路径: {project}/{name}")

    # 开始训练
    print("\n开始训练...")
    print("-" * 60)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device if device != "auto" else None,
        project=project,
        name=name,
        pretrained=pretrained,
        resume=resume,
        # 数据增强
        mosaic=1.0,  # Mosaic 数据增强
        mixup=0.0,  # MixUp 数据增强
        copy_paste=0.0,  # Copy-Paste 数据增强
        # 优化器设置
        optimizer="auto",  # 自动选择优化器
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率 (lr0 * lrf)
        momentum=0.937,  # SGD 动量
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # 其他设置
        amp=True,  # 混合精度
        patience=50,  # 早停
        save_period=10,  # 保存周期
        workers=4,
        verbose=True,
        **kwargs,
    )

    print("-" * 60)
    print("训练完成!")
    print("=" * 60)

    # 输出结果
    print(f"\n训练结果:")
    print(f"  最佳模型: {project}/{name}/weights/best.pt")
    print(f"  最后模型: {project}/{name}/weights/last.pt")

    return {"model_path": f"{project}/{name}/weights/best.pt", "results": results}


def train_with_config(config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """
    使用配置对象进行训练

    Args:
        config: 训练配置 (None 使用默认配置)

    Returns:
        训练结果字典
    """
    if config is None:
        config = TrainConfig()

    return train(
        epochs=config.epochs,
        batch_size=config.batch_size,
        device=config.device,
        project=config.project,
        name=config.name,
        amp=config.amp,
        pretrained=config.pretrained,
    )


def quick_train(
    data_yaml: str = "coco128.yaml", epochs: int = 10, model_size: str = "n"
) -> Dict[str, Any]:
    """
    快速训练 (用于测试)

    Args:
        data_yaml: 数据集配置文件
        epochs: 训练轮数 (默认 10)
        model_size: 模型大小

    Returns:
        训练结果字典
    """
    print("🚀 快速训练模式 (用于测试)")
    return train(
        data_yaml=data_yaml,
        model_size=model_size,
        epochs=epochs,
        batch_size=8,
        name="quick_train",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 目标检测训练")
    parser.add_argument(
        "--data", type=str, default="coco128.yaml", help="数据集配置文件"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="模型大小",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument(
        "--device", type=str, default="auto", help="设备 (auto/cuda/cpu/mps)"
    )
    parser.add_argument("--name", type=str, default="train", help="实验名称")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--quick", action="store_true", help="快速训练模式 (10 轮)")

    args = parser.parse_args()

    if args.quick:
        quick_train(data_yaml=args.data, epochs=10, model_size=args.model)
    else:
        train(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device,
            name=args.name,
            resume=args.resume,
        )
