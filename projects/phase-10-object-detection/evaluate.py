"""
评估脚本
========

评估 YOLOv8 目标检测模型性能。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import InferenceConfig, RESULT_DIR


def evaluate(
    model_path: str,
    data_yaml: str = "coco128.yaml",
    image_size: int = 640,
    batch_size: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "auto",
    save_dir: Optional[str] = None,
    plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    评估模型性能

    Args:
        model_path: 模型权重路径
        data_yaml: 数据集配置文件
        image_size: 输入图像尺寸
        batch_size: 批量大小
        conf: 置信度阈值 (评估时通常设较低)
        iou: IoU 阈值
        device: 评估设备
        save_dir: 结果保存目录
        plots: 是否生成可视化图表
        verbose: 是否显示详细信息

    Returns:
        评估指标字典
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    print("=" * 60)
    print("YOLOv8 模型评估")
    print("=" * 60)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置保存目录
    if save_dir is None:
        save_dir = str(RESULT_DIR / "eval")

    # 打印配置
    print(f"\n评估配置:")
    print(f"  数据集: {data_yaml}")
    print(f"  图像尺寸: {image_size}")
    print(f"  置信度阈值: {conf}")
    print(f"  IoU 阈值: {iou}")
    print(f"  设备: {device}")

    # 执行评估
    print("\n开始评估...")
    print("-" * 60)

    metrics = model.val(
        data=data_yaml,
        imgsz=image_size,
        batch=batch_size,
        conf=conf,
        iou=iou,
        device=device if device != "auto" else None,
        plots=plots,
        save_dir=save_dir,
        verbose=verbose,
    )

    print("-" * 60)
    print("评估完成!")
    print("=" * 60)

    # 解析结果
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "mAP75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_ap50": metrics.box.ap50.tolist()
        if hasattr(metrics.box, "ap50")
        else [],
    }

    # 打印结果
    print(f"\n📊 评估结果:")
    print(f"  mAP@0.5: {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP50-95']:.4f}")
    print(f"  mAP@0.75: {results['mAP75']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")

    if plots:
        print(f"\n可视化结果保存在: {save_dir}")
        print("  - confusion_matrix.png (混淆矩阵)")
        print("  - PR_curve.png (PR 曲线)")
        print("  - F1_curve.png (F1 曲线)")
        print("  - results.png (训练曲线)")

    return results


def compare_models(
    model_paths: list, data_yaml: str = "coco128.yaml", device: str = "auto"
) -> None:
    """
    对比多个模型的性能

    Args:
        model_paths: 模型路径列表
        data_yaml: 数据集配置文件
        device: 评估设备
    """
    print("=" * 60)
    print("模型对比评估")
    print("=" * 60)

    results = []
    for path in model_paths:
        print(f"\n评估模型: {path}")
        result = evaluate(
            model_path=path,
            data_yaml=data_yaml,
            device=device,
            plots=False,
            verbose=False,
        )
        result["model"] = path
        results.append(result)

    # 打印对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'模型':<40} {'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 60)
    for r in results:
        model_name = Path(r["model"]).stem
        print(f"{model_name:<40} {r['mAP50']:>10.4f} {r['mAP50-95']:>10.4f}")


def analyze_predictions(
    model_path: str, image_dir: str, conf: float = 0.25, save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    分析模型预测结果

    Args:
        model_path: 模型权重路径
        image_dir: 图像目录
        conf: 置信度阈值
        save_dir: 结果保存目录

    Returns:
        分析结果字典
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    import numpy as np
    from collections import Counter

    print("=" * 60)
    print("预测结果分析")
    print("=" * 60)

    # 加载模型
    model = YOLO(model_path)

    # 获取图像文件
    from utils import get_image_files

    image_files = get_image_files(image_dir)
    print(f"找到 {len(image_files)} 张图像")

    # 统计信息
    total_detections = 0
    class_counts = Counter()
    confidence_scores = []

    # 处理每张图像
    for img_path in image_files:
        results = model.predict(str(img_path), conf=conf, verbose=False)

        for result in results:
            boxes = result.boxes
            total_detections += len(boxes)

            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                class_counts[cls_name] += 1
                confidence_scores.append(float(box.conf[0]))

    # 分析结果
    analysis = {
        "total_images": len(image_files),
        "total_detections": total_detections,
        "avg_detections_per_image": total_detections / len(image_files)
        if image_files
        else 0,
        "class_distribution": dict(class_counts),
        "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0,
        "min_confidence": np.min(confidence_scores) if confidence_scores else 0,
        "max_confidence": np.max(confidence_scores) if confidence_scores else 0,
    }

    # 打印结果
    print(f"\n📊 分析结果:")
    print(f"  图像总数: {analysis['total_images']}")
    print(f"  检测总数: {analysis['total_detections']}")
    print(f"  平均每张图像检测数: {analysis['avg_detections_per_image']:.2f}")
    print(f"  平均置信度: {analysis['avg_confidence']:.4f}")

    print(f"\n类别分布 (前 10):")
    for cls, count in class_counts.most_common(10):
        print(f"  {cls}: {count}")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 模型评估")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument(
        "--data", type=str, default="coco128.yaml", help="数据集配置文件"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument(
        "--device", type=str, default="auto", help="设备 (auto/cuda/cpu/mps)"
    )
    parser.add_argument("--no-plots", action="store_true", help="不生成可视化图表")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_yaml=args.data,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        plots=not args.no_plots,
    )
