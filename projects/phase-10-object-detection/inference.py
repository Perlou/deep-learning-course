"""
推理脚本
========

使用训练好的 YOLOv8 模型进行目标检测推理。
"""

import sys
from pathlib import Path
from typing import Optional, Union, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import InferenceConfig, RESULT_DIR


def predict_image(
    source: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    save: bool = True,
    show: bool = False,
    save_dir: Optional[str] = None,
    device: str = "auto",
) -> None:
    """
    对单张图片进行目标检测

    Args:
        source: 图片路径
        model_path: 模型权重路径
        conf: 置信度阈值
        iou: NMS IoU 阈值
        save: 是否保存结果
        show: 是否显示结果
        save_dir: 保存目录
        device: 推理设备
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    print("=" * 60)
    print("YOLOv8 图片检测")
    print("=" * 60)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置保存目录
    if save_dir is None:
        save_dir = str(RESULT_DIR)

    print(f"检测图片: {source}")
    print(f"置信度阈值: {conf}")

    # 执行检测
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=save,
        show=show,
        project=save_dir,
        name="predict",
        device=device if device != "auto" else None,
    )

    # 打印结果
    for result in results:
        boxes = result.boxes
        print(f"\n检测到 {len(boxes)} 个物体:")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(
                f"  {i + 1}. {cls_name}: {conf_score:.2f} @ [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]"
            )

    if save:
        print(f"\n结果保存在: {save_dir}/predict")


def predict_batch(
    source: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    save: bool = True,
    save_dir: Optional[str] = None,
    device: str = "auto",
) -> None:
    """
    批量图片检测

    Args:
        source: 图片目录路径
        model_path: 模型权重路径
        conf: 置信度阈值
        save: 是否保存结果
        save_dir: 保存目录
        device: 推理设备
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    print("=" * 60)
    print("YOLOv8 批量检测")
    print("=" * 60)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置保存目录
    if save_dir is None:
        save_dir = str(RESULT_DIR)

    # 获取图片列表
    from utils import get_image_files

    image_files = get_image_files(source)
    print(f"找到 {len(image_files)} 张图片")

    # 执行检测
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        project=save_dir,
        name="batch",
        device=device if device != "auto" else None,
    )

    # 统计
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n总计检测到 {total_detections} 个物体")

    if save:
        print(f"结果保存在: {save_dir}/batch")


def predict_video(
    source: Union[str, int],
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    save: bool = True,
    show: bool = True,
    save_dir: Optional[str] = None,
    device: str = "auto",
) -> None:
    """
    视频/摄像头检测

    Args:
        source: 视频路径或摄像头 ID (0 为默认摄像头)
        model_path: 模型权重路径
        conf: 置信度阈值
        save: 是否保存结果
        show: 是否显示实时画面
        save_dir: 保存目录
        device: 推理设备
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    print("=" * 60)
    print("YOLOv8 视频检测")
    print("=" * 60)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置保存目录
    if save_dir is None:
        save_dir = str(RESULT_DIR)

    source_type = "摄像头" if isinstance(source, int) else "视频"
    print(f"检测{source_type}: {source}")
    print("按 'q' 退出")

    # 执行检测
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=show,
        stream=True,
        project=save_dir,
        name="video",
        device=device if device != "auto" else None,
    )

    # 流式处理
    frame_count = 0
    for result in results:
        frame_count += 1
        if frame_count % 30 == 0:  # 每 30 帧打印一次
            print(f"已处理 {frame_count} 帧, 当前帧检测到 {len(result.boxes)} 个物体")

    print(f"\n处理完成, 共 {frame_count} 帧")
    if save:
        print(f"结果保存在: {save_dir}/video")


def realtime_detect(
    model_path: str = "yolov8n.pt",
    camera_id: int = 0,
    conf: float = 0.25,
    device: str = "auto",
) -> None:
    """
    实时摄像头检测

    Args:
        model_path: 模型权重路径
        camera_id: 摄像头 ID
        conf: 置信度阈值
        device: 推理设备
    """
    predict_video(
        source=camera_id,
        model_path=model_path,
        conf=conf,
        save=False,
        show=True,
        device=device,
    )


def export_detections(
    source: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    output_file: str = "detections.json",
    device: str = "auto",
) -> None:
    """
    导出检测结果为 JSON 格式

    Args:
        source: 图片目录或单张图片
        model_path: 模型权重路径
        conf: 置信度阈值
        output_file: 输出 JSON 文件路径
        device: 推理设备
    """
    import json

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    print("=" * 60)
    print("导出检测结果")
    print("=" * 60)

    # 加载模型
    model = YOLO(model_path)

    # 执行检测
    results = model.predict(
        source=source,
        conf=conf,
        verbose=False,
        device=device if device != "auto" else None,
    )

    # 构建输出数据
    output_data = []
    for result in results:
        image_data = {
            "image": str(result.path),
            "shape": result.orig_shape,
            "detections": [],
        }

        for box in result.boxes:
            detection = {
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox_xyxy": box.xyxy[0].tolist(),
                "bbox_xywh": box.xywh[0].tolist(),
            }
            image_data["detections"].append(detection)

        output_data.append(image_data)

    # 保存 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 检测结果已导出: {output_file}")
    print(f"  处理图片数: {len(output_data)}")
    total_dets = sum(len(d["detections"]) for d in output_data)
    print(f"  总检测数: {total_dets}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 目标检测推理")
    parser.add_argument(
        "--source", type=str, required=True, help="图片/视频路径，或摄像头 ID (如 0)"
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(
        "--device", type=str, default="auto", help="设备 (auto/cuda/cpu/mps)"
    )
    parser.add_argument("--save", action="store_true", help="保存结果")
    parser.add_argument("--show", action="store_true", help="显示结果")
    parser.add_argument(
        "--export-json", type=str, default=None, help="导出为 JSON 文件"
    )

    args = parser.parse_args()

    # 判断输入类型
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
        if args.export_json:
            export_detections(
                source=source,
                model_path=args.model,
                conf=args.conf,
                output_file=args.export_json,
                device=args.device,
            )
        else:
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
