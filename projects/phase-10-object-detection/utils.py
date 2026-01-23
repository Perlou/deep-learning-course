"""
工具函数模块
============

提供目标检测项目常用的工具函数。
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np

# 设置日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    name: str = "object_detection",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径 (可选)

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)

    Args:
        box1: 边界框1 [x1, y1, x2, y2]
        box2: 边界框2 [x1, y1, x2, y2]

    Returns:
        IoU 值 (0-1)
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 并集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """
    将边界框从 [x1, y1, x2, y2] 格式转换为 [x_center, y_center, width, height]

    Args:
        box: [x1, y1, x2, y2] 格式的边界框

    Returns:
        [x_center, y_center, width, height] 格式的边界框
    """
    x1, y1, x2, y2 = box
    return np.array(
        [
            (x1 + x2) / 2,  # x_center
            (y1 + y2) / 2,  # y_center
            x2 - x1,  # width
            y2 - y1,  # height
        ]
    )


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """
    将边界框从 [x_center, y_center, width, height] 格式转换为 [x1, y1, x2, y2]

    Args:
        box: [x_center, y_center, width, height] 格式的边界框

    Returns:
        [x1, y1, x2, y2] 格式的边界框
    """
    x_center, y_center, width, height = box
    return np.array(
        [
            x_center - width / 2,  # x1
            y_center - height / 2,  # y1
            x_center + width / 2,  # x2
            y_center + height / 2,  # y2
        ]
    )


def normalize_box(box: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """
    将边界框坐标归一化到 0-1 范围

    Args:
        box: [x_center, y_center, width, height] 格式的边界框 (像素坐标)
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        归一化的边界框
    """
    x_center, y_center, width, height = box
    return np.array(
        [
            x_center / image_width,
            y_center / image_height,
            width / image_width,
            height / image_height,
        ]
    )


def denormalize_box(box: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """
    将归一化的边界框坐标转换为像素坐标

    Args:
        box: [x_center, y_center, width, height] 归一化边界框
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        像素坐标的边界框
    """
    x_center, y_center, width, height = box
    return np.array(
        [
            x_center * image_width,
            y_center * image_height,
            width * image_width,
            height * image_height,
        ]
    )


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制检测结果

    Args:
        image: 输入图像 (BGR 格式)
        boxes: 边界框 [N, 4] (xyxy 格式)
        class_ids: 类别 ID [N]
        scores: 置信度分数 [N]
        class_names: 类别名称列表
        colors: 类别颜色列表 (可选)
        thickness: 线条粗细

    Returns:
        绘制了检测结果的图像
    """
    import cv2

    image = image.copy()

    # 生成颜色
    if colors is None:
        np.random.seed(42)
        colors = [
            tuple(int(c) for c in np.random.randint(0, 255, 3))
            for _ in range(len(class_names))
        ]

    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(class_id) % len(colors)]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签
        label = f"{class_names[int(class_id)]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # 标签背景
        cv2.rectangle(
            image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1
        )

        # 标签文字
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return image


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """
    非极大值抑制 (NMS)

    Args:
        boxes: 边界框 [N, 4] (xyxy 格式)
        scores: 置信度分数 [N]
        iou_threshold: IoU 阈值

    Returns:
        保留的边界框索引列表
    """
    # 按置信度降序排序
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(int(current))

        if len(indices) == 1:
            break

        # 计算当前框与其他框的 IoU
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]

        ious = np.array(
            [calculate_iou(current_box, other_box) for other_box in other_boxes]
        )

        # 保留 IoU 小于阈值的框
        indices = indices[1:][ious < iou_threshold]

    return keep


def convert_coco_to_yolo(
    coco_annotation: dict, image_width: int, image_height: int
) -> str:
    """
    将 COCO 格式标注转换为 YOLO 格式

    Args:
        coco_annotation: COCO 格式的标注 (包含 bbox 和 category_id)
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        YOLO 格式的标注字符串
    """
    # COCO bbox: [x, y, width, height] (左上角坐标)
    x, y, w, h = coco_annotation["bbox"]

    # 转换为中心点坐标并归一化
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    width = w / image_width
    height = h / image_height

    # category_id - 1 因为 YOLO 类别从 0 开始
    class_id = coco_annotation["category_id"] - 1

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    获取目录中的所有图像文件

    Args:
        directory: 目录路径
        recursive: 是否递归搜索

    Returns:
        图像文件路径列表
    """
    directory = Path(directory)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    if recursive:
        files = directory.rglob("*")
    else:
        files = directory.glob("*")

    return [f for f in files if f.suffix.lower() in extensions]


if __name__ == "__main__":
    # 测试工具函数
    print("=" * 50)
    print("工具函数测试")
    print("=" * 50)

    # IoU 测试
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 5, 15, 15])
    iou = calculate_iou(box1, box2)
    print(f"\nIoU 测试:")
    print(f"  Box1: {box1}")
    print(f"  Box2: {box2}")
    print(f"  IoU: {iou:.4f}")

    # 坐标转换测试
    box_xyxy = np.array([10, 20, 50, 80])
    box_xywh = xyxy_to_xywh(box_xyxy)
    box_back = xywh_to_xyxy(box_xywh)
    print(f"\n坐标转换测试:")
    print(f"  原始 (xyxy): {box_xyxy}")
    print(f"  转换 (xywh): {box_xywh}")
    print(f"  还原 (xyxy): {box_back}")

    # NMS 测试
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]])
    scores = np.array([0.9, 0.8, 0.95])
    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"\nNMS 测试:")
    print(f"  输入框数: {len(boxes)}")
    print(f"  保留索引: {keep}")

    print("\n✓ 所有测试通过!")
