"""
数据集模块
==========

处理目标检测数据集，支持 YOLO 格式数据准备和转换。
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import yaml

from config import PROJECT_ROOT, DataConfig


def create_dataset_yaml(
    data_dir: str,
    class_names: List[str],
    train_path: str = "images/train",
    val_path: str = "images/val",
    output_path: Optional[str] = None,
) -> str:
    """
    创建 YOLO 格式的数据集配置文件

    Args:
        data_dir: 数据集根目录
        class_names: 类别名称列表
        train_path: 训练集相对路径
        val_path: 验证集相对路径
        output_path: 输出路径 (默认为 data_dir/dataset.yaml)

    Returns:
        配置文件路径
    """
    config = {
        "path": str(Path(data_dir).absolute()),
        "train": train_path,
        "val": val_path,
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if output_path is None:
        output_path = str(Path(data_dir) / "dataset.yaml")

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"✓ 数据集配置文件已创建: {output_path}")
    return output_path


def prepare_sample_dataset(data_dir: Optional[str] = None) -> str:
    """
    准备示例数据集 (使用 COCO128)

    COCO128 是 COCO 数据集的 128 张图片子集，适合快速测试和学习。

    Args:
        data_dir: 数据目录 (默认为项目 data 目录)

    Returns:
        数据集配置文件路径
    """
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data")

    # COCO128 可以通过 ultralytics 自动下载
    print("=" * 50)
    print("准备示例数据集 (COCO128)")
    print("=" * 50)
    print("""
COCO128 数据集信息:
  - 128 张图片 (COCO 子集)
  - 80 个类别
  - 适合快速验证和学习

使用方法:
  在训练时指定 data='coco128.yaml'
  ultralytics 会自动下载数据集
    """)

    return "coco128.yaml"


def create_custom_dataset_structure(data_dir: str, class_names: List[str]) -> str:
    """
    创建自定义数据集目录结构

    Args:
        data_dir: 数据集根目录
        class_names: 类别名称列表

    Returns:
        数据集配置文件路径
    """
    data_path = Path(data_dir)

    # 创建目录结构
    (data_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (data_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (data_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (data_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print(f"✓ 创建目录结构:")
    print(f"  {data_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  ├── labels/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  └── dataset.yaml")

    # 创建配置文件
    yaml_path = create_dataset_yaml(data_dir=data_dir, class_names=class_names)

    # 创建标注说明
    readme_content = f"""# 数据集标注说明

## 目录结构

将图片放入 `images/train` 和 `images/val` 目录。
对应的标注文件放入 `labels/train` 和 `labels/val` 目录。

## 标注格式 (YOLO 格式)

每个图片对应一个同名的 `.txt` 文件，每行一个物体：
```
<class_id> <x_center> <y_center> <width> <height>
```

- class_id: 类别索引 (从 0 开始)
- x_center, y_center: 边界框中心点 (归一化到 0-1)
- width, height: 边界框宽高 (归一化到 0-1)

## 类别列表

| ID | 类别名 |
|----|--------|
""" + "\n".join(f"| {i} | {name} |" for i, name in enumerate(class_names))

    readme_path = data_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✓ 创建标注说明: {readme_path}")

    return yaml_path


def convert_coco_to_yolo_dataset(
    coco_json_path: str, images_dir: str, output_dir: str, split: str = "train"
) -> None:
    """
    将 COCO 格式数据集转换为 YOLO 格式

    Args:
        coco_json_path: COCO 标注 JSON 文件路径
        images_dir: 图片目录路径
        output_dir: 输出目录路径
        split: 数据集划分 (train/val)
    """
    print(f"转换 COCO 数据集 -> YOLO 格式 ({split})")

    # 读取 COCO 标注
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # 创建输出目录
    output_path = Path(output_dir)
    images_out = output_path / "images" / split
    labels_out = output_path / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # 创建图片 ID 到文件名的映射
    image_id_to_info = {img["id"]: img for img in coco_data["images"]}

    # 创建类别 ID 映射 (COCO ID -> 连续 ID)
    category_id_map = {cat["id"]: i for i, cat in enumerate(coco_data["categories"])}

    # 按图片分组标注
    image_annotations: Dict[int, List] = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    # 转换每张图片
    converted = 0
    for image_id, image_info in image_id_to_info.items():
        file_name = image_info["file_name"]
        img_width = image_info["width"]
        img_height = image_info["height"]

        # 复制图片
        src_image = Path(images_dir) / file_name
        if src_image.exists():
            dst_image = images_out / file_name
            shutil.copy(src_image, dst_image)
        else:
            continue

        # 转换标注
        annotations = image_annotations.get(image_id, [])
        label_lines = []

        for ann in annotations:
            if "bbox" not in ann:
                continue

            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # 转换为 YOLO 格式 (归一化的中心点坐标)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            # 获取类别 ID
            class_id = category_id_map.get(ann["category_id"], 0)

            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # 保存标注文件
        label_file = labels_out / (Path(file_name).stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))

        converted += 1

    print(f"✓ 转换完成: {converted} 张图片")


def visualize_annotations(
    image_path: str,
    label_path: str,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    可视化标注

    Args:
        image_path: 图片路径
        label_path: 标注文件路径
        class_names: 类别名称列表
        save_path: 保存路径 (可选)
    """
    import cv2
    import numpy as np
    from utils import draw_detections, xywh_to_xyxy

    # 读取图片
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 读取标注
    boxes = []
    class_ids = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # 反归一化
                x_center *= w
                y_center *= h
                width *= w
                height *= h

                # 转换为 xyxy 格式
                box = xywh_to_xyxy(np.array([x_center, y_center, width, height]))
                boxes.append(box)
                class_ids.append(class_id)

    # 绘制
    if boxes:
        boxes = np.array(boxes)
        class_ids = np.array(class_ids)
        scores = np.ones(len(class_ids))  # 真实标注置信度为 1

        image = draw_detections(image, boxes, class_ids, scores, class_names)

    # 显示或保存
    if save_path:
        cv2.imwrite(save_path, image)
        print(f"✓ 保存可视化结果: {save_path}")
    else:
        cv2.imshow("Annotations", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_dataset_stats(data_yaml_path: str) -> Dict:
    """
    获取数据集统计信息

    Args:
        data_yaml_path: 数据集配置文件路径

    Returns:
        统计信息字典
    """
    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = Path(config["path"])
    train_labels = data_path / "labels" / "train"
    val_labels = data_path / "labels" / "val"

    stats = {
        "train_images": 0,
        "val_images": 0,
        "train_objects": 0,
        "val_objects": 0,
        "class_distribution": {},
    }

    # 统计训练集
    if train_labels.exists():
        for label_file in train_labels.glob("*.txt"):
            stats["train_images"] += 1
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        stats["train_objects"] += 1
                        class_id = int(parts[0])
                        class_name = config["names"].get(class_id, f"class_{class_id}")
                        stats["class_distribution"][class_name] = (
                            stats["class_distribution"].get(class_name, 0) + 1
                        )

    # 统计验证集
    if val_labels.exists():
        for label_file in val_labels.glob("*.txt"):
            stats["val_images"] += 1
            with open(label_file, "r") as f:
                for line in f:
                    if len(line.strip().split()) >= 5:
                        stats["val_objects"] += 1

    return stats


if __name__ == "__main__":
    # 演示数据集功能
    print("=" * 50)
    print("数据集模块演示")
    print("=" * 50)

    # 准备示例数据集
    prepare_sample_dataset()

    # 创建自定义数据集结构示例
    print("\n创建自定义数据集结构示例:")
    demo_classes = ["cat", "dog", "bird"]
    demo_dir = PROJECT_ROOT / "data" / "demo_dataset"

    print(f"\n如需创建自定义数据集，运行:")
    print(f'  create_custom_dataset_structure("{demo_dir}", {demo_classes})')
