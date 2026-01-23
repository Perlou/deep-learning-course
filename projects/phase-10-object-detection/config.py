"""
配置参数模块
============

定义项目所有可配置参数，包括模型、数据、训练和推理配置。
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"


@dataclass
class ModelConfig:
    """模型配置"""

    # 模型版本: n(nano), s(small), m(medium), l(large), x(xlarge)
    model_size: str = "n"

    # 预训练权重路径 (None 表示使用官方预训练权重)
    pretrained_weights: Optional[str] = None

    # 类别数 (使用 COCO 数据集时为 80)
    num_classes: int = 80

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return f"yolov8{self.model_size}.pt"


@dataclass
class DataConfig:
    """数据配置"""

    # 图像尺寸
    image_size: int = 640

    # 数据集路径
    data_dir: str = str(PROJECT_ROOT / "data")

    # 数据集配置文件
    data_yaml: Optional[str] = None

    # 类别名称 (COCO 80 类)
    class_names: List[str] = field(
        default_factory=lambda: [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
    )


@dataclass
class TrainConfig:
    """训练配置"""

    # 训练轮数
    epochs: int = 100

    # 批量大小
    batch_size: int = 16

    # 初始学习率
    learning_rate: float = 0.01

    # 权重衰减
    weight_decay: float = 0.0005

    # 优化器动量
    momentum: float = 0.937

    # 早停耐心值
    patience: int = 50

    # 工作进程数
    workers: int = 4

    # 是否使用预训练权重
    pretrained: bool = True

    # 保存周期
    save_period: int = 10

    # 项目名称
    project: str = str(MODEL_DIR)

    # 实验名称
    name: str = "train"

    # 是否使用混合精度
    amp: bool = True

    # 设备 (cuda / cpu / mps)
    device: str = "auto"


@dataclass
class InferenceConfig:
    """推理配置"""

    # 置信度阈值
    conf_threshold: float = 0.25

    # IoU 阈值 (NMS)
    iou_threshold: float = 0.45

    # 最大检测数
    max_det: int = 300

    # 设备
    device: str = "auto"

    # 是否保存结果
    save: bool = True

    # 是否显示结果
    show: bool = False

    # 保存目录
    save_dir: str = str(RESULT_DIR)


@dataclass
class Config:
    """完整配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# 默认配置实例
default_config = Config()


def get_device() -> str:
    """自动检测可用设备"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


if __name__ == "__main__":
    # 测试配置
    config = Config()
    print("=" * 50)
    print("目标检测系统配置")
    print("=" * 50)
    print(f"模型: {config.model.model_name}")
    print(f"图像尺寸: {config.data.image_size}")
    print(f"训练轮数: {config.train.epochs}")
    print(f"批量大小: {config.train.batch_size}")
    print(f"置信度阈值: {config.inference.conf_threshold}")
    print(f"可用设备: {get_device()}")
