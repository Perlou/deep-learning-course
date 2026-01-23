"""
配置文件
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """训练配置"""

    # 数据
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4

    # 模型
    model_name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = False

    # 优化器
    optimizer: str = "adamw"  # sgd, adam, adamw
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.01

    # 学习率调度
    scheduler: str = "cosine"  # step, cosine, onecycle, none
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # 训练
    epochs: int = 50
    grad_clip: Optional[float] = 1.0

    # 混合精度
    use_amp: bool = True

    # 梯度累积
    gradient_accumulation_steps: int = 1

    # 保存
    output_dir: str = "projects/phase-9-training-benchmark/outputs"
    save_every: int = 10
    log_every: int = 50

    # 设备
    device: str = "auto"  # auto, cuda, cpu


# 预定义配置
CONFIGS = {
    "baseline": TrainConfig(
        optimizer="sgd",
        lr=0.1,
        scheduler="step",
        use_amp=False,
        warmup_epochs=0,
    ),
    "adamw_cosine": TrainConfig(
        optimizer="adamw",
        lr=0.001,
        scheduler="cosine",
        use_amp=False,
        warmup_epochs=5,
    ),
    "adamw_onecycle": TrainConfig(
        optimizer="adamw",
        lr=0.01,
        scheduler="onecycle",
        use_amp=False,
        warmup_epochs=0,
    ),
    "full_optimization": TrainConfig(
        optimizer="adamw",
        lr=0.001,
        scheduler="cosine",
        use_amp=True,
        warmup_epochs=5,
        gradient_accumulation_steps=2,
        grad_clip=1.0,
    ),
}


def get_config(name: str = "full_optimization") -> TrainConfig:
    """获取预定义配置"""
    if name in CONFIGS:
        return CONFIGS[name]
    return TrainConfig()
