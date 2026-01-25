"""
配置参数
========

医学图像分割项目的所有配置参数集中管理。
"""

import os
from pathlib import Path

# ===========================================
# 路径配置
# ===========================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ===========================================
# 数据集配置
# ===========================================

# Montgomery County Chest X-Ray Dataset
DATASET_CONFIG = {
    "name": "montgomery",
    "url": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/MontgomerySet.zip",
    "image_dir": "CXR_png",
    "mask_left_dir": "ManualMask/leftMask",
    "mask_right_dir": "ManualMask/rightMask",
    "image_size": (256, 256),  # 训练时的图像大小
    "train_split": 0.8,
}


# ===========================================
# 模型配置
# ===========================================

MODEL_CONFIG = {
    "name": "unet",
    "in_channels": 1,  # 灰度图
    "out_channels": 1,  # 二分类 (肺/背景)
    "features": [64, 128, 256, 512],  # 各层特征数
    "bilinear": True,  # 使用双线性上采样
}


# ===========================================
# 训练配置
# ===========================================

TRAIN_CONFIG = {
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "scheduler": "cosine",  # cosine, step, plateau
    "early_stopping_patience": 10,
    "save_best_only": True,
    # 损失函数权重
    "bce_weight": 0.5,
    "dice_weight": 0.5,
    # 数据增强
    "augmentation": True,
    "aug_flip_prob": 0.5,
    "aug_rotate_degrees": 15,
    "aug_scale_range": (0.9, 1.1),
}


# ===========================================
# 设备配置
# ===========================================

import torch


def get_device():
    """自动选择最佳设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# ===========================================
# 日志配置
# ===========================================

LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_interval": 10,  # 每 N 个 batch 打印一次
}
