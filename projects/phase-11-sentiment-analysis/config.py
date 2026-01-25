"""
配置参数
========

情感分析项目的所有配置参数集中管理。
"""

import os
from pathlib import Path

import torch

# ===========================================
# 路径配置
# ===========================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ===========================================
# 数据集配置
# ===========================================

DATASET_CONFIG = {
    "name": "imdb",  # "imdb" 或 "chinese"
    "max_length": 256,
    "train_split": 0.8,
    "vocab_size": 20000,
}


# ===========================================
# 模型配置
# ===========================================

# TextCNN 配置
TEXTCNN_CONFIG = {
    "embed_dim": 128,
    "kernel_sizes": [3, 4, 5],
    "num_filters": 100,
    "dropout": 0.5,
}

# BiLSTM 配置
LSTM_CONFIG = {
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.5,
}

# BERT 配置
BERT_CONFIG = {
    "model_name": "distilbert-base-uncased",  # 轻量版 BERT
    "max_length": 256,
    "freeze_bert": False,
}


# ===========================================
# 训练配置
# ===========================================

TRAIN_CONFIG = {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-3,  # TextCNN/LSTM
    "bert_learning_rate": 2e-5,  # BERT
    "weight_decay": 1e-4,
    "early_stopping_patience": 5,
    "scheduler": "cosine",
}


# ===========================================
# 设备配置
# ===========================================


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
# 标签配置
# ===========================================

LABELS = {
    0: "负面 (Negative)",
    1: "正面 (Positive)",
}

NUM_CLASSES = len(LABELS)
