"""
工具函数
========

包含常用的辅助函数。
"""

import os
import random
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """设置随机种子以保证可复现性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """获取可用的计算设备

    Returns:
        torch.device: CUDA设备（如果可用）或CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数数量

    Args:
        model: PyTorch模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """格式化数字为可读形式

    Args:
        num: 数字

    Returns:
        格式化的字符串，如 "1.5M" 或 "2.3B"
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
):
    """保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
        path: 保存路径
        **kwargs: 其他需要保存的数据
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs,
    }

    torch.save(checkpoint, path)
    print(f"检查点已保存: {path}")


def load_checkpoint(
    path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """加载训练检查点

    Args:
        path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）

    Returns:
        检查点数据字典
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"检查点已加载: {path}")
    return checkpoint


def save_lora_weights(model: nn.Module, path: str):
    """只保存 LoRA 权重

    Args:
        model: 包含LoRA的模型
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.data.clone()

    torch.save(lora_state_dict, path)
    print(f"LoRA权重已保存: {path} ({len(lora_state_dict)} 个参数)")


def load_lora_weights(model: nn.Module, path: str):
    """加载 LoRA 权重

    Args:
        model: 模型
        path: LoRA权重路径
    """
    lora_state_dict = torch.load(path, map_location="cpu")

    model_state = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name] = param

    model.load_state_dict(model_state)
    print(f"LoRA权重已加载: {path}")


class AverageMeter:
    """计算和存储平均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: 容忍多少轮没有改善
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """打印模型信息

    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"\n{'=' * 60}")
    print(f"{model_name} 信息")
    print(f"{'=' * 60}")
    print(f"总参数量: {format_number(total_params)} ({total_params:,})")
    print(f"可训练参数: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"可训练比例: {100 * trainable_params / total_params:.2f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # 测试工具函数
    print(f"设备: {get_device()}")
    print(f"格式化: {format_number(1234567)}")
    print(f"格式化: {format_number(1234567890)}")

    # 测试 AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"平均值: {meter.avg}")
