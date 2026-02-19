"""
launch_ddp.py — DDP 多 GPU 训练启动脚本
========================================

使用 PyTorch DistributedDataParallel 在多 GPU 上并行训练。

用法:
  # 单机 2 GPU
  torchrun --nproc_per_node=2 scripts/launch_ddp.py --config configs/pretrain.yaml

  # 单机 4 GPU
  torchrun --nproc_per_node=4 scripts/launch_ddp.py --config configs/pretrain.yaml
"""

import os
import sys
import argparse

import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.gpt import GPT
from src.training.trainer_utils import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    wrap_model_ddp,
    get_dtype,
    CosineWarmupScheduler,
    clip_grad_norm,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
    create_grad_scaler,
)


def main():
    parser = argparse.ArgumentParser(description="DDP 多 GPU 预训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    # DDP 环境变量 (torchrun 自动设置)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 初始化 DDP
    setup_ddp(local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        print(f"🚀 DDP 启动: {world_size} GPU(s)")

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dtype = get_dtype(device, config.get("dtype", "float32"))

    # 创建模型
    model = GPT(config).to(device)
    model = wrap_model_ddp(model, device_id=local_rank)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数量: {total_params:,}")

    # 这里需要根据实际情况加载数据集
    # train_dataset = YourDataset(...)
    # sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    # train_loader = DataLoader(train_dataset, batch_size=..., sampler=sampler)

    if is_main_process():
        print("⚠️  这是 DDP 启动模板, 请根据实际需求配置数据集和训练循环")
        print("   参考 src/training/pretrain.py 的训练逻辑")

    # 清理
    cleanup_ddp()


if __name__ == "__main__":
    main()
