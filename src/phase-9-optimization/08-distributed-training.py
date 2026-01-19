"""
分布式训练基础 (Distributed Training)
=====================================

学习目标：
    1. 理解分布式训练的原理
    2. 学会使用 DistributedDataParallel (DDP)
    3. 了解多机多卡训练

核心概念：
    - DDP: 分布式数据并行
    - world_size: 总进程数
    - rank: 当前进程编号
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def introduction():
    print("=" * 60)
    print("分布式训练基础")
    print("=" * 60)

    print("""
为什么用 DDP 而不是 DataParallel？

    DataParallel:
        - 单进程多线程
        - 受 GIL 限制
        - 主卡负载重

    DDP (DistributedDataParallel):
        - 多进程
        - 更高效的通信 (Ring AllReduce)
        - 负载均衡

DDP 原理：

    ┌───────────────────────────────────────────┐
    │   每个 GPU 运行独立进程                    │
    │                                           │
    │   Process 0    Process 1    Process 2    │
    │   GPU 0        GPU 1        GPU 2        │
    │   Model副本    Model副本    Model副本     │
    │   Data子集     Data子集     Data子集      │
    │                                           │
    │         Ring AllReduce 同步梯度           │
    │                                           │
    └───────────────────────────────────────────┘
    """)


def ddp_template():
    """DDP 代码模板"""
    print("\n" + "=" * 60)
    print("DDP 代码模板")
    print("=" * 60)

    print('''
# ddp_train.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """初始化分布式环境"""
    dist.init_process_group(
        backend="nccl",    # GPU 用 nccl
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """清理"""
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # 创建模型
    model = YourModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 分布式采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler)

    # 训练
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 重要！
        for data, target in loader:
            ...

    cleanup()

# 启动命令：
# torchrun --nproc_per_node=4 ddp_train.py
    ''')


def key_concepts():
    """关键概念"""
    print("\n" + "=" * 60)
    print("关键概念")
    print("=" * 60)

    print("""
1. 初始化
    dist.init_process_group(backend="nccl")

2. 包装模型
    model = DDP(model, device_ids=[rank])

3. 分布式采样器
    sampler = DistributedSampler(dataset)
    每个进程只看到数据的一部分

4. 设置 epoch
    sampler.set_epoch(epoch)
    确保每个 epoch 数据 shuffle 不同

5. 启动方式
    # 推荐使用 torchrun
    torchrun --nproc_per_node=NUM_GPUS train.py

常见后端：
    - nccl: GPU 推荐
    - gloo: CPU 或跨平台
    - mpi: 高性能计算环境
    """)


def main():
    introduction()
    ddp_template()
    key_concepts()

    print("\n" + "=" * 60)
    print("关键要点")
    print("=" * 60)
    print("""
    ✓ DDP 比 DataParallel 更高效
    ✓ 每个 GPU 一个进程
    ✓ 使用 DistributedSampler
    ✓ 用 torchrun 启动
    """)


if __name__ == "__main__":
    main()
