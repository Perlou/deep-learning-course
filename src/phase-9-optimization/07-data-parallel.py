"""
数据并行 (Data Parallel)
========================

学习目标：
    1. 理解数据并行的原理
    2. 学会使用 DataParallel
    3. 了解多 GPU 训练基础

核心概念：
    - 数据并行: 数据分到多卡，模型复制
    - DataParallel: 简单的单机多卡方案
"""

import torch
import torch.nn as nn


def introduction():
    print("=" * 60)
    print("数据并行")
    print("=" * 60)

    print("""
数据并行原理：

    ┌─────────────────────────────────────────┐
    │ 一个 batch 的数据分给多个 GPU            │
    │                                         │
    │   Batch (64)                            │
    │      ↓                                  │
    │   ┌──────┬──────┬──────┬──────┐        │
    │   │ 16   │ 16   │ 16   │ 16   │        │
    │   │ GPU0 │ GPU1 │ GPU2 │ GPU3 │        │
    │   └──────┴──────┴──────┴──────┘        │
    │      ↓      ↓      ↓      ↓            │
    │   各自前向传播，计算梯度                  │
    │      ↓      ↓      ↓      ↓            │
    │   ─────────────────────────            │
    │         汇总梯度 (GPU0)                  │
    │   ─────────────────────────            │
    │      ↓                                  │
    │   更新参数，广播到所有 GPU                │
    └─────────────────────────────────────────┘

DataParallel 使用：

    model = MyModel()
    model = nn.DataParallel(model)  # 包装
    model = model.to('cuda')        # 移到 GPU
    """)


def dataparallel_demo():
    """DataParallel 演示"""
    print("\n" + "=" * 60)
    print("DataParallel 使用")
    print("=" * 60)

    # 检查 GPU
    num_gpus = torch.cuda.device_count()
    print(f"\n可用 GPU 数量: {num_gpus}")

    if num_gpus == 0:
        print("没有可用 GPU，展示 API 用法")
    else:
        print(f"GPU 列表: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

    # 创建模型
    model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 10))

    if num_gpus > 1:
        model = nn.DataParallel(model)
        print("\n✓ 已启用 DataParallel")
    elif num_gpus == 1:
        print("\n单 GPU，无需 DataParallel")

    device = torch.device("cuda" if num_gpus > 0 else "cpu")
    model = model.to(device)

    # 测试
    x = torch.randn(64, 100).to(device)
    output = model(x)
    print(f"输入: {x.shape} → 输出: {output.shape}")


def code_template():
    """代码模板"""
    print("\n" + "=" * 60)
    print("完整代码模板")
    print("=" * 60)

    print("""
# DataParallel 训练模板

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 创建模型
model = YourModel()

# 多 GPU 包装
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU")
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练
for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 保存模型（注意去掉 DataParallel 包装）
if hasattr(model, 'module'):
    torch.save(model.module.state_dict(), 'model.pth')
else:
    torch.save(model.state_dict(), 'model.pth')
    """)


def limitations():
    """局限性"""
    print("\n" + "=" * 60)
    print("DataParallel 的局限性")
    print("=" * 60)

    print("""
DataParallel 的问题：

    1. 单进程多线程，受 GIL 限制
    2. GPU0 负载不均衡（汇总梯度）
    3. 效率不如 DistributedDataParallel

推荐：
    单机多卡：使用 DistributedDataParallel
    多机多卡：使用 torch.distributed

下一课将介绍 DistributedDataParallel
    """)


def main():
    introduction()
    dataparallel_demo()
    code_template()
    limitations()

    print("\n" + "=" * 60)
    print("关键要点")
    print("=" * 60)
    print("""
    ✓ DataParallel 简单易用
    ✓ 数据分到多卡，梯度汇总
    ✓ 保存模型时注意 .module
    ✓ 更推荐使用 DDP
    """)


if __name__ == "__main__":
    main()
