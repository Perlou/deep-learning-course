"""
混合精度训练 (Mixed Precision Training)
=======================================

学习目标：
    1. 理解 FP16 混合精度训练的原理
    2. 学会使用 PyTorch AMP
    3. 了解损失缩放的作用

核心概念：
    - FP32/FP16: 不同精度浮点数
    - autocast: 自动混合精度上下文
    - GradScaler: 损失缩放器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time


def introduction():
    """混合精度介绍"""
    print("=" * 60)
    print("混合精度训练")
    print("=" * 60)

    print("""
FP16 vs FP32:
┌──────────┬────────┬──────────────┐
│   类型    │  位数  │   内存占用    │
├──────────┼────────┼──────────────┤
│ FP32     │ 32 位  │ 4 字节       │
│ FP16     │ 16 位  │ 2 字节       │
└──────────┴────────┴──────────────┘

混合精度优势：
    1. 内存减半 → 更大 batch size
    2. 计算加速 → Tensor Core 优化
    3. 通信减半 → 分布式更快

核心思想：
    ✓ 前向传播: FP16（快）
    ✓ 权重更新: FP32（精度）
    """)


def amp_usage():
    """AMP 使用方法"""
    print("\n" + "=" * 60)
    print("PyTorch AMP 使用")
    print("=" * 60)

    print("""
标准用法：

    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    for data, target in dataloader:
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

注意事项：
    - backward() 不放在 autocast 里
    - 需要 GPU 支持 (Volta+)
    """)


def demo():
    """演示"""
    print("\n" + "=" * 60)
    print("API 演示")
    print("=" * 60)

    # 创建模型
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 创建数据
    x = torch.randn(32, 256).to(device)
    y = torch.randint(0, 10, (32,)).to(device)

    if torch.cuda.is_available():
        scaler = GradScaler()

        optimizer.zero_grad()
        with autocast():
            output = model(x)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"✓ 混合精度训练完成")
        print(f"  损失: {loss.item():.4f}")
        print(f"  缩放因子: {scaler.get_scale():.0f}")
    else:
        print("CUDA 不可用，展示 FP32 训练")
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"✓ FP32 训练完成，损失: {loss.item():.4f}")


def main():
    introduction()
    amp_usage()
    demo()

    print("\n" + "=" * 60)
    print("关键要点")
    print("=" * 60)
    print("""
    ✓ FP16 节省内存，加速计算
    ✓ 使用 autocast + GradScaler
    ✓ 需要 GPU Tensor Core 支持
    """)


if __name__ == "__main__":
    main()
