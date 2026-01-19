"""
学习率调度器 (Learning Rate Schedulers)
=======================================

学习目标：
    1. 理解学习率调度的重要性
    2. 掌握常用学习率调度策略
    3. 学会使用 Warmup 技术
    4. 能够选择合适的调度策略

核心概念：
    - 学习率衰减: 训练后期降低学习率
    - Warmup: 训练初期逐渐增加学习率
    - 余弦退火: 平滑的周期性调度
    - One Cycle: 超收敛的调度策略

前置知识：
    - 01-sgd-momentum.py: SGD 优化器
    - 02-adam-variants.py: Adam 优化器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    LambdaLR,
)
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：为什么需要学习率调度 ====================


def introduction():
    """学习率调度介绍"""
    print("=" * 60)
    print("第一部分：为什么需要学习率调度")
    print("=" * 60)

    print("""
学习率的作用：

    学习率是最重要的超参数之一，直接影响训练效果。

    ┌─────────────────────────────────────────────────────────┐
    │ 学习率太大：                                             │
    │   - 损失震荡不收敛                                       │
    │   - 跳过最优点                                           │
    │                                                         │
    │ 学习率太小：                                             │
    │   - 收敛太慢                                             │
    │   - 容易困在局部最优                                     │
    └─────────────────────────────────────────────────────────┘

为什么需要调度？

    训练不同阶段需要不同的学习率：

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  初期：可以用较大学习率，快速探索                         │
    │  中期：逐渐降低，稳定收敛                                 │
    │  后期：很小的学习率，精细调整                             │
    │                                                         │
    │  学习率                                                  │
    │    ↑                                                    │
    │    │ ▓▓▓                                                │
    │    │    ▓▓▓                                             │
    │    │       ▓▓▓▓                                         │
    │    │           ▓▓▓▓▓▓                                   │
    │    │                 ▓▓▓▓▓▓▓▓▓▓                         │
    │    └────────────────────────────→ epochs                │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

常用调度策略：

    1. Step 衰减: 每隔 N 个 epoch 衰减
    2. 指数衰减: 每个 epoch 乘以衰减因子
    3. 余弦退火: 平滑的周期性变化
    4. Warmup: 初期从小到大
    5. One Cycle: 先增后减
    """)


# ==================== 第二部分：阶梯式衰减 ====================


def step_decay_demo():
    """阶梯式衰减演示"""
    print("\n" + "=" * 60)
    print("第二部分：阶梯式衰减 (Step/MultiStep)")
    print("=" * 60)

    print("""
StepLR:
    每隔 step_size 个 epoch，学习率乘以 gamma

    lr = initial_lr * gamma^(epoch // step_size)

MultiStepLR:
    在指定的 milestones 衰减

    例如: milestones=[30, 60, 90], gamma=0.1
    在 epoch 30, 60, 90 时学习率变为原来的 0.1 倍
    """)

    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # StepLR
    scheduler1 = StepLR(optimizer, step_size=30, gamma=0.1)

    # MultiStepLR
    optimizer2 = optim.SGD(model.parameters(), lr=0.1)
    scheduler2 = MultiStepLR(optimizer2, milestones=[30, 60, 90], gamma=0.1)

    # 记录学习率
    lrs1, lrs2 = [], []
    for epoch in range(100):
        lrs1.append(scheduler1.get_last_lr()[0])
        lrs2.append(scheduler2.get_last_lr()[0])
        scheduler1.step()
        scheduler2.step()

    # 可视化
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(lrs1)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("StepLR (step_size=30, gamma=0.1)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(lrs2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("MultiStepLR (milestones=[30,60,90])")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("step_lr.png", dpi=150, bbox_inches="tight")
    print("\n✓ StepLR 对比图已保存到 step_lr.png")


# ==================== 第三部分：指数衰减 ====================


def exponential_decay_demo():
    """指数衰减演示"""
    print("\n" + "=" * 60)
    print("第三部分：指数衰减 (ExponentialLR)")
    print("=" * 60)

    print("""
ExponentialLR:
    每个 epoch 学习率乘以 gamma

    lr = initial_lr * gamma^epoch

特点：
    - 平滑衰减
    - gamma 通常接近 1 (如 0.99, 0.95)
    """)

    model = nn.Linear(10, 2)

    # 不同衰减率
    gammas = [0.99, 0.95, 0.9]

    plt.figure(figsize=(8, 5))

    for gamma in gammas:
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        lrs = []
        for epoch in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        plt.plot(lrs, label=f"gamma={gamma}")

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("ExponentialLR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("exponential_lr.png", dpi=150, bbox_inches="tight")
    print("\n✓ ExponentialLR 图已保存到 exponential_lr.png")


# ==================== 第四部分：余弦退火 ====================


def cosine_annealing_demo():
    """余弦退火演示"""
    print("\n" + "=" * 60)
    print("第四部分：余弦退火 (Cosine Annealing)")
    print("=" * 60)

    print("""
CosineAnnealingLR:
    学习率按余弦函数变化

    lr = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2

CosineAnnealingWarmRestarts:
    周期性重启的余弦退火

    每 T_0 个 epoch 重启一次
    T_mult 控制周期增长

优点：
    - 平滑变化，训练更稳定
    - 周期性重启可能跳出局部最优
    """)

    model = nn.Linear(10, 2)

    # CosineAnnealingLR
    optimizer1 = optim.SGD(model.parameters(), lr=0.1)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=50, eta_min=0.001)

    # CosineAnnealingWarmRestarts
    optimizer2 = optim.SGD(model.parameters(), lr=0.1)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=20, T_mult=2)

    lrs1, lrs2 = [], []
    for epoch in range(100):
        lrs1.append(scheduler1.get_last_lr()[0])
        lrs2.append(scheduler2.get_last_lr()[0])
        scheduler1.step()
        scheduler2.step()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(lrs1)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("CosineAnnealingLR (T_max=50)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(lrs2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("CosineAnnealingWarmRestarts (T_0=20)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cosine_lr.png", dpi=150, bbox_inches="tight")
    print("\n✓ Cosine 调度图已保存到 cosine_lr.png")


# ==================== 第五部分：Warmup ====================


def warmup_demo():
    """Warmup 演示"""
    print("\n" + "=" * 60)
    print("第五部分：Warmup (预热)")
    print("=" * 60)

    print("""
Warmup 的作用：

    训练初期模型权重是随机的，梯度可能很大且方向不稳定。
    直接使用大学习率可能导致训练不稳定。

    ┌─────────────────────────────────────────────────────────┐
    │ Warmup 策略：                                           │
    │                                                         │
    │ 1. 线性 Warmup: lr = base_lr * (step / warmup_steps)   │
    │ 2. 指数 Warmup: lr = base_lr * (1 - e^(-step/τ))      │
    │                                                         │
    │ 典型设置: warmup_steps = 总步数的 5-10%                 │
    └─────────────────────────────────────────────────────────┘

常见组合：

    Warmup + Cosine Decay (Transformer 标配)
    Warmup + Linear Decay
    Warmup + Step Decay
    """)

    # 自定义 Warmup + Cosine
    def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 warmup
                return epoch / warmup_epochs
            else:
                # 余弦衰减
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    model = nn.Linear(10, 2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = create_warmup_cosine_scheduler(
        optimizer, warmup_epochs=10, total_epochs=100
    )

    lrs = []
    for epoch in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plt.figure(figsize=(8, 5))
    plt.plot(lrs)
    plt.axvline(x=10, color="r", linestyle="--", label="Warmup 结束")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Warmup + Cosine Decay")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("warmup_cosine.png", dpi=150, bbox_inches="tight")
    print("\n✓ Warmup + Cosine 图已保存到 warmup_cosine.png")


# ==================== 第六部分：One Cycle ====================


def one_cycle_demo():
    """One Cycle 演示"""
    print("\n" + "=" * 60)
    print("第六部分：One Cycle (超收敛)")
    print("=" * 60)

    print("""
OneCycleLR (超收敛):

    来自论文 "Super-Convergence" (Smith & Topin, 2018)

    策略：
    1. 学习率先从小增大到最大值
    2. 然后从最大值降低到很小
    3. 同时动量做相反变化

    ┌─────────────────────────────────────────────────────────┐
    │ 学习率:                                                  │
    │     ↗↗↗↘↘↘↘↘↘                                           │
    │   /          ↘↘↘↘                                       │
    │  /                ↘↘↘                                   │
    │                                                         │
    │ 动量:                                                    │
    │  ↘↘↘↗↗↗↗↗↗                                              │
    │       ↘↗↗↗↗↗                                            │
    └─────────────────────────────────────────────────────────┘

优点：
    - 可以用更大的学习率
    - 训练更快收敛
    - 通常只需要一个周期
    """)

    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 假设每个 epoch 100 步，共 20 epochs
    steps_per_epoch = 100
    epochs = 20
    total_steps = steps_per_epoch * epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,  # 最大学习率
        total_steps=total_steps,
        pct_start=0.3,  # 30% 时间上升
        anneal_strategy="cos",
    )

    lrs = []
    for step in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plt.figure(figsize=(8, 5))
    plt.plot(lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("OneCycleLR")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("one_cycle.png", dpi=150, bbox_inches="tight")
    print("\n✓ OneCycleLR 图已保存到 one_cycle.png")


# ==================== 第七部分：ReduceLROnPlateau ====================


def plateau_demo():
    """ReduceLROnPlateau 演示"""
    print("\n" + "=" * 60)
    print("第七部分：ReduceLROnPlateau (自适应衰减)")
    print("=" * 60)

    print("""
ReduceLROnPlateau:

    当指标不再改善时，自动降低学习率。

    参数：
    - mode: 'min' 或 'max'（监控指标是越小越好还是越大越好）
    - factor: 衰减因子
    - patience: 等待多少个 epoch 不改善后降低
    - threshold: 认为改善的阈值

PyTorch 使用：

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',      # 监控 loss
        factor=0.1,      # 学习率乘以 0.1
        patience=10,     # 等待 10 epochs
        verbose=True     # 打印信息
    )

    # 在验证后调用
    scheduler.step(val_loss)

注意：
    这是唯一需要传入指标值的调度器！
    """)


# ==================== 第八部分：调度策略选择指南 ====================


def scheduler_guide():
    """调度策略选择指南"""
    print("\n" + "=" * 60)
    print("第八部分：调度策略选择指南")
    print("=" * 60)

    print("""
调度策略选择建议：

┌─────────────────┬──────────────────────────────────────────┐
│     策略         │              推荐场景                     │
├─────────────────┼──────────────────────────────────────────┤
│ StepLR          │ • 经典 CV 训练 (ResNet 等)                │
│ MultiStepLR     │ • 已知最优衰减点                          │
├─────────────────┼──────────────────────────────────────────┤
│ CosineAnnealing │ • 通用首选                                │
│                 │ • 平滑衰减                                │
├─────────────────┼──────────────────────────────────────────┤
│ Warmup + Cosine │ • Transformer 模型                        │
│                 │ • 大模型训练                              │
├─────────────────┼──────────────────────────────────────────┤
│ OneCycleLR      │ • 快速训练                                │
│                 │ • 追求收敛速度                            │
├─────────────────┼──────────────────────────────────────────┤
│ ReduceLROnPlat  │ • 不确定最优 epoch 数                     │
│                 │ • 早停配合使用                            │
└─────────────────┴──────────────────────────────────────────┘

经验法则：

    1. 不知道用什么？先试 CosineAnnealingLR
    2. 训练 Transformer？用 Warmup + Cosine/Linear
    3. 想快速收敛？用 OneCycleLR
    4. 有验证集监控？用 ReduceLROnPlateau
    """)


# ==================== 第九部分：练习 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：可视化所有调度器
    任务: 在一张图上对比所有调度器的学习率曲线

练习 2：实现自定义 Warmup
    任务: 使用 LambdaLR 实现
    - 线性 Warmup
    - 指数 Warmup
    - Warmup + 多种衰减策略

练习 3：CIFAR-10 实验
    任务: 在 CIFAR-10 上对比不同调度策略
    记录收敛速度和最终准确率

练习 4：找最优学习率
    任务: 实现学习率范围测试 (LR Range Test)
    找到合适的初始学习率

思考题 1：为什么 Warmup 对大模型重要？
    提示: 考虑初始梯度的稳定性

思考题 2：CosineAnnealing 为什么效果好？
    提示: 考虑逃离局部最优
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    step_decay_demo()
    exponential_decay_demo()
    cosine_annealing_demo()
    warmup_demo()
    one_cycle_demo()
    plateau_demo()
    scheduler_guide()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 04-gradient-clipping.py: 梯度裁剪

关键要点回顾：
    ✓ 学习率调度对训练效果至关重要
    ✓ CosineAnnealing 是通用首选
    ✓ Transformer 用 Warmup + Cosine
    ✓ OneCycleLR 可以加速收敛
    ✓ ReduceLROnPlateau 用于自适应调整
    """)


if __name__ == "__main__":
    main()
