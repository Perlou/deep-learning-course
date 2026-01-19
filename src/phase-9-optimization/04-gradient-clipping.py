"""
梯度裁剪 (Gradient Clipping)
============================

学习目标：
    1. 理解梯度爆炸问题
    2. 掌握梯度裁剪的两种方法
    3. 学会在实践中应用梯度裁剪
    4. 了解梯度裁剪的最佳实践

核心概念：
    - 梯度爆炸: 梯度过大导致训练不稳定
    - 按值裁剪: 直接限制梯度值范围
    - 按范数裁剪: 限制梯度向量的范数
    - 阈值选择: 如何确定裁剪阈值

前置知识：
    - Phase 4: 反向传播
    - Phase 6: RNN/LSTM 训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：梯度爆炸问题 ====================


def introduction():
    """梯度爆炸问题介绍"""
    print("=" * 60)
    print("第一部分：梯度爆炸问题")
    print("=" * 60)

    print("""
什么是梯度爆炸？

    在深度网络或循环网络中，梯度在反向传播时可能指数级增长。

    ┌─────────────────────────────────────────────────────────┐
    │ 梯度爆炸的表现：                                         │
    │                                                         │
    │ 1. 损失突然变成 NaN 或 Inf                              │
    │ 2. 损失剧烈震荡                                         │
    │ 3. 模型权重变得非常大                                    │
    └─────────────────────────────────────────────────────────┘

为什么会发生梯度爆炸？

    以 RNN 为例：

    ∂L/∂W = Σₜ ∂L/∂hₜ * ∂hₜ/∂hₜ₋₁ * ... * ∂h₁/∂W

    如果 ||∂hₜ/∂hₜ₋₁|| > 1，梯度会指数级增长！

常见场景：

    1. RNN/LSTM 处理长序列
    2. 非常深的网络
    3. 学习率过大
    4. 权重初始化不当

解决方案：

    ✓ 梯度裁剪 (Gradient Clipping) ← 本课重点
    ✓ 合适的权重初始化
    ✓ 使用 LSTM/GRU 替代普通 RNN
    ✓ 批归一化
    ✓ 残差连接
    """)


# ==================== 第二部分：按值裁剪 ====================


def clip_by_value_demo():
    """按值裁剪演示"""
    print("\n" + "=" * 60)
    print("第二部分：按值裁剪 (Clip by Value)")
    print("=" * 60)

    print("""
按值裁剪：

    将每个梯度值限制在 [-clip_value, clip_value] 范围内。

    g_i = clamp(g_i, -clip_value, clip_value)

PyTorch 实现：

    torch.nn.utils.clip_grad_value_(parameters, clip_value)

示例：
    clip_value = 1.0
    原始梯度: [0.5, 2.3, -1.8, 0.1]
    裁剪后:   [0.5, 1.0, -1.0, 0.1]

优点：简单直观
缺点：可能改变梯度方向
    """)

    # 演示
    print("\n示例：按值裁剪\n")

    # 创建模型
    model = nn.Linear(10, 2)

    # 模拟大梯度
    x = torch.randn(4, 10) * 10
    y = torch.randn(4, 2)

    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    # 裁剪前
    grad_before = model.weight.grad.clone()
    print(f"裁剪前梯度范围: [{grad_before.min():.4f}, {grad_before.max():.4f}]")

    # 按值裁剪
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

    grad_after = model.weight.grad
    print(f"裁剪后梯度范围: [{grad_after.min():.4f}, {grad_after.max():.4f}]")


# ==================== 第三部分：按范数裁剪 ====================


def clip_by_norm_demo():
    """按范数裁剪演示"""
    print("\n" + "=" * 60)
    print("第三部分：按范数裁剪 (Clip by Norm)")
    print("=" * 60)

    print("""
按范数裁剪（更常用！）：

    如果梯度向量的范数超过阈值，则按比例缩小。

    if ||g|| > max_norm:
        g = g * max_norm / ||g||

    保持梯度方向不变，只缩小大小。

PyTorch 实现：

    torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    返回值：裁剪前的梯度范数（用于监控）

常用阈值：
    - 1.0: 常见设置
    - 0.5: 更保守
    - 5.0: 较宽松

优点：
    ✓ 保持梯度方向
    ✓ 更平滑地限制梯度
    ✓ 更常用于实践
    """)

    # 演示
    print("\n示例：按范数裁剪\n")

    model = nn.Linear(10, 2)

    # 模拟大梯度
    x = torch.randn(4, 10) * 10
    y = torch.randn(4, 2)

    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    # 裁剪前范数
    total_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.norm(2).item() ** 2
    total_norm_before = total_norm_before**0.5
    print(f"裁剪前梯度范数: {total_norm_before:.4f}")

    # 按范数裁剪
    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # 裁剪后范数
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.norm(2).item() ** 2
    total_norm_after = total_norm_after**0.5
    print(f"裁剪后梯度范数: {total_norm_after:.4f}")
    print(f"max_norm 阈值: {max_norm}")


# ==================== 第四部分：完整训练示例 ====================


def training_example():
    """完整训练示例"""
    print("\n" + "=" * 60)
    print("第四部分：完整训练示例")
    print("=" * 60)

    print("""
标准训练循环中使用梯度裁剪：

    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()

        # 梯度裁剪（在 optimizer.step() 之前）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

关键点：
    梯度裁剪必须在 backward() 之后、step() 之前！
    """)

    # 创建简易 RNN 演示梯度爆炸
    print("\n示例：RNN 训练中的梯度监控\n")

    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    # 创建数据
    seq_len = 50
    batch_size = 16

    model = SimpleRNN(input_size=10, hidden_size=32, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练并监控梯度
    grad_norms_no_clip = []
    grad_norms_with_clip = []

    for i in range(50):
        x = torch.randn(batch_size, seq_len, 10)
        y = torch.randn(batch_size, 2)

        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        # 记录裁剪前的梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        grad_norms_no_clip.append(total_norm**0.5)

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 记录裁剪后的梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        grad_norms_with_clip.append(total_norm**0.5)

        optimizer.step()

    # 可视化
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(grad_norms_no_clip)
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("裁剪前的梯度范数")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(grad_norms_with_clip)
    plt.axhline(y=1.0, color="r", linestyle="--", label="max_norm=1.0")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("裁剪后的梯度范数")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gradient_clipping.png", dpi=150, bbox_inches="tight")
    print("✓ 梯度监控图已保存到 gradient_clipping.png")


# ==================== 第五部分：最佳实践 ====================


def best_practices():
    """最佳实践"""
    print("\n" + "=" * 60)
    print("第五部分：最佳实践")
    print("=" * 60)

    print("""
梯度裁剪最佳实践：

1. 何时使用梯度裁剪？

    ✓ RNN/LSTM/GRU 训练（几乎必须）
    ✓ Transformer 训练
    ✓ 非常深的网络
    ✓ 训练不稳定时

2. 如何选择 max_norm？

    常见值：0.5, 1.0, 5.0, 10.0

    方法 1：先不裁剪，监控梯度范数的分布
            选择 90-99 百分位作为阈值

    方法 2：从 1.0 开始，观察训练稳定性

3. 推荐使用按范数裁剪

    clip_grad_norm_ 比 clip_grad_value_ 更常用
    因为它保持梯度方向

4. 监控梯度范数

    记录每步的梯度范数，用于诊断训练问题

代码模板：

    def train_step(model, batch, optimizer, max_norm=1.0):
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()

        # 梯度裁剪并记录范数
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm
        )

        optimizer.step()
        return loss.item(), grad_norm.item()
    """)


# ==================== 第六部分：练习 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：监控梯度范数
    任务: 在任意模型训练中添加梯度范数监控
    绘制梯度范数随训练步数的变化曲线

练习 2：对比有无梯度裁剪
    任务: 训练一个 LSTM 语言模型
    对比使用和不使用梯度裁剪的效果

练习 3：寻找最优 max_norm
    任务: 尝试不同的 max_norm 值
    找到最优设置

练习 4：实现梯度范数监控器
    任务: 创建一个类，自动记录每步的梯度范数
    支持可视化和统计分析

思考题 1：梯度裁剪会影响收敛吗？
    提示: 考虑梯度方向和大小的关系

思考题 2：为什么按范数裁剪更好？
    提示: 考虑梯度向量的几何意义
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    clip_by_value_demo()
    clip_by_norm_demo()
    training_example()
    best_practices()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 05-mixed-precision.py: 混合精度训练

关键要点回顾：
    ✓ 梯度爆炸在深度/循环网络中常见
    ✓ 按范数裁剪更常用，保持梯度方向
    ✓ 裁剪在 backward() 之后、step() 之前
    ✓ max_norm 常用值: 1.0
    ✓ 监控梯度范数有助于诊断问题
    """)


if __name__ == "__main__":
    main()
