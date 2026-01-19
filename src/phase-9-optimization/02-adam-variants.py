"""
Adam 优化器家族 (Adam Variants)
================================

学习目标：
    1. 理解 Adam 的自适应学习率原理
    2. 掌握 Adam、AdamW 的区别
    3. 了解其他 Adam 变体 (RAdam, AdaGrad, RMSprop)
    4. 学会选择合适的优化器

核心概念：
    - 自适应学习率: 不同参数不同学习率
    - 一阶矩 (m): 梯度的指数移动平均
    - 二阶矩 (v): 梯度平方的指数移动平均
    - 偏差修正: 解决初始化偏差

前置知识：
    - 01-sgd-momentum.py: SGD 和动量
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：自适应学习率优化器 ====================


def introduction():
    """自适应学习率优化器介绍"""
    print("=" * 60)
    print("第一部分：自适应学习率优化器")
    print("=" * 60)

    print("""
为什么需要自适应学习率？

    SGD 的问题：所有参数使用相同的学习率

    ┌─────────────────────────────────────────────────────────┐
    │ 问题场景：                                               │
    │                                                         │
    │ - 稀疏特征：出现频率低的特征需要更大的学习率              │
    │ - 不同层：浅层和深层可能需要不同学习率                    │
    │ - 不同方向：某些方向曲率大，某些方向曲率小                │
    │                                                         │
    │ 解决方案：为每个参数自适应调整学习率                      │
    └─────────────────────────────────────────────────────────┘

自适应优化器发展历程：

    AdaGrad (2011)
         ↓
      RMSprop (2012)
         ↓
       Adam (2014) ← 最常用
         ↓
    AdamW / RAdam / AdaFactor ...
    """)


# ==================== 第二部分：AdaGrad ====================


def adagrad_demo():
    """AdaGrad 演示"""
    print("\n" + "=" * 60)
    print("第二部分：AdaGrad")
    print("=" * 60)

    print("""
AdaGrad (Adaptive Gradient Algorithm):

    根据历史梯度调整学习率。

公式：
    g_t = ∇L(θ_t)                    # 当前梯度
    G_t = G_{t-1} + g_t²             # 累积梯度平方
    θ_{t+1} = θ_t - η / √(G_t + ε) * g_t

特点：
    ✓ 对稀疏特征效果好
    ✗ 学习率单调递减，后期可能过小
    ✗ 需要手动设置初始学习率

PyTorch 使用：

    optimizer = optim.Adagrad(
        model.parameters(),
        lr=0.01,
        lr_decay=0,
        weight_decay=0
    )
    """)


# ==================== 第三部分：RMSprop ====================


def rmsprop_demo():
    """RMSprop 演示"""
    print("\n" + "=" * 60)
    print("第三部分：RMSprop")
    print("=" * 60)

    print("""
RMSprop (Root Mean Square Propagation):

    解决 AdaGrad 学习率单调递减的问题。

公式：
    g_t = ∇L(θ_t)
    v_t = β * v_{t-1} + (1-β) * g_t²    # 指数移动平均
    θ_{t+1} = θ_t - η / √(v_t + ε) * g_t

    其中 β 通常为 0.9

与 AdaGrad 的区别：
    ┌─────────────────────────────────────────────────────────┐
    │ AdaGrad: 累积所有历史梯度平方 (越来越大)                 │
    │ RMSprop: 只关注近期梯度 (指数衰减)                       │
    └─────────────────────────────────────────────────────────┘

PyTorch 使用：

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=0.01,
        alpha=0.99,     # 平滑系数 β
        eps=1e-8,
        momentum=0      # 可选择添加动量
    )
    """)


# ==================== 第四部分：Adam ====================


def adam_demo():
    """Adam 演示"""
    print("\n" + "=" * 60)
    print("第四部分：Adam")
    print("=" * 60)

    print("""
Adam (Adaptive Moment Estimation):

    结合 Momentum 和 RMSprop 的优点。

公式：
    g_t = ∇L(θ_t)

    # 一阶矩（动量）
    m_t = β₁ * m_{t-1} + (1-β₁) * g_t

    # 二阶矩（自适应学习率）
    v_t = β₂ * v_{t-1} + (1-β₂) * g_t²

    # 偏差修正（重要！）
    m̂_t = m_t / (1 - β₁ᵗ)
    v̂_t = v_t / (1 - β₂ᵗ)

    # 参数更新
    θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)

默认超参数（来自论文）：
    - β₁ = 0.9      (一阶矩系数)
    - β₂ = 0.999    (二阶矩系数)
    - ε = 1e-8      (数值稳定性)
    - η = 0.001     (学习率)

为什么需要偏差修正？
    ┌─────────────────────────────────────────────────────────┐
    │ 问题：m₀ = 0, v₀ = 0，初始时 m_t 和 v_t 偏向于 0        │
    │ 解决：除以 (1 - βᵗ) 进行修正                            │
    │       当 t 较大时，(1 - βᵗ) ≈ 1，修正效果消失            │
    └─────────────────────────────────────────────────────────┘

PyTorch 使用：

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0      # L2 正则化
    )
    """)

    # 演示 Adam 的一阶矩和二阶矩
    print("\n示例：Adam 内部状态\n")

    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 模拟几步更新
    for step in range(5):
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        loss = nn.MSELoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 查看优化器状态
    state = optimizer.state[list(model.parameters())[0]]
    print(f"当前步数: {state['step']}")
    print(f"一阶矩 (exp_avg) 形状: {state['exp_avg'].shape}")
    print(f"二阶矩 (exp_avg_sq) 形状: {state['exp_avg_sq'].shape}")


# ==================== 第五部分：AdamW ====================


def adamw_demo():
    """AdamW 演示"""
    print("\n" + "=" * 60)
    print("第五部分：AdamW (解耦权重衰减)")
    print("=" * 60)

    print("""
AdamW vs Adam：

    问题：Adam 中的 weight_decay 实现有问题！

    Adam 的权重衰减（L2 正则化）：
    ┌─────────────────────────────────────────────────────────┐
    │ g_t = ∇L(θ_t) + λ * θ_t     <- 正则化加到梯度上         │
    │ 然后计算 m_t 和 v_t                                      │
    │                                                         │
    │ 问题：正则化项被自适应学习率缩放了！                      │
    └─────────────────────────────────────────────────────────┘

    AdamW 的解耦权重衰减：
    ┌─────────────────────────────────────────────────────────┐
    │ 正常计算 Adam 更新                                       │
    │ θ_{t+1} = θ_t - η * (Adam更新 + λ * θ_t)                │
    │                                                         │
    │ 权重衰减不经过自适应学习率！                              │
    └─────────────────────────────────────────────────────────┘

为什么 AdamW 更好？

    1. 正则化效果更稳定
    2. 在大模型训练中效果更好
    3. 是当前大语言模型的标准选择

PyTorch 使用：

    # Adam + L2 正则化（不推荐）
    optimizer = optim.Adam(params, lr=0.001, weight_decay=0.01)

    # AdamW（推荐）
    optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    """)

    # 对比实验
    compare_adam_adamw()


def compare_adam_adamw():
    """对比 Adam 和 AdamW"""
    print("\n示例：Adam vs AdamW 权重衰减效果\n")

    torch.manual_seed(42)

    # 创建简单模型
    def train_and_check_weights(opt_class, weight_decay, steps=100):
        model = nn.Linear(10, 2)
        optimizer = opt_class(model.parameters(), lr=0.01, weight_decay=weight_decay)

        weight_norms = [model.weight.norm().item()]

        for _ in range(steps):
            x = torch.randn(4, 10)
            y = torch.randn(4, 2)
            loss = nn.MSELoss()(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            weight_norms.append(model.weight.norm().item())

        return weight_norms

    # 对比
    wd = 0.1
    adam_norms = train_and_check_weights(optim.Adam, wd)
    adamw_norms = train_and_check_weights(optim.AdamW, wd)

    plt.figure(figsize=(8, 5))
    plt.plot(adam_norms, label="Adam (weight_decay=0.1)", alpha=0.8)
    plt.plot(adamw_norms, label="AdamW (weight_decay=0.1)", alpha=0.8)
    plt.xlabel("Steps")
    plt.ylabel("Weight Norm")
    plt.title("Adam vs AdamW: 权重衰减效果")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("adam_vs_adamw.png", dpi=150, bbox_inches="tight")
    print("✓ 对比图已保存到 adam_vs_adamw.png")


# ==================== 第六部分：优化器选择指南 ====================


def optimizer_guide():
    """优化器选择指南"""
    print("\n" + "=" * 60)
    print("第六部分：优化器选择指南")
    print("=" * 60)

    print("""
何时使用哪个优化器？

┌────────────────┬────────────────────────────────────────────┐
│    优化器       │              推荐场景                       │
├────────────────┼────────────────────────────────────────────┤
│ SGD+Momentum   │ • CV 任务 (ResNet, VGG 等)                  │
│                │ • 追求泛化性能                               │
│                │ • 训练时间充足                               │
├────────────────┼────────────────────────────────────────────┤
│ Adam           │ • NLP 任务                                  │
│                │ • 快速原型开发                               │
│                │ • 稀疏梯度问题                               │
├────────────────┼────────────────────────────────────────────┤
│ AdamW          │ • Transformer 模型                          │
│                │ • 大语言模型                                 │
│                │ • 需要正则化时                               │
├────────────────┼────────────────────────────────────────────┤
│ RMSprop        │ • RNN/LSTM 训练                             │
│                │ • 强化学习                                   │
└────────────────┴────────────────────────────────────────────┘

经验法则：

    1. 不知道用什么？先试 Adam/AdamW
    2. CV 任务追求最佳性能？用 SGD+Momentum + 学习率调度
    3. 训练 Transformer？用 AdamW
    4. 训练不稳定？降低学习率，检查梯度

常用超参数设置：

    SGD:
        lr=0.1, momentum=0.9, weight_decay=1e-4

    Adam/AdamW:
        lr=1e-3 或 3e-4, weight_decay=0.01
    """)


# ==================== 第七部分：实验对比 ====================


def full_comparison():
    """完整对比实验"""
    print("\n" + "=" * 60)
    print("第七部分：完整对比实验")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 创建数据
    X = torch.randn(500, 20)
    y = torch.randn(500, 1)

    def create_model():
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def train(model, optimizer, epochs=100):
        criterion = nn.MSELoss()
        losses = []
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    # 对比不同优化器
    optimizers = {
        "SGD": lambda p: optim.SGD(p, lr=0.01),
        "SGD+Momentum": lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
        "RMSprop": lambda p: optim.RMSprop(p, lr=0.01),
        "Adam": lambda p: optim.Adam(p, lr=0.01),
        "AdamW": lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.01),
    }

    plt.figure(figsize=(10, 6))

    for name, opt_fn in optimizers.items():
        model = create_model()
        optimizer = opt_fn(model.parameters())
        losses = train(model, optimizer)
        plt.plot(losses, label=name, alpha=0.8)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("优化器对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✓ 对比图已保存到 optimizer_comparison.png")


# ==================== 第八部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：学习率敏感性
    任务: 对比 Adam 和 SGD 对学习率的敏感性
    问题: 哪个优化器对学习率更鲁棒？

练习 2：CIFAR-10 实验
    任务: 在 CIFAR-10 上分别用 SGD 和 Adam 训练 ResNet
    对比最终准确率和收敛速度

练习 3：手动实现 Adam
    不使用 PyTorch 优化器，手动实现 Adam
    包括偏差修正

练习 4：超参数实验
    任务: 调整 β₁ 和 β₂ 的值
    观察对收敛的影响

思考题 1：为什么 Adam 对学习率不那么敏感？
    提示: 考虑自适应学习率的作用

思考题 2：AdamW 中 weight_decay 和 lr 的关系
    如何设置这两个超参数？

思考题 3：为什么 CV 任务常用 SGD？
    提示: 考虑泛化性能
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    adagrad_demo()
    rmsprop_demo()
    adam_demo()
    adamw_demo()
    optimizer_guide()
    full_comparison()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 03-lr-schedulers.py: 学习率调度

关键要点回顾：
    ✓ Adam 结合了动量和自适应学习率
    ✓ AdamW 使用解耦的权重衰减
    ✓ CV 常用 SGD，NLP 常用 Adam
    ✓ 默认试 Adam，追求性能用 SGD
    ✓ 大模型用 AdamW
    """)


if __name__ == "__main__":
    main()
