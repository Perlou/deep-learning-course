"""
SGD 与动量优化器 (SGD with Momentum)
====================================

学习目标：
    1. 理解随机梯度下降 (SGD) 的基本原理
    2. 掌握动量 (Momentum) 的作用和原理
    3. 了解 Nesterov 动量的改进
    4. 对比不同优化器在实际任务中的表现

核心概念：
    - 梯度下降: 沿负梯度方向更新参数
    - 动量: 累积历史梯度，加速收敛
    - Nesterov 动量: 前瞻性梯度计算
    - 学习率: 步长控制

前置知识：
    - Phase 2: 微积分基础（梯度）
    - Phase 3: PyTorch 基础
    - Phase 4: 神经网络训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ==================== 第一部分：SGD 原理 ====================


def introduction():
    """SGD 原理介绍"""
    print("=" * 60)
    print("第一部分：SGD 原理")
    print("=" * 60)

    print("""
随机梯度下降 (Stochastic Gradient Descent, SGD)：

    最基本的优化算法，是深度学习的基石。

基本公式：
    θ_{t+1} = θ_t - η * ∇L(θ_t)

    其中：
    - θ: 模型参数
    - η: 学习率 (learning rate)
    - ∇L: 损失函数关于参数的梯度

SGD 的问题：

    1. 学习率选择困难
       ┌─────────────────────────────────────────────────────┐
       │ η 太大: 损失震荡，不收敛                              │
       │ η 太小: 收敛太慢，容易困在局部最优                     │
       └─────────────────────────────────────────────────────┘

    2. 对所有参数使用相同学习率
       - 不同参数可能需要不同的更新速率

    3. 容易陷入鞍点
       - 在鞍点附近梯度接近 0

    4. 在沟壑（ravine）处震荡
       - 某些方向曲率大，某些方向曲率小
    """)


# ==================== 第二部分：动量优化器 ====================


def momentum_demo():
    """动量优化器演示"""
    print("\n" + "=" * 60)
    print("第二部分：动量优化器")
    print("=" * 60)

    print("""
动量 (Momentum) 的核心思想：

    累积历史梯度，像"小球滚下山坡"一样加速收敛。

公式：
    v_t = γ * v_{t-1} + η * ∇L(θ_t)
    θ_{t+1} = θ_t - v_t

    其中：
    - v: 速度（累积梯度）
    - γ: 动量系数（通常 0.9）
    - η: 学习率

动量的作用：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │ 1. 加速收敛                                              │
    │    - 当梯度方向一致时，动量累积，加速前进                  │
    │                                                         │
    │ 2. 减少震荡                                              │
    │    - 当梯度方向变化时，动量抵消部分震荡                    │
    │                                                         │
    │ 3. 帮助逃离局部最优                                      │
    │    - 积累的动量可以帮助越过小的局部最优                    │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

可视化对比：

    SGD (无动量):
    ~~~~~~~~~~~~~~~~~~~~~~→ (震荡前进)

    SGD + Momentum:
    ────────────────────→ (平滑前进)
    """)

    # 可视化动量的效果
    visualize_momentum()


def visualize_momentum():
    """可视化动量效果"""
    print("\n示例：2D 优化问题可视化\n")

    # 定义一个沟壑函数 (Rosenbrock-like)
    def loss_fn(x, y):
        return (1 - x) ** 2 + 10 * (y - x**2) ** 2

    # 创建网格
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_fn(X, Y)

    # 手动实现 SGD 和 SGD+Momentum
    def optimize(lr, momentum=0, steps=100, start=(-1.5, 2.0)):
        x, y = start
        v_x, v_y = 0, 0
        path = [(x, y)]

        for _ in range(steps):
            # 计算梯度
            grad_x = -2 * (1 - x) + 10 * 2 * (y - x**2) * (-2 * x)
            grad_y = 10 * 2 * (y - x**2)

            # 更新速度
            v_x = momentum * v_x + lr * grad_x
            v_y = momentum * v_y + lr * grad_y

            # 更新参数
            x = x - v_x
            y = y - v_y

            path.append((x, y))

        return np.array(path)

    # 运行优化
    path_sgd = optimize(lr=0.001, momentum=0, steps=200)
    path_momentum = optimize(lr=0.001, momentum=0.9, steps=200)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, path, title in [
        (axes[0], path_sgd, "SGD (无动量)"),
        (axes[1], path_momentum, "SGD + Momentum (γ=0.9)"),
    ]:
        ax.contour(X, Y, Z, levels=50, cmap="viridis")
        ax.plot(path[:, 0], path[:, 1], "r.-", markersize=2, linewidth=0.5)
        ax.plot(path[0, 0], path[0, 1], "go", markersize=10, label="起点")
        ax.plot(1, 1, "r*", markersize=15, label="最优点")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig("momentum_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ 动量对比图已保存到 momentum_comparison.png")


# ==================== 第三部分：Nesterov 动量 ====================


def nesterov_demo():
    """Nesterov 动量演示"""
    print("\n" + "=" * 60)
    print("第三部分：Nesterov 动量")
    print("=" * 60)

    print("""
Nesterov 加速梯度 (NAG)：

    在动量的基础上，先"看一步"再计算梯度。

标准动量：
    v_t = γ * v_{t-1} + η * ∇L(θ_t)
    θ_{t+1} = θ_t - v_t

Nesterov 动量：
    v_t = γ * v_{t-1} + η * ∇L(θ_t - γ * v_{t-1})  <- 先走一步再算梯度
    θ_{t+1} = θ_t - v_t

直觉理解：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │ 标准动量: 先计算当前梯度，再结合动量                      │
    │                                                         │
    │ Nesterov: 先按动量方向走一步，                           │
    │           在预测位置计算梯度，                           │
    │           这样能更早"预见"梯度变化                        │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

PyTorch 使用：

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=True  # 启用 Nesterov
    )
    """)

    # 对比代码示例
    print("\n示例：创建 SGD 优化器\n")

    model = nn.Linear(10, 2)

    # 普通 SGD
    sgd = optim.SGD(model.parameters(), lr=0.01)
    print(f"SGD: lr={sgd.defaults['lr']}")

    # SGD + Momentum
    sgd_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(
        f"SGD+Momentum: lr={sgd_momentum.defaults['lr']}, momentum={sgd_momentum.defaults['momentum']}"
    )

    # SGD + Nesterov
    sgd_nesterov = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    print(f"SGD+Nesterov: nesterov={sgd_nesterov.defaults['nesterov']}")


# ==================== 第四部分：权重衰减 ====================


def weight_decay_demo():
    """权重衰减演示"""
    print("\n" + "=" * 60)
    print("第四部分：权重衰减 (L2 正则化)")
    print("=" * 60)

    print("""
权重衰减 (Weight Decay)：

    在损失函数中添加 L2 正则化项，防止过拟合。

公式：
    L_total = L_original + (λ/2) * ||θ||²

    等价的参数更新：
    θ_{t+1} = θ_t - η * (∇L + λ * θ_t)
            = θ_t * (1 - η * λ) - η * ∇L
              ↑
              权重衰减

PyTorch 使用：

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4  # L2 正则化系数
    )

常用 weight_decay 值：
    - 1e-4 到 1e-5: 常见设置
    - 0: 不使用正则化
    """)

    # 演示权重衰减的效果
    print("\n示例：权重衰减参数更新\n")

    # 创建简单模型
    model = nn.Linear(10, 2)
    initial_weight = model.weight.data.clone()

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)

    # 创建假数据
    x = torch.randn(4, 10)
    y = torch.randn(4, 2)

    # 一步更新
    loss = nn.MSELoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    weight_change = (model.weight.data - initial_weight).abs().mean()
    print(f"权重变化 (带 weight_decay): {weight_change:.6f}")


# ==================== 第五部分：实际训练对比 ====================


def training_comparison():
    """实际训练对比"""
    print("\n" + "=" * 60)
    print("第五部分：实际训练对比")
    print("=" * 60)

    # 创建简单数据集
    np.random.seed(42)
    torch.manual_seed(42)

    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    # 定义模型
    def create_model():
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def train(model, optimizer, epochs=100):
        criterion = nn.MSELoss()
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    # 训练不同优化器
    configs = [
        ("SGD (lr=0.01)", {"lr": 0.01}),
        ("SGD (lr=0.1)", {"lr": 0.1}),
        ("SGD+Momentum", {"lr": 0.01, "momentum": 0.9}),
        ("SGD+Nesterov", {"lr": 0.01, "momentum": 0.9, "nesterov": True}),
    ]

    plt.figure(figsize=(10, 6))

    for name, kwargs in configs:
        model = create_model()
        optimizer = optim.SGD(model.parameters(), **kwargs)
        losses = train(model, optimizer)
        plt.plot(losses, label=name, alpha=0.8)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SGD 变体对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("sgd_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✓ 训练对比图已保存到 sgd_comparison.png")


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：动量系数实验
    任务: 尝试不同的动量系数 (0, 0.5, 0.9, 0.99)
    观察对收敛速度和稳定性的影响

练习 2：学习率敏感性
    任务: 对比 SGD 和 SGD+Momentum 对学习率的敏感性
    问题: 使用动量时可以用更大的学习率吗？

练习 3：实现手动 SGD
    不使用 PyTorch 优化器，手动实现:
    - SGD
    - SGD + Momentum
    - SGD + Nesterov

练习 4：CIFAR-10 训练
    任务: 在 CIFAR-10 上训练 ResNet
    对比 SGD+Momentum 和 Adam 的效果

思考题 1：为什么动量系数通常是 0.9？
    太大或太小会怎样？

思考题 2：动量与学习率的关系
    使用动量时，有效步长是多少？

思考题 3：何时用 SGD，何时用 Adam？
    提示: 考虑任务类型和训练稳定性
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    momentum_demo()
    nesterov_demo()
    weight_decay_demo()
    training_comparison()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 02-adam-variants.py: Adam 优化器家族

关键要点回顾：
    ✓ SGD 是最基本的优化算法
    ✓ 动量累积历史梯度，加速收敛
    ✓ Nesterov 动量有前瞻性
    ✓ 权重衰减防止过拟合
    ✓ 动量系数通常设为 0.9
    """)


if __name__ == "__main__":
    main()
