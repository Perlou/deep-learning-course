"""
梯度累积 (Gradient Accumulation)
================================

学习目标：
    1. 理解梯度累积的原理和用途
    2. 学会实现梯度累积
    3. 了解何时使用梯度累积

核心概念：
    - 虚拟 batch size: 通过累积实现更大 batch
    - 梯度累积步数: 累积多少个 mini-batch
"""

import torch
import torch.nn as nn
import torch.optim as optim


def introduction():
    print("=" * 60)
    print("梯度累积")
    print("=" * 60)

    print("""
为什么需要梯度累积？

    问题: 显存不足以使用大 batch size
    解决: 多次前向/反向传播后再更新权重

原理：
    目标 batch_size = 64
    实际 batch_size = 16
    累积步数 = 64 / 16 = 4

    for i, (data, target) in enumerate(loader):
        loss = model(data).mean()
        loss = loss / accumulation_steps  # 平均损失
        loss.backward()  # 梯度累积

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # 更新权重
            optimizer.zero_grad()

关键点：
    ✓ 损失除以累积步数
    ✓ 不立即 zero_grad()
    ✓ 累积达到步数后再 step()
    """)


def gradient_accumulation_demo():
    """梯度累积演示"""
    print("\n" + "=" * 60)
    print("梯度累积演示")
    print("=" * 60)

    # 创建模型
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 参数
    accumulation_steps = 4
    actual_batch_size = 8
    effective_batch_size = actual_batch_size * accumulation_steps

    print(f"\n实际 batch size: {actual_batch_size}")
    print(f"累积步数: {accumulation_steps}")
    print(f"等效 batch size: {effective_batch_size}")

    # 模拟训练
    total_steps = 12

    print("\n训练过程:")
    for step in range(total_steps):
        # 创建数据
        x = torch.randn(actual_batch_size, 10)
        y = torch.randn(actual_batch_size, 2)

        # 前向传播
        output = model(x)
        loss = criterion(output, y)
        loss = loss / accumulation_steps  # 关键：损失需要平均

        # 反向传播（梯度累积）
        loss.backward()

        # 更新权重
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"  Step {step + 1}: 权重更新 ✓")
        else:
            print(f"  Step {step + 1}: 梯度累积中...")


def complete_template():
    """完整代码模板"""
    print("\n" + "=" * 60)
    print("完整代码模板")
    print("=" * 60)

    print("""
def train_with_accumulation(model, loader, optimizer, accumulation_steps):
    model.train()
    optimizer.zero_grad()

    for step, (data, target) in enumerate(loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 损失缩放
        loss = loss / accumulation_steps
        loss.backward()

        # 累积够了就更新
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # 处理最后不足一个累积周期的梯度
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    """)


def main():
    introduction()
    gradient_accumulation_demo()
    complete_template()

    print("\n" + "=" * 60)
    print("关键要点")
    print("=" * 60)
    print("""
    ✓ 梯度累积模拟大 batch size
    ✓ 损失要除以累积步数
    ✓ 累积达到步数后再更新
    ✓ 适用于显存不足的场景
    """)


if __name__ == "__main__":
    main()
