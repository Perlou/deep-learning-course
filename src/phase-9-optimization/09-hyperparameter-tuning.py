"""
超参数调优 (Hyperparameter Tuning)
==================================

学习目标：
    1. 理解超参数调优的重要性
    2. 掌握网格搜索和随机搜索
    3. 了解贝叶斯优化基础

核心概念：
    - 超参数: 训练前设定的参数
    - 网格搜索: 穷举所有组合
    - 随机搜索: 随机采样组合
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
import random


def introduction():
    print("=" * 60)
    print("超参数调优")
    print("=" * 60)

    print("""
什么是超参数？

    超参数 vs 参数：
    ┌────────────────┬────────────────────────────────┐
    │    超参数       │            参数                │
    ├────────────────┼────────────────────────────────┤
    │ 训练前设定     │ 训练中学习                      │
    │ 学习率、层数   │ 权重、偏置                      │
    │ 需要调优       │ 自动优化                        │
    └────────────────┴────────────────────────────────┘

常见超参数：

    学习率: 0.1, 0.01, 0.001, ...
    批量大小: 16, 32, 64, 128
    网络层数: 2, 4, 6, 8
    隐藏单元: 64, 128, 256, 512
    Dropout率: 0.1, 0.3, 0.5
    正则化强度: 1e-3, 1e-4, 1e-5
    """)


def grid_search_demo():
    """网格搜索演示"""
    print("\n" + "=" * 60)
    print("网格搜索 (Grid Search)")
    print("=" * 60)

    print("""
网格搜索：穷举所有超参数组合

    优点: 保证找到搜索空间中的最优
    缺点: 计算量随维度指数增长

示例：
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [32, 64]
    hidden_dims = [128, 256]

    组合数: 3 × 2 × 2 = 12 次实验
    """)

    # 定义搜索空间
    search_space = {
        "lr": [0.1, 0.01, 0.001],
        "hidden_dim": [64, 128],
        "dropout": [0.1, 0.3],
    }

    # 生成所有组合
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(product(*values))

    print(f"\n搜索空间: {search_space}")
    print(f"总组合数: {len(combinations)}")
    print("\n前 5 个组合:")
    for i, combo in enumerate(combinations[:5]):
        config = dict(zip(keys, combo))
        print(f"  {i + 1}. {config}")

    # 模拟训练
    print("\n模拟网格搜索...")
    best_score = float("inf")
    best_config = None

    for combo in combinations:
        config = dict(zip(keys, combo))
        # 模拟验证损失
        score = abs(np.log10(config["lr"]) + 2) + random.random() * 0.1
        if score < best_score:
            best_score = score
            best_config = config

    print(f"\n最优配置: {best_config}")
    print(f"最优得分: {best_score:.4f}")


def random_search_demo():
    """随机搜索演示"""
    print("\n" + "=" * 60)
    print("随机搜索 (Random Search)")
    print("=" * 60)

    print("""
随机搜索：随机采样超参数组合

    优点：
    - 效率高于网格搜索
    - 可以搜索连续空间
    - 对重要超参数采样更密集

    研究表明随机搜索通常优于网格搜索！
    """)

    def sample_config():
        """随机采样一个配置"""
        return {
            "lr": 10 ** np.random.uniform(-4, -1),  # log 均匀分布
            "hidden_dim": np.random.choice([64, 128, 256, 512]),
            "dropout": np.random.uniform(0.0, 0.5),
            "weight_decay": 10 ** np.random.uniform(-5, -2),
        }

    # 采样并评估
    n_trials = 20
    print(f"\n运行 {n_trials} 次随机搜索...")

    best_score = float("inf")
    best_config = None

    for i in range(n_trials):
        config = sample_config()
        # 模拟验证损失
        score = abs(np.log10(config["lr"]) + 3) + random.random() * 0.1
        if score < best_score:
            best_score = score
            best_config = config

    print(f"\n最优配置:")
    for k, v in best_config.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"最优得分: {best_score:.4f}")


def practical_tips():
    """实用技巧"""
    print("\n" + "=" * 60)
    print("超参数调优技巧")
    print("=" * 60)

    print("""
1. 优先调优学习率
    最重要的超参数，影响最大

2. 使用对数尺度
    lr: [1e-5, 1e-4, 1e-3, 1e-2]
    不要用线性尺度

3. 先粗搜索后细搜索
    第一轮：大范围，少样本
    第二轮：缩小到有希望的区域

4. 早停
    快速淘汰差的配置
    节省计算资源

5. 使用验证集
    永远不要在测试集上调优！

6. 记录实验
    使用 MLflow/W&B 记录结果
    方便分析和复现
    """)


def main():
    introduction()
    grid_search_demo()
    random_search_demo()
    practical_tips()

    print("\n" + "=" * 60)
    print("关键要点")
    print("=" * 60)
    print("""
    ✓ 学习率是最重要的超参数
    ✓ 随机搜索通常优于网格搜索
    ✓ 使用对数尺度采样学习率
    ✓ 先粗后细两阶段搜索
    """)


if __name__ == "__main__":
    main()
