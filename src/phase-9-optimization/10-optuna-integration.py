"""
Optuna 自动调参 (Optuna Integration)
====================================

学习目标：
    1. 学会使用 Optuna 进行自动超参数优化
    2. 理解贝叶斯优化的基本原理
    3. 掌握 Optuna 的常用功能

核心概念：
    - Optuna: 自动超参数优化框架
    - Study: 优化实验
    - Trial: 单次试验
    - Pruning: 提前终止差的试验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 检查 Optuna 是否安装
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def introduction():
    print("=" * 60)
    print("Optuna 自动调参")
    print("=" * 60)

    print("""
Optuna 简介：

    一个自动超参数优化框架，使用贝叶斯优化。

安装：
    pip install optuna

核心概念：

    Study: 整个优化实验
    Trial: 一次超参数组合的尝试
    Objective: 目标函数（返回要优化的指标）

基本用法：

    import optuna

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        hidden = trial.suggest_int('hidden', 32, 256)

        model = create_model(hidden)
        val_loss = train_and_eval(model, lr)

        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    """)


def optuna_demo():
    """Optuna 使用演示"""
    print("\n" + "=" * 60)
    print("Optuna 演示")
    print("=" * 60)

    if not OPTUNA_AVAILABLE:
        print("\nOptuna 未安装。请运行: pip install optuna")
        print("\n以下是使用示例代码：")
        show_example_code()
        return

    # 创建简单数据
    X = torch.randn(200, 10)
    y = torch.randn(200, 1)

    def objective(trial):
        # 建议超参数
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # 创建模型
        model = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # 简化训练
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # 剪枝：提前终止差的试验
            trial.report(loss.item(), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return loss.item()

    # 创建 study 并优化
    print("\n运行 Optuna 优化 (20 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    # 输出结果
    print("\n最优超参数:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print(f"\n最优值: {study.best_value:.6f}")


def show_example_code():
    """显示示例代码"""
    print("""
import optuna
import torch.nn as nn
import torch.optim as optim

def objective(trial):
    # 定义搜索空间
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden = trial.suggest_int('hidden', 32, 512)
    layers = trial.suggest_int('layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    # 创建模型
    model = create_model(hidden, layers, dropout)

    # 选择优化器
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 训练并返回验证损失
    val_loss = train_and_evaluate(model, optimizer)
    return val_loss

# 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")

# 可视化（需要 plotly）
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
    """)


def suggest_methods():
    """参数建议方法"""
    print("\n" + "=" * 60)
    print("Optuna 参数建议方法")
    print("=" * 60)

    print("""
常用参数建议方法：

    # 浮点数
    trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # 对数尺度
    trial.suggest_float('dropout', 0.0, 0.5)          # 线性尺度

    # 整数
    trial.suggest_int('hidden', 32, 512, step=32)
    trial.suggest_int('layers', 1, 4)

    # 分类变量
    trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    trial.suggest_categorical('activation', ['relu', 'tanh', 'gelu'])

高级功能：

    # Pruning（提前终止）
    for epoch in range(100):
        ...
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 多目标优化
    study = optuna.create_study(
        directions=['minimize', 'minimize']  # loss 和 latency
    )
    """)


def main():
    introduction()
    optuna_demo()
    suggest_methods()

    print("\n" + "=" * 60)
    print("关键要点")
    print("=" * 60)
    print("""
    ✓ Optuna 使用贝叶斯优化
    ✓ 比随机搜索更高效
    ✓ 支持 Pruning 提前终止
    ✓ 可视化帮助理解结果
    ✓ 支持分布式优化
    """)


if __name__ == "__main__":
    main()
