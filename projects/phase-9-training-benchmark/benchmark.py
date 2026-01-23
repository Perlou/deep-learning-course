"""
基准测试 - 对比多种优化配置
"""

import time
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIGS, TrainConfig
from model import get_model
from dataset import get_dataloaders
from trainer import Trainer


def run_benchmark(config_names=None, epochs=20):
    """运行基准测试"""
    if config_names is None:
        config_names = [
            "baseline",
            "adamw_cosine",
            "adamw_onecycle",
            "full_optimization",
        ]

    print("=" * 60)
    print("Phase 9 基准测试: 优化策略对比")
    print("=" * 60)

    # 数据 (所有配置共享)
    train_loader, test_loader = get_dataloaders(batch_size=128)

    results = {}

    for name in config_names:
        print(f"\n{'=' * 60}")
        print(f"配置: {name}")
        print("=" * 60)

        # 获取配置
        config = CONFIGS.get(name, TrainConfig())
        config.epochs = epochs  # 统一 epochs
        config.output_dir = f"projects/phase-9-training-benchmark/outputs/{name}"

        # 模型
        model = get_model(config.model_name, config.num_classes)

        # 训练器
        trainer = Trainer(model, config, train_loader, test_loader)

        # 计时
        start_time = time.time()

        # 训练
        history = trainer.train()

        # 结果
        elapsed_time = time.time() - start_time
        results[name] = {
            "history": history,
            "best_acc": trainer.best_acc,
            "time": elapsed_time,
            "config": config,
        }

        print(
            f"\n{name}: 最佳准确率={trainer.best_acc:.2f}%, 用时={elapsed_time / 60:.2f}分钟"
        )

    return results


def plot_comparison(
    results, save_path="projects/phase-9-training-benchmark/outputs/logs/benchmark.png"
):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Loss 对比
    for i, (name, data) in enumerate(results.items()):
        axes[0, 0].plot(data["history"]["test_loss"], label=name, color=colors[i])
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Test Loss")
    axes[0, 0].set_title("测试损失对比")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy 对比
    for i, (name, data) in enumerate(results.items()):
        axes[0, 1].plot(data["history"]["test_acc"], label=name, color=colors[i])
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Test Accuracy (%)")
    axes[0, 1].set_title("测试准确率对比")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning Rate 对比
    for i, (name, data) in enumerate(results.items()):
        axes[1, 0].plot(data["history"]["lr"], label=name, color=colors[i])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("学习率变化对比")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 最终结果柱状图
    names = list(results.keys())
    accs = [results[n]["best_acc"] for n in names]
    times = [results[n]["time"] / 60 for n in names]

    x = np.arange(len(names))
    width = 0.35

    ax2 = axes[1, 1]
    bars1 = ax2.bar(x - width / 2, accs, width, label="准确率 (%)", color="steelblue")
    ax2.set_ylabel("准确率 (%)")
    ax2.set_xlabel("配置")
    ax2.set_title("最终结果对比")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)

    ax3 = ax2.twinx()
    bars2 = ax3.bar(x + width / 2, times, width, label="时间 (分钟)", color="coral")
    ax3.set_ylabel("时间 (分钟)")

    ax2.legend(loc="upper left")
    ax3.legend(loc="upper right")

    # 添加数值标签
    for bar, acc in zip(bars1, accs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n对比图已保存到 {save_path}")


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("基准测试结果摘要")
    print("=" * 60)

    print(f"\n{'配置':<25} {'准确率':<12} {'时间':<12} {'优化器':<10} {'调度器':<12}")
    print("-" * 75)

    baseline_time = results.get("baseline", {}).get("time", 1)

    for name, data in results.items():
        config = data["config"]
        acc = data["best_acc"]
        time_min = data["time"] / 60
        speedup = baseline_time / data["time"]

        print(
            f"{name:<25} {acc:.2f}%       {time_min:.2f}分       {config.optimizer:<10} {config.scheduler:<12}"
        )

    # 找出最佳配置
    best_name = max(results, key=lambda x: results[x]["best_acc"])
    print(f"\n🏆 最佳配置: {best_name} (准确率: {results[best_name]['best_acc']:.2f}%)")


def main():
    # 快速测试 (减少 epochs)
    results = run_benchmark(
        config_names=["baseline", "adamw_cosine", "full_optimization"],
        epochs=10,  # 快速测试用 10 个 epoch
    )

    # 绘制对比图
    plot_comparison(results)

    # 打印摘要
    print_summary(results)


if __name__ == "__main__":
    main()
