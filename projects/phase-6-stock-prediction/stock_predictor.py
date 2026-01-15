"""
Phase 6 实战项目 - LSTM 股票价格预测

使用 LSTM 进行时间序列预测，预测股票价格走势

核心技术：
- 滑动窗口（Sliding Window）构建监督学习数据
- 数据归一化（MinMaxScaler）
- 多层 LSTM 时间序列模型
- 序列预测与可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import argparse

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ==================== 1. 数据处理 ====================


class StockDataset(Dataset):
    """时间序列数据集 - 滑动窗口"""

    def __init__(self, data, window_size=30):
        """
        Args:
            data: 归一化后的时间序列数据 (numpy array)
            window_size: 时间窗口大小
        """
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """
        返回 (X, y) 对
        X: 过去 window_size 天的价格
        y: 下一天的价格
        """
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]

        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),  # (window_size, 1)
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),  # (1,)
        )


def load_and_preprocess_data(data_path, train_ratio=0.8, window_size=30):
    """
    加载和预处理股票数据

    Returns:
        train_loader, test_loader, scaler, train_size
    """
    # 读取数据
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 使用收盘价
    prices = df["Close"].values.reshape(-1, 1)

    print("\n数据信息:")
    print(f"  总样本数: {len(prices)}")
    print(f"  日期范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"  价格范围: {prices.min():.2f} - {prices.max():.2f}")

    # 归一化到 [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # 划分训练集和测试集
    train_size = int(len(prices_scaled) * train_ratio)
    train_data = prices_scaled[:train_size]
    test_data = prices_scaled[train_size:]

    print(f"  训练集大小: {train_size}")
    print(f"  测试集大小: {len(test_data)}")

    # 创建数据集
    train_dataset = StockDataset(train_data, window_size=window_size)
    test_dataset = StockDataset(test_data, window_size=window_size)

    return train_dataset, test_dataset, scaler, train_size


# ==================== 2. 模型定义 ====================


class StockLSTM(nn.Module):
    """LSTM 时间序列预测模型"""

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: 输入特征数（这里是1，只有收盘价）
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            dropout: Dropout 比例
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_size)

        Returns:
            output: (batch_size, 1)
        """
        # LSTM forward
        # lstm_out: (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # 只取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # 全连接层
        output = self.fc(last_output)  # (batch_size, 1)

        return output


# ==================== 3. 训练和评估 ====================


def train_epoch(model, train_loader, criterion, optimizer):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # 前向传播
        output = model(x)
        loss = criterion(output, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(y.cpu().numpy())

    return total_loss / len(test_loader), np.array(predictions), np.array(targets)


# ==================== 4. 可视化 ====================


def plot_predictions(
    train_losses, test_losses, predictions, targets, scaler, output_dir
):
    """绘制训练曲线和预测结果"""

    # 1. 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(train_losses, label="训练损失")
    axes[0].plot(test_losses, label="测试损失")
    axes[0].set_title("训练和测试损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True)

    # 2. 预测 vs 真实值（归一化）
    axes[1].plot(targets, label="真实值", alpha=0.7)
    axes[1].plot(predictions, label="预测值", alpha=0.7)
    axes[1].set_title("预测结果对比（归一化）")
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("归一化价格")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    print(f"✓ 训练曲线已保存到 {output_dir / 'training_curves.png'}")

    # 3. 预测 vs 真实值（原始价格）
    plt.figure(figsize=(15, 6))

    # 反归一化
    predictions_original = scaler.inverse_transform(predictions)
    targets_original = scaler.inverse_transform(targets)

    plt.plot(targets_original, label="真实价格", linewidth=2, alpha=0.8)
    plt.plot(predictions_original, label="预测价格", linewidth=2, alpha=0.8)
    plt.title("股票价格预测对比", fontsize=14, fontweight="bold")
    plt.xlabel("时间步", fontsize=12)
    plt.ylabel("价格", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "predictions.png", dpi=300, bbox_inches="tight")
    print(f"✓ 预测结果已保存到 {output_dir / 'predictions.png'}")


# ==================== 5. 主训练流程 ====================


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/stock_data.csv", help="数据文件路径"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    parser.add_argument("--window", type=int, default=30, help="时间窗口大小")
    parser.add_argument("--hidden", type=int, default=128, help="LSTM隐藏层大小")
    parser.add_argument("--layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument(
        "--test", action="store_true", help="测试模式（只训练2个epoch）"
    )
    args = parser.parse_args()

    if args.test:
        args.epochs = 2
        print("⚠️  测试模式：只训练 2 个 epoch")

    # 配置
    config = {
        "window_size": args.window,
        "hidden_size": args.hidden,
        "num_layers": args.layers,
        "dropout": 0.2,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "train_ratio": 0.8,
    }

    # 创建输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print("LSTM 股票价格预测")
    print(f"{'=' * 60}\n")

    # 检查数据文件
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        print("\n请先运行以下命令下载数据:")
        print("  python download_data.py")
        return

    # 加载数据
    train_dataset, test_dataset, scaler, train_size = load_and_preprocess_data(
        data_path, train_ratio=config["train_ratio"], window_size=config["window_size"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # 创建模型
    model = StockLSTM(
        input_size=1,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    print(f"\n模型参数:")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 训练循环
    print(f"\n开始训练...")
    print(f"{'=' * 60}\n")

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)

        # 评估
        test_loss, _, _ = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)

        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}] "
            f"训练损失: {train_loss:.6f}, 测试损失: {test_loss:.6f}"
        )

        # 学习率调度
        scheduler.step(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_loss,
                    "config": config,
                },
                checkpoint_dir / "best_model.pth",
            )

    print(f"\n{'=' * 60}")
    print("训练完成!")
    print(f"最佳测试损失: {best_test_loss:.6f}")
    print(f"{'=' * 60}\n")

    # 加载最佳模型进行最终评估
    checkpoint = torch.load(checkpoint_dir / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    # 最终预测
    _, predictions, targets = evaluate(model, test_loader, criterion)

    # 计算评估指标（原始价格）
    predictions_original = scaler.inverse_transform(predictions)
    targets_original = scaler.inverse_transform(targets)

    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_original, predictions_original)

    print("评估指标:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

    # 绘制结果
    plot_predictions(
        train_losses, test_losses, predictions, targets, scaler, predictions_dir
    )

    # 保存评估报告
    with open(predictions_dir / "evaluation_report.txt", "w") as f:
        f.write("LSTM 股票价格预测 - 评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型配置:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n评估指标:\n")
        f.write(f"  MSE:  {mse:.2f}\n")
        f.write(f"  MAE:  {mae:.2f}\n")
        f.write(f"  RMSE: {rmse:.2f}\n")
        f.write(f"  R²:   {r2:.4f}\n")

    print(f"\n✓ 评估报告已保存到 {predictions_dir / 'evaluation_report.txt'}")

    print(f"\n{'=' * 60}")
    print("项目完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
