"""
09-time-series.py - 时间序列预测

本节学习:
1. 时间序列预测任务
2. 数据准备方法
3. LSTM 时间序列预测
4. 多步预测策略
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第9节: 时间序列预测")
print("=" * 60)

# =============================================================================
# 1. 时间序列预测任务
# =============================================================================
print("""
📚 时间序列预测

任务类型:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  单步预测: 给定 [x₁, x₂, ..., xₜ] 预测 xₜ₊₁                  │
│                                                              │
│  多步预测: 给定 [x₁, x₂, ..., xₜ] 预测 [xₜ₊₁, ..., xₜ₊ₙ]    │
│                                                              │
│  常见应用:                                                    │
│    • 股票价格预测                                             │
│    • 天气预报                                                 │
│    • 能源消耗预测                                             │
│    • 设备故障预测                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 2. 生成示例数据
# =============================================================================
print("\n" + "=" * 60)
print("📌 2. 生成示例数据")
print("-" * 60)


def generate_synthetic_data(n_samples=1000, noise=0.1):
    """生成合成时间序列数据 (正弦波 + 趋势 + 噪声)"""
    t = np.linspace(0, 100, n_samples)

    # 主成分: 正弦波
    signal = np.sin(t * 0.5)

    # 添加趋势
    trend = t * 0.01

    # 添加季节性
    seasonal = 0.3 * np.sin(t * 0.1)

    # 添加噪声
    noise_component = noise * np.random.randn(n_samples)

    data = signal + trend + seasonal + noise_component

    return data.astype(np.float32)


# 生成数据
data = generate_synthetic_data()
print(f"数据形状: {data.shape}")
print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")

# 可视化
plt.figure(figsize=(12, 4))
plt.plot(data[:500], "b-", alpha=0.7)
plt.xlabel("时间步")
plt.ylabel("值")
plt.title("合成时间序列数据")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/time_series_data.png", dpi=100)
plt.close()
print("时间序列数据已保存: outputs/time_series_data.png")


# =============================================================================
# 3. 数据准备: 滑动窗口
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 数据准备: 滑动窗口")
print("-" * 60)


def create_sequences(data, seq_length, pred_length=1):
    """
    创建训练序列 (滑动窗口)
    Args:
        data: 原始时间序列
        seq_length: 输入序列长度
        pred_length: 预测长度
    Returns:
        X: [n_samples, seq_length, 1]
        y: [n_samples, pred_length]
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + pred_length])

    X = np.array(X)[:, :, np.newaxis]  # 添加特征维度
    y = np.array(y)

    return X, y


seq_length = 50
pred_length = 1

X, y = create_sequences(data, seq_length, pred_length)
print(f"输入序列形状: {X.shape}")
print(f"目标形状: {y.shape}")

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\n训练集: X={X_train.shape}, y={y_train.shape}")
print(f"测试集: X={X_test.shape}, y={y_test.shape}")

# 转换为 PyTorch 张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


# =============================================================================
# 4. LSTM 时间序列模型
# =============================================================================
print("\n" + "=" * 60)
print("📌 4. LSTM 时间序列模型")
print("-" * 60)


class LSTMPredictor(nn.Module):
    """LSTM 时间序列预测模型"""

    def __init__(
        self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            out: [batch, output_dim]
        """
        # LSTM 编码
        lstm_out, _ = self.lstm(x)

        # 取最后一个时刻的输出
        last_output = lstm_out[:, -1, :]

        # 预测
        out = self.fc(last_output)

        return out


model = LSTMPredictor(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
print(f"模型结构:\n{model}")
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试前向传播
test_input = torch.randn(4, 50, 1)
test_output = model(test_input)
print(f"测试输入: {test_input.shape} → 输出: {test_output.shape}")


# =============================================================================
# 5. 训练模型
# =============================================================================
print("\n" + "=" * 60)
print("📌 5. 训练模型")
print("-" * 60)

# 超参数
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# 数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

# 模型、损失、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPredictor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
train_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

# 训练曲线
plt.figure(figsize=(10, 4))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("训练损失曲线")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/time_series_training.png", dpi=100)
plt.close()
print("训练曲线已保存: outputs/time_series_training.png")


# =============================================================================
# 6. 评估和预测
# =============================================================================
print("\n" + "=" * 60)
print("📌 6. 评估和预测")
print("-" * 60)

model.eval()
with torch.no_grad():
    X_test_device = X_test.to(device)
    predictions = model(X_test_device).cpu().numpy()

    test_loss = criterion(torch.FloatTensor(predictions), y_test).item()

print(f"测试集 MSE: {test_loss:.6f}")
print(f"测试集 RMSE: {np.sqrt(test_loss):.6f}")

# 可视化预测结果
plt.figure(figsize=(12, 5))
plt.plot(y_test.numpy()[:200], "b-", label="真实值", alpha=0.7)
plt.plot(predictions[:200], "r--", label="预测值", alpha=0.7)
plt.xlabel("时间步")
plt.ylabel("值")
plt.title("时间序列预测结果")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/time_series_prediction.png", dpi=100)
plt.close()
print("预测结果已保存: outputs/time_series_prediction.png")


# =============================================================================
# 7. 多步预测
# =============================================================================
print("\n" + "=" * 60)
print("📌 7. 多步预测策略")
print("-" * 60)

print("""
多步预测策略:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  1. 直接多输出 (Direct)                                      │
│     模型直接输出多步: [xₜ₊₁, xₜ₊₂, ..., xₜ₊ₙ]               │
│     优点: 简单高效                                           │
│     缺点: 难以建模步间依赖                                   │
│                                                              │
│  2. 递归预测 (Recursive)                                     │
│     预测 xₜ₊₁ → 作为输入 → 预测 xₜ₊₂ → ...                  │
│     优点: 可用单步模型                                       │
│     缺点: 误差累积                                           │
│                                                              │
│  3. Seq2Seq                                                  │
│     编码器处理输入 → 解码器生成多步输出                      │
│     优点: 建模能力强                                         │
│     缺点: 模型复杂                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")


def recursive_predict(model, initial_seq, n_steps):
    """递归多步预测"""
    model.eval()
    predictions = []
    current_seq = initial_seq.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current_seq)
            predictions.append(pred.item())

            # 滑动窗口: 移除最早的，添加新预测
            current_seq = torch.roll(current_seq, -1, dims=1)
            current_seq[0, -1, 0] = pred

    return predictions


# 多步预测示例
initial = X_test[0:1].to(device)
multi_preds = recursive_predict(model, initial, n_steps=50)

plt.figure(figsize=(12, 4))
plt.plot(range(50), y_test[:50].numpy().flatten(), "b-", label="真实值", alpha=0.7)
plt.plot(range(50), multi_preds, "r--", label="多步预测", alpha=0.7)
plt.xlabel("时间步")
plt.ylabel("值")
plt.title("递归多步预测 (50步)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/multi_step_prediction.png", dpi=100)
plt.close()
print("多步预测结果已保存: outputs/multi_step_prediction.png")


# =============================================================================
# 8. 练习
# =============================================================================
print("\n" + "=" * 60)
print("📝 练习题")
print("-" * 60)

print("""
1. 滑动窗口序列长度如何选择？
   答：取决于数据的周期性；一般包含 1-2 个完整周期

2. 递归多步预测的主要问题是什么？
   答：误差累积，早期的预测误差会传播到后续步骤

3. 时间序列预测中为什么 LSTM 优于普通 RNN？
   答：能捕捉长期依赖，如季节性模式

4. 如何处理多变量时间序列？
   答：将 input_dim 设为变量数量，每个时间步输入多个特征
""")

print("\n✅ 第9节完成！")
print("=" * 60)
print("🎉 恭喜完成 Phase 6: RNN/LSTM 全部课程！")
print("=" * 60)
