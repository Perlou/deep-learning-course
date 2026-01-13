"""
房价预测 MLP 回归模型
Phase 4 实战项目

学习目标：
1. MLP 在回归任务中的应用
2. 特征标准化和数据预处理
3. 正则化技术：Dropout, BatchNorm, L2
4. 回归评估指标：MSE, RMSE, MAE, R²
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("Phase 4 实战项目：房价预测 MLP 回归")
print("=" * 60)


# =============================================================================
# 1. 配置
# =============================================================================
class Config:
    # 数据
    test_size = 0.2
    val_size = 0.1

    # 模型
    hidden_dims = [128, 64, 32]
    dropout_rate = 0.2

    # 训练
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 1e-4  # L2 正则化
    num_epochs = 100
    patience = 15  # 早停

    # 保存
    save_dir = "./outputs"

    # 设备
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


config = Config()
os.makedirs(config.save_dir, exist_ok=True)
print(f"\n使用设备: {config.device}")


# =============================================================================
# 2. 数据准备
# =============================================================================
print("\n" + "=" * 60)
print("【1. 数据准备】")

# 加载 California Housing 数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

print(f"\n数据集信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  特征名: {feature_names}")
print(f"  目标范围: {y.min():.2f} ~ {y.max():.2f} (单位: $100,000)")

# 划分数据集
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=config.test_size, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=config.val_size / (1 - config.test_size), random_state=42
)

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)}")
print(f"  验证集: {len(X_val)}")
print(f"  测试集: {len(X_test)}")

# 特征标准化（非常重要！）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# 目标值也标准化，有助于训练稳定性
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"\n特征标准化后:")
print(f"  X 均值: {X_train_scaled.mean():.4f}")
print(f"  X 标准差: {X_train_scaled.std():.4f}")


# 转换为 PyTorch 张量
def to_tensor(X, y, device):
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.FloatTensor(y).unsqueeze(1).to(device)
    return TensorDataset(X_t, y_t)


train_dataset = to_tensor(X_train_scaled, y_train_scaled, config.device)
val_dataset = to_tensor(X_val_scaled, y_val_scaled, config.device)
test_dataset = to_tensor(X_test_scaled, y_test_scaled, config.device)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)


# =============================================================================
# 3. 可视化数据分布
# =============================================================================
print("\n" + "=" * 60)
print("【2. 数据可视化】")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i, (name, ax) in enumerate(zip(feature_names, axes)):
    ax.hist(X[:, i], bins=30, edgecolor="black", alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

plt.suptitle("特征分布", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(
    f"{config.save_dir}/feature_distributions.png", dpi=100, bbox_inches="tight"
)
plt.close()
print(f"特征分布图已保存: {config.save_dir}/feature_distributions.png")

# 目标分布
plt.figure(figsize=(8, 5))
plt.hist(y, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
plt.xlabel("房价中位数 ($100,000)")
plt.ylabel("样本数")
plt.title("目标变量分布")
plt.axvline(y.mean(), color="red", linestyle="--", label=f"均值: {y.mean():.2f}")
plt.legend()
plt.savefig(f"{config.save_dir}/target_distribution.png", dpi=100)
plt.close()
print(f"目标分布图已保存: {config.save_dir}/target_distribution.png")


# =============================================================================
# 4. 模型定义
# =============================================================================
print("\n" + "=" * 60)
print("【3. 模型定义】")


class MLPRegressor(nn.Module):
    """
    多层感知机回归模型

    特点:
    - He 初始化 (适合 ReLU)
    - BatchNorm (加速训练、稳定梯度)
    - Dropout (防止过拟合)
    - 线性输出层 (回归任务)
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            linear = nn.Linear(prev_dim, hidden_dim)

            # He 初始化
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)

            layers.append(linear)

            # BatchNorm (除了最后一层)
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # 激活函数
            layers.append(nn.ReLU(inplace=True))

            # Dropout (除了最后一层)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)

        # 输出层 - 线性（回归任务无激活函数）
        self.output = nn.Linear(hidden_dims[-1], 1)
        nn.init.kaiming_normal_(
            self.output.weight, mode="fan_in", nonlinearity="linear"
        )
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x


# 创建模型
input_dim = X.shape[1]
model = MLPRegressor(
    input_dim=input_dim,
    hidden_dims=config.hidden_dims,
    dropout_rate=config.dropout_rate,
).to(config.device)

print(f"\n模型结构:\n{model}")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")


# =============================================================================
# 5. 损失函数和优化器
# =============================================================================
print("\n" + "=" * 60)
print("【4. 训练配置】")

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,  # L2 正则化
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

print(f"损失函数: MSE")
print(f"优化器: Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})")
print(f"学习率调度: ReduceLROnPlateau")


# =============================================================================
# 6. 训练和验证函数
# =============================================================================
def train_epoch(model, loader, criterion, optimizer):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
            predictions.append(pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, predictions, targets


# =============================================================================
# 7. 训练循环
# =============================================================================
print("\n" + "=" * 60)
print("【5. 开始训练】")

history = {"train_loss": [], "val_loss": [], "lr": []}

best_val_loss = float("inf")
best_model_state = None
epochs_without_improvement = 0
start_time = time.time()

for epoch in range(config.num_epochs):
    # 训练
    train_loss = train_epoch(model, train_loader, criterion, optimizer)

    # 验证
    val_loss, _, _ = evaluate(model, val_loader, criterion)

    # 记录历史
    current_lr = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["lr"].append(current_lr)

    # 学习率调度
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # 打印进度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1:3d}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    # 早停
    if epochs_without_improvement >= config.patience:
        print(f"\n早停触发! 验证损失已 {config.patience} 个 epoch 未改善")
        break

elapsed_time = time.time() - start_time
print(f"\n训练完成! 用时: {elapsed_time:.2f}s")
print(f"最佳验证损失: {best_val_loss:.4f}")


# =============================================================================
# 8. 绘制训练曲线
# =============================================================================
print("\n" + "=" * 60)
print("【6. 训练可视化】")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1 = axes[0]
ax1.plot(history["train_loss"], label="训练损失", color="blue")
ax1.plot(history["val_loss"], label="验证损失", color="orange")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.set_title("训练曲线")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 学习率曲线
ax2 = axes[1]
ax2.plot(history["lr"], color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning Rate")
ax2.set_title("学习率变化")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{config.save_dir}/training_curves.png", dpi=100)
plt.close()
print(f"训练曲线已保存: {config.save_dir}/training_curves.png")


# =============================================================================
# 9. 测试评估
# =============================================================================
print("\n" + "=" * 60)
print("【7. 测试评估】")

# 加载最佳模型
model.load_state_dict(best_model_state)

# 测试
test_loss, predictions_scaled, targets_scaled = evaluate(model, test_loader, criterion)

# 反标准化预测值
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
targets = scaler_y.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

# 计算回归指标
mse = mean_squared_error(targets, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)

print(f"\n测试集评估指标:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

# 解释 R²
if r2 > 0.8:
    r2_comment = "优秀"
elif r2 > 0.6:
    r2_comment = "良好"
elif r2 > 0.4:
    r2_comment = "一般"
else:
    r2_comment = "较差"
print(f"\n模型表现: {r2_comment} (R² = {r2:.4f})")


# =============================================================================
# 10. 预测可视化
# =============================================================================
print("\n" + "=" * 60)
print("【8. 预测可视化】")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 预测 vs 真实值散点图
ax1 = axes[0]
ax1.scatter(targets, predictions, alpha=0.5, s=10)
ax1.plot(
    [targets.min(), targets.max()],
    [targets.min(), targets.max()],
    "r--",
    linewidth=2,
    label="理想预测",
)
ax1.set_xlabel("真实房价 ($100,000)")
ax1.set_ylabel("预测房价 ($100,000)")
ax1.set_title(f"预测 vs 真实值 (R² = {r2:.4f})")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 残差分布
ax2 = axes[1]
residuals = predictions - targets
ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
ax2.axvline(0, color="red", linestyle="--", linewidth=2)
ax2.set_xlabel("残差 (预测 - 真实)")
ax2.set_ylabel("样本数")
ax2.set_title(f"残差分布 (均值: {residuals.mean():.4f})")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{config.save_dir}/predictions.png", dpi=100)
plt.close()
print(f"预测可视化已保存: {config.save_dir}/predictions.png")


# =============================================================================
# 11. 特征重要性分析
# =============================================================================
print("\n" + "=" * 60)
print("【9. 特征重要性分析】")

# 使用第一层权重的绝对值作为简单的特征重要性估计
first_layer_weight = model.hidden[0].weight.data.cpu().numpy()
importance = np.abs(first_layer_weight).mean(axis=0)
importance = importance / importance.sum()  # 归一化

# 排序
sorted_idx = np.argsort(importance)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = importance[sorted_idx]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_features)), sorted_importance[::-1], color="steelblue")
plt.yticks(range(len(sorted_features)), sorted_features[::-1])
plt.xlabel("相对重要性")
plt.title("特征重要性 (基于第一层权重)")
plt.grid(True, alpha=0.3, axis="x")

# 添加数值标签
for bar, val in zip(bars, sorted_importance[::-1]):
    plt.text(
        val + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(f"{config.save_dir}/feature_importance.png", dpi=100)
plt.close()
print(f"特征重要性已保存: {config.save_dir}/feature_importance.png")

print("\n特征重要性排名:")
for i, (name, imp) in enumerate(zip(sorted_features, sorted_importance), 1):
    print(f"  {i}. {name}: {imp:.4f}")


# =============================================================================
# 12. 残差分析
# =============================================================================
print("\n" + "=" * 60)
print("【10. 残差分析】")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 残差 vs 预测值
ax1 = axes[0]
ax1.scatter(predictions, residuals, alpha=0.5, s=10)
ax1.axhline(0, color="red", linestyle="--", linewidth=2)
ax1.set_xlabel("预测值")
ax1.set_ylabel("残差")
ax1.set_title("残差 vs 预测值")
ax1.grid(True, alpha=0.3)

# Q-Q 图 (简化版)
ax2 = axes[1]
sorted_residuals = np.sort(residuals)
theoretical_quantiles = np.linspace(0.001, 0.999, len(sorted_residuals))
from scipy import stats

theoretical_values = (
    stats.norm.ppf(theoretical_quantiles) * residuals.std() + residuals.mean()
)
ax2.scatter(theoretical_values, sorted_residuals, alpha=0.5, s=10)
ax2.plot(
    [theoretical_values.min(), theoretical_values.max()],
    [theoretical_values.min(), theoretical_values.max()],
    "r--",
    linewidth=2,
)
ax2.set_xlabel("理论分位数")
ax2.set_ylabel("样本分位数")
ax2.set_title("Q-Q 图 (正态性检验)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{config.save_dir}/residuals.png", dpi=100)
plt.close()
print(f"残差分析已保存: {config.save_dir}/residuals.png")


# =============================================================================
# 13. 保存模型
# =============================================================================
print("\n" + "=" * 60)
print("【11. 保存模型】")

# 保存最佳模型
model_path = f"{config.save_dir}/best_model.pth"
torch.save(
    {
        "model_state_dict": best_model_state,
        "config": {
            "input_dim": input_dim,
            "hidden_dims": config.hidden_dims,
            "dropout_rate": config.dropout_rate,
        },
        "scaler_X_mean": scaler_X.mean_,
        "scaler_X_scale": scaler_X.scale_,
        "scaler_y_mean": scaler_y.mean_,
        "scaler_y_scale": scaler_y.scale_,
        "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
    },
    model_path,
)
print(f"模型已保存: {model_path}")


# =============================================================================
# 14. 推理示例
# =============================================================================
print("\n" + "=" * 60)
print("【12. 推理示例】")


def predict_house_price(model, features, scaler_X, scaler_y, device):
    """预测单个样本的房价"""
    model.eval()
    with torch.no_grad():
        # 标准化
        features_scaled = scaler_X.transform(features.reshape(1, -1))
        # 转为张量
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        # 预测
        prediction_scaled = model(features_tensor).cpu().numpy()
        # 反标准化
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    return prediction[0, 0]


# 随机选择几个测试样本
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), 5, replace=False)

print("\n预测示例:")
print("-" * 60)
for idx in sample_indices:
    sample_features = X_test[idx]
    true_price = y_test[idx]
    pred_price = predict_house_price(
        model, sample_features, scaler_X, scaler_y, config.device
    )
    error = abs(pred_price - true_price)
    print(
        f"真实: ${true_price * 100000:,.0f} | 预测: ${pred_price * 100000:,.0f} | 误差: ${error * 100000:,.0f}"
    )


# =============================================================================
# 15. 总结
# =============================================================================
print("\n" + "=" * 60)
print("【项目总结】")
print("=" * 60)

print(f"""
应用的 Phase 4 知识点:
  ✅ 多层感知机 (MLP) 架构
  ✅ He 初始化 (适合 ReLU 激活)
  ✅ ReLU 激活函数 (隐藏层)
  ✅ BatchNorm (加速训练、稳定梯度)
  ✅ Dropout (防止过拟合)
  ✅ L2 正则化 (weight_decay)
  ✅ Adam 优化器
  ✅ 学习率调度
  ✅ 早停策略

回归任务特点:
  • 输出层无激活函数 (线性输出)
  • MSE 损失函数
  • 评估指标: RMSE, MAE, R²

生成文件:
  📊 {config.save_dir}/feature_distributions.png
  📊 {config.save_dir}/target_distribution.png
  📊 {config.save_dir}/training_curves.png
  📊 {config.save_dir}/predictions.png
  📊 {config.save_dir}/feature_importance.png
  📊 {config.save_dir}/residuals.png
  💾 {config.save_dir}/best_model.pth
""")

print("✅ Phase 4 实战项目完成！")
