# Phase 6 实战项目 - LSTM 股票价格预测

## 📋 项目概述

使用 **多层 LSTM** 进行股票价格的时间序列预测，通过历史价格数据预测未来走势。

## 🎯 学习目标

- 理解时间序列预测的基本原理
- 掌握滑动窗口（Sliding Window）构建监督学习数据
- 学习 LSTM 在时间序列任务中的应用
- 了解数据归一化的重要性
- 掌握时间序列模型的评估方法

## 📊 数据集

**股票历史数据**（通过 yfinance 下载）：

- 默认使用 S&P 500 指数 (^GSPC)
- 时间范围：2020-01-01 至今
- 特征：开盘价、收盘价、最高价、最低价、成交量
- 本项目使用：收盘价（Close）

## 🏗️ 模型架构

**LSTM 时间序列预测模型**：

```
输入: 过去30天的价格 (30, 1)
    ↓
LSTM(1 → 128, num_layers=2, dropout=0.2)
    ↓
取最后时间步输出 (128,)
    ↓
Linear(128 → 1)
    ↓
输出: 下一天的价格 (1,)
```

### 关键组件

1. **滑动窗口**：将时间序列转换为监督学习问题

   - 输入 X：过去 30 天的价格
   - 目标 y：第 31 天的价格

2. **数据归一化**：使用 MinMaxScaler 归一化到 [0, 1]
3. **双层 LSTM**：捕捉时间序列的复杂模式
4. **序列预测**：基于历史数据预测未来值

## 🚀 运行方式

### 1. 安装依赖

```bash
# yfinance 用于下载股票数据
pip install yfinance
```

### 2. 下载数据

```bash
cd projects/phase-6-stock-prediction

# 下载 S&P 500 指数数据（默认）
python download_data.py

# 或下载其他股票
python download_data.py --symbol AAPL --start 2020-01-01

# 其他示例：
# 道琼斯指数: --symbol ^DJI
# 特斯拉: --symbol TSLA
# 上证指数: --symbol 000001.SS
```

### 3. 训练模型

```bash
python stock_predictor.py
```

可选参数：

```bash
python stock_predictor.py \
    --epochs 100 \
    --batch-size 32 \
    --window 30 \
    --hidden 128 \
    --layers 2 \
    --lr 0.001
```

### 4. 查看结果

训练完成后会生成：

- `outputs/predictions/training_curves.png` - 训练曲线和预测对比
- `outputs/predictions/predictions.png` - 预测价格 vs 真实价格
- `outputs/predictions/evaluation_report.txt` - 评估指标报告
- `outputs/checkpoints/best_model.pth` - 最佳模型权重

## 📈 预期结果

- **训练时间**：~3-5 分钟 (CPU) / ~1-2 分钟 (GPU)
- **评估指标**：
  - MAE (平均绝对误差)：通常在价格的 1-3% 范围内
  - RMSE (均方根误差)：取决于价格波动性
  - R² (决定系数)：通常在 0.85-0.95 之间

### 示例结果

```
评估指标:
  MSE:  245.32
  MAE:  12.45
  RMSE: 15.66
  R²:   0.9234
```

## ✅ 关键知识点

### 1. 滑动窗口

将时间序列转换为监督学习问题：

```python
# 原始时间序列: [p1, p2, p3, p4, p5, p6, ...]
# 窗口大小 = 3

# 训练样本:
# X1 = [p1, p2, p3], y1 = p4
# X2 = [p2, p3, p4], y2 = p5
# X3 = [p3, p4, p5], y3 = p6
```

### 2. 数据归一化

为什么需要归一化？

- 不同特征的尺度差异大
- 加快训练收敛速度
- 避免数值稳定性问题

```python
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 预测后需要反归一化
predictions_original = scaler.inverse_transform(predictions_scaled)
```

### 3. 时间序列数据划分

⚠️ **重要**：不能随机打乱！

```python
# ✓ 正确：按时间顺序划分
train_data = data[:train_size]
test_data = data[train_size:]

# ✗ 错误：随机划分会导致数据泄露
```

### 4. 评估指标

- **MAE**：平均绝对误差，易于解释
- **RMSE**：均方根误差，对大误差更敏感
- **R²**：决定系数，衡量模型解释方差的能力

## 📁 项目结构

```
phase-6-stock-prediction/
├── stock_predictor.py      # 主训练和预测程序
├── download_data.py         # 数据下载脚本
├── data/
│   └── stock_data.csv       # 股票历史数据
├── outputs/
│   ├── predictions/         # 预测结果和图表
│   │   ├── training_curves.png
│   │   ├── predictions.png
│   │   └── evaluation_report.txt
│   └── checkpoints/         # 模型权重
│       └── best_model.pth
└── README.md
```

## 🔧 超参数配置

| 参数            | 值    | 说明            |
| --------------- | ----- | --------------- |
| `window_size`   | 30    | 时间窗口（天）  |
| `hidden_size`   | 128   | LSTM 隐藏层大小 |
| `num_layers`    | 2     | LSTM 层数       |
| `dropout`       | 0.2   | Dropout 比例    |
| `batch_size`    | 32    | 批量大小        |
| `learning_rate` | 0.001 | 初始学习率      |
| `num_epochs`    | 100   | 训练轮数        |
| `train_ratio`   | 0.8   | 训练集比例      |

## 🚀 扩展建议

1. **多变量预测**：

   - 同时使用开盘价、最高价、最低价、成交量
   - 添加技术指标（MA, RSI, MACD）

2. **多步预测**：

   - 预测未来多天的价格
   - 使用 Seq2Seq 架构

3. **注意力机制**：

   - 添加注意力层关注重要时间步
   - 使用 Transformer 替代 LSTM

4. **双向 LSTM**：

   - 虽然实时预测不能使用未来信息
   - 但可用于离线分析和回测

5. **集成学习**：
   - 训练多个模型进行集成
   - 结合 LSTM、GRU、Transformer

## ⚠️ 免责声明

- 本项目仅用于学习深度学习和时间序列预测技术
- **不构成任何投资建议**
- 股票市场受多种因素影响，历史价格不能完全预测未来
- 实际投资请咨询专业金融顾问

## 📖 参考资料

- [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Forecasting with LSTM](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

## 💡 常见问题

**Q: 为什么预测结果总是滞后一天？**  
A: 这是时间序列预测的常见问题。模型学会了"复制"前一天的值。可以尝试：

- 增加窗口大小
- 添加更多特征
- 使用差分（预测变化量而非绝对值）

**Q: 如何避免过拟合？**  
A:

- 使用 Dropout
- 早停（Early Stopping）
- 减少模型容量（减少隐藏层大小或层数）

**Q: 可以预测股票涨跌吗？**  
A: 可以！将问题转换为分类任务：

- 标签：涨(1) / 跌(0)
- 损失函数：交叉熵
- 评估指标：准确率、精确率、召回率

---

**完成时间**：2024-01-15  
**Phase**：6 - RNN/LSTM  
**难度**：⭐⭐⭐⭐
