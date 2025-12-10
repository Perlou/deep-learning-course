# Phase 4: 神经网络基础

> **目标**：理解神经网络核心原理  
> **预计时长**：1 周  
> **前置条件**：Phase 1-3 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解感知机和多层感知机 (MLP) 原理
2. 掌握前向传播和反向传播的完整过程
3. 理解各种激活函数的特性和选择
4. 掌握正则化技术（Dropout、BatchNorm）
5. 理解权重初始化的重要性

---

## 📚 核心概念

### 感知机

最基本的神经网络单元：

```
y = f(w·x + b)
```

### 多层感知机 (MLP)

由多个全连接层组成：

```
输入 → 隐藏层1 → 隐藏层2 → ... → 输出
```

### 激活函数

引入非线性：

- **ReLU**: max(0, x)，最常用
- **Sigmoid**: 1/(1+e^-x)，输出层二分类
- **GELU**: Transformer 中常用

### 正则化

防止过拟合：

- **Dropout**: 随机丢弃神经元
- **BatchNorm**: 归一化中间层
- **L1/L2 正则化**: 权重惩罚

---

## 📁 文件列表

| 文件                          | 描述                      | 状态 |
| ----------------------------- | ------------------------- | ---- |
| `01-perceptron.py`            | 单层感知机                | ⏳   |
| `02-mlp-basic.py`             | 多层感知机                | ⏳   |
| `03-forward-backward.py`      | 前向传播与反向传播        | ⏳   |
| `04-activation-functions.py`  | ReLU、Sigmoid、Tanh、GELU | ⏳   |
| `05-activation-comparison.py` | 激活函数对比分析          | ⏳   |
| `06-dropout.py`               | Dropout 原理与实现        | ⏳   |
| `07-batch-normalization.py`   | BatchNorm 详解            | ⏳   |
| `08-weight-regularization.py` | L1/L2 正则化              | ⏳   |
| `09-weight-init.py`           | Xavier、He 初始化         | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-4-neural-networks/01-perceptron.py
python src/phase-4-neural-networks/02-mlp-basic.py
```

---

## 📖 推荐资源

- 《深度学习》花书 - 第 6-7 章
- [CS231n 神经网络讲义](https://cs231n.github.io/)

---

## ✅ 完成检查

- [ ] 能够手动实现感知机
- [ ] 理解 MLP 的结构和作用
- [ ] 能够手动推导反向传播
- [ ] 理解各种激活函数的优缺点
- [ ] 理解 Dropout 的工作原理
- [ ] 能够正确使用 BatchNorm
- [ ] 理解权重初始化的重要性
- [ ] 完成房价预测回归项目
