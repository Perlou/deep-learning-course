# Phase 4: 神经网络基础

> **目标**：理解神经网络核心原理  
> **预计时长**：1 周  
> **前置条件**：Phase 3 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解感知机和多层网络的原理
2. 手动实现前向传播和反向传播
3. 掌握各种激活函数的特点和选择
4. 理解正则化技术：Dropout、BatchNorm、L1/L2
5. 掌握权重初始化的重要性

---

## 📁 文件列表

| 文件                          | 描述               | 状态 |
| ----------------------------- | ------------------ | ---- |
| `01-perceptron.py`            | 单层感知机         | ✅   |
| `02-mlp-basic.py`             | 多层感知机         | ✅   |
| `03-forward-backward.py`      | 前向传播与反向传播 | ✅   |
| `04-activation-functions.py`  | 激活函数详解       | ✅   |
| `05-activation-comparison.py` | 激活函数对比分析   | ✅   |
| `06-dropout.py`               | Dropout 原理与实现 | ✅   |
| `07-batch-normalization.py`   | BatchNorm 详解     | ✅   |
| `08-weight-regularization.py` | L1/L2 正则化       | ✅   |
| `09-weight-init.py`           | Xavier、He 初始化  | ✅   |

---

## 🚀 运行方式

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行单个文件
python src/phase-4-neural-network-basics/01-perceptron.py
```

---

## 🏆 实战项目

**房价预测（回归任务）**

- 位置: `projects/phase-4-regression/`
- 目标: 使用 MLP 解决回归问题

---

## ✅ 完成检查

- [ ] 理解感知机的决策边界
- [ ] 能手动实现反向传播
- [ ] 知道激活函数的梯度和特点
- [ ] 理解 Dropout 的训练/推理差异
- [ ] 理解 BatchNorm 的作用
- [ ] 能选择合适的权重初始化方法
