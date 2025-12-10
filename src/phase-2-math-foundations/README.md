# Phase 2: 深度学习数学基础

> **目标**：掌握深度学习必需的数学知识  
> **预计时长**：1-2 周  
> **前置条件**：Phase 1 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解向量、矩阵运算在深度学习中的应用
2. 掌握梯度、链式法则（反向传播的数学基础）
3. 理解概率论基础和信息论概念
4. 能够手动推导简单神经网络的反向传播

---

## 📚 核心概念

### 线性代数

- **向量与矩阵**：神经网络的基本运算单位
- **矩阵乘法**：全连接层的核心操作
- **特征分解与 SVD**：数据降维、理解模型

### 微积分

- **偏导数与梯度**：优化的方向
- **链式法则**：反向传播的数学基础
- **梯度下降**：参数更新的核心算法

### 概率论与信息论

- **概率分布**：理解数据生成过程
- **交叉熵**：分类任务的损失函数
- **KL 散度**：分布间的距离度量

---

## 📁 文件列表

| 文件                          | 描述                | 状态 |
| ----------------------------- | ------------------- | ---- |
| `01-vectors-matrices.py`      | 向量空间、矩阵运算  | ⏳   |
| `02-eigenvalue-svd.py`        | 特征分解、SVD       | ⏳   |
| `03-derivatives-gradients.py` | 偏导数、梯度        | ⏳   |
| `04-chain-rule.py`            | 链式法则            | ⏳   |
| `05-optimization-basics.py`   | 梯度下降可视化      | ⏳   |
| `06-probability-basics.py`    | 概率分布            | ⏳   |
| `07-entropy-kl-divergence.py` | 熵、KL 散度、交叉熵 | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-2-math-foundations/01-vectors-matrices.py
python src/phase-2-math-foundations/05-optimization-basics.py
```

---

## 📖 推荐资源

- 《深度学习》花书 - 第 2-4 章
- [3Blue1Brown 线性代数系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown 微积分系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

---

## ✅ 完成检查

- [ ] 能够进行矩阵乘法和转置运算
- [ ] 理解特征值和特征向量的意义
- [ ] 能够计算简单函数的梯度
- [ ] 理解链式法则并能手动推导
- [ ] 能够实现梯度下降算法
- [ ] 理解交叉熵损失的含义
