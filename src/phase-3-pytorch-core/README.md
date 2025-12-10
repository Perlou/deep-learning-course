# Phase 3: PyTorch 核心技能

> **目标**：深入理解 PyTorch 框架  
> **预计时长**：1 周  
> **前置条件**：Phase 1-2 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 熟练操作 PyTorch Tensor
2. 理解自动微分 (Autograd) 机制
3. 使用 nn.Module 构建神经网络
4. 实现完整的训练循环
5. 完成 MNIST 手写数字分类项目

---

## 📚 核心概念

### Tensor

PyTorch 的核心数据结构：

- 类似 NumPy 的 ndarray
- 支持 GPU 加速
- 支持自动微分

### Autograd

自动微分机制：

- `requires_grad=True` 追踪计算图
- `.backward()` 自动计算梯度
- `.grad` 获取梯度值

### nn.Module

神经网络的基类：

- `__init__` 定义层
- `forward` 定义前向传播
- 自动管理参数

### 训练循环

```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

## 📁 文件列表

| 文件                       | 描述                  | 状态 |
| -------------------------- | --------------------- | ---- |
| `01-tensor-basics.py`      | 创建、属性、设备      | ⏳   |
| `02-tensor-operations.py`  | 运算、索引、变形      | ⏳   |
| `03-tensor-autograd.py`    | 自动微分机制          | ⏳   |
| `04-nn-module.py`          | nn.Module 深入        | ⏳   |
| `05-loss-functions.py`     | 损失函数详解          | ⏳   |
| `06-optimizers.py`         | 优化器原理与使用      | ⏳   |
| `07-dataset-dataloader.py` | Dataset 和 DataLoader | ⏳   |
| `08-data-augmentation.py`  | 数据增强技术          | ⏳   |
| `09-training-loop.py`      | 完整训练流程          | ⏳   |
| `10-model-save-load.py`    | 模型保存与加载        | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-3-pytorch-core/01-tensor-basics.py
python src/phase-3-pytorch-core/09-training-loop.py
```

---

## 📖 推荐资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [动手学深度学习 PyTorch 版](https://d2l.ai/)
- [PyTorch 中文文档](https://pytorch-cn.readthedocs.io/)

---

## ✅ 完成检查

- [ ] 能够创建和操作 Tensor
- [ ] 理解 GPU 和 CPU 之间的数据转移
- [ ] 理解计算图和自动微分
- [ ] 能够使用 nn.Module 定义网络
- [ ] 理解各种损失函数的适用场景
- [ ] 能够实现完整的训练循环
- [ ] 完成 MNIST 分类项目
