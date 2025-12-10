# Phase 9: 训练技巧与优化

> **目标**：掌握工业级训练技巧  
> **预计时长**：1-2 周  
> **前置条件**：Phase 1-8 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 深入理解各种优化器的原理
2. 掌握学习率调度策略
3. 使用混合精度训练加速
4. 理解分布式训练的基本原理
5. 使用 Optuna 进行超参数调优

---

## 📚 核心概念

### 优化器

| 优化器       | 特点         | 适用场景   |
| ------------ | ------------ | ---------- |
| SGD+Momentum | 经典稳定     | CV 模型    |
| Adam         | 自适应学习率 | 通用首选   |
| AdamW        | 解耦权重衰减 | 大模型训练 |

### 学习率调度

- **StepLR**: 阶梯式衰减
- **CosineAnnealingLR**: 余弦退火
- **OneCycleLR**: 超收敛
- **WarmupLR**: 预热 + 衰减

### 混合精度训练

使用 FP16 加速训练：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📁 文件列表

| 文件                          | 描述               | 状态 |
| ----------------------------- | ------------------ | ---- |
| `01-sgd-momentum.py`          | SGD 及动量         | ⏳   |
| `02-adam-variants.py`         | Adam、AdamW        | ⏳   |
| `03-lr-schedulers.py`         | 学习率调度         | ⏳   |
| `04-gradient-clipping.py`     | 梯度裁剪           | ⏳   |
| `05-mixed-precision.py`       | 混合精度训练       | ⏳   |
| `06-gradient-accumulation.py` | 梯度累积           | ⏳   |
| `07-data-parallel.py`         | 数据并行           | ⏳   |
| `08-distributed-training.py`  | 分布式训练基础     | ⏳   |
| `09-hyperparameter-tuning.py` | 网格搜索、随机搜索 | ⏳   |
| `10-optuna-integration.py`    | Optuna 自动调参    | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-9-optimization/01-sgd-momentum.py
python src/phase-9-optimization/05-mixed-precision.py
```

---

## 📖 推荐资源

- [PyTorch 分布式训练文档](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Optuna 官方文档](https://optuna.org/)

---

## ✅ 完成检查

- [ ] 理解各种优化器的区别
- [ ] 能够选择合适的学习率调度策略
- [ ] 能够使用混合精度训练
- [ ] 理解梯度累积的作用
- [ ] 了解数据并行的基本原理
- [ ] 能够使用 Optuna 进行超参数调优
