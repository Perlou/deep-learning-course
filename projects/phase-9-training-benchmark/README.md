# Phase 9 实战项目：训练优化基准测试

## 项目简介

本项目综合运用 Phase 9 学习的所有训练优化技巧，通过在 CIFAR-10 数据集上训练 ResNet 模型，对比不同优化策略的效果。

## 技术栈

- Python 3.10+
- PyTorch 2.x
- torchvision
- matplotlib
- tqdm

## 项目结构

```
phase-9-training-benchmark/
├── README.md           # 项目说明
├── config.py           # 配置参数
├── model.py            # ResNet 模型
├── dataset.py          # 数据加载
├── trainer.py          # 训练器（核心）
├── train.py            # 训练入口
├── evaluate.py         # 评估脚本
├── benchmark.py        # 基准测试对比
└── outputs/
    ├── models/         # 保存的模型
    └── logs/           # 训练日志
```

## 快速开始

### 1. 训练单个配置

```bash
python projects/phase-9-training-benchmark/train.py
```

### 2. 运行基准测试（对比多种优化策略）

```bash
python projects/phase-9-training-benchmark/benchmark.py
```

### 3. 评估模型

```bash
python projects/phase-9-training-benchmark/evaluate.py
```

## 优化技巧演示

本项目演示以下优化技巧：

| 技巧         | 说明                                |
| ------------ | ----------------------------------- |
| 优化器对比   | SGD vs Adam vs AdamW                |
| 学习率调度   | StepLR / CosineAnnealing / OneCycle |
| 混合精度训练 | FP16 加速                           |
| 梯度累积     | 模拟大 Batch                        |
| 梯度裁剪     | 防止梯度爆炸                        |
| 权重衰减     | L2 正则化                           |
| Warmup       | 学习率预热                          |

## 实验结果示例

| 配置                   | 准确率 | 训练时间 |
| ---------------------- | ------ | -------- |
| SGD + StepLR           | ~85%   | 基准     |
| AdamW + Cosine         | ~87%   | 1.1x     |
| AdamW + OneCycle + AMP | ~88%   | 0.7x     |

## 学习要点

1. **优化器选择**：AdamW 通常比 Adam 更适合现代网络
2. **学习率调度**：OneCycleLR 可以实现超收敛
3. **混合精度**：可显著加速训练（~2x）且几乎不损失精度
4. **梯度累积**：小显存也能用大 Batch Size
5. **Warmup**：对大学习率训练很重要

## 参考资料

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [1cycle policy](https://arxiv.org/abs/1708.07120)
