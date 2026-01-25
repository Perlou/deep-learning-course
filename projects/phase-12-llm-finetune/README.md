# LLM 微调实战项目

> Phase 12: 大模型与前沿技术 - 实战项目

## 项目简介

本项目实现了一个完整的 LLM 指令微调流程，包括：

- **LoRA 高效微调**：只训练 0.5% 的参数，大幅减少显存和计算需求
- **指令数据处理**：ChatML 对话模板格式化
- **完整训练流程**：梯度累积、学习率调度、早停机制
- **评估与推理**：困惑度计算、Top-k/Top-p 采样、交互对话

## 技术栈

- Python 3.10+
- PyTorch 2.x
- 纯手工实现（无外部LLM库依赖）

## 项目结构

```
phase-12-llm-finetune/
├── README.md           # 本文档
├── config.py           # 配置管理
├── model.py            # 模型定义 + LoRA
├── dataset.py          # 数据集处理
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── inference.py        # 推理脚本
├── utils.py            # 工具函数
├── main.py             # 主入口
└── outputs/            # 输出目录
    ├── models/         # 模型权重
    └── logs/           # 训练日志
```

## 快速开始

### 1. 环境准备

```bash
# 在项目根目录激活虚拟环境
source venv/bin/activate
```

### 2. 运行演示

```bash
python projects/phase-12-llm-finetune/main.py demo
```

### 3. 训练模型

```bash
python projects/phase-12-llm-finetune/main.py train --epochs 3
```

### 4. 评估模型

```bash
python projects/phase-12-llm-finetune/main.py eval
```

### 5. 交互对话

```bash
python projects/phase-12-llm-finetune/main.py chat
```

## 核心概念

### LoRA (Low-Rank Adaptation)

LoRA 是一种高效的模型微调方法，核心思想是：

```
原始权重 W 固定，只训练低秩增量 ΔW = B × A

W: (d × d) 矩阵，参数量 d²
ΔW = B × A
A: (d × r), B: (r × d), 参数量 2dr

当 r << d 时，参数量大幅减少
```

**参数量对比**（d=4096, r=8）：
| 方法 | 参数量 | 比例 |
|------|--------|------|
| 全量微调 | 16.7M | 100% |
| LoRA | 65K | 0.4% |

### ChatML 对话模板

```
<|im_start|>system
你是AI助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的吗？<|im_end|>
```

### 模型架构

本项目实现的 TinyLLM 采用现代 LLM 架构：

- **Decoder-Only**: 自回归语言模型
- **Pre-Norm**: RMSNorm 归一化
- **SwiGLU**: 高效激活函数
- **因果注意力**: 下三角掩码

## 配置说明

| 配置项        | 默认值 | 说明             |
| ------------- | ------ | ---------------- |
| hidden_size   | 512    | 隐藏层维度       |
| num_layers    | 6      | Transformer 层数 |
| num_heads     | 8      | 注意力头数       |
| lora_rank     | 8      | LoRA 秩          |
| lora_alpha    | 16     | LoRA 缩放因子    |
| batch_size    | 4      | 批次大小         |
| learning_rate | 1e-4   | 学习率           |
| num_epochs    | 3      | 训练轮数         |

## 学习要点

1. **LoRA 原理**：理解低秩分解如何减少参数量
2. **指令微调**：学习如何将预训练模型转化为对话模型
3. **ChatML 模板**：掌握对话数据的标准格式
4. **训练技巧**：梯度累积、学习率warmup、早停
5. **采样策略**：Top-k/Top-p sampling 的原理和实现

## 扩展方向

- [ ] 接入真实数据集（如 Alpaca、ShareGPT）
- [ ] 使用 Hugging Face Transformers 加载预训练模型
- [ ] 实现 QLoRA（量化 + LoRA）
- [ ] 添加 DPO 偏好学习
- [ ] 部署推理服务

## 参考资料

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [LLaMA 论文](https://arxiv.org/abs/2302.13971)
- [InstructGPT 论文](https://arxiv.org/abs/2203.02155)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
