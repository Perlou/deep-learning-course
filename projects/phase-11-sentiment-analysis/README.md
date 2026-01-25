# Phase 11 实战项目：情感分析系统

## 项目简介

本项目是 Phase 11 NLP 阶段的实战项目，实现多模型情感分析系统，支持 TextCNN、BiLSTM 和 BERT。

## 技术栈

- Python 3.10+
- PyTorch 2.x
- HuggingFace Transformers (可选)

## 项目结构

```
phase-11-sentiment-analysis/
├── README.md           # 项目说明
├── config.py           # 配置参数
├── dataset.py          # 数据集加载
├── model.py            # TextCNN, BiLSTM, BERT
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── inference.py        # 推理脚本
├── main.py             # CLI 入口
└── outputs/
    ├── models/         # 保存的模型
    └── results/        # 评估结果
```

## 快速开始

### 1. 环境准备

```bash
source venv/bin/activate
```

### 2. 运行演示

```bash
python projects/phase-11-sentiment-analysis/main.py demo
```

### 3. 训练模型

```bash
# TextCNN 快速训练
python projects/phase-11-sentiment-analysis/main.py train --model textcnn --quick

# BiLSTM
python projects/phase-11-sentiment-analysis/main.py train --model lstm --quick

# BERT (需要 transformers)
pip install transformers
python projects/phase-11-sentiment-analysis/main.py train --model bert --quick
```

### 4. 评估模型

```bash
python projects/phase-11-sentiment-analysis/main.py eval --model textcnn
```

### 5. 情感预测

```bash
# 单句预测
python projects/phase-11-sentiment-analysis/main.py predict --text "I love this movie!"

# 交互模式
python projects/phase-11-sentiment-analysis/main.py predict --interactive
```

## 支持的模型

| 模型    | 特点               | 参数量 |
| ------- | ------------------ | ------ |
| TextCNN | 多尺度卷积，训练快 | ~1.5M  |
| BiLSTM  | 双向编码，捕捉序列 | ~2M    |
| BERT    | 预训练微调，效果好 | ~66M   |

## 模型架构

### TextCNN

```
词嵌入 → [Conv3, Conv4, Conv5] → MaxPool → Concat → FC → Softmax
```

### BiLSTM

```
词嵌入 → BiLSTM × 2 → MeanPool → FC → Softmax
```

### BERT

```
Input → DistilBERT → [CLS] → Dropout → FC → Softmax
```

## 学习要点

1. **TextCNN**: 多尺度卷积核捕捉 n-gram 特征
2. **BiLSTM**: 双向编码捕捉上下文
3. **BERT 微调**: 预训练-微调范式
4. **评估指标**: Accuracy, F1, Confusion Matrix

## 参考资料

- [TextCNN 论文](https://arxiv.org/abs/1408.5882)
- [HuggingFace 教程](https://huggingface.co/course)
- [Phase 11 课程](../../src/phase-11-nlp/)
