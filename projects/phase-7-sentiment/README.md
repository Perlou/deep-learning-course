# Phase 7 - BERT 中文情感分类项目

## 项目概述

使用预训练的 BERT 模型进行中文情感分类，展示迁移学习和模型微调的强大能力。

## 学习目标

1. 理解 BERT 的预训练-微调范式
2. 掌握 HuggingFace Transformers 库的使用
3. 学习文本分类任务的完整流程
4. 实践迁移学习的最佳实践

## 项目结构

```
phase-7-sentiment/
├── README.md
├── bert_classifier.py     # BERT分类器
├── train.py              # 训练脚本
├── data/                 # 数据目录
└── outputs/
    ├── checkpoints/      # 模型检查点
    └── results/          # 评估结果
```

## 模型架构

### BERT 配置

- **预训练模型**: bert-base-chinese
- **参数量**: ~110M
- **隐藏层维度**: 768
- **注意力头数**: 12
- **编码器层数**: 12

### 分类器结构

```
输入文本
    ↓
BERT Tokenizer (分词)
    ↓
BERT Encoder (编码)
    ↓
[CLS] Token Output
    ↓
Dropout (0.1)
    ↓
Linear (768 → 2)
    ↓
Softmax
    ↓
分类结果 (正面/负面)
```

## 数据集

本项目使用内置示例数据：

- **正面评论**: 5 条
- **负面评论**: 5 条
- **总计**: 10 条样本

### 示例数据

```python
正面: "这部电影太精彩了，我非常喜欢！"
负面: "剧情拖沓无聊，浪费时间。"
```

> **注意**: 这是演示用的小数据集。实际应用需要更大数据集（如 ChnSentiCorp、豆瓣评论等）。

## 运行方式

### 1. 安装依赖

```bash
pip install torch transformers
```

### 2. 下载 BERT 模型

首次运行会自动下载 `bert-base-chinese`。

如果下载慢，可使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 测试模型

```bash
python bert_classifier.py
```

这将：

- 加载示例数据
- 创建 BERT 分类器
- 测试前向传播
- 显示未训练的预测结果

### 4. 训练模型

```bash
python train.py
```

训练过程：

- 10 个 epochs
- AdamW 优化器（学习率 2e-5）
- 自动保存最佳模型

## 关键知识点

### 1. BERT Tokenization

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 分词
text = "这是一个例子"
tokens = tokenizer.tokenize(text)
# ['这', '是', '一', '个', '例', '子']

# 编码（添加特殊token）
encodings = tokenizer(text, return_tensors='pt')
# input_ids: [101, 6821, 3221, 671, 702, 862, 1126, 102]
# [CLS], 这, 是, 一, 个, 例, 子, [SEP]
```

### 2. 预训练 vs 微调

**预训练（BERT 已完成）**：

- 在大规模语料上训练
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

**微调（我们要做的）**：

- 添加任务特定的头（分类层）
- 在目标任务数据上训练
- 使用较小的学习率

### 3. 使用[CLS] Token

BERT 将第一个 token `[CLS]` 的输出作为整个句子的语义表示：

```python
outputs = bert(input_ids, attention_mask)
cls_output = outputs.pooler_output  # [CLS]的输出
logits = classifier(cls_output)      # 分类
```

### 4. 微调技巧

**小学习率**：

```python
optimizer = AdamW(model.parameters(), lr=2e-5)
```

**梯度裁剪**（可选）：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Warmup**（推荐）：

```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)
```

## 扩展建议

### 1. 使用真实数据集

**ChnSentiCorp**：

```python
# 中文情感分析标准数据集
# 约12k样本
```

**豆瓣电影评论**：

```python
from datasets import load_dataset
dataset = load_dataset('douban_movie')
```

### 2. 数据增强

```python
# 同义词替换
# 回译（中→英→中）
# 随机删除/插入
```

### 3. 模型改进

- 尝试其他 BERT 变体：
  - **RoBERTa-Chinese**
  - **ERNIE**
  - **MacBERT**

### 4. 评估指标

```python
from sklearn.metrics import classification_report, confusion_matrix

# 精确率、召回率、F1
report = classification_report(y_true, y_pred)

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
```

### 5. 推理优化

**模型导出**：

```python
# ONNX格式（加速推理）
torch.onnx.export(model, ...)
```

**量化**：

```python
# INT8量化（减小模型大小）
model_int8 = torch.quantization.quantize_dynamic(model, ...)
```

## 常见问题

### Q1: 为什么使用 bert-base-chinese？

A: 专门为中文预训练的 BERT 模型，在中文任务上效果更好。

### Q2: 数据量不够怎么办？

A: BERT 的优势就是即使数据少也能 work：

- 预训练提供了语言理解能力
- 微调只需要少量标注数据
- 可以尝试数据增强

### Q3: 训练很慢？

A: BERT 模型较大(110M 参数)，建议：

- 使用 GPU 训练
- 减小 batch size（如果 OOM）
- 考虑使用更小的模型（如 TinyBERT）

### Q4: 如何提高准确率？

A:

- 增加训练数据
- 调整学习率和训练轮数
- 尝试不同的 BERT 变体
- 添加更复杂的分类头

### Q5: BERT vs 传统方法？

A: BERT 优势：

- 上下文感知
- 语义理解更深
- 迁移学习能力强
- 无需人工特征工程

## HuggingFace 资源

- **模型库**: https://huggingface.co/models?language=zh
- **文档**: https://huggingface.co/docs/transformers
- **课程**: https://huggingface.co/course

## 参考资源

- 论文: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- [中文 BERT 模型](https://github.com/ymcui/Chinese-BERT-wwm)

## 下一步

完成本项目后，可以：

1. 尝试多分类任务
2. 实现命名实体识别（NER）
3. 尝试问答系统（QA）
4. 探索其他预训练模型
