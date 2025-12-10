# Phase 11: 自然语言处理 NLP

> **目标**：掌握 NLP 核心技术  
> **预计时长**：2 周  
> **前置条件**：Phase 1-10 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解词向量和文本表示
2. 掌握 NLP 核心任务（分类、NER、问答）
3. 熟练使用 HuggingFace Transformers
4. 掌握参数高效微调 (LoRA/PEFT)
5. 完成情感分析和问答系统项目

---

## 📚 核心概念

### 文本表示

- **词袋模型**: 忽略顺序，统计词频
- **Word2Vec**: 词嵌入，捕捉语义
- **Contextual Embeddings**: BERT 等，考虑上下文

### NLP 核心任务

| 任务         | 描述               | 模型        |
| ------------ | ------------------ | ----------- |
| 文本分类     | 情感分析、主题分类 | BERT        |
| 命名实体识别 | 提取人名、地名等   | BERT-NER    |
| 问答系统     | 从文档中找答案     | BERT-QA     |
| 机器翻译     | 语言转换           | Transformer |

### HuggingFace 生态

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")
```

### 参数高效微调

- **LoRA**: 低秩适配
- **Prefix Tuning**: 前缀微调
- **Adapter**: 适配器层

---

## 📁 文件列表

| 文件                           | 描述             | 状态 |
| ------------------------------ | ---------------- | ---- |
| `01-word2vec.py`               | 词向量训练       | ⏳   |
| `02-embeddings-advanced.py`    | 词嵌入分析       | ⏳   |
| `03-text-classification.py`    | 文本分类         | ⏳   |
| `04-ner.py`                    | 命名实体识别     | ⏳   |
| `05-question-answering.py`     | 问答系统         | ⏳   |
| `06-huggingface-basics.py`     | HuggingFace 入门 | ⏳   |
| `07-transformer-finetuning.py` | Transformer 微调 | ⏳   |
| `08-peft-lora.py`              | 参数高效微调     | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-11-nlp/01-word2vec.py
python src/phase-11-nlp/06-huggingface-basics.py
```

---

## 📖 推荐资源

- [HuggingFace 官方教程](https://huggingface.co/course)
- [CS224n 课程](https://web.stanford.edu/class/cs224n/)

---

## ✅ 完成检查

- [ ] 理解词向量的原理
- [ ] 能够进行文本分类任务
- [ ] 能够进行命名实体识别
- [ ] 熟练使用 HuggingFace Transformers
- [ ] 理解 LoRA 的原理
- [ ] 完成情感分析项目
