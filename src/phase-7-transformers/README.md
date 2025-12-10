# Phase 7: 注意力机制与 Transformer

> **目标**：深入理解现代深度学习核心架构  
> **预计时长**：2-3 周  
> **前置条件**：Phase 1-6 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 深入理解自注意力 (Self-Attention) 机制
2. 从零实现完整的 Transformer 架构
3. 理解 BERT 和 GPT 的区别与联系
4. 能够使用预训练模型进行微调
5. 完成机器翻译和文本分类项目

---

## 📚 核心概念

### 自注意力 (Self-Attention)

```python
# Scaled Dot-Product Attention
Attention(Q, K, V) = softmax(QK^T / √d_k) V

# Q: Query, K: Key, V: Value
# d_k: Key 的维度
```

### 多头注意力 (Multi-Head Attention)

并行多个注意力头，捕捉不同子空间的信息：

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
```

### 位置编码 (Positional Encoding)

为序列添加位置信息：

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Transformer 架构

```
输入 → Embedding + PE → [Encoder Layers] → 编码输出
                                    ↓
解码输入 → Embedding + PE → [Decoder Layers] → 输出
```

---

## 📁 文件列表

| 文件                         | 描述             | 状态 |
| ---------------------------- | ---------------- | ---- |
| `01-self-attention.py`       | 自注意力从零实现 | ⏳   |
| `02-multi-head-attention.py` | 多头注意力       | ⏳   |
| `03-positional-encoding.py`  | 位置编码         | ⏳   |
| `04-transformer-encoder.py`  | 编码器实现       | ⏳   |
| `05-transformer-decoder.py`  | 解码器实现       | ⏳   |
| `06-transformer-full.py`     | 完整 Transformer | ⏳   |
| `07-bert-architecture.py`    | BERT 结构理解    | ⏳   |
| `08-bert-finetuning.py`      | BERT 微调实践    | ⏳   |
| `09-gpt-architecture.py`     | GPT 架构         | ⏳   |
| `10-gpt-generation.py`       | 文本生成         | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-7-transformers/01-self-attention.py
python src/phase-7-transformers/06-transformer-full.py
```

---

## 📖 推荐资源

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- 论文：Attention Is All You Need, BERT, GPT

---

## ✅ 完成检查

- [ ] 能够手动计算自注意力
- [ ] 理解 Q, K, V 的含义
- [ ] 能够解释为什么需要多头注意力
- [ ] 理解位置编码的作用
- [ ] 能够从零实现 Transformer
- [ ] 理解 BERT 和 GPT 的区别
- [ ] 能够微调 BERT 进行分类任务
- [ ] 完成机器翻译项目
