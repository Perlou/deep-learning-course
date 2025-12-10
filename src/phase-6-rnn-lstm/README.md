# Phase 6: 循环神经网络 RNN/LSTM

> **目标**：掌握序列建模技术  
> **预计时长**：2 周  
> **前置条件**：Phase 1-4 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解 RNN 的结构和时序建模原理
2. 掌握 LSTM 和 GRU 的门控机制
3. 理解梯度消失/爆炸问题及解决方案
4. 实现序列到序列 (Seq2Seq) 模型
5. 完成文本生成和时间序列预测项目

---

## 📚 核心概念

### RNN 基础

```python
# RNN 单元
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b)
y_t = W_hy @ h_t + b_y
```

### LSTM

门控机制解决长期依赖：

- **遗忘门 (Forget Gate)**: 决定丢弃什么信息
- **输入门 (Input Gate)**: 决定存储什么信息
- **输出门 (Output Gate)**: 决定输出什么信息

### GRU

简化版 LSTM：

- **更新门 (Update Gate)**
- **重置门 (Reset Gate)**

### Seq2Seq

编码器-解码器架构：

```
输入序列 → 编码器 → 上下文向量 → 解码器 → 输出序列
```

---

## 📁 文件列表

| 文件                       | 描述               | 状态 |
| -------------------------- | ------------------ | ---- |
| `01-rnn-basics.py`         | RNN 结构与前向传播 | ⏳   |
| `02-bptt.py`               | 时间反向传播       | ⏳   |
| `03-vanishing-gradient.py` | 梯度消失问题       | ⏳   |
| `04-lstm.py`               | LSTM 门控机制      | ⏳   |
| `05-gru.py`                | GRU 简化结构       | ⏳   |
| `06-bidirectional.py`      | 双向 RNN           | ⏳   |
| `07-seq2seq.py`            | 序列到序列模型     | ⏳   |
| `08-attention-basic.py`    | 基础注意力机制     | ⏳   |
| `09-time-series.py`        | 时间序列预测       | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-6-rnn-lstm/01-rnn-basics.py
python src/phase-6-rnn-lstm/04-lstm.py
```

---

## 📖 推荐资源

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- 论文：LSTM, Seq2Seq with Attention

---

## ✅ 完成检查

- [ ] 理解 RNN 的展开结构
- [ ] 能够解释梯度消失问题
- [ ] 理解 LSTM 的门控机制
- [ ] 能够比较 LSTM 和 GRU 的区别
- [ ] 理解双向 RNN 的作用
- [ ] 能够实现 Seq2Seq 模型
- [ ] 完成文本生成项目
- [ ] 完成时间序列预测项目
