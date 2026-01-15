# Phase 7 - 中英机器翻译项目

## 项目概述

使用 Transformer 模型实现中英双向机器翻译，展示 Encoder-Decoder 架构的实际应用。

## 学习目标

1. 理解 Transformer 在序列到序列任务中的应用
2. 掌握机器翻译的数据处理流程
3. 学习中文文本的分词和词表构建
4. 实践训练技巧：学习率调度、梯度裁剪等
5. 了解 BLEU 评分等翻译质量指标

## 项目结构

```
phase-7-translation/
├── README.md
├── translation_model.py    # Transformer模型定义
├── train.py               # 训练脚本
├── data/                  # 数据目录
└── outputs/
    ├── checkpoints/       # 模型检查点
    └── translations/      # 翻译结果
```

## 模型架构

### Transformer 配置

- **模型维度**: 256
- **注意力头数**: 8
- **编码器层数**: 4
- **解码器层数**: 4
- **前馈网络维度**: 1024
- **Dropout**: 0.1

### 关键组件

1. **词嵌入层**：

   - 中文词表 + 英文词表
   - 可学习的词向量

2. **位置编码**：

   - 正弦位置编码
   - 为序列添加位置信息

3. **Encoder**：

   - Multi-Head Self-Attention
   - Feed-Forward Network
   - 残差连接 + LayerNorm

4. **Decoder**：

   - Masked Self-Attention
   - Cross-Attention
   - Feed-Forward Network

5. **输出层**：
   - Linear 投影到目标词表大小

## 数据集

本项目使用内置的简化数据集进行演示：

- **训练样本**: 10 个中英句对
- **目的**: 快速验证模型和训练流程
- **实际应用**: 可替换为大规模数据集（如 WMT、UN Parallel Corpus）

### 示例数据

```
中文: 我 爱 学习 深度 学习
英文: i love learning deep learning

中文: 今天 天气 很 好
英文: the weather is nice today
```

## 运行方式

### 1. 安装依赖

```bash
pip install torch jieba
```

### 2. 测试模型

```bash
python translation_model.py
```

这将：

- 加载示例数据
- 构建词表
- 创建模型
- 测试前向传播

### 3. 训练模型

```bash
python train.py
```

训练过程：

- 100 个 epochs
- 每 10 个 epoch 打印一次
- 自动保存最佳模型

### 4. 查看结果

训练完成后会自动展示翻译样例

## 关键知识点

### 1. 中文处理

**分词**：

```python
import jieba
tokens = jieba.cut("我爱学习深度学习")
# ['我', '爱', '学习', '深度', '学习']
```

**词表构建**：

- 特殊 token: `<pad>`, `<sos>`, `<eos>`, `<unk>`
- 最小词频过滤
- OOV 处理

### 2. 序列到序列训练

**Teacher Forcing**：

- 训练时使用真实目标序列
- 解码器输入：`<sos> + 目标序列[:-1]`
- 解码器输出：与`目标序列[1:]`对比

**掩码**：

```python
# 因果掩码（防止看到未来）
tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

# Padding掩码
padding_mask = (seq == PAD_IDX)
```

### 3. 训练技巧

**梯度裁剪**：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**学习率**：

- 使用较小的学习率（1e-4）
- 可以使用 warmup 策略

**损失函数**：

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

### 4. 推理

**贪心解码**：

```python
# 逐步生成
tgt = [<sos>]
for step in range(max_len):
    output = model(src, tgt)
    next_word = output[-1].argmax()
    tgt.append(next_word)
    if next_word == <eos>:
        break
```

## 扩展建议

### 1. 使用真实数据集

```python
# WMT中英翻译数据
from datasets import load_dataset
dataset = load_dataset('wmt19', 'zh-en')
```

### 2. 改进分词

- 使用 BPE(Byte Pair Encoding)
- 使用 SentencePiece
- 字符级编码

### 3. 更好的解码策略

- **Beam Search**: 保留 top-k 个候选
- **Length Penalty**: 惩罚过短/过长的翻译
- **Coverage Penalty**: 避免重复

### 4. 评估指标

**BLEU 分数**：

```python
from sacrebleu import corpus_bleu
bleu = corpus_bleu(hypotheses, [references])
```

### 5. 注意力可视化

可视化翻译时的注意力权重：

- 查看源语言和目标语言的对齐关系
- 理解模型的翻译过程

## 常见问题

### Q1: 为什么使用简化数据集？

A: 便于快速实验和理解流程。实际应用需要大规模数据集（如 10 万+句对）。

### Q2: 如何提高翻译质量？

A:

- 增加训练数据
- 增大模型（更多层、更大维度）
- 更长时间训练
- 使用预训练模型

### Q3: 中文分词的重要性？

A: 中文没有天然的词分界，分词质量直接影响：

- 词表大小
- 模型学习难度
- 翻译质量

### Q4: GPU 内存不足怎么办？

A:

- 减小 batch size
- 减小模型维度
- 使用梯度累积
- 使用混合精度训练

## 参考资源

- 论文: "Attention Is All You Need" (Vaswani et al., 2017)
- [PyTorch Transformer 教程](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## 下一步

完成本项目后，可以：

1. 尝试其他语言对的翻译
2. 实现更复杂的解码策略
3. 学习预训练的翻译模型（如 MarianMT）
4. 进入 Phase 7 的第二个项目：BERT 文本分类
