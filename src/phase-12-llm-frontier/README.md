# Phase 12: 大模型与前沿技术

> **目标**：理解大模型训练与前沿技术  
> **预计时长**：2-3 周  
> **前置条件**：Phase 1-11 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 深入理解 LLM 架构（GPT、LLaMA 等）
2. 理解预训练、指令微调、RLHF 流程
3. 掌握模型量化和推理优化技术
4. 了解多模态和 Agent 前沿方向
5. 完成 LLM 微调项目

---

## 📚 核心概念

### LLM 架构

现代 LLM 的关键组件：

- **Tokenizer**: BPE, SentencePiece
- **Positional Encoding**: RoPE, ALiBi
- **Attention**: Flash Attention, Multi-Query Attention
- **Normalization**: RMSNorm, Pre-Norm

### 训练流程

```
预训练 (Pre-training)
    ↓
指令微调 (Instruction Tuning)
    ↓
RLHF (Reinforcement Learning from Human Feedback)
```

### 高效推理

- **量化**: INT8, INT4, GPTQ, AWQ
- **KV Cache**: 缓存注意力计算
- **投机解码**: 小模型预测 + 大模型验证

### 前沿方向

- **多模态**: 图文理解、视频理解
- **Agent**: 工具调用、规划执行
- **RAG**: 检索增强生成

---

## 📁 文件列表

| 文件                           | 描述                | 状态 |
| ------------------------------ | ------------------- | ---- |
| `01-llm-architecture.py`       | LLM 架构理解        | ⏳   |
| `02-tokenization-advanced.py`  | 分词器详解          | ⏳   |
| `03-flash-attention.py`        | FlashAttention 原理 | ⏳   |
| `04-pre-training-basics.py`    | 预训练基础          | ⏳   |
| `05-instruction-tuning.py`     | 指令微调            | ⏳   |
| `06-rlhf-basics.py`            | RLHF 原理           | ⏳   |
| `07-quantization.py`           | 模型量化            | ⏳   |
| `08-inference-optimization.py` | 推理优化            | ⏳   |
| `09-multimodal.py`             | 多模态模型          | ⏳   |
| `10-agents-tools.py`           | Agent 与工具调用    | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-12-llm-frontier/01-llm-architecture.py
python src/phase-12-llm-frontier/07-quantization.py
```

---

## 📖 推荐资源

- [LLaMA 论文](https://arxiv.org/abs/2302.13971)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [InstructGPT 论文](https://arxiv.org/abs/2203.02155)

---

## ✅ 完成检查

- [ ] 理解现代 LLM 的架构
- [ ] 理解预训练和微调的区别
- [ ] 理解 RLHF 的基本流程
- [ ] 能够进行模型量化
- [ ] 了解多模态模型的架构
- [ ] 完成 LLM 微调项目
