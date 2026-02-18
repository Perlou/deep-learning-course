# MiniMind - 产品需求文档 (PRD)

> 版本: v1.0  
> 更新日期: 2026-02-18  
> 作者: Deep Learning Course Capstone Project

---

## 1. 项目背景

### 1.1 项目定位

MiniMind 是一个从零训练大语言模型的教育性实战项目。通过复刻 GPT-4 / Gemini 的完整训练流水线（Pre-training → SFT → DPO），在 MacBook 上训练一个 ~26M 参数的语言模型，让学习者彻底理解 LLM 的每一个技术环节。

### 1.2 项目动机

| 对比维度 | 现有 Capstone (RAG QA) | 本项目 (MiniMind)  |
| -------- | ---------------------- | ------------------ |
| 层次     | 应用层 — 使用现有模型  | 模型层 — 从零构建  |
| 重点     | 检索增强生成、系统集成 | 模型架构、训练流程 |
| 技能     | 工程集成能力           | 深度学习核心能力   |

两个项目互补，构成完整的 LLM 能力闭环：**既能造模型，也能用模型**。

### 1.3 项目愿景

> "用最少的资源，走完大模型训练的每一步"

---

## 2. 功能需求

### 2.1 核心功能模块

```
┌─────────────────────────────────────────────────────────────┐
│                    MiniMind 功能架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  模型构建     │  │  训练流水线   │  │  推理交互     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │               │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐       │
│  │ · RoPE       │  │ · 预训练     │  │ · 文本生成    │       │
│  │ · RMSNorm    │  │ · SFT 微调   │  │ · 交互对话    │       │
│  │ · SwiGLU     │  │ · DPO 对齐   │  │ · 评估指标    │       │
│  │ · GQA        │  │ · 日志可视化  │  │ · 采样策略    │       │
│  │ · Transformer│  │              │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │  数据处理     │  │  工具链       │                         │
│  └──────┬───────┘  └──────┬───────┘                         │
│         │                 │                                 │
│  ┌──────▼───────┐  ┌──────▼───────┐                         │
│  │ · BPE 分词器  │  │ · 配置管理   │                         │
│  │ · 预训练数据  │  │ · Checkpoint │                         │
│  │ · SFT 数据   │  │ · 训练日志   │                         │
│  │ · DPO 数据   │  │              │                         │
│  └──────────────┘  └──────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 功能详细说明

#### 2.2.1 模型架构模块

| 功能项               | 描述                                      | 优先级 |
| -------------------- | ----------------------------------------- | ------ |
| RoPE 旋转位置编码    | 从零实现 Rotary Position Embedding        | P0     |
| RMSNorm              | 实现 Root Mean Square Layer Normalization | P0     |
| SwiGLU 激活函数      | 实现 SwiGLU FFN (替代 GELU)               | P0     |
| Multi-Head Attention | 支持标准 MHA 和 GQA                       | P0     |
| Transformer Block    | 组装完整的 Decoder Block                  | P0     |
| GPT 完整模型         | Token Embedding → N × Block → LM Head     | P0     |
| 模型参数统计         | 自动统计和打印参数量                      | P1     |
| 多规模配置           | 支持 Small (26M) / Medium (200M) 切换     | P1     |

#### 2.2.2 数据处理模块

| 功能项         | 描述                                  | 优先级 |
| -------------- | ------------------------------------- | ------ |
| BPE 分词器训练 | 基于 sentencepiece 训练中英混合分词器 | P0     |
| 预训练数据集   | 下载并处理中英文本语料                | P0     |
| SFT 数据集     | 加载指令-回复对话数据                 | P0     |
| DPO 数据集     | 加载 chosen/rejected 偏好对数据       | P1     |
| 数据预处理脚本 | 自动下载、清洗、打包数据              | P0     |

#### 2.2.3 训练流水线模块

| 功能项                | 描述                                         | 优先级 |
| --------------------- | -------------------------------------------- | ------ |
| 预训练 (Pre-training) | Next-token prediction，AdamW + cosine LR     | P0     |
| 指令微调 (SFT)        | 在对话数据上继续训练，mask 非 assistant 部分 | P0     |
| 对齐训练 (DPO)        | Direct Preference Optimization               | P1     |
| 梯度累积              | 模拟大 batch size                            | P0     |
| 混合精度训练          | FP16/BF16 加速 (MPS/CUDA)                    | P1     |
| Checkpoint 保存/恢复  | 定期保存训练状态                             | P0     |
| 训练日志              | Loss、LR、速度等指标记录                     | P0     |

#### 2.2.4 推理交互模块

| 功能项     | 描述                                | 优先级 |
| ---------- | ----------------------------------- | ------ |
| 文本生成   | 支持 top-k、top-p、temperature 采样 | P0     |
| 重复惩罚   | Repetition penalty 避免重复生成     | P1     |
| 交互式对话 | 终端 CLI 对话界面                   | P0     |
| 对话模板   | 自动格式化多轮对话                  | P0     |

---

## 3. 非功能需求

### 3.1 硬件兼容性

| 指标      | 要求                                 |
| --------- | ------------------------------------ |
| 最低配置  | MacBook (Apple Silicon M1+, 8GB RAM) |
| CPU 训练  | Small 配置必须支持纯 CPU 训练        |
| MPS 加速  | 自动检测并使用 Apple MPS 后端        |
| CUDA 支持 | 有 NVIDIA GPU 时自动启用             |

### 3.2 性能指标 (Small 配置, MacBook)

| 指标       | 要求                  |
| ---------- | --------------------- |
| 模型初始化 | < 5 秒                |
| 预训练速度 | ≥ 50 tokens/sec (MPS) |
| 推理速度   | ≥ 20 tokens/sec       |
| 内存占用   | < 4GB                 |

### 3.3 教育性要求

| 指标       | 要求                                           |
| ---------- | ---------------------------------------------- |
| 代码可读性 | 每个组件独立文件，详细中文注释                 |
| 渐进式学习 | 脚本编号 01-06，按步骤执行                     |
| 原理文档   | 每个核心组件配套架构文档                       |
| 零依赖模型 | 模型架构部分只依赖 PyTorch，不使用 HuggingFace |

---

## 4. 训练流水线设计

### 4.1 完整流程

```mermaid
graph LR
    A[原始语料] --> B[BPE 分词器训练]
    B --> C[数据预处理]
    C --> D[Pre-training]
    D --> E[SFT 微调]
    E --> F[DPO 对齐]
    F --> G[Chat 推理]

    style A fill:#e1f5fe
    style D fill:#fff3e0
    style E fill:#e8f5e9
    style F fill:#fce4ec
    style G fill:#f3e5f5
```

### 4.2 各阶段对比

| 阶段         | 目标               | 数据              | 输出              | MacBook 耗时 |
| ------------ | ------------------ | ----------------- | ----------------- | ------------ |
| Tokenizer    | 构建中英混合分词器 | 文本语料样本      | `tokenizer.model` | ~10 分钟     |
| Pre-training | 学习语言知识       | 中英文本 (~100MB) | `pretrain.pth`    | 3-6 小时     |
| SFT          | 学习对话格式       | 指令数据 (~50K条) | `sft.pth`         | 1-2 小时     |
| DPO          | 偏好对齐           | 偏好对 (~10K条)   | `dpo.pth`         | 30-60 分钟   |

---

## 5. 项目范围

### 5.1 本期包含 (In Scope)

- ✅ 完整的 Decoder-only Transformer 从零实现
- ✅ BPE 分词器训练 (sentencepiece)
- ✅ 三阶段训练流水线 (Pre-training → SFT → DPO)
- ✅ 终端交互式对话
- ✅ MacBook (CPU/MPS) 完整支持
- ✅ 详细的架构文档和训练指南
- ✅ Small (26M) 和 Medium (200M) 配置

### 5.2 本期不包含 (Out of Scope)

- ❌ Web UI 界面
- ❌ 多机分布式训练
- ❌ RLHF (使用 DPO 替代, 无需 reward model)
- ❌ 多模态 (图片/音频)
- ❌ 模型部署服务化 (API Server)
- ❌ 量化推理 (INT4/INT8)

### 5.3 未来规划 (Future)

- 🔮 Streamlit / Gradio Web UI
- 🔮 模型量化部署
- 🔮 知识蒸馏 (大模型→小模型)
- 🔮 多模态扩展 (Vision-Language)
- 🔮 MoE (Mixture of Experts) 架构

---

## 6. 验收标准

### 6.1 功能验收

- [ ] 模型能在 MacBook 上完成初始化和前向传播
- [ ] BPE 分词器能正确编解码中英文
- [ ] 预训练后模型能生成连贯文本 (非随机)
- [ ] SFT 后模型能进行基本的指令跟随对话
- [ ] DPO 后模型回复质量有可观测的提升
- [ ] 交互式对话界面能正常加载模型并实时生成

### 6.2 性能验收

- [ ] Small 模型参数量在 20-30M 范围内
- [ ] 预训练 loss 能持续下降并收敛
- [ ] 推理速度 ≥ 10 tokens/sec (MacBook CPU)
- [ ] 完整训练流程 (Pre-train + SFT + DPO) 可在 24h 内完成

### 6.3 教育验收

- [ ] 每个核心组件 (RoPE, RMSNorm, SwiGLU, Attention) 独立实现并有详细注释
- [ ] 架构文档清晰解释每个组件的数学原理
- [ ] 训练指南能让零基础用户跟着完成全流程
- [ ] 代码不依赖 HuggingFace transformers (模型架构部分)

---

## 7. 风险评估

| 风险项             | 可能性 | 影响 | 缓解措施                                                  |
| ------------------ | ------ | ---- | --------------------------------------------------------- |
| MacBook 训练速度慢 | 高     | 中   | 控制 Small 配置在 26M，减少训练 steps                     |
| 小模型生成质量差   | 高     | 低   | 项目核心是教育价值而非模型质量，文档中明确说明            |
| 数据下载困难       | 中     | 中   | 提供多个数据源 (HuggingFace + 镜像)，支持小数据集快速验证 |
| MPS 后端兼容性     | 中     | 中   | 所有功能确保 CPU fallback 可用                            |
| 内存不足 (8GB Mac) | 低     | 高   | Small 配置控制内存占用 < 4GB，支持梯度累积                |

---

## 附录

### A. 术语表

| 术语         | 说明                                                                       |
| ------------ | -------------------------------------------------------------------------- |
| Pre-training | 预训练，在海量文本上进行 next-token prediction                             |
| SFT          | Supervised Fine-Tuning，监督微调，让模型学会对话格式                       |
| DPO          | Direct Preference Optimization，直接偏好优化，无需 reward model 的对齐方法 |
| RLHF         | Reinforcement Learning from Human Feedback，基于人类反馈的强化学习         |
| RoPE         | Rotary Position Embedding，旋转位置编码                                    |
| RMSNorm      | Root Mean Square Normalization，均方根归一化                               |
| SwiGLU       | Swish-Gated Linear Unit，门控线性单元激活函数                              |
| GQA          | Grouped Query Attention，分组查询注意力                                    |
| BPE          | Byte Pair Encoding，字节对编码分词算法                                     |
| MPS          | Metal Performance Shaders，Apple 的 GPU 加速框架                           |

### B. 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Llama 架构
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO 论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE 论文
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy 的 nanoGPT 项目
- [MiniMind](https://github.com/jingyaogong/minimind) - 开源 MiniMind 参考实现
