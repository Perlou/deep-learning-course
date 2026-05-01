# MiniMind - 技术架构设计文档

> 版本: v1.1  
> 更新日期: 2026-02-23  
> 作者: Deep Learning Course Capstone Project

---

## 1. 系统架构总览

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ClearMind 系统架构                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      Scripts Layer (入口脚本)                     │   │
│  │                                                                   │   │
│  │   download_data.py (按 profile 拉 minimind 数据)                  │   │
│  │   smoke_test.py (端到端最小链路 + 推理验证)                        │   │
│  │   train.py --stage pretrain → sft → dpo                           │   │
│  │   chat.py (chat_template 多轮对话 + open_thinking)                │   │
│  │   launch_ddp.py (torchrun 多 GPU 预训练)                          │   │
│  │   autodl_train.sh (AutoDL Base/Plus 一键)                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      Training Layer (训练逻辑)                    │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │   pretrain   │  │     sft      │  │     dpo      │           │   │
│  │  │              │  │              │  │              │           │   │
│  │  │ · AdamW      │  │ · 加载预训练  │  │ · 加载 SFT   │           │   │
│  │  │ · Cosine LR  │  │ · Mask 策略  │  │ · 偏好优化   │           │   │
│  │  │ · Grad Accum │  │ · 对话格式   │  │ · β 控制     │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      Model Layer (模型架构)                       │   │
│  │                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐     │   │
│  │  │                     GPT Model                            │     │   │
│  │  │  Token Embedding → N × TransformerBlock → RMSNorm → LMHead│   │   │
│  │  └─────────────────────────────────────────────────────────┘     │   │
│  │                              │                                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Attention │  │   FFN    │  │ RMSNorm  │  │   RoPE   │         │   │
│  │  │ MHA/GQA  │  │ SwiGLU   │  │          │  │          │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      Data Layer (数据处理)                        │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │  Tokenizer   │  │ PretrainData │  │  SFT / DPO   │           │   │
│  │  │              │  │              │  │   Dataset     │           │   │
│  │  │ sentencepiece│  │ 中英文本语料  │  │ 指令/偏好数据 │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 技术选型

| 层级 | 组件         | 技术选型             | 选型理由                       |
| ---- | ------------ | -------------------- | ------------------------------ |
| 框架 | 深度学习框架 | PyTorch 2.x          | 灵活性、动态图、MPS 支持       |
| 分词 | Tokenizer    | sentencepiece        | 高效 BPE、中文支持好、C++ 实现 |
| 数据 | 数据集       | HuggingFace datasets | 丰富的中英文数据源             |
| 配置 | 配置管理     | YAML + dataclass     | 清晰、类型安全                 |
| 日志 | 训练日志     | TensorBoard (可选)   | 标准工具、可视化丰富           |
| 设备 | 加速后端     | CPU / MPS / CUDA     | 自动检测、渐进式加速           |

---

## 2. 模型架构设计

### 2.1 整体架构

MiniMind 采用 **Decoder-only Transformer** 架构，与 GPT / Llama / Gemini 一致。

```
┌─────────────────────────────────────────────────────────────┐
│                        GPT Model                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Token IDs: [101, 2023, 8024, ...]                    │
│                         │                                    │
│                         ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Token Embedding (vocab × d_model)       │   │
│   └─────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Transformer Block × N layers            │   │
│   │                                                      │   │
│   │   ┌──────────────────────────────────────────────┐   │   │
│   │   │  RMSNorm → Multi-Head Attention (+ RoPE)     │   │   │
│   │   │              ↓ (residual connection)          │   │   │
│   │   │  RMSNorm → FFN (SwiGLU)                      │   │   │
│   │   │              ↓ (residual connection)          │   │   │
│   │   └──────────────────────────────────────────────┘   │   │
│   │                                                      │   │
│   └─────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              RMSNorm (final)                         │   │
│   └─────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              LM Head (d_model → vocab_size)          │   │
│   └─────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│   Output Logits: [vocab_size] per position                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模型配置

```python
@dataclass
class ModelConfig:
    # 模型维度
    d_model: int = 512        # 隐藏层维度
    n_heads: int = 8          # 注意力头数
    n_kv_heads: int = 8       # KV 头数 (GQA, 等于 n_heads 时为标准 MHA)
    n_layers: int = 8         # Transformer 层数
    d_ff: int = 1408          # FFN 中间维度 (≈ 2.75 × d_model)
    vocab_size: int = 8000    # 词表大小
    max_seq_len: int = 512    # 最大序列长度
    dropout: float = 0.1      # Dropout 比率
    norm_eps: float = 1e-6    # RMSNorm epsilon
```

#### Small 配置 (~26M 参数)

| 参数        | 值   | 说明                 |
| ----------- | ---- | -------------------- |
| d_model     | 512  | 隐藏层维度           |
| n_heads     | 8    | 注意力头数           |
| n_kv_heads  | 8    | KV 头数 (标准 MHA)   |
| n_layers    | 8    | Transformer 层数     |
| d_ff        | 1408 | FFN 维度             |
| vocab_size  | 8000 | 词表大小             |
| max_seq_len | 512  | 最大序列长度         |
| 总参数量    | ~26M | 适合 MacBook CPU/MPS |

#### Medium 配置 (~200M 参数)

| 参数        | 值    | 说明                    |
| ----------- | ----- | ----------------------- |
| d_model     | 1024  | 隐藏层维度              |
| n_heads     | 16    | 注意力头数              |
| n_kv_heads  | 8     | KV 头数 (GQA, 2:1 分组) |
| n_layers    | 16    | Transformer 层数        |
| d_ff        | 2816  | FFN 维度                |
| vocab_size  | 32000 | 词表大小                |
| max_seq_len | 1024  | 最大序列长度            |
| 总参数量    | ~200M | 需要 GPU (8GB+ VRAM)    |

### 2.3 核心组件详解

#### 2.3.1 RoPE — 旋转位置编码

**原理**: 将位置信息编码为向量旋转，使注意力分数自然包含相对位置关系。

```
数学公式:
  f(x, pos) = x · e^(i·pos·θ)

其中 θ_k = 1 / 10000^(2k/d), k = 0, 1, ..., d/2-1

实现要点:
  1. 预计算频率矩阵 freqs = θ × position_ids
  2. 将 hidden states 视为复数对 (x_2k, x_2k+1)
  3. 应用旋转: (x·cos(θ) - y·sin(θ), x·sin(θ) + y·cos(θ))
```

**优势**: 支持外推到更长序列、无需学习参数。

#### 2.3.2 RMSNorm — 均方根归一化

```
数学公式:
  RMSNorm(x) = x / RMS(x) · γ
  RMS(x) = √(mean(x²) + ε)

对比 LayerNorm:
  LayerNorm(x) = (x - mean(x)) / √(var(x) + ε) · γ + β

优势: 省去 mean 计算和 bias，计算量减少 ~10%
```

#### 2.3.3 SwiGLU — 门控线性激活

```
数学公式:
  SwiGLU(x) = (x · W₁) ⊙ SiLU(x · W_gate) · W₂

对比标准 FFN:
  FFN(x) = GELU(x · W₁) · W₂

优势: 门控机制提升表达能力，Llama/Gemini 均采用
注意: 由于多一个权重矩阵 W_gate，d_ff 需要适当缩小
      通常 d_ff ≈ 2.67 × d_model (而标准 FFN 为 4 × d_model)
```

#### 2.3.4 Multi-Head Attention + GQA

```
标准 MHA (Small):
  Q: [batch, seq, n_heads, head_dim]       # 8 heads
  K: [batch, seq, n_heads, head_dim]       # 8 heads
  V: [batch, seq, n_heads, head_dim]       # 8 heads

GQA (Medium):
  Q: [batch, seq, n_heads, head_dim]       # 16 heads
  K: [batch, seq, n_kv_heads, head_dim]    #  8 heads (shared)
  V: [batch, seq, n_kv_heads, head_dim]    #  8 heads (shared)

  每 2 个 Q head 共享 1 组 K/V → KV Cache 减少 50%

Causal Mask:
  使用上三角 mask 确保每个 token 只能看到之前的 token
  mask[i][j] = -inf if j > i, else 0
```

---

## 3. 数据流设计

### 3.1 分词器训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Tokenizer 训练流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   中英混合文本语料                                            │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ "Deep learning is amazing. 深度学习非常有趣。"         │   │
│   │ "The transformer architecture changed NLP..."         │   │
│   │ "注意力机制是 Transformer 的核心..."                    │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │        sentencepiece BPE Training                    │   │
│   │                                                      │   │
│   │  · model_type: bpe                                   │   │
│   │  · vocab_size: 8000 (small) / 32000 (medium)        │   │
│   │  · character_coverage: 0.9995                        │   │
│   │  · 特殊 token: <s>, </s>, <unk>, <pad>              │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   输出: tokenizer.model + tokenizer.vocab                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 预训练数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    Pre-training 数据流                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   原始文本 → Tokenize → 拼接 → 切分固定长度                    │
│                                                              │
│   "深度学习是..." → [1203, 445, 78, ...]                      │
│   "Transformers..." → [8832, 2234, ...]                      │
│                           │                                  │
│                           ▼                                  │
│   拼接为连续序列:                                              │
│   [1203, 445, 78, ..., 8832, 2234, ..., ...]                 │
│                           │                                  │
│                           ▼                                  │
│   切分固定长度 (max_seq_len = 512):                            │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│   │ sample 0      │  │ sample 1      │  │ sample N      │   │
│   │ [512 tokens]  │  │ [512 tokens]  │  │ [512 tokens]  │   │
│   └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                              │
│   训练目标: input[:-1] → target[1:] (next token prediction)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 SFT 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    SFT 数据格式                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   原始指令数据:                                                │
│   {                                                          │
│     "instruction": "请解释什么是梯度下降",                       │
│     "output": "梯度下降是一种优化算法..."                       │
│   }                                                          │
│                           │                                  │
│                           ▼                                  │
│   格式化为对话模板:                                             │
│   <s>Human: 请解释什么是梯度下降\n                              │
│   Assistant: 梯度下降是一种优化算法...</s>                      │
│                           │                                  │
│                           ▼                                  │
│   Loss Mask 策略:                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ Human: 请解释...   │ Assistant: 梯度下降是...          │  │
│   │ ████ MASK (不计算loss) ████ │ ✅ 计算 loss ✅            │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   只对 Assistant 回复部分计算 loss，避免学习用户输入            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 DPO 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    DPO 数据格式                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   偏好数据对:                                                  │
│   {                                                          │
│     "prompt": "什么是机器学习？",                               │
│     "chosen": "机器学习是人工智能的一个分支，它允许...",          │
│     "rejected": "机器学习就是让机器去学习东西"                  │
│   }                                                          │
│                           │                                  │
│                           ▼                                  │
│   DPO Loss:                                                  │
│   L = -log σ(β · (log π(chosen) - log π_ref(chosen)         │
│                   - log π(rejected) + log π_ref(rejected)))  │
│                                                              │
│   π     = 当前模型概率                                        │
│   π_ref = 参考模型概率 (冻结的 SFT 模型)                       │
│   β     = 温度参数 (default: 0.1)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 训练策略设计

### 4.1 预训练配置

```yaml
pretrain:
  # 优化器
  optimizer: adamw
  lr: 3e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]

  # 学习率调度
  scheduler: cosine
  warmup_steps: 500
  min_lr: 3e-5

  # 训练参数
  batch_size: 16 # 每个 step 的 batch size
  gradient_accumulation: 4 # 梯度累积步数 (有效 batch = 64)
  max_steps: 10000 # 总训练步数

  # Checkpoint
  save_every: 1000 # 每 1000 步保存
  eval_every: 500 # 每 500 步评估

  # 精度
  dtype: float32 # MacBook CPU 使用 float32
  # dtype: bfloat16         # MPS/CUDA 使用混合精度
```

### 4.2 SFT 配置

```yaml
sft:
  # 基于预训练 checkpoint
  pretrained_path: outputs/pretrain/final.pth

  # 优化器 (较小学习率)
  lr: 1e-5
  weight_decay: 0.01

  # 训练参数
  batch_size: 8
  gradient_accumulation: 4
  epochs: 3
  max_seq_len: 512
```

### 4.3 DPO 配置

```yaml
dpo:
  # 基于 SFT checkpoint
  sft_path: outputs/sft/final.pth

  # DPO 参数
  beta: 0.1 # DPO 温度
  lr: 5e-6 # 更小的学习率

  # 训练参数
  batch_size: 4
  gradient_accumulation: 8
  epochs: 1
```

---

## 5. 推理设计

### 5.1 生成策略

```python
class GenerationConfig:
    max_new_tokens: int = 256    # 最大生成长度
    temperature: float = 0.7     # 温度 (越高越随机)
    top_k: int = 50              # Top-K 采样
    top_p: float = 0.9           # Nucleus 采样
    repetition_penalty: float = 1.1  # 重复惩罚
    do_sample: bool = True       # True: 采样, False: 贪心
```

### 5.2 对话模板

```
对话格式:
  <s>Human: {user_message_1}
  Assistant: {assistant_response_1}
  Human: {user_message_2}
  Assistant: {assistant_response_2}</s>

推理时:
  1. 拼接历史对话
  2. 添加 "Assistant: " 前缀
  3. 自回归生成直到 </s> 或达到 max_new_tokens
```

---

## 6. 目录结构设计

```
capstone-llm-from-scratch/
├── README.md                        # 项目总览
├── requirements.txt                 # Python 依赖
│
├── configs/                         # 配置文件
│   ├── small.yaml                   # 26M 参数配置 (MacBook)
│   └── medium.yaml                  # 200M 参数配置 (GPU)
│
├── src/                             # 源代码
│   ├── __init__.py
│   │
│   ├── model/                       # 模型架构 (从零实现)
│   │   ├── __init__.py
│   │   ├── config.py                # ModelConfig dataclass
│   │   ├── rope.py                  # RoPE 旋转位置编码
│   │   ├── normalization.py         # RMSNorm
│   │   ├── activation.py            # SwiGLU
│   │   ├── attention.py             # MHA + GQA + Causal Mask
│   │   ├── feedforward.py           # SwiGLU FFN
│   │   ├── transformer.py           # Transformer Block
│   │   └── gpt.py                   # 完整 GPT 模型
│   │
│   ├── data/                        # 数据处理
│   │   ├── __init__.py
│   │   ├── tokenizer.py             # BPE Tokenizer 封装
│   │   ├── pretrain_dataset.py      # 预训练数据集
│   │   ├── sft_dataset.py           # SFT 数据集
│   │   └── dpo_dataset.py           # DPO 数据集
│   │
│   ├── training/                    # 训练逻辑
│   │   ├── __init__.py
│   │   ├── pretrain.py              # 预训练 Trainer
│   │   ├── sft.py                   # SFT Trainer
│   │   ├── dpo.py                   # DPO Trainer
│   │   └── trainer_utils.py         # LR Scheduler, Logger, etc.
│   │
│   └── inference/                   # 推理
│       ├── __init__.py
│       ├── generate.py              # 文本生成引擎
│       └── chat.py                  # 交互式对话
│
├── scripts/                         # 入口脚本
│   ├── train.py                     # 统一训练 (--stage pretrain/sft/dpo)
│   ├── chat.py                      # 对话（chat_template + open_thinking）
│   ├── smoke_test.py                # 端到端最小链路冒烟
│   ├── download_data.py             # minimind 数据按 profile 下载
│   ├── launch_ddp.py                # DDP 多 GPU 入口
│   └── autodl_train.sh              # AutoDL 一键 Base/Plus
│
├── tokenizer/                       # 仓库自带 HF tokenizer（不在 data/ 下，便于 git 追踪）
│   └── minimind/
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── evaluate/                        # 评估
│   ├── eval_perplexity.py           # 困惑度（含 --compare 三阶段对比）
│   ├── eval_generation.py           # 生成质量（Distinct-N + 重复率）
│   ├── eval_instruction.py          # 指令跟随（5 类 + 拒绝识别）
│   └── eval_benchmark.py            # 一键综合报告 + markdown
│
├── deploy/                          # 部署
│   ├── api_server.py                # OpenAI 兼容 SSE API
│   ├── web_demo.py                  # Gradio Web 对话
│   ├── export_model.py              # safetensors 导出
│   └── Dockerfile
│
├── outputs/                         # 输出 (gitignore)
│   ├── pretrain/                    # final.pth + _resume.pth
│   ├── sft/
│   └── dpo/
│
├── data/                            # 训练数据 (gitignore，用户从 modelscope/HF 下载)
│   ├── pretrain_t2t_mini.jsonl
│   ├── sft_t2t_mini.jsonl
│   └── dpo.jsonl                    # 等
│
└── docs/                            # 文档
    ├── PRD.md
    ├── TECHNICAL_DESIGN.md          # 本文档
    ├── PROGRESS_TRACKER.md
    ├── DEPLOY.md
    └── AUTODL_GUIDE.md
```

---

## 7. 设备适配

### 7.1 自动设备检测

```python
def get_device():
    """自动检测最优计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

### 7.2 各设备训练对比

| 设备             | Small (26M) 速度 | Medium (200M) 速度 | 内存占用    |
| ---------------- | ---------------- | ------------------ | ----------- |
| MacBook CPU (M1) | ~30 tokens/s     | 不推荐             | ~2GB / ~8GB |
| MacBook MPS (M1) | ~80 tokens/s     | ~20 tokens/s       | ~2GB / ~6GB |
| RTX 3060 (12GB)  | ~200 tokens/s    | ~80 tokens/s       | ~2GB / ~5GB |
| A100 (80GB)      | ~500 tokens/s    | ~300 tokens/s      | ~2GB / ~5GB |

---

## 附录

### A. 依赖列表

```txt
# requirements.txt

# Core
torch>=2.1.0
sentencepiece>=0.2.0

# Data
datasets>=2.16.0            # HuggingFace datasets
tqdm>=4.66.0

# Config
pyyaml>=6.0.0

# Logging (optional)
tensorboard>=2.15.0

# Evaluation (optional)
numpy>=1.24.0
matplotlib>=3.8.0
```

### B. 参数量计算公式

```
Total params =
  + Token Embedding:      vocab_size × d_model
  + N × Transformer Block:
    + Attention:
      + W_q: d_model × d_model
      + W_k: d_model × (d_model × n_kv_heads / n_heads)
      + W_v: d_model × (d_model × n_kv_heads / n_heads)
      + W_o: d_model × d_model
    + FFN (SwiGLU):
      + W_gate: d_model × d_ff
      + W_up:   d_model × d_ff
      + W_down: d_ff × d_model
    + RMSNorm × 2: d_model × 2
  + Final RMSNorm: d_model
  + LM Head: d_model × vocab_size (可与 Embedding 共享权重)

Small (512, 8, 8, 1408, 8000):
  ≈ 4M + 8 × (1M + 1M + 1M + 2.2M + 1K) + 4M ≈ 26M
```
