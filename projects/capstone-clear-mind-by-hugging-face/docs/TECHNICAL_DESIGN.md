# ClearMind-HF 技术架构文档

> **版本:** v1.0
> **日期:** 2026-03-21
> **作者:** ClearMind Team
> **姊妹项目:** [ClearMind (from-scratch)](../../capstone-llm-from-scratch/)

---

## 1. 系统架构总览

### 1.1 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 Application                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Notebook  │ │  Gradio  │ │   CLI    │ │   HF Hub         │   │
│  │ 对比教学  │ │ Web Demo │ │  Chat    │ │ push_to_hub      │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        推理层 Inference                         │
│  ┌──────────────────┐ ┌─────────────────┐ ┌────────────────┐   │
│  │ model.generate() │ │ pipeline()      │ │ GGUF Export    │   │
│  │ GenerationMixin  │ │ text-generation │ │ llama.cpp      │   │
│  └──────────────────┘ └─────────────────┘ └────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        训练层 Training                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ HF Trainer   │ │ TRL SFTTrainer│ │TRL DPOTrainer│            │
│  │ (Pretrain)   │ │ (SFT)        │ │ (DPO)        │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ PEFT LoRA    │ │ accelerate   │ │ DataCollator │            │
│  │ QLoRA        │ │ 分布式训练    │ │ 数据整理      │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                        模型层 Model                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              ClearMindForCausalLM (PreTrainedModel)       │  │
│  │  ┌─────────────┐ ┌──────────────────────────────────────┐│  │
│  │  │ Embedding   │ │ TransformerBlock × n_layers          ││  │
│  │  │ + Dropout   │ │ ┌────────────┐ ┌──────────────────┐ ││  │
│  │  └─────────────┘ │ │ RMSNorm    │ │ RMSNorm          │ ││  │
│  │  ┌─────────────┐ │ │ Attention  │ │ FeedForward      │ ││  │
│  │  │ RMSNorm     │ │ │ (GQA+RoPE) │ │ (SwiGLU)         │ ││  │
│  │  │ + LM Head   │ │ └────────────┘ └──────────────────┘ ││  │
│  │  └─────────────┘ └──────────────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        数据层 Data                              │
│  ┌──────────────────┐ ┌─────────────────┐ ┌────────────────┐   │
│  │ tokenizers       │ │ datasets        │ │ DataCollator   │   │
│  │ ByteLevelBPE     │ │ load_dataset()  │ │ LM / SFT / DPO│   │
│  │ + ChatTemplate   │ │ .map()          │ │                │   │
│  └──────────────────┘ └─────────────────┘ └────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        评估层 Evaluation                        │
│  ┌──────────────────┐ ┌─────────────────┐ ┌────────────────┐   │
│  │ lm-eval-harness  │ │ Perplexity      │ │ 阶段对比       │   │
│  │ 标准 Benchmark   │ │ 困惑度评估       │ │ 可视化         │   │
│  └──────────────────┘ └─────────────────┘ └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 技术选型表

| 组件 | from-scratch 选型 | HF 选型 | 选型理由 |
|------|-------------------|---------|----------|
| 模型基类 | nn.Module | PreTrainedModel | 支持 save/load/generate/push_to_hub |
| 配置管理 | dataclass + YAML | PretrainedConfig + YAML | 支持 from_pretrained / Auto 注册 |
| 分词器 | sentencepiece | tokenizers + PreTrainedTokenizerFast | Rust 加速 + chat template 支持 |
| 预训练 | 手写 training loop | transformers.Trainer | 封装完整训练逻辑 |
| SFT | 手写 SFTTrainer | trl.SFTTrainer | 内置 DataCollatorForCompletionOnlyLM |
| DPO | 手写 DPOTrainer | trl.DPOTrainer | 内置 ref model 管理和 DPO loss |
| LoRA | 手写 LoRALinear | peft.LoraConfig | 丰富的 adapter 支持 |
| 多卡 | torch DDP | accelerate | 一套代码适配多种并行策略 |
| 数据加载 | torch Dataset | datasets + DataCollator | 高效数据处理 + 动态 padding |
| 评估 | 手写 eval 脚本 | lm-eval-harness | 标准化 benchmark |
| 部署 | FastAPI | pipeline + Gradio | 开箱即用 |

### 1.3 组件依赖图

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
│  tokenizers  │────→│  ClearMindConfig  │────→│ ClearMindFor  │
│  (训练分词器) │     │ (PretrainedConfig)│     │ CausalLM      │
└─────────────┘     └──────────────────┘     │(PreTrainedModel)│
                                              └───────┬───────┘
                                                      │
              ┌───────────────────────────────────────┼───────────────┐
              │                                       │               │
              ▼                                       ▼               ▼
     ┌────────────────┐                    ┌──────────────┐  ┌────────────┐
     │ HF Trainer     │                    │ TRL SFTTrainer│  │TRL DPOTrainer│
     │ (预训练)        │                    │ (SFT)        │  │ (DPO)        │
     └────────────────┘                    └──────────────┘  └────────────┘
              │                                       │               │
              │              ┌────────────────────────┘               │
              ▼              ▼                                        ▼
     ┌────────────────────────────┐                    ┌──────────────────┐
     │ peft (LoRA / QLoRA)       │                    │ 评估 & 部署       │
     │ get_peft_model()          │                    │ lm-eval / Gradio  │
     └────────────────────────────┘                    └──────────────────┘
```

---

## 2. 模型架构设计

### 2.1 ClearMindConfig

继承自 `PretrainedConfig`，与 from-scratch 版 `ModelConfig` 的字段映射：

```python
class ClearMindConfig(PretrainedConfig):
    model_type = "clearmind"

    def __init__(
        self,
        hidden_size=512,          # from-scratch: d_model
        num_attention_heads=8,     # from-scratch: n_heads
        num_key_value_heads=8,     # from-scratch: n_kv_heads
        num_hidden_layers=8,       # from-scratch: n_layers
        intermediate_size=1408,    # from-scratch: d_ff
        vocab_size=8000,           # 相同
        max_position_embeddings=512,  # from-scratch: max_seq_len
        hidden_dropout_prob=0.1,   # from-scratch: dropout
        rms_norm_eps=1e-6,         # from-scratch: norm_eps
        rope_theta=10000.0,        # from-scratch: 硬编码 10000.0
        use_cache=True,            # 新增：控制 KV Cache
        tie_word_embeddings=True,  # from-scratch: 硬编码 True
        **kwargs,
    ):
        super().__init__(**kwargs)
        # ... 赋值
```

**字段映射表：**

| from-scratch (ModelConfig) | HF (ClearMindConfig) | 说明 |
|---------------------------|----------------------|------|
| `d_model` | `hidden_size` | 隐藏层维度 |
| `n_heads` | `num_attention_heads` | 注意力头数 |
| `n_kv_heads` | `num_key_value_heads` | KV 头数 (GQA) |
| `n_layers` | `num_hidden_layers` | Transformer 层数 |
| `d_ff` | `intermediate_size` | FFN 中间维度 |
| `vocab_size` | `vocab_size` | 词表大小 |
| `max_seq_len` | `max_position_embeddings` | 最大序列长度 |
| `dropout` | `hidden_dropout_prob` | Dropout 概率 |
| `norm_eps` | `rms_norm_eps` | RMSNorm epsilon |
| (硬编码 10000) | `rope_theta` | RoPE 基础频率 |
| `head_dim` (property) | `head_dim` (property) | = hidden_size // num_attention_heads |
| `n_kv_groups` (property) | `num_key_value_groups` (property) | = num_attention_heads // num_key_value_heads |
| (无) | `use_cache` | 控制是否使用 KV Cache |
| (硬编码 True) | `tie_word_embeddings` | Embedding-LMHead 权重共享 |

**工厂方法保留：**

```python
@classmethod
def tiny(cls):
    return cls(hidden_size=128, num_attention_heads=4, num_key_value_heads=4,
               num_hidden_layers=4, intermediate_size=352, vocab_size=2000,
               max_position_embeddings=128)

@classmethod
def small(cls):
    return cls()  # 默认值即 small 配置
```

### 2.2 ClearMindForCausalLM

继承自 `PreTrainedModel`，实现 `CausalLM` 接口：

```
ClearMindForCausalLM(PreTrainedModel)
├── config: ClearMindConfig
├── model: ClearMindModel
│   ├── embed_tokens: nn.Embedding(vocab_size, hidden_size)
│   ├── embed_dropout: nn.Dropout
│   ├── layers: nn.ModuleList[ClearMindDecoderLayer × num_hidden_layers]
│   │   └── ClearMindDecoderLayer
│   │       ├── input_layernorm: ClearMindRMSNorm        # from-scratch: attn_norm
│   │       ├── self_attn: ClearMindAttention             # from-scratch: attention
│   │       │   ├── q_proj: nn.Linear(hidden_size, hidden_size, bias=False)
│   │       │   ├── k_proj: nn.Linear(hidden_size, kv_dim, bias=False)
│   │       │   ├── v_proj: nn.Linear(hidden_size, kv_dim, bias=False)
│   │       │   ├── o_proj: nn.Linear(hidden_size, hidden_size, bias=False)
│   │       │   └── rotary_emb: ClearMindRotaryEmbedding
│   │       ├── post_attention_layernorm: ClearMindRMSNorm  # from-scratch: ffn_norm
│   │       └── mlp: ClearMindMLP                          # from-scratch: feedforward
│   │           ├── gate_proj: nn.Linear(hidden_size, intermediate_size, bias=False)
│   │           ├── up_proj: nn.Linear(hidden_size, intermediate_size, bias=False)
│   │           └── down_proj: nn.Linear(intermediate_size, hidden_size, bias=False)
│   └── norm: ClearMindRMSNorm                            # from-scratch: final_norm
└── lm_head: nn.Linear(hidden_size, vocab_size, bias=False)  # tied with embed_tokens
```

**与 from-scratch 的命名映射：**

| from-scratch 命名 | HF 命名 | 说明 |
|-------------------|---------|------|
| `token_embedding` | `embed_tokens` | HF 标准命名 |
| `layers` | `layers` | 相同 |
| `attn_norm` | `input_layernorm` | HF Llama 风格 |
| `attention` | `self_attn` | HF 标准命名 |
| `w_q` / `w_k` / `w_v` / `w_o` | `q_proj` / `k_proj` / `v_proj` / `o_proj` | HF 标准命名 |
| `ffn_norm` | `post_attention_layernorm` | HF Llama 风格 |
| `feedforward` | `mlp` | HF 标准命名 |
| `w_gate` / `w_up` / `w_down` | `gate_proj` / `up_proj` / `down_proj` | HF 标准命名 |
| `final_norm` | `norm` | HF Llama 风格 |
| `lm_head` | `lm_head` | 相同 |

**关键方法：**

```python
class ClearMindForCausalLM(PreTrainedModel):
    config_class = ClearMindConfig
    _no_split_modules = ["ClearMindDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]

    def forward(self, input_ids, attention_mask=None, labels=None,
                past_key_values=None, use_cache=None, **kwargs):
        """
        与 from-scratch 版 GPT.forward() 对应：
        - input_ids: 相同
        - attention_mask: 相同（1=有效, 0=padding）
        - labels: 对应 from-scratch 的 targets
        - past_key_values: 对应 from-scratch 的 kv_caches
        - use_cache: 相同
        返回 CausalLMOutputWithPast(loss, logits, past_key_values)
        """

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """GenerationMixin 需要的方法，处理 KV Cache 场景下的输入裁剪"""

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Beam Search 需要的缓存重排"""
```

### 2.3 AutoClass 注册

```python
# 在 __init__.py 或单独的 auto_register.py 中
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("clearmind", ClearMindConfig)
AutoModelForCausalLM.register(ClearMindConfig, ClearMindForCausalLM)
```

注册后可使用：
```python
model = AutoModelForCausalLM.from_pretrained("path/to/clearmind")
config = AutoConfig.from_pretrained("path/to/clearmind")
```

---

## 3. 数据流设计

### 3.1 Tokenizer 训练流程

```
原始文本 ──→ tokenizers.ByteLevelBPETokenizer ──→ 训练 BPE
                                                    │
                                                    ▼
                                         vocab.json + merges.txt
                                                    │
                                                    ▼
                                    PreTrainedTokenizerFast 封装
                                    + special_tokens_map.json
                                    + tokenizer_config.json
                                    + chat_template (Jinja2)
                                                    │
                                                    ▼
                                         tokenizer.save_pretrained()
```

**from-scratch 对比：**

| 步骤 | from-scratch | HF |
|------|-------------|-----|
| BPE 训练 | sentencepiece.SentencePieceTrainer.train() | ByteLevelBPETokenizer.train() |
| 产物文件 | tokenizer.model (单文件) | vocab.json + merges.txt + configs |
| 封装类 | ClearMindTokenizer (手动 encode/decode) | PreTrainedTokenizerFast (内置方法) |
| 特殊 token | 手动管理 id | special_tokens_map.json 声明 |
| Chat 格式 | 硬编码 `Human: ... Assistant: ...` | Jinja2 chat_template |
| 保存/加载 | shutil.copy .model 文件 | save_pretrained() / from_pretrained() |

### 3.2 预训练数据流

```
原始文本 ──→ datasets.load_dataset("text") ──→ tokenizer(text, truncation, max_length)
                                                │
                                                ▼
                                     tokenized dataset (input_ids)
                                                │
                                                ▼
                                DataCollatorForLanguageModeling(mlm=False)
                                    - 动态 padding
                                    - labels = input_ids.clone()
                                    - padding token 的 label 设为 -100
                                                │
                                                ▼
                                         Trainer.train()
```

**from-scratch 对比：**

| 步骤 | from-scratch | HF |
|------|-------------|-----|
| 数据格式 | .bin / .jsonl / .txt | datasets 支持所有格式 |
| 分词 | 手动在 `__getitem__` 中调用 | `dataset.map(tokenize_fn, batched=True)` |
| 批处理 | DataLoader + 手动 padding | DataCollatorForLanguageModeling |
| Labels | `input_ids[1:]` 手动构造 | DataCollator 自动生成 |

### 3.3 SFT 数据流

```
指令数据 ──→ datasets.load_dataset() ──→ apply_chat_template 格式化
(Alpaca/ShareGPT)                         │
                                          ▼
                               formatted conversation string
                               "<s>Human: ...\nAssistant: ...</s>"
                                          │
                                          ▼
                           tokenizer(text, truncation, max_length)
                                          │
                                          ▼
                      DataCollatorForCompletionOnlyLM
                         - 自动识别 response_template
                         - prompt 部分 label 设为 -100
                         - 只有 assistant 回复计算 loss
                                          │
                                          ▼
                                   SFTTrainer.train()
```

**from-scratch 对比：**

| 步骤 | from-scratch | HF |
|------|-------------|-----|
| 格式化 | `build_prompt()` 手动拼接 | `apply_chat_template()` Jinja2 渲染 |
| Loss Mask | 手动计算 prompt_len，设 labels[:prompt_len]=-100 | DataCollatorForCompletionOnlyLM 自动处理 |
| 多轮对话 | 手动拼接 turn | chat_template 内置多轮支持 |

### 3.4 DPO 数据流

```
偏好数据 ──→ datasets.load_dataset() ──→ 标准格式：
(chosen/rejected)                        {
                                           "prompt": "...",
                                           "chosen": "...",
                                           "rejected": "..."
                                         }
                                                │
                                                ▼
                                      DPOTrainer 内置处理
                                       - 自动 tokenize prompt + chosen/rejected
                                       - 自动创建 ref model
                                       - 自动计算 DPO loss
                                                │
                                                ▼
                                         DPOTrainer.train()
```

**from-scratch 对比：**

| 步骤 | from-scratch | HF |
|------|-------------|-----|
| 数据格式 | DPODataset 手动 tokenize chosen/rejected | DPOTrainer 内置 tokenize |
| Ref Model | `copy.deepcopy(model)` 手动冻结 | DPOTrainer 自动管理 ref model |
| Log Probs | 手动计算 `log_softmax + gather + mask` | DPOTrainer 内置计算 |
| DPO Loss | `-logsigmoid(beta * margin).mean()` | DPOTrainer 内置多种 loss 变体 |

---

## 4. 训练策略设计

### 4.1 TrainingArguments 与 from-scratch YAML 的映射

**预训练配置映射：**

| from-scratch YAML | TrainingArguments | 说明 |
|-------------------|-------------------|------|
| `pretrain.batch_size` | `per_device_train_batch_size` | 单卡 batch size |
| `pretrain.gradient_accumulation_steps` | `gradient_accumulation_steps` | 梯度累积步数 |
| `pretrain.max_steps` | `max_steps` | 最大训练步数 |
| `pretrain.learning_rate` | `learning_rate` | 峰值学习率 |
| `pretrain.min_lr` | `lr_scheduler_kwargs={"min_lr": ...}` | 最低学习率 |
| `pretrain.warmup_steps` | `warmup_steps` | 预热步数 |
| `pretrain.dtype` | `bf16=True` 或 `fp16=True` | 混合精度 |
| `pretrain.weight_decay` | `weight_decay` | 权重衰减 |
| `pretrain.grad_clip` | `max_grad_norm` | 梯度裁剪 |
| `pretrain.patience` | `EarlyStoppingCallback(patience=N)` | 早停 |

**SFT 配置 (SFTConfig)：**

| from-scratch YAML | SFTConfig | 说明 |
|-------------------|-----------|------|
| `sft.batch_size` | `per_device_train_batch_size` | 单卡 batch size |
| `sft.epochs` | `num_train_epochs` | 训练轮数 |
| `sft.learning_rate` | `learning_rate` | 学习率 |
| `sft.dtype` | `bf16=True` | 混合精度 |
| (手动 loss mask) | `dataset_text_field` + `DataCollatorForCompletionOnlyLM` | 自动 loss mask |

**DPO 配置 (DPOConfig)：**

| from-scratch YAML | DPOConfig | 说明 |
|-------------------|-----------|------|
| `dpo.batch_size` | `per_device_train_batch_size` | 单卡 batch size |
| `dpo.epochs` | `num_train_epochs` | 训练轮数 |
| `dpo.learning_rate` | `learning_rate` | 学习率 |
| `dpo.beta` | `beta` | DPO 温度系数 |
| (deepcopy ref model) | `ref_model` 或自动创建 | 参考模型 |

**LoRA 配置 (LoraConfig)：**

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # from-scratch: rank
    lora_alpha=16,          # from-scratch: alpha
    lora_dropout=0.05,      # from-scratch: dropout
    target_modules=[        # from-scratch: ["w_q", "w_k", "w_v", "w_o"]
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    bias="none",            # from-scratch: 所有 Linear bias=False
)
```

### 4.2 HF Trainer 内部机制详解

HF `Trainer.train()` 内部流程与 from-scratch 的对应关系：

```
┌─────────────────────────────────────────────────────────────────┐
│              Trainer.train() 内部 10 个步骤                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① 初始化优化器 create_optimizer()                               │
│     from-scratch: AdamW(param_groups, lr, weight_decay)         │
│     HF: 相同的 AdamW，自动分离 decay/no-decay 参数组              │
│                                                                 │
│  ② 初始化学习率调度器 create_scheduler()                          │
│     from-scratch: CosineWarmupScheduler (手写)                  │
│     HF: get_scheduler("cosine", warmup_steps=N)                │
│                                                                 │
│  ③ 初始化混合精度 GradScaler + autocast                          │
│     from-scratch: torch.amp.GradScaler + autocast 手动包裹       │
│     HF: TrainingArguments(bf16=True) 自动处理                    │
│                                                                 │
│  ④ 训练循环开始 training_loop                                    │
│     from-scratch: for step in range(max_steps) 或 for epoch     │
│     HF: 内部 epoch 循环，支持 max_steps 中断                     │
│                                                                 │
│  ⑤ 前向传播 compute_loss()                                      │
│     from-scratch: logits, loss, _ = model(input_ids, targets)   │
│     HF: outputs = model(**inputs); loss = outputs.loss          │
│                                                                 │
│  ⑥ 反向传播 + 梯度累积                                           │
│     from-scratch: loss.backward(); if step % accum == 0: step() │
│     HF: accelerator.backward(loss); 自动管理累积计数              │
│                                                                 │
│  ⑦ 梯度裁剪 + 优化器步进                                         │
│     from-scratch: clip_grad_norm_(params, max_norm)             │
│     HF: 相同，由 max_grad_norm 控制                              │
│                                                                 │
│  ⑧ 日志记录 log()                                                │
│     from-scratch: TrainingLogger (手写)                         │
│     HF: 自动记录到 TensorBoard / WandB / console                 │
│                                                                 │
│  ⑨ 评估 evaluate()                                              │
│     from-scratch: 手动 eval loop                                │
│     HF: eval_strategy="steps"/"epoch" 自动触发                   │
│                                                                 │
│  ⑩ Checkpoint 保存 save_model()                                 │
│     from-scratch: torch.save(state_dict, path)                  │
│     HF: save_strategy="steps"/"epoch" 自动保存                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Callback 体系

| from-scratch 组件 | HF Callback | 触发时机 |
|-------------------|-------------|----------|
| EarlyStopping | EarlyStoppingCallback | on_evaluate |
| TrainingLogger.log() | 内置 logging | on_log |
| checkpoint 保存 | 内置 save_strategy | on_save |
| (无) | TensorBoardCallback | on_log |
| (无) | WandbCallback | on_log |
| (无) | ProgressCallback | on_step_end |

---

## 5. 推理设计

### 5.1 model.generate() 集成

from-scratch 版的 `generate()` 函数实现了自回归生成，HF 版通过继承 `GenerationMixin` 获得更丰富的生成能力：

| 功能 | from-scratch | HF GenerationMixin |
|------|-------------|-------------------|
| Greedy | 手动 argmax | `do_sample=False` |
| Top-k | 手动排序 + 过滤 | `top_k=50` |
| Top-p | 手动累积概率过滤 | `top_p=0.9` |
| Temperature | 手动 `logits / temp` | `temperature=0.7` |
| Repetition Penalty | 手动惩罚已生成 token | `repetition_penalty=1.1` |
| Beam Search | 未实现 | `num_beams=N` |
| KV Cache | 手动拼接 cached K/V | `use_cache=True` 自动管理 |
| EOS 停止 | 手动检查 | 内置 stopping_criteria |
| Batch 生成 | 未实现 | 内置支持 |

**关键实现：** `prepare_inputs_for_generation()`

```python
def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                   attention_mask=None, **kwargs):
    if past_key_values is not None:
        # KV Cache 模式：只取最后一个 token
        input_ids = input_ids[:, -1:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
```

### 5.2 Chat Template

```jinja2
{% for message in messages %}
{% if message['role'] == 'user' %}
Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
Assistant:
{% endif %}
```

**使用方式：**

```python
messages = [
    {"role": "user", "content": "你好"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# 输出: "Human: 你好\nAssistant: "
```

### 5.3 HF Pipeline

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="path/to/clearmind", tokenizer="path/to/clearmind")
output = pipe("Human: 你好\nAssistant: ", max_new_tokens=100, temperature=0.7)
```

---

## 6. 目录结构设计

```
capstone-clear-mind-by-hugging-face/
├── CLAUDE.md                          # AI 助手指南
├── README.md                          # 项目首页
├── requirements.txt                   # 依赖列表
├── run.sh                             # 一键启动脚本
├── configs/
│   ├── tiny.yaml                      # 教学/测试配置
│   ├── small.yaml                     # 入门训练配置
│   ├── medium.yaml                    # 标准训练配置
│   └── large.yaml                     # 完整训练配置
├── docs/
│   ├── PRD.md                         # 产品需求文档
│   ├── TECHNICAL_DESIGN.md            # 技术架构文档
│   ├── PROGRESS_TRACKER.md            # 开发进度表
│   ├── DEPLOY.md                      # 部署文档
│   └── AUTODL_GUIDE.md               # AutoDL 训练指南
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── configuration_clearmind.py # ClearMindConfig (PretrainedConfig)
│   │   ├── modeling_clearmind.py      # ClearMindForCausalLM (PreTrainedModel)
│   │   │                              #   含 Attention, MLP, RMSNorm, RoPE
│   │   └── auto_register.py          # AutoClass 注册
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py              # Tokenizer 训练 + PreTrainedTokenizerFast 封装
│   │   ├── prepare_data.py           # 数据下载/预处理
│   │   └── data_utils.py             # 数据格式化、chat template 应用
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py               # HF Trainer 预训练
│   │   ├── sft.py                    # TRL SFTTrainer 微调
│   │   ├── dpo.py                    # TRL DPOTrainer 对齐
│   │   └── callbacks.py             # 自定义 Callback
│   └── inference/
│       ├── __init__.py
│       ├── generate.py               # model.generate() 封装
│       └── chat.py                   # 交互式对话
├── scripts/
│   ├── train_tokenizer.py            # Tokenizer 训练入口
│   ├── train.py                      # 统一训练入口 (--stage pretrain/sft/dpo)
│   ├── chat.py                       # 交互式对话入口
│   ├── smoke_test.py                 # 冒烟测试
│   └── prepare_data.py              # 数据准备入口
├── notebooks/
│   ├── 01_tokenizer_comparison.ipynb  # Tokenizer 对比
│   ├── 02_model_comparison.ipynb      # 模型架构对比
│   ├── 03_data_comparison.ipynb       # 数据处理对比
│   ├── 04_pretrain_comparison.ipynb   # 预训练对比
│   ├── 05_sft_comparison.ipynb        # SFT 对比
│   ├── 06_dpo_comparison.ipynb        # DPO 对比
│   ├── 07_lora_comparison.ipynb       # LoRA 对比
│   └── 08_eval_comparison.ipynb       # 评估对比
├── evaluate/
│   ├── eval_perplexity.py            # Perplexity 评估
│   ├── eval_generation.py            # 生成质量评估
│   └── eval_benchmark.py            # lm-eval-harness 评估
├── deploy/
│   ├── web_demo.py                   # Gradio Web Demo
│   ├── export_gguf.py                # GGUF 格式导出
│   └── Dockerfile                    # Docker 部署
├── tests/
│   ├── conftest.py                   # 测试 fixture
│   ├── test_config.py                # 配置测试
│   ├── test_model.py                 # 模型测试
│   ├── test_tokenizer.py            # 分词器测试
│   ├── test_data.py                  # 数据处理测试
│   ├── test_training.py             # 训练流程测试
│   ├── test_generate.py             # 生成测试
│   └── test_lora.py                  # LoRA 测试
└── outputs/                          # 训练产物 (gitignored)
    ├── tokenizer/
    ├── pretrain/
    ├── sft/
    └── dpo/
```

---

## 7. 设备适配

### 7.1 accelerate 集成

```python
# accelerate config 示例 (单卡 GPU)
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
```

```python
# 多卡 DDP
accelerate launch --multi_gpu --num_processes 2 scripts/train.py --stage pretrain
```

| 场景 | from-scratch | HF + accelerate |
|------|-------------|-----------------|
| CPU 训练 | `device = "cpu"` | 自动检测 |
| 单卡 GPU | `device = "cuda"` | 自动检测 |
| Apple MPS | `device = "mps"` | 自动检测 |
| 多卡 DDP | `torchrun --nproc_per_node=2` | `accelerate launch --multi_gpu` |
| 混合精度 | 手动 `autocast + GradScaler` | `TrainingArguments(bf16=True)` |

---

## 附录

### 附录 A：依赖列表

```
# 核心
torch>=2.1.0
transformers>=4.40.0
tokenizers>=0.19.0
datasets>=2.18.0
accelerate>=0.28.0
trl>=0.8.0
peft>=0.10.0

# 评估
lm-eval>=0.4.0

# 工具
pyyaml>=6.0
numpy>=1.24.0
tqdm>=4.65.0
tensorboard>=2.15.0

# 部署
gradio>=4.0.0

# 开发
pytest>=7.0.0
ruff>=0.3.0
```

### 附录 B：版本兼容矩阵

| 组件 | 最低版本 | 推荐版本 | 说明 |
|------|---------|---------|------|
| Python | 3.10 | 3.11 | f-string / match 语法 |
| PyTorch | 2.1.0 | 2.2+ | SDPA 支持 |
| Transformers | 4.40.0 | 4.44+ | 最新 AutoClass 机制 |
| TRL | 0.8.0 | 0.12+ | 最新 SFTTrainer/DPOTrainer API |
| PEFT | 0.10.0 | 0.12+ | 稳定 LoRA 支持 |
| tokenizers | 0.19.0 | 0.20+ | ByteLevel BPE |
| datasets | 2.18.0 | 2.20+ | 高效 map/filter |
| accelerate | 0.28.0 | 0.33+ | 稳定分布式支持 |

### 附录 C：参数量验证

模型参数量公式（与 from-scratch 相同）：

```
Embedding:       vocab_size × hidden_size
每层 Attention:  hidden_size × hidden_size × 2                    (Q, O)
               + hidden_size × (hidden_size // num_kv_groups) × 2  (K, V)
每层 MLP:       hidden_size × intermediate_size × 3               (gate, up, down)
每层 Norm:      hidden_size × 2                                   (input_layernorm, post_attention_layernorm)
Final Norm:     hidden_size
LM Head:        与 Embedding 共享（不额外计算）

总参数量 ≈ vocab_size × hidden_size
         + num_hidden_layers × (4 × hidden_size² / group_factor + 3 × hidden_size × intermediate_size + 2 × hidden_size)
         + hidden_size
```

| 配置 | hidden_size | layers | heads | kv_heads | d_ff | vocab | 预估参数量 |
|------|-------------|--------|-------|----------|------|-------|-----------|
| Tiny | 128 | 4 | 4 | 4 | 352 | 2,000 | ~0.6M |
| Small | 512 | 8 | 8 | 8 | 1,408 | 8,000 | ~26M |
| Medium | 1,024 | 16 | 16 | 8 | 2,816 | 32,000 | ~200M |
| Large | 2,048 | 24 | 32 | 8 | 5,632 | 64,000 | ~930M |
