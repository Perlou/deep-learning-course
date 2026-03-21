# 🤗 ClearMind-HF — HuggingFace 生态下的 LLM 全流程训练

> **ClearMind (from-scratch) 的 HuggingFace 姊妹项目** — 相同的模型架构（RoPE + RMSNorm + SwiGLU + GQA），不同的实现哲学。从手写每一行到站在巨人肩膀上。

---

## ✨ 项目亮点

- 🤗 **HF 全家桶** — transformers / tokenizers / datasets / accelerate / trl / peft，一站式体验
- 🔄 **一一对应** — 每个模块都有 from-scratch 版对照，理解 HF 封装了什么
- 🧱 **自定义模型** — 不用现成的 GPT2/Llama，自己注册 PreTrainedModel，学习 HF 模型机制
- ⚡ **TRL 训练** — SFTTrainer / DPOTrainer，工业级训练流程
- 🔧 **PEFT 微调** — LoRA / QLoRA，参数高效微调
- 📊 **lm-eval 评估** — 标准化 benchmark，不再手写 eval
- 🚀 **Hub 集成** — push_to_hub 一键发布，全球开发者可用
- 📓 **对比 Notebook** — 8 个 from-scratch vs HuggingFace 对比教学 notebook
- 🎓 **教育导向** — 中英双语注释，每个 HF API 标注对应的 from-scratch 实现
- 🖥️ **多设备** — 通过 accelerate 自动适配 CPU / CUDA / MPS / 多卡
- 📐 **4 档配置** — Tiny(0.6M) → Mini(26M) → ClearMind(200M) → Plus(930M)

---

## 🔀 姊妹项目对比

| 维度 | ClearMind (from-scratch) | ClearMind-HF |
|------|--------------------------|--------------|
| **模型定义** | `GPT(nn.Module)` 手写所有组件 | `ClearMindForCausalLM(PreTrainedModel)` 自定义注册 |
| **分词器** | sentencepiece + 手动封装 | HF tokenizers + PreTrainedTokenizerFast |
| **训练循环** | 手写 step/epoch-based loop | HF Trainer + TRL (SFTTrainer / DPOTrainer) |
| **LoRA** | 手写 LoRALinear + apply_lora | PEFT LoraConfig + get_peft_model |
| **多卡训练** | torchrun DDP 手动管理 | accelerate launch 一键分布式 |
| **数据处理** | 自定义 Dataset + 手动 padding | datasets + DataCollator 动态批处理 |
| **评估** | 手写 eval 脚本 | lm-eval-harness 标准 benchmark |
| **部署** | 手写 FastAPI + export | pipeline + Gradio + push_to_hub |

> [!NOTE]
> 两个项目使用 **完全相同的模型架构**（RoPE + RMSNorm + SwiGLU + GQA），参数量在相同配置下一致。区别仅在于实现方式和生态集成。

---

## 🏷️ 模型家族

| 代号 | 配置 | hidden_size | layers | heads | kv_heads | 参数量 | 推荐设备 |
|------|------|-------------|--------|-------|----------|--------|---------|
| ClearMind-Tiny | tiny.yaml | 128 | 4 | 4 | 4 (MHA) | ~0.6M | CPU |
| ClearMind-Mini | small.yaml | 512 | 8 | 8 | 8 (MHA) | ~26M | CPU / GPU |
| ClearMind | medium.yaml | 1,024 | 16 | 16 | 8 (GQA) | ~200M | GPU 8GB+ |
| ClearMind-Plus | large.yaml | 2,048 | 24 | 32 | 8 (GQA) | ~930M | GPU 24GB+ |

---

## 📁 项目结构

```
capstone-clear-mind-by-hugging-face/
├── 📄 CLAUDE.md                          # AI 助手指南
├── 📄 README.md                          # 项目首页（本文件）
├── 📄 requirements.txt                   # 依赖列表
├── 🚀 run.sh                             # 一键启动脚本
│
├── ⚙️ configs/                            # 模型配置
│   ├── tiny.yaml                         # 教学/测试 (~0.6M)
│   ├── small.yaml                        # 入门训练 (~26M)
│   ├── medium.yaml                       # 标准训练 (~200M)
│   └── large.yaml                        # 完整训练 (~930M)
│
├── 📚 docs/                               # 项目文档
│   ├── PRD.md                            # 产品需求文档
│   ├── TECHNICAL_DESIGN.md               # 技术架构文档
│   ├── PROGRESS_TRACKER.md               # 开发进度表
│   ├── DEPLOY.md                         # 部署文档
│   └── AUTODL_GUIDE.md                   # AutoDL 训练指南
│
├── 🧠 src/                                # 核心源代码
│   ├── model/
│   │   ├── configuration_clearmind.py    # ClearMindConfig (PretrainedConfig)
│   │   ├── modeling_clearmind.py         # ClearMindForCausalLM (PreTrainedModel)
│   │   └── auto_register.py             # AutoClass 注册
│   ├── data/
│   │   ├── tokenizer.py                 # Tokenizer 训练 + 封装
│   │   ├── prepare_data.py              # 数据下载/预处理
│   │   └── data_utils.py                # 数据格式化工具
│   ├── training/
│   │   ├── pretrain.py                  # HF Trainer 预训练
│   │   ├── sft.py                       # TRL SFTTrainer
│   │   ├── dpo.py                       # TRL DPOTrainer
│   │   └── callbacks.py                 # 自定义 Callback
│   └── inference/
│       ├── generate.py                  # model.generate() 封装
│       └── chat.py                      # 交互式对话
│
├── 📜 scripts/                            # 入口脚本
│   ├── train_tokenizer.py               # Tokenizer 训练
│   ├── train.py                         # 统一训练入口 (--stage)
│   ├── chat.py                          # 交互式对话
│   ├── smoke_test.py                    # 冒烟测试
│   └── prepare_data.py                  # 数据准备
│
├── 📓 notebooks/                          # 对比教学 Notebook
│   ├── 01_tokenizer_comparison.ipynb    # Tokenizer 对比
│   ├── 02_model_comparison.ipynb        # 模型架构对比
│   ├── 03_data_comparison.ipynb         # 数据处理对比
│   ├── 04_pretrain_comparison.ipynb     # 预训练对比
│   ├── 05_sft_comparison.ipynb          # SFT 对比
│   ├── 06_dpo_comparison.ipynb          # DPO 对比
│   ├── 07_lora_comparison.ipynb         # LoRA 对比
│   └── 08_eval_comparison.ipynb         # 评估对比
│
├── 📊 evaluate/                           # 评估脚本
│   ├── eval_perplexity.py               # Perplexity 评估
│   ├── eval_generation.py               # 生成质量评估
│   └── eval_benchmark.py                # lm-eval-harness
│
├── 🚀 deploy/                             # 部署相关
│   ├── web_demo.py                      # Gradio Web Demo
│   ├── export_gguf.py                   # GGUF 导出
│   └── Dockerfile                       # Docker 部署
│
├── 🧪 tests/                              # 单元测试
│   ├── conftest.py                      # 测试 fixture
│   ├── test_config.py                   # 配置测试
│   ├── test_model.py                    # 模型测试
│   ├── test_tokenizer.py               # 分词器测试
│   ├── test_data.py                     # 数据测试
│   ├── test_training.py                 # 训练测试
│   ├── test_generate.py                 # 生成测试
│   └── test_lora.py                     # LoRA 测试
│
└── 📦 outputs/                            # 训练产物 (gitignored)
    ├── tokenizer/
    ├── pretrain/
    ├── sft/
    └── dpo/
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd capstone-clear-mind-by-hugging-face

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 一键启动

```bash
# 交互式菜单（推荐）
bash run.sh
```

### 手动步骤

```bash
# 1. 准备数据
python scripts/prepare_data.py

# 2. 训练 Tokenizer
python scripts/train_tokenizer.py --vocab_size 8000

# 3. 预训练
python scripts/train.py --stage pretrain --config configs/tiny.yaml

# 4. SFT 微调
python scripts/train.py --stage sft --config configs/tiny.yaml

# 5. DPO 对齐
python scripts/train.py --stage dpo --config configs/tiny.yaml

# 6. 交互式对话
python scripts/chat.py --model_path outputs/dpo/
```

### 冒烟测试

```bash
# 验证全流程（Tiny 配置，1 步训练）
python scripts/smoke_test.py --max_steps 1
```

### LoRA 微调

```bash
# 使用 PEFT LoRA 微调
python scripts/train.py --stage sft --config configs/tiny.yaml --use_lora

# QLoRA 微调（需要 GPU + bitsandbytes）
python scripts/train.py --stage sft --config configs/tiny.yaml --use_qlora
```

### 多卡训练

```bash
# 使用 accelerate 启动多卡训练
accelerate launch --multi_gpu --num_processes 2 scripts/train.py --stage pretrain --config configs/small.yaml
```

---

## 🏗️ 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    ClearMind-HF Pipeline                 │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌─────┐  ┌─────┐         │
│  │Tokenizer │→ │ Pretrain  │→ │ SFT │→ │ DPO │         │
│  │tokenizers│  │ Trainer   │  │ TRL │  │ TRL │         │
│  └──────────┘  └──────────┘  └─────┘  └─────┘         │
│       │              │           │         │            │
│       ▼              ▼           ▼         ▼            │
│  ┌──────────────────────────────────────────────┐      │
│  │    ClearMindForCausalLM (PreTrainedModel)     │      │
│  │    RoPE + RMSNorm + SwiGLU + GQA              │      │
│  └──────────────────────────────────────────────┘      │
│       │              │           │         │            │
│       ▼              ▼           ▼         ▼            │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────┐      │
│  │ peft    │  │ lm-eval  │  │pipeline│  │ Hub  │      │
│  │LoRA    │  │ 评估     │  │ Gradio │  │ push │      │
│  └─────────┘  └──────────┘  └────────┘  └──────┘      │
└─────────────────────────────────────────────────────────┘
```

### 模型架构

```
Input Token IDs
       │
       ▼
┌──────────────┐
│ embed_tokens  │  nn.Embedding(vocab_size, hidden_size)
│ + dropout     │  权重与 lm_head 共享
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│         ClearMindDecoderLayer × N         │  (N = num_hidden_layers)
│  ┌────────────────────────────────────┐  │
│  │ input_layernorm (RMSNorm)          │  │  Pre-Norm
│  │         │                          │  │
│  │ self_attn (GQA + RoPE + KV Cache)  │  │  q_proj / k_proj / v_proj / o_proj
│  │         │                          │  │  F.scaled_dot_product_attention
│  │     + residual                     │  │
│  ├────────────────────────────────────┤  │
│  │ post_attention_layernorm (RMSNorm) │  │  Pre-Norm
│  │         │                          │  │
│  │ mlp (SwiGLU)                       │  │  gate_proj / up_proj / down_proj
│  │   silu(gate) * up → down           │  │
│  │         │                          │  │
│  │     + residual                     │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ norm (RMSNorm)│  Final LayerNorm
│ + lm_head     │  nn.Linear(hidden_size, vocab_size)
└──────┬───────┘
       │
       ▼
   Logits → Loss (CrossEntropy, ignore_index=-100)
```

---

## 📊 模型配置详解

| 参数 | Tiny | Mini | ClearMind | Plus |
|------|------|------|-----------|------|
| hidden_size | 128 | 512 | 1,024 | 2,048 |
| num_attention_heads | 4 | 8 | 16 | 32 |
| num_key_value_heads | 4 | 8 | 8 | 8 |
| num_hidden_layers | 4 | 8 | 16 | 24 |
| intermediate_size | 352 | 1,408 | 2,816 | 5,632 |
| vocab_size | 2,000 | 8,000 | 32,000 | 64,000 |
| max_position_embeddings | 128 | 512 | 1,024 | 2,048 |
| hidden_dropout_prob | 0.1 | 0.1 | 0.05 | 0.0 |
| rms_norm_eps | 1e-6 | 1e-6 | 1e-6 | 1e-5 |
| 注意力机制 | MHA | MHA | GQA 2:1 | GQA 4:1 |
| 训练精度 | float32 | float32 | bfloat16 | bfloat16 |
| 预估参数量 | ~0.6M | ~26M | ~200M | ~930M |

---

## 📊 模型评估

```bash
# Perplexity 评估
python evaluate/eval_perplexity.py --model_path outputs/pretrain/

# 生成质量评估
python evaluate/eval_generation.py --model_path outputs/sft/

# lm-eval-harness 标准评估
python evaluate/eval_benchmark.py --model_path outputs/dpo/ --tasks hellaswag,arc_easy
```

---

## 🚀 部署上线

```bash
# Gradio Web Demo
python deploy/web_demo.py --model_path outputs/dpo/

# 推送到 HuggingFace Hub
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('outputs/dpo/')
tokenizer = AutoTokenizer.from_pretrained('outputs/tokenizer/')
model.push_to_hub('your-username/clearmind-hf')
tokenizer.push_to_hub('your-username/clearmind-hf')
"

# GGUF 导出（供 llama.cpp 使用）
python deploy/export_gguf.py --model_path outputs/dpo/ --output clearmind.gguf
```

---

## 🧪 单元测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_model.py -v
python -m pytest tests/test_tokenizer.py -v

# 带覆盖率
python -m pytest tests/ --cov=src --cov-report=term-missing
```

| 测试文件 | 覆盖模块 | 说明 |
|---------|---------|------|
| test_config.py | ClearMindConfig | 配置创建、验证、YAML 加载、from_pretrained |
| test_model.py | ClearMindForCausalLM | forward/backward、KV Cache、save/load_pretrained |
| test_tokenizer.py | Tokenizer | encode/decode、special tokens、chat template |
| test_data.py | Data Utils | 数据格式化、tokenize、DataCollator |
| test_training.py | Training | Trainer 集成、SFT loss mask、DPO |
| test_generate.py | Generate | model.generate()、pipeline |
| test_lora.py | PEFT | LoRA apply/merge/save/load |

---

## 📖 HF 生态学习路线

1. **tokenizers** — 理解 BPE 分词原理，学习 PreTrainedTokenizerFast 机制
2. **PreTrainedModel** — 自定义模型注册，理解 save/load/generate 框架
3. **datasets** — 高效数据处理，map/filter/shuffle，DataCollator
4. **Trainer** — 深入理解训练循环封装，TrainingArguments 配置
5. **TRL** — SFTTrainer / DPOTrainer，RLHF 流程
6. **PEFT** — LoRA/QLoRA，参数高效微调
7. **Hub** — push_to_hub，Model Card，开源协作

---

## 📝 License

MIT License
