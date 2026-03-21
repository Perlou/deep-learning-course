# CLAUDE.md — ClearMind-HF 项目指南

## 项目概述

ClearMind-HF 是 ClearMind (from-scratch) 的 HuggingFace 生态姊妹项目。使用相同的模型架构（RoPE + RMSNorm + SwiGLU + GQA），通过 HF 全家桶实现 Tokenizer → Pre-training → SFT → DPO 全流程。

## 姊妹项目

- **from-scratch 版:** `../capstone-llm-from-scratch/`
- **HF 版 (本项目):** `../capstone-clear-mind-by-hugging-face/`

## 架构

- **模型**: ClearMindForCausalLM (PreTrainedModel)，RoPE + RMSNorm + SwiGLU + GQA + KV Cache
- **训练**: HF Trainer (Pretrain) / TRL SFTTrainer (SFT) / TRL DPOTrainer (DPO)
- **推理**: model.generate() (GenerationMixin) + pipeline + Gradio
- **数据**: HF tokenizers (ByteLevelBPE) + datasets + DataCollator
- **微调**: PEFT LoRA / QLoRA

## 关键路径

| 模块 | 路径 |
|------|------|
| 模型配置 | `src/model/configuration_clearmind.py` |
| 模型定义 | `src/model/modeling_clearmind.py` |
| AutoClass 注册 | `src/model/auto_register.py` |
| Tokenizer | `src/data/tokenizer.py` |
| 数据处理 | `src/data/data_utils.py`, `src/data/prepare_data.py` |
| 预训练 | `src/training/pretrain.py` |
| SFT 微调 | `src/training/sft.py` |
| DPO 对齐 | `src/training/dpo.py` |
| 推理 | `src/inference/generate.py`, `src/inference/chat.py` |
| 入口脚本 | `scripts/train.py` (统一训练入口，--stage pretrain/sft/dpo) |
| 配置文件 | `configs/` (tiny.yaml, small.yaml, medium.yaml, large.yaml) |
| 对比 Notebook | `notebooks/` (8 个 from-scratch vs HF 对比 notebook) |
| 评估 | `evaluate/` |
| 测试 | `tests/` |

## 开发命令

```bash
# 运行测试
./venv/bin/python -m pytest tests/ -v

# Lint
./venv/bin/python -m ruff check src/ scripts/ tests/

# 冒烟测试
./venv/bin/python scripts/smoke_test.py --max_steps 1

# 训练
./venv/bin/python scripts/train.py --stage pretrain --config configs/tiny.yaml
./venv/bin/python scripts/train.py --stage sft --config configs/tiny.yaml
./venv/bin/python scripts/train.py --stage dpo --config configs/tiny.yaml

# LoRA 微调
./venv/bin/python scripts/train.py --stage sft --config configs/tiny.yaml --use_lora

# 多卡训练
accelerate launch --multi_gpu --num_processes 2 scripts/train.py --stage pretrain --config configs/small.yaml
```

## 代码规范

- Python 3.10+，使用 Ruff 做 lint/format
- 中英文双语注释，docstring 解释"为什么"
- **HF 标准命名**：`q_proj`, `k_proj`, `v_proj`, `o_proj`（不是 from-scratch 的 `w_q` 等）
- **HF 标准命名**：`gate_proj`, `up_proj`, `down_proj`（不是 from-scratch 的 `w_gate` 等）
- **HF 标准命名**：`hidden_size`, `num_attention_heads`, `num_hidden_layers`（不是 `d_model` 等）
- 配置继承 PretrainedConfig，模型继承 PreTrainedModel
- 训练使用 HF Trainer / TRL，不手写 training loop
- Tiny 配置 vocab_size=2000，与 tiny.yaml 一致

## 配置文件结构

```yaml
# configs/tiny.yaml 示例
model:
  hidden_size: 128
  num_attention_heads: 4
  num_key_value_heads: 4
  num_hidden_layers: 4
  intermediate_size: 352
  vocab_size: 2000
  max_position_embeddings: 128
  hidden_dropout_prob: 0.1
  rms_norm_eps: 1.0e-6

pretrain:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  max_steps: 200
  learning_rate: 5.0e-4
  warmup_steps: 20
  bf16: false

sft:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  num_train_epochs: 2
  learning_rate: 2.0e-5

dpo:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  learning_rate: 1.0e-5
  beta: 0.1
```

## 测试

使用 `conftest.py` 中的 `tiny_config` 和 `tiny_model` fixture。
测试不需要 GPU，全部在 CPU 上运行。

## 依赖

主要依赖: torch, transformers, tokenizers, datasets, accelerate, trl, peft, pyyaml, numpy, tqdm
评估: lm-eval
部署: gradio
虚拟环境: `./venv/`

## 与 from-scratch 版本的文件映射

| from-scratch | HF 版 | 变化说明 |
|-------------|-------|---------|
| `src/model/config.py` (ModelConfig) | `src/model/configuration_clearmind.py` (ClearMindConfig) | 继承 PretrainedConfig |
| `src/model/gpt.py` (GPT) | `src/model/modeling_clearmind.py` (ClearMindForCausalLM) | 继承 PreTrainedModel |
| `src/model/attention.py` (Attention) | `src/model/modeling_clearmind.py` (ClearMindAttention) | 合并到单文件 |
| `src/model/feedforward.py` (FeedForward) | `src/model/modeling_clearmind.py` (ClearMindMLP) | 合并到单文件 |
| `src/model/normalization.py` (RMSNorm) | `src/model/modeling_clearmind.py` (ClearMindRMSNorm) | 合并到单文件 |
| `src/model/rope.py` | `src/model/modeling_clearmind.py` (ClearMindRotaryEmbedding) | 合并到单文件 |
| `src/model/transformer.py` (TransformerBlock) | `src/model/modeling_clearmind.py` (ClearMindDecoderLayer) | 合并到单文件 |
| (无) | `src/model/auto_register.py` | 新增：AutoClass 注册 |
| `src/data/tokenizer.py` (ClearMindTokenizer) | `src/data/tokenizer.py` (PreTrainedTokenizerFast) | tokenizers 库替代 sentencepiece |
| `src/data/pretrain_dataset.py` | `src/data/data_utils.py` + datasets | DataCollatorForLanguageModeling |
| `src/data/sft_dataset.py` | `src/data/data_utils.py` + datasets | DataCollatorForCompletionOnlyLM |
| `src/data/dpo_dataset.py` | `src/data/data_utils.py` + datasets | DPOTrainer 内置处理 |
| `src/training/base_trainer.py` | (不需要) | HF Trainer 提供基类 |
| `src/training/trainer_utils.py` | `src/training/callbacks.py` | EarlyStoppingCallback 等 |
| `src/training/pretrain.py` (PreTrainer) | `src/training/pretrain.py` (HF Trainer) | Trainer + TrainingArguments |
| `src/training/sft.py` (SFTTrainer) | `src/training/sft.py` (TRL SFTTrainer) | SFTTrainer + SFTConfig |
| `src/training/dpo.py` (DPOTrainer) | `src/training/dpo.py` (TRL DPOTrainer) | DPOTrainer + DPOConfig |
| `src/training/lora.py` (LoRALinear) | PEFT 库 (LoraConfig) | get_peft_model 替代 apply_lora |
| `src/inference/generate.py` | `src/inference/generate.py` | model.generate() 替代手写生成 |
| `src/inference/chat.py` | `src/inference/chat.py` | pipeline 替代手写推理 |
| `scripts/train_tokenizer.py` | `scripts/train_tokenizer.py` | tokenizers 库替代 sentencepiece |
| `scripts/train.py` | `scripts/train.py` | Trainer API 替代手写 loop |
| `scripts/launch_ddp.py` | accelerate launch | accelerate 替代 torchrun |
| `evaluate/eval_benchmark.py` | `evaluate/eval_benchmark.py` | lm-eval-harness |
| `deploy/web_demo.py` | `deploy/web_demo.py` | pipeline + Gradio |
