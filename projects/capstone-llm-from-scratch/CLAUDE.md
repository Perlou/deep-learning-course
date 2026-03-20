# CLAUDE.md — ClearMind 项目指南

## 项目概述

ClearMind 是一个从零构建的 LLM 训练项目，涵盖 Tokenizer → Pre-training → SFT → DPO 全流程。纯 PyTorch 实现，教育导向。

## 架构

- **模型**: GPT (RoPE + RMSNorm + SwiGLU + GQA + KV Cache)
- **训练**: PreTrainer / SFTTrainer / DPOTrainer，继承自 BaseTrainer
- **推理**: KV Cache 自回归生成，Top-k/Top-p 采样
- **数据**: BPE 分词器 (sentencepiece)，三种数据集格式

## 关键路径

| 模块 | 路径 |
|------|------|
| 模型定义 | `src/model/` (config.py, gpt.py, attention.py, rope.py) |
| 训练逻辑 | `src/training/` (base_trainer.py, pretrain.py, sft.py, dpo.py, lora.py) |
| 数据处理 | `src/data/` (tokenizer.py, pretrain_dataset.py, sft_dataset.py, dpo_dataset.py) |
| 推理 | `src/inference/` (generate.py, chat.py) |
| 入口脚本 | `scripts/train.py` (统一训练入口，--stage pretrain/sft/dpo) |
| 配置文件 | `configs/` (tiny.yaml, small.yaml, medium.yaml, large.yaml) |
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
```

## 代码规范

- Python 3.10+，使用 Ruff 做 lint/format
- 中英文双语注释，docstring 解释"为什么"
- 模型属性名: `w_q`, `w_k`, `w_v`, `w_o` (不是 HuggingFace 风格的 q_proj)
- Trainer 通过 BaseTrainer 共享逻辑，子类实现 `train()` 方法
- 配置通过 YAML + dataclass (ModelConfig)，支持工厂方法 (tiny/small/medium)
- Tiny 配置 vocab_size=2000，与 tiny.yaml 一致

## 测试

89 个测试覆盖所有核心模块。使用 `conftest.py` 中的 `tiny_config` 和 `tiny_model` fixture。
测试不需要 GPU，全部在 CPU 上运行。

## 依赖

主要依赖: torch, sentencepiece, pyyaml, numpy, tqdm
虚拟环境: `./venv/`
