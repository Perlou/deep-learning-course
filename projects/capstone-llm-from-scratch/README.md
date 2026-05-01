# 🧠 ClearMind

> 从零实现的中文 LLM 训练项目，基于 [minimind](https://github.com/jingyaogong/minimind) 的数据/tokenizer 生态，**通过更扎实的工程基础与若干标准化架构改进追求同等规模的效果反超**，并发布到 HuggingFace 与 ModelScope。

[![Tests](https://img.shields.io/badge/tests-141%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()
[![Status](https://img.shields.io/badge/Phase%201--5-✅-brightgreen)]()

## 🎯 发布矩阵

| 模型 | 参数量 | 对标 | 推荐显卡 | 训练时长 | 总成本 | Config |
|---|---|---|---|---|---|---|
| **ClearMind-Base** | **68.8M** dense | minimind-3 (64M dense) | RTX 4090 24G | ~28-32h | **¥65-80** | `configs/main.yaml` |
| **ClearMind-Plus** | **486.3M** dense | minimind-3-moe (198M-A64M) | A100-PCIE 40G | ~45-55h | **¥155-200** | `configs/plus.yaml` |

两个模型共享同一份训练代码、tokenizer、数据，仅 yaml 规格不同。完整成本/选型分析见 [docs/AUTODL_GUIDE.md](docs/AUTODL_GUIDE.md#step-1--注册-autodl--选-gpu)。

## 🚀 一键上手

```bash
# 1) 环境
git clone <this-repo> && cd capstone-llm-from-scratch
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2) 数据（从 minimind 镜像下载，国内推荐 modelscope）
python scripts/download_data.py --profile base --source modelscope

# 3) 本地 5 分钟冒烟（CPU/MPS 即可）
bash run.sh                                # 交互式：选 Tiny → 全流程

# 4) AutoDL 正式训练（断 SSH 不影响）
bash scripts/autodl/preflight.sh --profile base    # 9 项强制自检
bash scripts/autodl/launch.sh tiny  all            # 5 min 冒烟
bash scripts/autodl/launch.sh small all            # 30 min 验证
bash scripts/autodl/launch.sh base  all            # 12-18h 正式

# 5) 评估
python evaluate/benchmarks/ceval.py --config configs/main.yaml
python evaluate/benchmarks/cmmlu.py --config configs/main.yaml
python evaluate/benchmarks/alignbench.py --config configs/main.yaml

# 6) 发布到 HF / ModelScope
bash scripts/release.sh base --stage dpo --push-hf you/ClearMind-Base
```

## 🏗️ 架构

```
Token IDs ──► Embedding ──► N × TransformerBlock ──► RMSNorm ──► LM Head ──► Logits
                              ├─ RMSNorm + (QK-Norm) → MHA/GQA + RoPE(θ=1e6,YaRN) + KV Cache + SDPA
                              └─ RMSNorm → SwiGLU FFN
```

- **位置编码**：RoPE θ=1e6（Qwen3 对齐），可选 YaRN 长上下文外推
- **归一化**：RMSNorm + QK-Norm（Llama-3 / Gemma2 同款）
- **注意力**：GQA + KV Cache + Sliding Window + Flash Attention（SDPA 自动选）
- **FFN**：SwiGLU，`d_ff = ⌈d_model · π / 64⌉ · 64`（TensorCore 对齐）
- **残差初始化**：1/√(2L) 缩放（GPT-2/Llama 标准做法）
- **Tokenizer**：复用 minimind ByteLevel BPE（vocab=6400）
- **发布**：训练用 ClearMind GPT，发布时通过 `convert_to_qwen3.py` 转 Qwen3ForCausalLM 兼容 → ollama / vllm / llama.cpp 即用

## 📐 配置矩阵

| Config | 参数 | d_model / heads / kv_heads / layers | d_ff | seq_len | 用途 |
|---|---|---|---|---|---|
| `tiny.yaml` | 0.5M | 64 / 4 / 2 / 2 | 256 | 128 | CPU/MPS 冒烟 |
| `small.yaml` | 26M | 512 / 8 / 2 / 8 | 1664 | 1024 | 单卡（对齐 minimind2-small） |
| `main.yaml` | **68.8M** | 768 / 8 / 4 / 8 | 2432 | 1024 | **ClearMind-Base 发布版** |
| `plus.yaml` | **486.3M** | 1280 / 16 / 4 / 24 | 4032 | 1024 | **ClearMind-Plus 发布版** |

## ✅ 完成度

| Phase | 状态 | 说明 |
|---|---|---|
| 重构 1: 数据 + tokenizer | ✅ | HF tokenizer + chat_template + minimind 数据生态 |
| 重构 2: scripts/eval/deploy | ✅ | 统一入口 + 评测 + API 服务 |
| Phase 1: 地基 bugfix | ✅ | NaN 三层防御 + RoPE buffer 共享 + 残差缩放 + 原子 ckpt |
| Phase 2: 架构升级 | ✅ | QK-Norm + RoPE θ=1e6 + YaRN + d_ff 对齐 |
| Phase 3: 训练扩展 | 🟢 | DPO ✅ Distillation ✅ GRPO+CISPO ✅ Rollout 引擎 ✅（PPO/Agentic RL 待补） |
| Phase 4: 工程化 | ✅ | torch.compile + fused AdamW + DDP no_sync + activation ckpt + wandb/swanlab |
| Phase 5: 发布闭环 | ✅ | safetensors + Qwen3 export + HF/MS push + OpenAI 兼容 API + `release.sh` 端到端流水线 |
| Phase E1: 评测体系 | ✅ | C-Eval + CMMLU + AlignBench + LLM-as-Judge + 多模型对照 |
| AutoDL 上线工具链 | ✅ | preflight + launch(tmux) + status + save + release |

## 📁 项目结构

```
capstone-llm-from-scratch/
├── src/
│   ├── model/            # GPT 架构（config/rope/attention/transformer/...）
│   ├── data/             # HF tokenizer + 4 个 dataset 类
│   ├── training/         # base_trainer + pretrain/sft/dpo/distillation/grpo + rollout_engine
│   └── inference/        # generate.py + chat.py
├── scripts/
│   ├── train.py          # 统一训练入口（--stage pretrain/sft/dpo/distillation/grpo）
│   ├── release.sh        # ⭐ 端到端发布流水线（convert + 验证 + 打包 + push）
│   ├── autodl/           # ⭐ AutoDL 一键工具链
│   │   ├── preflight.sh  #   9 项自检
│   │   ├── launch.sh     #   tmux 启动器（断 SSH 不影响）
│   │   ├── status.sh     #   状态查询
│   │   └── save_outputs.sh
│   └── {convert_to_qwen3,push_to_hub,push_to_modelscope,smoke_test}.py
├── evaluate/             # ⭐ Phase E1 评测体系
│   ├── benchmarks/{ceval,cmmlu,alignbench}.py
│   ├── judge/llm_judge.py
│   └── eval_compare.py
├── deploy/               # OpenAI 兼容 API + Web demo + Dockerfile
├── configs/              # tiny/small/main/plus.yaml
├── tests/                # 141 个单元测试
└── docs/
    ├── AUTODL_GUIDE.md   # ⭐ 8 步攻略（rent → train → eval → release）
    ├── PRD.md / TECHNICAL_DESIGN.md / PROGRESS_TRACKER.md
    └── DEPLOY.md
```

## 📚 详细文档

- **[docs/AUTODL_GUIDE.md](docs/AUTODL_GUIDE.md)** — 完整上线攻略（rent + 自检 + 训练 + 评估 + 发布）
- **[evaluate/README.md](evaluate/README.md)** — 评测体系与 OpenCompass 对齐说明
- **[docs/PRD.md](docs/PRD.md)** — 产品需求与目标
- **[docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)** — 架构与各模块设计
- **[docs/PROGRESS_TRACKER.md](docs/PROGRESS_TRACKER.md)** — 阶段进度追踪
- **[docs/DEPLOY.md](docs/DEPLOY.md)** — 部署到 HF / ModelScope
- **[CLAUDE.md](CLAUDE.md)** — Claude Code 协作的项目指南

## 🧪 测试

```bash
./venv/bin/python -m pytest tests/ -v   # 141 passed
```

覆盖：模型前后向 / attention（GQA/KV cache/SWA）/ 配置 / HF tokenizer / 文本生成 / 三种 dataset / DPO loss / Distillation KL / GRPO reward / Rollout 引擎 / LoRA / 训练边界条件 / Resume。

## 🙏 致谢

- 数据集 / tokenizer / chat_template 模板：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）
- 架构灵感：Llama / Qwen / Gemma
- 算法：RoFormer (RoPE) / GQA / SwiGLU / DPO / GRPO / CISPO

## 📝 License

Apache-2.0（与 minimind 数据/tokenizer 来源协议保持一致）。

---

> **声明**：本项目复用 minimind 的开源数据集与 tokenizer 资产（Apache-2.0），独立实现了模型架构、trainer、dataset 适配层、评测体系与发布工具链。所有 minimind 借鉴点在代码注释中均有标注；所有"超越路径"改进项在 PRD 与 plan 文件中有详细论证。
