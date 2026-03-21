# 📊 ClearMind-HF 开发进度表

> **项目:** ClearMind-HF (HuggingFace 生态版)
> **姊妹项目:** [ClearMind (from-scratch)](../../capstone-llm-from-scratch/)
> **开始日期:** 2026-03-21
> **最后更新:** 2026-03-21

---

## 📊 总体进度

| 阶段 | 进度 | 状态 |
|------|------|------|
| 1. 项目搭建与基础设施 | `████████████████████ 100%` | ✅ Complete |
| 2. Tokenizer 训练 | `████████████████████ 100%` | ✅ Complete |
| 3. 模型定义 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 4. 数据处理 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 5. 预训练 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 6. SFT 微调 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 7. DPO 对齐 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 8. PEFT 微调 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 9. 评估体系 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 10. 推理与部署 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 11. 工程质量与文档 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |
| 12. HF Hub 集成 | `░░░░░░░░░░░░░░░░░░░░ 0%` | 🔜 Pending |

---

## 阶段 1：项目搭建与基础设施 ✅

> **目标:** 搭建项目骨架，配置开发环境

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 1.1 | 创建项目目录结构 (src/, scripts/, configs/, tests/, notebooks/, docs/) | 🔥 P0 | ✅ Complete | 2026-03-21 | 15 个目录, 53 个文件 |
| 1.2 | 编写 requirements.txt | 🔥 P0 | ✅ Complete | 2026-03-21 | torch, transformers, trl, peft, datasets, accelerate |
| 1.3 | 创建 configs/ 配置文件 (tiny/small/medium/large.yaml) | 🔥 P0 | ✅ Complete | 2026-03-21 | HF 命名约定 (hidden_size 等) |
| 1.4 | 编写 run.sh 一键启动脚本 | ⚡ P1 | ✅ Complete | 2026-03-21 | 9 个选项, 含 LoRA 微调 |
| 1.5 | 编写 CLAUDE.md AI 助手指南 | 🔥 P0 | ✅ Complete | 2026-03-21 | 文档规划阶段已完成 |
| 1.6 | 配置 .gitignore | 🔥 P0 | ✅ Complete | 2026-03-21 | outputs/, venv/, .cache/, runs/ |
| 1.7 | 初始化 Git 仓库 | ⚡ P1 | ✅ Complete | 2026-03-21 | monorepo 内, 无需 git init |

---

## 阶段 2：Tokenizer 训练 ✅

> **目标:** 使用 HF tokenizers 训练 BPE 分词器，封装为 PreTrainedTokenizerFast
> **from-scratch 对应:** `src/data/tokenizer.py` + `scripts/train_tokenizer.py`

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 2.1 | 使用 tokenizers 库训练 ByteLevelBPE | 🔥 P0 | ✅ Complete | 2026-03-21 | BPE + ByteLevel pre-tokenizer |
| 2.2 | 封装为 PreTrainedTokenizerFast | 🔥 P0 | ✅ Complete | 2026-03-21 | save_pretrained / from_pretrained |
| 2.3 | 配置特殊 token (bos/eos/pad/unk) | 🔥 P0 | ✅ Complete | 2026-03-21 | TemplateProcessing 自动 BOS/EOS |
| 2.4 | 实现 chat template (Jinja2) | ⚡ P1 | ✅ Complete | 2026-03-21 | Human/Assistant 格式 |
| 2.5 | 编写 scripts/train_tokenizer.py | 🔥 P0 | ✅ Complete | 2026-03-21 | 读 YAML config, 训练+保存 |
| 2.6 | 编写 Tokenizer 单元测试 | ⚡ P1 | ✅ Complete | 2026-03-21 | 20 个测试全部通过 |
| 2.7 | 编写 01_tokenizer_comparison.ipynb | 🟡 P2 | ✅ Complete | 2026-03-21 | sentencepiece vs HF tokenizers 对比 |

---

## 阶段 3：模型定义 🔜

> **目标:** 实现 ClearMindConfig + ClearMindForCausalLM，注册 AutoClass
> **from-scratch 对应:** `src/model/` 全部文件

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 3.1 | 实现 ClearMindConfig (PretrainedConfig) | 🔥 P0 | 🔜 Pending | - | 字段映射见 TECHNICAL_DESIGN.md |
| 3.2 | 实现 ClearMindRMSNorm | 🔥 P0 | 🔜 Pending | - | from-scratch: RMSNorm |
| 3.3 | 实现 ClearMindRotaryEmbedding (RoPE) | 🔥 P0 | 🔜 Pending | - | from-scratch: precompute_rope_frequencies |
| 3.4 | 实现 ClearMindAttention (GQA + RoPE + KV Cache) | 🔥 P0 | 🔜 Pending | - | q_proj/k_proj/v_proj/o_proj 命名 |
| 3.5 | 实现 ClearMindMLP (SwiGLU) | 🔥 P0 | 🔜 Pending | - | gate_proj/up_proj/down_proj 命名 |
| 3.6 | 实现 ClearMindDecoderLayer | 🔥 P0 | 🔜 Pending | - | from-scratch: TransformerBlock |
| 3.7 | 实现 ClearMindModel + ClearMindForCausalLM | 🔥 P0 | 🔜 Pending | - | 继承 PreTrainedModel |
| 3.8 | 实现 prepare_inputs_for_generation() | 🔥 P0 | 🔜 Pending | - | GenerationMixin 集成 |
| 3.9 | AutoClass 注册 (AutoConfig, AutoModelForCausalLM) | ⚡ P1 | 🔜 Pending | - | auto_register.py |
| 3.10 | 参数量验证 (与 from-scratch 版对比) | ⚡ P1 | 🔜 Pending | - | 4 档配置全部验证 |
| 3.11 | 编写模型单元测试 | ⚡ P1 | 🔜 Pending | - | forward/backward/KV Cache/save_pretrained |
| 3.12 | 编写 02_model_comparison.ipynb | 🟡 P2 | 🔜 Pending | - | nn.Module vs PreTrainedModel 对比 |

---

## 阶段 4：数据处理 🔜

> **目标:** 使用 datasets 库处理预训练/SFT/DPO 数据
> **from-scratch 对应:** `src/data/pretrain_dataset.py`, `sft_dataset.py`, `dpo_dataset.py`

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 4.1 | 编写 scripts/prepare_data.py 数据下载/准备 | 🔥 P0 | ✅ Complete | 2026-03-21 | small/medium/large 三档 |
| 4.2 | 预训练数据处理 (datasets.map + tokenizer) | 🔥 P0 | 🔜 Pending | - | 替代 PretrainDataset |
| 4.3 | SFT 数据处理 (chat template 格式化) | 🔥 P0 | 🔜 Pending | - | apply_chat_template + Alpaca/ShareGPT |
| 4.4 | DPO 数据处理 (prompt/chosen/rejected 格式) | 🔥 P0 | 🔜 Pending | - | DPOTrainer 标准输入格式 |
| 4.5 | 编写数据处理单元测试 | ⚡ P1 | 🔜 Pending | - | 数据格式验证、tokenize 正确性 |
| 4.6 | 编写 03_data_comparison.ipynb | 🟡 P2 | 🔜 Pending | - | 手写 Dataset vs datasets 库对比 |

---

## 阶段 5：预训练 🔜

> **目标:** 使用 HF Trainer 进行预训练
> **from-scratch 对应:** `src/training/pretrain.py`

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 5.1 | 配置 TrainingArguments (从 YAML 加载) | 🔥 P0 | 🔜 Pending | - | 字段映射见 TECHNICAL_DESIGN.md |
| 5.2 | 实现 DataCollatorForLanguageModeling 集成 | 🔥 P0 | 🔜 Pending | - | mlm=False, 自动生成 labels |
| 5.3 | 编写预训练脚本 (scripts/train.py --stage pretrain) | 🔥 P0 | 🔜 Pending | - | 统一入口 |
| 5.4 | 实现自定义 Callback (日志/评估) | ⚡ P1 | 🔜 Pending | - | from-scratch: TrainingLogger |
| 5.5 | accelerate 多卡支持 | 🟡 P2 | 🔜 Pending | - | from-scratch: launch_ddp.py |
| 5.6 | 冒烟测试 (Tiny 配置 max_steps=1) | 🔥 P0 | 🔜 Pending | - | 验证完整流程 |
| 5.7 | 编写 04_pretrain_comparison.ipynb | 🟡 P2 | 🔜 Pending | - | 手写 loop vs HF Trainer 对比 |

---

## 阶段 6：SFT 微调 🔜

> **目标:** 使用 TRL SFTTrainer 进行指令微调
> **from-scratch 对应:** `src/training/sft.py`

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 6.1 | 配置 SFTConfig | 🔥 P0 | 🔜 Pending | - | 从 YAML 加载 SFT 参数 |
| 6.2 | 集成 DataCollatorForCompletionOnlyLM | 🔥 P0 | 🔜 Pending | - | 替代手写 loss mask |
| 6.3 | 编写 SFT 训练脚本 (scripts/train.py --stage sft) | 🔥 P0 | 🔜 Pending | - | TRL SFTTrainer |
| 6.4 | 验证 loss mask 正确性 | ⚡ P1 | 🔜 Pending | - | 只有 Assistant 回复计算 loss |
| 6.5 | 编写 05_sft_comparison.ipynb | 🟡 P2 | 🔜 Pending | - | 手写 SFTTrainer vs TRL 对比 |

---

## 阶段 7：DPO 对齐 🔜

> **目标:** 使用 TRL DPOTrainer 进行偏好对齐
> **from-scratch 对应:** `src/training/dpo.py`

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 7.1 | 配置 DPOConfig | 🔥 P0 | 🔜 Pending | - | beta, loss_type 等 |
| 7.2 | 编写 DPO 训练脚本 (scripts/train.py --stage dpo) | 🔥 P0 | 🔜 Pending | - | TRL DPOTrainer |
| 7.3 | 验证 ref model 管理 | ⚡ P1 | 🔜 Pending | - | from-scratch: deepcopy |
| 7.4 | 验证 DPO loss 计算 | ⚡ P1 | 🔜 Pending | - | 与 from-scratch 数值对比 |
| 7.5 | 编写 06_dpo_comparison.ipynb | 🟡 P2 | 🔜 Pending | - | 手写 DPO vs TRL DPO 对比 |

---

## 阶段 8：PEFT 微调 🔜

> **目标:** 使用 PEFT 库实现 LoRA/QLoRA 微调
> **from-scratch 对应:** `src/training/lora.py`

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 8.1 | 配置 LoraConfig (target_modules, r, alpha) | 🔥 P0 | 🔜 Pending | - | target: q_proj, k_proj, v_proj, o_proj |
| 8.2 | 集成 get_peft_model() | 🔥 P0 | 🔜 Pending | - | 替代 apply_lora() |
| 8.3 | 实现 QLoRA (BitsAndBytesConfig + LoRA) | ⚡ P1 | 🔜 Pending | - | from-scratch 无对应 |
| 8.4 | 实现 merge_and_unload() | 🔥 P0 | 🔜 Pending | - | from-scratch: merge_lora() |
| 8.5 | LoRA adapter 保存/加载 | ⚡ P1 | 🔜 Pending | - | save_pretrained / from_pretrained |
| 8.6 | 编写 LoRA 单元测试 | ⚡ P1 | 🔜 Pending | - | apply/merge/save/load |
| 8.7 | 编写 07_lora_comparison.ipynb | 🟡 P2 | 🔜 Pending | - | 手写 LoRA vs PEFT 对比 |

---

## 阶段 9：评估体系 🔜

> **目标:** 集成 lm-eval-harness，建立完整评估流程
> **from-scratch 对应:** `evaluate/` 目录

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 9.1 | Perplexity 评估脚本 | 🔥 P0 | 🔜 Pending | - | from-scratch: eval_perplexity.py |
| 9.2 | lm-eval-harness 集成 | ⚡ P1 | 🔜 Pending | - | 标准 benchmark 评估 |
| 9.3 | 阶段对比评估 (Pretrain vs SFT vs DPO) | ⚡ P1 | 🔜 Pending | - | 可视化对比各阶段表现 |
| 9.4 | 编写 08_eval_comparison.ipynb | 🟡 P2 | 🔜 Pending | - | 手写 eval vs lm-eval 对比 |

---

## 阶段 10：推理与部署 🔜

> **目标:** 实现 pipeline 推理、Gradio Demo、GGUF 导出
> **from-scratch 对应:** `deploy/` + `src/inference/`

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 10.1 | pipeline("text-generation") 推理封装 | 🔥 P0 | 🔜 Pending | - | 一行代码推理 |
| 10.2 | Gradio Web Demo | ⚡ P1 | 🔜 Pending | - | from-scratch: web_demo.py |
| 10.3 | CLI 交互式对话 (scripts/chat.py) | ⚡ P1 | 🔜 Pending | - | from-scratch: chat.py |
| 10.4 | GGUF 格式导出 | 🟡 P2 | 🔜 Pending | - | llama.cpp 兼容 |

---

## 阶段 11：工程质量与文档 🔜

> **目标:** 完善测试、文档、类型提示

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 11.1 | 补齐所有模块单元测试 | 🔥 P0 | 🔜 Pending | - | 目标：覆盖所有核心模块 |
| 11.2 | 添加类型提示 (type hints) | ⚡ P1 | 🔜 Pending | - | 所有公开 API |
| 11.3 | 整理 Notebook 系列 (8 个对比 notebook) | ⚡ P1 | 🔜 Pending | - | 完整的学习路线 |
| 11.4 | 编写 AutoDL 训练指南 | 🟡 P2 | 🔜 Pending | - | docs/AUTODL_GUIDE.md |
| 11.5 | 编写部署文档 | 🟡 P2 | 🔜 Pending | - | docs/DEPLOY.md |
| 11.6 | 冒烟测试脚本 (smoke_test.py) | 🔥 P0 | 🔜 Pending | - | 全流程端到端验证 |

---

## 阶段 12：HF Hub 集成 🔜

> **目标:** 推送模型和数据集到 HuggingFace Hub

| # | 任务 | 优先级 | 状态 | 日期 | 备注 |
|---|------|--------|------|------|------|
| 12.1 | 创建 Model Card (README.md for Hub) | ⚡ P1 | 🔜 Pending | - | 模型信息、训练细节、使用方式 |
| 12.2 | push_to_hub (模型 + Tokenizer) | ⚡ P1 | 🔜 Pending | - | model.push_to_hub() |
| 12.3 | 创建 Dataset Card | 🟡 P2 | 🔜 Pending | - | 数据集描述 |
| 12.4 | push_to_hub (数据集) | 🟡 P2 | 🔜 Pending | - | dataset.push_to_hub() |

---

## 📈 进度统计

| 指标 | 数量 |
|------|------|
| 总任务数 | 58 |
| ✅ 已完成 | 15 |
| 🔜 待开始 | 43 |
| 完成率 | 26% |

**按优先级分布：**

| 优先级 | 数量 | 说明 |
|--------|------|------|
| 🔥 P0 | 28 | 核心功能，必须完成 |
| ⚡ P1 | 22 | 重要功能，应该完成 |
| 🟡 P2 | 8 | 增强功能，可延后 |

---

## 📝 模型家族

| 代号 | 配置文件 | 预估参数量 | 推荐设备 |
|------|---------|-----------|---------|
| ClearMind-Tiny | tiny.yaml | ~0.6M | CPU |
| ClearMind-Mini | small.yaml | ~26M | CPU / GPU |
| ClearMind | medium.yaml | ~200M | GPU (8GB+) |
| ClearMind-Plus | large.yaml | ~930M | GPU (24GB+) |

---

## 📋 每日更新日志

### 2026-03-21

- 🎉 项目启动
- 📄 创建项目规划文档 (README.md, CLAUDE.md, PRD.md, TECHNICAL_DESIGN.md, PROGRESS_TRACKER.md)
- 📋 定义 12 个开发阶段，58 个子任务
- ✅ 阶段 1 完成：项目搭建与基础设施 (目录结构、配置文件、run.sh、.gitignore、requirements.txt)
- ✅ 阶段 2 完成：Tokenizer 训练 (ByteLevelBPE, PreTrainedTokenizerFast, chat template, 20 个测试通过)
- ✅ scripts/prepare_data.py 数据准备脚本 (small/medium/large 三档)
