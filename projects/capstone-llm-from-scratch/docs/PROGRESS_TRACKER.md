# ClearMind - 开发计划进度表

> 开始日期: 2026-02-18  
> 最近更新: 2026-02-19  
> 负责人: Perlou

---

## 📊 总体进度

```
整体进度: ███████████████░░░░░ 75%

Phase 1:  ██████████ 100%  项目搭建与模型架构
Phase 2:  ██████████ 100%  数据处理模块
Phase 3:  ██████████ 100%  预训练
Phase 4:  ██████████ 100%  SFT 指令微调
Phase 5:  ██████████ 100%  DPO 对齐训练
Phase 6:  ██████████ 100%  推理与对话
Phase 7:  ██████████ 100%  评估体系
Phase 8:  ██████████ 100%  部署上线
Phase 9:  █████░░░░░  50%  推理性能优化
Phase 10: ░░░░░░░░░░   0%  训练改进
Phase 11: ░░░░░░░░░░   0%  工程质量
```

---

## 📅 Phase 1: 项目搭建与模型架构 ✅

**状态**: ✅ 已完成  
**目标**: 从零实现完整的 Decoder-only Transformer 架构

| #    | 任务                                     | 状态 | 完成日期   | 备注                           |
| ---- | ---------------------------------------- | ---- | ---------- | ------------------------------ |
| 1.1  | 创建项目目录结构                         | ✅   | 2026-02-18 | configs/, src/, scripts/, docs |
| 1.2  | 创建 requirements.txt                    | ✅   | 2026-02-18 | torch, sentencepiece, datasets |
| 1.3  | 创建配置文件 (small/medium/large.yaml)   | ✅   | 2026-02-18 | 三档配置: Mini/标准/Plus       |
| 1.4  | 实现 ModelConfig (config.py)             | ✅   | 2026-02-18 | dataclass + from_yaml          |
| 1.5  | 实现 RoPE 旋转位置编码 (rope.py)         | ✅   | 2026-02-18 | 预计算频率 + 旋转应用          |
| 1.6  | 实现 RMSNorm (normalization.py)          | ✅   | 2026-02-18 | 对比 LayerNorm 验证            |
| 1.7  | 实现 SwiGLU 激活函数 (activation.py)     | ✅   | 2026-02-18 | 纯激活函数                     |
| 1.8  | 实现 Multi-Head Attention (attention.py) | ✅   | 2026-02-18 | MHA + GQA + Causal Mask + RoPE |
| 1.9  | 实现 FeedForward (feedforward.py)        | ✅   | 2026-02-18 | SwiGLU FFN (3 矩阵)            |
| 1.10 | 实现 TransformerBlock (transformer.py)   | ✅   | 2026-02-18 | Pre-Norm + Residual            |
| 1.11 | 实现 GPT 完整模型 (gpt.py)               | ✅   | 2026-02-18 | Embedding + N×Block + LMHead   |
| 1.12 | 模型验证脚本 (verify_model.py)           | ✅   | 2026-02-18 | 参数量统计 + shape 验证        |

---

## 📅 Phase 2: 数据处理模块 ✅

**状态**: ✅ 已完成  
**目标**: 实现 Tokenizer 训练和三种数据集的加载

| #   | 任务                                     | 状态 | 完成日期   | 备注                         |
| --- | ---------------------------------------- | ---- | ---------- | ---------------------------- |
| 2.1 | 实现数据准备脚本 (prepare_data.py)       | ✅   | 2026-02-18 | 样例数据 + HuggingFace 下载  |
| 2.2 | 实现 Tokenizer 训练 (train_tokenizer.py) | ✅   | 2026-02-18 | sentencepiece BPE            |
| 2.3 | 实现 Tokenizer 封装类 (tokenizer.py)     | ✅   | 2026-02-18 | encode/decode/special tokens |
| 2.4 | 实现预训练数据集 (pretrain_dataset.py)   | ✅   | 2026-02-18 | 文本拼接→固定长度切分        |
| 2.5 | 实现 SFT 数据集 (sft_dataset.py)         | ✅   | 2026-02-18 | 对话模板 + loss mask         |
| 2.6 | 实现 DPO 数据集 (dpo_dataset.py)         | ✅   | 2026-02-18 | chosen/rejected 对           |
| 2.7 | 真实数据集下载 (download_dataset.py)     | ✅   | 2026-02-18 | HuggingFace + 三档规模       |

---

## 📅 Phase 3: 预训练 ✅

**状态**: ✅ 已完成  
**目标**: 实现预训练流程

| #   | 任务                                | 状态 | 完成日期   | 备注                           |
| --- | ----------------------------------- | ---- | ---------- | ------------------------------ |
| 3.1 | 实现训练工具函数 (trainer_utils.py) | ✅   | 2026-02-18 | LR scheduler, grad clipping    |
| 3.2 | 实现预训练 Trainer (pretrain.py)    | ✅   | 2026-02-18 | AdamW + cosine LR + grad accum |
| 3.3 | 实现 Checkpoint 保存/恢复           | ✅   | 2026-02-18 | model + optimizer + step       |
| 3.4 | 实现训练日志记录                    | ✅   | 2026-02-18 | loss, lr, speed, ETA           |
| 3.5 | 统一训练入口 (train.py --stage)     | ✅   | 2026-02-18 | pretrain/sft/dpo 统一入口      |

---

## 📅 Phase 4: SFT 指令微调 ✅

**状态**: ✅ 已完成  
**目标**: 在预训练模型上进行指令微调

| #   | 任务                            | 状态 | 完成日期   | 备注                           |
| --- | ------------------------------- | ---- | ---------- | ------------------------------ |
| 4.1 | 实现 SFT Trainer (sft.py)       | ✅   | 2026-02-18 | 加载 pretrain ckpt + loss mask |
| 4.2 | SFT 训练 (train.py --stage sft) | ✅   | 2026-02-18 | 统一训练入口                   |

---

## 📅 Phase 5: DPO 对齐训练 ✅

**状态**: ✅ 已完成  
**目标**: 实现 DPO 算法并完成对齐训练

| #   | 任务                            | 状态 | 完成日期   | 备注                 |
| --- | ------------------------------- | ---- | ---------- | -------------------- |
| 5.1 | 实现 DPO Trainer (dpo.py)       | ✅   | 2026-02-18 | DPO loss + ref model |
| 5.2 | DPO 训练 (train.py --stage dpo) | ✅   | 2026-02-18 | 统一训练入口         |

---

## 📅 Phase 6: 推理与对话 ✅

**状态**: ✅ 已完成  
**目标**: 实现推理引擎和交互式对话

| #   | 任务                              | 状态 | 完成日期   | 备注                      |
| --- | --------------------------------- | ---- | ---------- | ------------------------- |
| 6.1 | 实现文本生成引擎 (generate.py)    | ✅   | 2026-02-18 | top-k, top-p, temperature |
| 6.2 | 实现交互式对话 (chat.py)          | ✅   | 2026-02-18 | 终端 CLI 对话界面         |
| 6.3 | 对话入口脚本 (scripts/chat.py)    | ✅   | 2026-02-18 | 自动查找最佳模型          |
| 6.4 | AutoDL 一键训练 (autodl_train.sh) | ✅   | 2026-02-18 | A100 部署脚本             |
| 6.5 | 编写 README.md                    | ✅   | 2026-02-18 | 项目总览 + 快速开始       |
| 6.6 | 更新 PROGRESS_TRACKER.md          | ✅   | 2026-02-18 | 全部进度更新              |

---

## 📅 Phase 7: 评估体系 ✅

**状态**: ✅ 已完成  
**目标**: 构建 PPL / 生成质量 / 指令跟随 全面评估体系

| #   | 任务                                   | 状态 | 完成日期   | 备注                           |
| --- | -------------------------------------- | ---- | ---------- | ------------------------------ |
| 7.1 | 增强困惑度评估 (eval_perplexity.py)    | ✅   | 2026-02-19 | 新增 --compare 阶段对比        |
| 7.2 | 实现生成质量评估 (eval_generation.py)  | ✅   | 2026-02-19 | Distinct-N / 重复率 / 平均长度 |
| 7.3 | 实现指令跟随评估 (eval_instruction.py) | ✅   | 2026-02-19 | 格式正确率 / 相关性 / 安全拒绝 |
| 7.4 | 实现综合评估报告 (eval_benchmark.py)   | ✅   | 2026-02-19 | 一键全面评测 + Markdown 报告   |

---

## 📅 Phase 8: 部署上线 ✅

**状态**: ✅ 已完成  
**目标**: REST API / Web UI / Docker 全链路部署

| #   | 任务                               | 状态 | 完成日期   | 备注                           |
| --- | ---------------------------------- | ---- | ---------- | ------------------------------ |
| 8.1 | 实现 REST API (api_server.py)      | ✅   | 2026-02-19 | FastAPI, 兼容 OpenAI 格式, SSE |
| 8.2 | 实现 Web 演示 (web_demo.py)        | ✅   | 2026-02-19 | Gradio 对话界面 + 参数面板     |
| 8.3 | 实现模型导出 (export_model.py)     | ✅   | 2026-02-19 | 权重瘦身 / TorchScript / INT8  |
| 8.4 | 容器化部署 (Dockerfile)            | ✅   | 2026-02-19 | Docker 一键构建 + GPU 支持     |
| 8.5 | 部署依赖 (requirements-deploy.txt) | ✅   | 2026-02-19 | fastapi, gradio, uvicorn       |

---

## 📅 Phase 9: 推理性能优化 🔜

**状态**: 🔄 进行中  
**目标**: 大幅提升推理速度和显存效率  
**优先级**: 🔥 高

| #   | 任务                          | 状态 | 优先级 | 备注                                      |
| --- | ----------------------------- | ---- | ------ | ----------------------------------------- |
| 9.1 | 实现 KV Cache                 | ✅   | 🔥 P0  | Prefill + Decode 模式，推理速度提升 5-10x |
| 9.2 | 集成 Flash Attention          | ✅   | 🔥 P0  | F.scaled_dot_product_attention()          |
| 9.3 | 实现 Sliding Window Attention | ⬜   | 🟡 P2  | 降低长序列显存，大模型配置可选            |
| 9.4 | GGUF 格式导出                 | ⬜   | 🟡 P2  | 支持 llama.cpp 纯 CPU 高效推理            |

---

## 📅 Phase 10: 训练改进 ✅

**状态**: ✅ 已完成  
**目标**: 提升训练质量和灵活性  
**优先级**: ⚡ 中高

| #    | 任务                        | 状态 | 优先级 | 备注                           |
| ---- | --------------------------- | ---- | ------ | ------------------------------ |
| 10.1 | 增加验证集 + Early Stopping | ✅   | 🔥 P0  | pretrain/sft/dpo 全部集成      |
| 10.2 | 混合精度 GradScaler         | ✅   | ⚡ P1  | CUDA FP16 下防止梯度下溢       |
| 10.3 | 多卡数据并行 (DDP)          | ✅   | ⚡ P1  | DDP 工具函数 + launch 脚本     |
| 10.4 | 实现 LoRA 微调              | ✅   | 🟡 P2  | lora.py: apply/merge/save/load |
| 10.5 | 集成 TensorBoard            | ✅   | 🟡 P2  | TrainingLogger 可选 TB 写入    |

---

## 📅 Phase 11: 工程质量 🔜

**状态**: ⏳ 计划中  
**目标**: 提升代码质量和可维护性  
**优先级**: 🛠️ 中

| #    | 任务                  | 状态 | 优先级 | 备注                              |
| ---- | --------------------- | ---- | ------ | --------------------------------- |
| 11.1 | 提取 Trainer 基类     | ⬜   | ⚡ P1  | 消除 Pre/SFT/DPO Trainer 重复代码 |
| 11.2 | 增加单元测试 (pytest) | ⬜   | ⚡ P1  | tests/ 目录，覆盖核心模块         |
| 11.3 | 完善类型提示          | ⬜   | 🟡 P2  | 补全返回值类型注解                |
| 11.4 | Tokenizer 鲁棒性      | ⬜   | 🟡 P2  | byte-fallback + OOV 覆盖率检测    |
| 11.5 | 配置校验增强          | ⬜   | 🟢 P3  | d_ff ≈ 8/3 × d_model 等规则校验   |

---

## 📈 进度统计

| 阶段     | 总任务数 | 已完成 | 完成率  |
| -------- | -------- | ------ | ------- |
| Phase 1  | 12       | 12     | 100%    |
| Phase 2  | 7        | 7      | 100%    |
| Phase 3  | 5        | 5      | 100%    |
| Phase 4  | 2        | 2      | 100%    |
| Phase 5  | 2        | 2      | 100%    |
| Phase 6  | 6        | 6      | 100%    |
| Phase 7  | 4        | 4      | 100%    |
| Phase 8  | 5        | 5      | 100%    |
| Phase 9  | 4        | 0      | 0%      |
| Phase 10 | 5        | 5      | 100%    |
| Phase 11 | 5        | 0      | 0%      |
| **总计** | **57**   | **48** | **84%** |

---

## 📝 模型家族

| 代号           | 配置文件            | 参数量 | 适合设备  |
| -------------- | ------------------- | ------ | --------- |
| ClearMind-Mini | configs/small.yaml  | ~26M   | MacBook   |
| ClearMind      | configs/medium.yaml | ~200M  | GPU 24GB+ |
| ClearMind-Plus | configs/large.yaml  | ~468M  | A100 80GB |

---

## 📋 每日更新日志

### 2026-02-19

- 📊 新增评估体系: eval_generation / eval_instruction / eval_benchmark
- 🔄 增强 eval_perplexity: --compare 阶段对比功能
- 🚀 新增部署模块: FastAPI API / Gradio Web / 模型导出 / Docker
- 📝 更新 README: 评估章节 + 部署章节 + 项目结构
- 📋 更新进度表: 新增 Phase 9-11 优化计划

### 2026-02-18

- 📝 创建项目文档 (PRD, TECHNICAL_DESIGN, PROGRESS_TRACKER)
- 🏗️ 搭建项目结构, 实现 requirements.txt 和三档配置
- 🧠 实现完整模型架构 (RoPE, RMSNorm, SwiGLU, GQA, TransformerBlock, GPT)
- 📦 实现数据处理模块 (Tokenizer, PretrainDataset, SFTDataset, DPODataset)
- 🏋️ 实现三阶段训练 (PreTrainer, SFTTrainer, DPOTrainer)
- 💬 实现推理和对话模块 (generate, chat)
- 📊 实现困惑度评估
- 🚀 新增 A100 适配: large.yaml + 真实数据下载 + AutoDL 部署脚本
- 🏷️ 模型命名: ClearMind 系列 (Mini / 标准 / Plus)
