# MiniMind - 开发计划进度表

> 开始日期: 2026-02-18  
> 预计完成: 2026-03-18  
> 负责人: Perlou

---

## 📊 总体进度

```
整体进度: ░░░░░░░░░░░░░░░░░░░░ 0%

Phase 1: ░░░░░░░░░░   0%  项目搭建与模型架构
Phase 2: ░░░░░░░░░░   0%  数据处理模块
Phase 3: ░░░░░░░░░░   0%  预训练
Phase 4: ░░░░░░░░░░   0%  SFT 指令微调
Phase 5: ░░░░░░░░░░   0%  DPO 对齐训练
Phase 6: ░░░░░░░░░░   0%  推理、评估与文档
```

---

## 📅 Phase 1: 项目搭建与模型架构 (Week 1)

**日期**: 2026-02-18 ~ 2026-02-24  
**状态**: ⬜ 待开始  
**目标**: 从零实现完整的 Decoder-only Transformer 架构

| #    | 任务                                     | 预计耗时 | 状态      | 完成日期 | 备注                           |
| ---- | ---------------------------------------- | -------- | --------- | -------- | ------------------------------ |
| 1.1  | 创建项目目录结构                         | 0.5h     | ⬜ 待开始 |          | 按 TECHNICAL_DESIGN 目录结构   |
| 1.2  | 创建 requirements.txt                    | 0.5h     | ⬜ 待开始 |          | torch, sentencepiece, datasets |
| 1.3  | 创建配置文件 (small.yaml, medium.yaml)   | 1h       | ⬜ 待开始 |          | ModelConfig dataclass          |
| 1.4  | 实现 ModelConfig (config.py)             | 1h       | ⬜ 待开始 |          | dataclass + from_yaml          |
| 1.5  | 实现 RoPE 旋转位置编码 (rope.py)         | 2h       | ⬜ 待开始 |          | 预计算频率 + 旋转应用          |
| 1.6  | 实现 RMSNorm (normalization.py)          | 1h       | ⬜ 待开始 |          | 对比 LayerNorm 验证            |
| 1.7  | 实现 SwiGLU 激活函数 (activation.py)     | 1h       | ⬜ 待开始 |          | gate + up + down 三矩阵        |
| 1.8  | 实现 Multi-Head Attention (attention.py) | 3h       | ⬜ 待开始 |          | MHA + GQA + Causal Mask + RoPE |
| 1.9  | 实现 FeedForward (feedforward.py)        | 1h       | ⬜ 待开始 |          | 基于 SwiGLU 的 FFN             |
| 1.10 | 实现 TransformerBlock (transformer.py)   | 1.5h     | ⬜ 待开始 |          | Attention + FFN + Residual     |
| 1.11 | 实现 GPT 完整模型 (gpt.py)               | 2h       | ⬜ 待开始 |          | Embedding + N×Block + LMHead   |
| 1.12 | 模型验证 — 前向传播测试                  | 1h       | ⬜ 待开始 |          | 参数量统计 + shape 验证        |

**本阶段交付物**:

- [ ] configs/small.yaml, configs/medium.yaml
- [ ] src/model/ 完整实现 (8 个文件)
- [ ] 模型前向传播通过验证

---

## 📅 Phase 2: 数据处理模块 (Week 2)

**日期**: 2026-02-25 ~ 2026-03-03  
**状态**: ⬜ 待开始  
**目标**: 实现 Tokenizer 训练和三种数据集的加载

| #   | 任务                                        | 预计耗时 | 状态      | 完成日期 | 备注                         |
| --- | ------------------------------------------- | -------- | --------- | -------- | ---------------------------- |
| 2.1 | 实现数据下载脚本 (01_prepare_data.py)       | 2h       | ⬜ 待开始 |          | HuggingFace datasets         |
| 2.2 | 实现 Tokenizer 训练 (02_train_tokenizer.py) | 2h       | ⬜ 待开始 |          | sentencepiece BPE            |
| 2.3 | 实现 Tokenizer 封装类 (tokenizer.py)        | 1.5h     | ⬜ 待开始 |          | encode/decode/special tokens |
| 2.4 | 实现预训练数据集 (pretrain_dataset.py)      | 2h       | ⬜ 待开始 |          | 文本拼接→固定长度切分        |
| 2.5 | 实现 SFT 数据集 (sft_dataset.py)            | 2h       | ⬜ 待开始 |          | 对话模板 + loss mask         |
| 2.6 | 实现 DPO 数据集 (dpo_dataset.py)            | 2h       | ⬜ 待开始 |          | chosen/rejected 对           |
| 2.7 | 数据流程验证                                | 1.5h     | ⬜ 待开始 |          | DataLoader 输出 shape 检查   |

**本阶段交付物**:

- [ ] scripts/01_prepare_data.py
- [ ] scripts/02_train_tokenizer.py
- [ ] src/data/ 完整实现 (4 个文件)
- [ ] 分词器能正确分词中英文

---

## 📅 Phase 3: 预训练 (Week 3)

**日期**: 2026-03-03 ~ 2026-03-09  
**状态**: ⬜ 待开始  
**目标**: 实现预训练流程，完成 Small 模型预训练

| #   | 任务                                | 预计耗时 | 状态      | 完成日期 | 备注                           |
| --- | ----------------------------------- | -------- | --------- | -------- | ------------------------------ |
| 3.1 | 实现训练工具函数 (trainer_utils.py) | 2h       | ⬜ 待开始 |          | LR scheduler, grad clipping    |
| 3.2 | 实现预训练 Trainer (pretrain.py)    | 3h       | ⬜ 待开始 |          | AdamW + cosine LR + grad accum |
| 3.3 | 实现 Checkpoint 保存/恢复           | 1.5h     | ⬜ 待开始 |          | model + optimizer + step       |
| 3.4 | 实现训练日志记录                    | 1h       | ⬜ 待开始 |          | loss, lr, speed, ETA           |
| 3.5 | 预训练入口脚本 (03_pretrain.py)     | 1h       | ⬜ 待开始 |          | CLI 参数解析                   |
| 3.6 | 快速验证训练 (max_steps=100)        | 1h       | ⬜ 待开始 |          | 确认 loss 下降                 |
| 3.7 | 完整预训练 (Small, ~10K steps)      | 观察     | ⬜ 待开始 |          | MacBook 预计 3-6 小时          |

**本阶段交付物**:

- [ ] src/training/pretrain.py
- [ ] src/training/trainer_utils.py
- [ ] scripts/03_pretrain.py
- [ ] outputs/pretrain/final.pth (预训练模型)

---

## 📅 Phase 4: SFT 指令微调 (Week 3-4)

**日期**: 2026-03-09 ~ 2026-03-12  
**状态**: ⬜ 待开始  
**目标**: 在预训练模型上进行指令微调

| #   | 任务                      | 预计耗时 | 状态      | 完成日期 | 备注                           |
| --- | ------------------------- | -------- | --------- | -------- | ------------------------------ |
| 4.1 | 实现 SFT Trainer (sft.py) | 2.5h     | ⬜ 待开始 |          | 加载 pretrain ckpt + loss mask |
| 4.2 | SFT 入口脚本 (04_sft.py)  | 1h       | ⬜ 待开始 |          | CLI 参数                       |
| 4.3 | 快速验证 SFT (100 steps)  | 0.5h     | ⬜ 待开始 |          | 确认 loss 下降                 |
| 4.4 | 完整 SFT 训练 (3 epochs)  | 观察     | ⬜ 待开始 |          | MacBook 预计 1-2 小时          |

**本阶段交付物**:

- [ ] src/training/sft.py
- [ ] scripts/04_sft.py
- [ ] outputs/sft/final.pth (SFT 模型)

---

## 📅 Phase 5: DPO 对齐训练 (Week 4)

**日期**: 2026-03-12 ~ 2026-03-14  
**状态**: ⬜ 待开始  
**目标**: 实现 DPO 算法并完成对齐训练

| #   | 任务                      | 预计耗时 | 状态      | 完成日期 | 备注                    |
| --- | ------------------------- | -------- | --------- | -------- | ----------------------- |
| 5.1 | 实现 DPO Trainer (dpo.py) | 3h       | ⬜ 待开始 |          | DPO loss + ref model    |
| 5.2 | DPO 入口脚本 (05_dpo.py)  | 1h       | ⬜ 待开始 |          | CLI 参数                |
| 5.3 | 快速验证 DPO (100 steps)  | 0.5h     | ⬜ 待开始 |          | 确认 loss 变化合理      |
| 5.4 | 完整 DPO 训练 (1 epoch)   | 观察     | ⬜ 待开始 |          | MacBook 预计 30-60 分钟 |

**本阶段交付物**:

- [ ] src/training/dpo.py
- [ ] scripts/05_dpo.py
- [ ] outputs/dpo/final.pth (DPO 模型)

---

## 📅 Phase 6: 推理、评估与文档 (Week 4)

**日期**: 2026-03-14 ~ 2026-03-18  
**状态**: ⬜ 待开始  
**目标**: 实现推理对话、评估脚本，完善全部文档

| #   | 任务                                | 预计耗时 | 状态      | 完成日期 | 备注                      |
| --- | ----------------------------------- | -------- | --------- | -------- | ------------------------- |
| 6.1 | 实现文本生成引擎 (generate.py)      | 2h       | ⬜ 待开始 |          | top-k, top-p, temperature |
| 6.2 | 实现交互式对话 (chat.py)            | 2h       | ⬜ 待开始 |          | 终端 CLI 对话界面         |
| 6.3 | 对话入口脚本 (06_chat.py)           | 1h       | ⬜ 待开始 |          | 多轮对话                  |
| 6.4 | 实现困惑度评估 (eval_perplexity.py) | 1.5h     | ⬜ 待开始 |          | 评估各阶段模型            |
| 6.5 | 对比测试 (pretrain vs sft vs dpo)   | 1h       | ⬜ 待开始 |          | 生成质量对比              |
| 6.6 | 编写 README.md                      | 2h       | ⬜ 待开始 |          | 项目总览                  |
| 6.7 | 编写 ARCHITECTURE.md                | 3h       | ⬜ 待开始 |          | 架构深度解析              |
| 6.8 | 编写 TRAINING_GUIDE.md              | 2h       | ⬜ 待开始 |          | 完整训练指南              |
| 6.9 | 更新课程项目列表                    | 0.5h     | ⬜ 待开始 |          | projects/README.md        |

**本阶段交付物**:

- [ ] src/inference/ 完整实现
- [ ] scripts/06_chat.py
- [ ] evaluate/eval_perplexity.py
- [ ] docs/ 完整文档
- [ ] README.md

---

## 📈 进度统计

### 任务统计

| 阶段     | 总任务数 | 已完成 | 进行中 | 完成率 |
| -------- | -------- | ------ | ------ | ------ |
| Phase 1  | 12       | 0      | 0      | 0%     |
| Phase 2  | 7        | 0      | 0      | 0%     |
| Phase 3  | 7        | 0      | 0      | 0%     |
| Phase 4  | 4        | 0      | 0      | 0%     |
| Phase 5  | 4        | 0      | 0      | 0%     |
| Phase 6  | 9        | 0      | 0      | 0%     |
| **总计** | **43**   | **0**  | **0**  | **0%** |

### 时间统计

| 指标       | 值         |
| ---------- | ---------- |
| 预计总工时 | ~65 小时   |
| 已投入工时 | 0 小时     |
| 项目状态   | ⬜ 待开始  |
| 预计完成   | 2026-03-18 |

---

## 📝 状态图例

| 图标 | 状态   | 说明         |
| ---- | ------ | ------------ |
| ⬜   | 待开始 | 尚未开始     |
| 🔄   | 进行中 | 正在开发     |
| ✅   | 已完成 | 开发完成     |
| ⚠️   | 阻塞   | 遇到阻塞问题 |
| ⏭️   | 跳过   | 任务跳过     |

---

## 📋 每日更新日志

### 2026-02-18

- 📝 创建项目需求文档 (PRD.md)
- 📝 创建技术设计文档 (TECHNICAL_DESIGN.md)
- 📝 创建开发进度表 (PROGRESS_TRACKER.md)

---

## 🚨 风险追踪

| 风险           | 状态      | 描述                           | 缓解措施                   |
| -------------- | --------- | ------------------------------ | -------------------------- |
| MacBook 训练慢 | ⬜ 待观察 | Small 模型预训练可能需 6+ 小时 | 减少 max_steps、使用 MPS   |
| MPS 兼容性     | ⬜ 待观察 | 某些 op 在 MPS 上可能不支持    | CPU fallback               |
| 数据下载       | ⬜ 待观察 | HuggingFace 国内可能较慢       | 使用 mirror 或准备小数据集 |
| 内存限制       | ⬜ 待观察 | 8GB Mac 训练时内存紧张         | 减小 batch_size + 梯度累积 |

---

## 💡 备注

- 每完成一个任务，将状态更新为 ✅ 并填写完成日期
- 遇到问题时，在备注栏记录并更新风险追踪
- 每周结束时更新进度统计
- 训练阶段标记为「观察」的任务耗时取决于硬件，不计入工时
