# ClearMind — 产品需求文档（PRD）

> 版本：v2.0（2026-05-01 重写，配合方向调整：教学项目 → 生产模型发布）
> 作者：Perlou

---

## 1. 项目定位

### 1.1 一句话

**从零实现的中文小语言模型，复用 [minimind](https://github.com/jingyaogong/minimind) 的数据/tokenizer 生态，凭借更扎实的工程基础与若干标准化的架构改进，目标是在同等参数规模上效果反超 minimind，并发布到 HuggingFace 与 ModelScope。**

### 1.2 双产品矩阵

| 产品 | 参数 | 对标 | 定位 |
|---|---|---|---|
| **ClearMind-Base** | 68.8M dense | minimind-3 (64M dense) | 端侧/低算力对话基线，证明同等规模也能超越 |
| **ClearMind-Plus** | 486.3M dense | minimind-3-moe (198M-A64M)，单 token 算力 7.1× | 质量旗舰，dense 路线压制 sparse |

### 1.3 与 minimind 的关系

**复用** minimind 的：tokenizer (vocab=6400 含 `<tool_call>`/`<think>` 与 16 个 buffer)、chat_template Jinja 模板、数据集（pretrain_t2t / sft_t2t / dpo / rlaif / agent_rl）。
**重写** ClearMind 自己的：模型架构（基本兼容 Qwen3 命名映射）、训练 trainer、dataset 适配层、推理引擎、发布工具链。
**超越** minimind 的具体技术点（见 §3）。

## 2. 目标用户与场景

| 用户群 | 场景 | 对应产品 |
|---|---|---|
| 小团队 / 个人开发者 | 端侧问答、Demo、教学示例 | ClearMind-Base |
| 中小企业 / 应用开发 | 私有化部署、轻量对话服务 | ClearMind-Plus |
| 研究 / 教学 | 复现现代 LLM 训练流水线 | 全链路代码 + docs |
| 社区 | HuggingFace / ModelScope 用户使用 | 发布的 HF/MS 仓库 |

## 3. 超越 minimind 的具体技术路径

| 维度 | ClearMind 做法 | minimind 现状 | 预期收益 |
|---|---|---|---|
| Trainer 抽象 | `BaseTrainer` 共享逻辑 | 9 个独立 ~200 行脚本 | 维护性 + 减少 bug 表面 |
| 优化器分组 | decay / no_decay 分组 | 单组 AdamW | 训练稳定性提升 |
| LR scheduler | 标准 cosine + warmup（state 可保存） | 非标准压缩公式 [0.1lr, 0.55lr] | 与 GPT-2/Llama 默认对齐 |
| Train/Val split + EarlyStopping | ✅ 实现 | ❌ | 防过拟合 + 节省算力 |
| 残差初始化 1/√(2L) 缩放 | 计划落地（Phase 1） | ❌ | 深层模型训练稳定性 |
| QK-Norm | 计划落地（Phase 2） | ❌ | Llama-3 / Gemma2 同款，长训 loss 更稳 |
| RoPE θ=1e6 + YaRN scaling | 计划落地（Phase 2） | ✅（已支持） | 长上下文外推 |
| LoRA + torch.compile 兼容 | 显式 LoRALinear 类 | monkey-patch（disable compile） | 微调时 +20-40% 速度 |
| AMP API | torch.amp 新 API | 旧 torch.cuda.amp | 跨版本兼容 |
| 跨平台设备 | CUDA / MPS / CPU 自动选择 | 仅 CUDA | MacBook 也能开发 |
| chat_template + token 序列扫描 mask | ✅（参考 minimind） | ✅ | 修 BPE 边界 mask 错位 |
| 修复 attention_mask 0×inf=NaN bug | ✅ 已修（冒烟时发现） | N/A | 阻塞性 bug |
| 发布生态打通 | 计划：safetensors + Qwen3 兼容导出 + OpenAI API server | ✅ 已有 | 与 minimind 持平 |

## 4. 关键功能需求

### 4.1 训练全栈（继承 minimind）

- **Pretrain**：next-token prediction，per_sample / packed 双模式
- **SFT**：chat_template 多角色（system/user/assistant/tool）+ 工具调用 + 自适应思考
- **DPO**：偏好对齐（chosen/rejected 单 forward 优化）
- **LoRA**：低秩高效微调（与 torch.compile 共存）
- **白盒蒸馏（Phase 3）**：KL + α·CE
- **PPO / GRPO / CISPO（Phase 3）**：含可插拔 Rollout 引擎（torch / sglang）
- **Agentic Tool-Use RL（Phase 3）**：多轮 tool-call rollout + mock 工具沙箱

### 4.2 数据 / Tokenizer

- 完全复用 minimind 数据集（modelscope/hf 双源）
- 扁平结构（与 minimind 发布形式一致）
- HF AutoTokenizer 加载（`tokenizer/minimind/`，仓库自带）
- chat_template Jinja 模板原生支持 system/user/assistant/tool 多角色 + tool_calls + reasoning_content + open_thinking

### 4.3 发布闭环（关键差异化）

- **safetensors** 落盘
- **Qwen3 兼容导出脚本**（`scripts/convert_to_qwen3.py`）：torch.pth → Qwen3ForCausalLM HF 权重
- 训练好的模型可被 `ollama` / `vllm` / `llama.cpp` / `Llama-Factory` 直接消费
- **HuggingFace Hub push**（`scripts/push_to_hub.py`）
- **ModelScope upload**（`scripts/push_to_modelscope.py`）
- **OpenAI 兼容 API server**（`deploy/openai_api.py`，FastAPI + SSE + tools + thinking）
- **模型卡**（README + LICENSE + tags + intended_use + 评测结果）

### 4.4 评测

- C-Eval / C-MMLU / OpenBookQA / GSM8K（subset）
- 与 minimind-3 / minimind-3-moe 同条件对比表
- tokens/sec 推理速度对比
- 困惑度（PPL）对比

## 5. 成功指标

| 指标 | 目标 |
|---|---|
| ClearMind-Base PPL（与 minimind-3 同测试集） | ≤ minimind-3 PPL |
| ClearMind-Base C-Eval / C-MMLU 平均 | ≥ minimind-3 |
| ClearMind-Plus PPL | ≤ minimind-3-moe PPL |
| ClearMind-Plus C-Eval / C-MMLU 平均 | 显著 > minimind-3-moe |
| 单卡 A100 bf16 推理 tokens/sec（Plus） | ≥ 80 tok/s（batch=1）|
| HF Hub 仓库可被 transformers AutoModelForCausalLM 直接加载 | ✅ |
| ModelScope 仓库可被国内用户访问 | ✅ |

## 6. 非目标（Not Goals）

- ❌ 不做多模态（视觉/音频）
- ❌ 不追求 GPT-4 级别能力（参数量决定上限）
- ❌ 不做 web 版聊天产品（只做 OpenAI 兼容 API + 简易 web demo）
- ❌ 不维护 sentencepiece tokenizer 路径作为主线（保留作 legacy 不删）

## 7. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| 数据规模不足以训出 Plus（486M 需要 ~10B tokens） | 中 | 用 minimind 完整版数据 + 多 epoch；评测仍优于 minimind-3-moe 即达标 |
| MoE 实现工作量大（plan #12） | 中 | 优先 dense 路径，MoE 放 Phase 3 后期 |
| 单卡 A100 80GB 不够 Plus 训练 | 低 | bf16 + grad checkpoint，最差降到 batch=8/seq=1024 |
| HF/ModelScope 发布认证流程 | 低 | 提前注册账号，用 token 自动化 |
| 评测结果不如 minimind | 中 | Phase 1+2 的标准化改进（残差初始化、QK-Norm）单独验证收益；保留 dense vs sparse 计算量优势叙事 |

## 8. 时间表

| 阶段 | 内容 | 状态 | 算力 |
|---|---|---|---|
| 重构 1 | minimind 数据/tokenizer 适配层 + 配置矩阵 + 冒烟通过 + NaN bug 修复 | ✅ 完成 | 本地 |
| 重构 2 | scripts/evaluate/deploy 重写 | ✅ 完成 | 本地 |
| Phase 1 | LR bug 修 + RoPE buffer 共享 + 残差初始化 + DPO 单 forward + 原子 checkpoint + NaN 三层防御 | ✅ 完成 | 本地 |
| Phase 2 | QK-Norm + RoPE θ=1e6 + YaRN | ✅ 完成 | 本地 |
| Phase 3 (核心) | DPO + 白盒蒸馏 + GRPO/CISPO + Rollout 引擎 | ✅ 完成 | 本地 |
| Phase 3 (扩展) | PPO + Agentic RL | ❌ 待补 | 本地+GPU |
| Phase 4 | torch.compile + fused AdamW + DDP no_sync + activation ckpt + wandb/swanlab | ✅ 完成 | 本地 |
| Phase 5 | safetensors + Qwen3 export + HF/MS push + OpenAI API + release.sh | ✅ 完成 | 本地 |
| Phase E1 | C-Eval + CMMLU + AlignBench + LLM-as-Judge + 多模型对照 | ✅ 完成 | 本地 |
| AutoDL 工具链 | preflight + tmux launch + status + save + release | ✅ 完成 | 本地 |
| **Base 训练** | RTX 4090 / A100 80GB ×1 卡 | ⏳ 待开 | 12-18h（¥150-300） |
| **Plus 训练** | A100 / A800 80GB ×1 卡 | ⏳ 待开 | 30-40h（¥400-700） |
| **发模型卡** | 评测 + 对照表 + 模型卡 → push HF / ModelScope | ⏳ 待开（脚本就绪） | 本地 |

## 9. 参考资料

- [minimind 项目](https://github.com/jingyaogong/minimind)
- [minimind 数据集 (modelscope)](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [minimind 数据集 (huggingface)](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)
- 路线图详细分析：`~/.claude/plans/cryptic-tumbling-dove.md`
- 架构设计：[docs/TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)
