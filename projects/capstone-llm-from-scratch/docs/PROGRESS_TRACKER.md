# ClearMind 开发进度表

> 起始日期: 2026-02-18
> 最近更新: 2026-05-02
> 负责人: Perlou

---

## 📊 总体进度

```
重构阶段 1（minimind 数据/tokenizer 接入）:    ██████████ 100%  ✅
重构阶段 2（scripts/evaluate/deploy 重写）:    ██████████ 100%  ✅
P0（断点续训稳定性升级）:                       ██████████ 100%  ✅
Phase 1（架构地基修复）:                        ██████████ 100%  ✅
Phase 2（QK-Norm + RoPE θ=1e6 + YaRN）:        ██████████ 100%  ✅
Phase 3（蒸馏 + GRPO + PPO + Agent RL）:       ███████░░░  70%  🟢 (核心完成；PPO/Agent RL 待补)
Phase 4（torch.compile + wandb + 多卡优化）:   ██████████ 100%  ✅
Phase 5（safetensors + Qwen3 export + push）:  ██████████ 100%  ✅
Phase E1（评测体系：C-Eval + LLM-Judge）:      ██████████ 100%  ✅
AutoDL 上线工具链（preflight/launch/save）:    ██████████ 100%  ✅
线上训练（AutoDL Base + Plus）:                 ░░░░░░░░░░  0%  待开（代码全部就绪）
```

| 总计 | 测试 |
|---|---|
| 全部代码 / 工具链就绪 | **147 个 pytest 全过** |

---

## ✅ Phase 1 — 架构地基修复（已完成）

| 修复 | 状态 |
|---|---|
| `GPT.forward` attention_mask `0*inf=NaN` bug | ✅ 已修 |
| SFT loss-mask BPE 边界错位 → 改用 token 序列扫描 | ✅ |
| `_optimizer_step` LR 错位（scaler 跳过更新仍 scheduler.step） | ✅ |
| 每层重复 RoPE buffer → 顶层共享 | ✅ |
| 冗余 causal_mask buffer → 按需动态构造 | ✅ |
| DPO chosen/rejected 双 forward + deepcopy → 单 forward + 共享 ref | ✅ |
| 残差 proj 初始化 1/√(2L) 缩放（GPT-2/Llama 标准） | ✅ |
| DataLoader 默认 num_workers=0 / 无 persistent_workers → 智能默认 | ✅ |
| Checkpoint 非原子保存 + 全 fp32 落盘 → atomic write + half_weights | ✅ |
| **NaN 三层防御**（dataset 尾部截断 + loss safe-aggregate + trainer 守卫） | ✅ |
| **chat.py truncation budget 边界 case**（max_new ≥ max_seq_len） | ✅ |
| **load_checkpoint 友好的 shape mismatch 报错** | ✅ |

### 🆕 2026-05-02 追加修复（AutoDL 上线前对比 minimind 发现）

| 修复 | 文件 | 影响 |
|---|---|---|
| **DPOTrainer 忽略 `config.max_steps`** → 对齐 sft.py 的 `min(cfg_max, full_max)` 写法 | `src/training/dpo.py:81-89` | yaml 里写的 max_steps 之前完全失效（实测 yaml=200 → 实跑 4077 step），AutoDL 上跑全量 DPO 时无法 cap 总时长，会烧多余的钱 |
| **RMSNorm 在 bf16 输入下输出退化为 fp32** → 内部强制 `x.float()` 计算 + `.type_as(x)` cast 回原 dtype（minimind 标准写法） | `src/model/normalization.py` | 旧实现因 weight 默认 fp32 广播提升，bf16 训练时 RMSNorm 之后所有张量退化为 fp32，破坏 autocast 收益；同时 `pow(2).mean(-1)` 在 bf16 下精度不稳，long context 易触发 loss 尖刺 |

回归测试：5 个 RMSNorm dtype 契约测试（`tests/test_normalization.py`）+ 1 个 DPO max_steps 回归测试（`tests/test_training_edge_cases.py::test_dpo_trainer_respects_config_max_steps`），141 → 147 全绿。

---

## ✅ Phase 2 — 架构升级（已完成）

| 升级 | 文件 | 状态 |
|---|---|---|
| QK-Norm（Llama-3 / Gemma2 同款） | `src/model/attention.py` | ✅ |
| RoPE θ=1e6（Qwen3 对齐，长上下文友好） | `src/model/rope.py` + ModelConfig.rope_theta=1e6 | ✅ |
| YaRN 长上下文外推 | `src/model/rope.py::precompute_rope_frequencies(rope_scaling=...)` | ✅ |
| `d_ff = ⌈d_model · π / 64⌉ · 64`（minimind/Qwen3 风格） | 4 个 yaml + ModelConfig 工厂 | ✅ |
| Sliding Window Attention | `src/model/attention.py` | ✅ |

---

## 🟢 Phase 3 — 训练阶段扩展（70%，核心完成）

| 阶段 | 文件 | 状态 | 备注 |
|---|---|---|---|
| Pretrain | `src/training/pretrain.py` | ✅ | step-based + per_sample/packed 双模式 |
| SFT | `src/training/sft.py` | ✅ | epoch-based + max_steps cap + NaN 守卫 |
| DPO | `src/training/dpo.py` (463 行) | ✅ | 单 forward + 共享 ref |
| **白盒蒸馏** | `src/training/distillation.py` (341 行) | ✅ | KL + T² 缩放 + α 混合 CE |
| **GRPO + CISPO** | `src/training/grpo.py` (382 行) | ✅ | DeepSeek-R1 同款 + 规则 reward |
| **Rollout 引擎** | `src/training/rollout_engine.py` (430 行) | ✅ | TorchBackend ✅ + SGLangBackend ✅ |
| LoRA | `src/training/lora.py` | ✅ | 显式 LoRALinear 类（torch.compile 兼容） |
| PPO | — | ❌ 待补 | GRPO 已能覆盖大多数用例；PPO 是后续延伸 |
| Agentic RL | — | ❌ 待补 | 需要工具调用模拟器；Phase 6 候选 |

---

## ✅ Phase 4 — 工程化补强（已完成）

| 优化 | 文件 | 配置开关 | 状态 |
|---|---|---|---|
| `torch.compile` | `base_trainer.py` | `use_compile: true` | ✅ CUDA 默认开 |
| **Fused AdamW** | `base_trainer.py` | `use_fused_adamw: true` | ✅ +10-15% optimizer step |
| **Activation checkpointing** | `gpt.py` (gradient_checkpointing_enable/disable) | `use_gradient_checkpointing: true` | ✅ plus 默认开 |
| **DDP no_sync** | `scripts/launch_ddp.py` | （DDP 自动） | ✅ 节省 30% 通信带宽 |
| SkipBatchSampler | `trainer_utils.py` | （续训自动） | ✅ |
| wandb / swanlab | `base_trainer.py` | `use_wandb: true` + `wandb_backend` | ✅ 双后端 |
| TensorBoard | `trainer_utils.py::TrainingLogger` | `use_tensorboard: true` | ✅ |
| pin_memory + persistent_workers + prefetch_factor | `base_trainer.py` | 自动 | ✅ |

---

## ✅ Phase 5 — 发布闭环（已完成）

| 模块 | 文件 | 状态 |
|---|---|---|
| Qwen3 兼容导出 | `scripts/convert_to_qwen3.py` (464 行) | ✅ 含 model card 自动生成 |
| safetensors 落盘 | `convert_to_qwen3.py::_save_safetensors` | ✅ |
| HuggingFace push | `scripts/push_to_hub.py` (211 行) | ✅ |
| ModelScope push | `scripts/push_to_modelscope.py` (188 行) | ✅ |
| OpenAI 兼容 API server | `deploy/api_server.py` (390 行) | ✅ |
| Web demo (Gradio) | `deploy/web_demo.py` (191 行) | ✅ |
| Dockerfile | `deploy/Dockerfile` | ✅ |
| **端到端发布流水线** | `scripts/release.sh` (8.9 KB) | ✅ 5 步：检查→转换→transformers 验证→打包→push |

---

## ✅ Phase E1 — 评测体系（已完成）

| 评测 | 文件 | 协议 |
|---|---|---|
| 困惑度 (PPL) | `evaluate/eval_perplexity.py` | 三阶段对比（pretrain/sft/dpo） |
| 生成多样性 | `evaluate/eval_generation.py` | Distinct-N + trigram 重复率 + EOS 命中 |
| 指令跟随（旧） | `evaluate/eval_instruction.py` | 14 题 keyword overlap（冒烟用） |
| **C-Eval** | `evaluate/benchmarks/ceval.py` | 5-shot loglikelihood（OpenCompass 对齐） |
| **CMMLU** | `evaluate/benchmarks/cmmlu.py` | 5-shot loglikelihood |
| **AlignBench-zh** | `evaluate/benchmarks/alignbench.py` | LLM-as-Judge 1-10 分 |
| **LLM-as-Judge 客户端** | `evaluate/judge/llm_judge.py` | OpenAI 兼容（DeepSeek/GPT/Qwen-72B） |
| **多模型对照** | `evaluate/eval_compare.py` | merge/run 双模式，输出 markdown |

详见 [`evaluate/README.md`](../evaluate/README.md)。

---

## ✅ AutoDL 上线工具链（已完成）

| 脚本 | 职责 |
|---|---|
| `scripts/autodl/preflight.sh` | 9 项强制自检（Python/依赖/GPU/磁盘/数据/tmux/单测/冒烟） |
| `scripts/autodl/launch.sh` | tmux 启动器，断 SSH 不影响；自动级联 pretrain→sft→dpo |
| `scripts/autodl/status.sh` | 训练状态查询（sessions/进程/GPU/最近 step/产物/磁盘） |
| `scripts/autodl/save_outputs.sh` | 归档 ckpt+log+eval+manifest+sha256 → tar.gz |
| `scripts/release.sh` | 端到端发布（convert→验证→打包→push HF/MS） |

详见 [`docs/AUTODL_GUIDE.md`](AUTODL_GUIDE.md)。

---

## ⏳ 待开（代码就绪，需要 GPU）

### 显卡选型（基于 2026-05 AutoDL 实价）

| 配置 | 推荐显卡 | ¥/h | 估时 | **总成本** |
|---|---|---|---|---|
| **Base (68.8M)** | RTX 4090 24G ⭐ | 2.29 | ~28-32h | **¥65-80** |
| **Plus (486.3M)** | A100-PCIE 40G ⭐ | 3.45 | ~45-55h | **¥155-200** |
| 两个都跑 | — | — | — | **≈ ¥240（充 ¥400 留 buffer）** |

> A800 80G（5.24/h）最稳但贵 ¥80+；4090 跑 Plus 需要 100+h 风险高，不推荐。详见 [AUTODL_GUIDE.md](AUTODL_GUIDE.md#step-1--注册-autodl--选-gpu)。

### 已确认 Base 实例配置（2026-05-02 选定）

| 项 | 配置 | 评估 |
|---|---|---|
| GPU | RTX 4090 24GB | ✅ 35% VRAM 利用，富余 17GB |
| CPU | Xeon Gold 6430 × 16 核 | ✅ 富余（8 worker 用得上） |
| 内存 | 120 GB | ✅ 巨富余 |
| 系统盘 | 30 GB | 🟡 venv/cache 必须重定向到数据盘 |
| 数据盘 | 50 GB | 🟡 够用（~30GB 占用），建议升到 80GB 留 buffer |

部署前必须先在 `~/.bashrc` 配置 cache 重定向（详见 AUTODL_GUIDE.md Step 2.2.1）。

### 1. 上 AutoDL 跑 Base 正式训练（RTX 4090，~¥70）

```bash
bash scripts/autodl/preflight.sh --profile base
bash scripts/autodl/launch.sh tiny  all       # 5 min 冒烟
bash scripts/autodl/launch.sh small all       # 30 min 验证
bash scripts/autodl/launch.sh base  all       # 12-18h 正式
```

### 2. 评测 + 发模型卡

```bash
python evaluate/benchmarks/ceval.py --config configs/main.yaml
python evaluate/benchmarks/alignbench.py --config configs/main.yaml
bash scripts/release.sh base --stage dpo --push-hf you/ClearMind-Base
```

### 3. Plus 训练 + 发布（A100 40G，~¥175）

A100 40G 需要先调 `configs/plus.yaml`：
```yaml
pretrain:
  batch_size: 8              # 默认 16 减半（A100 40G 显存预算）
  gradient_accumulation: 16  # 翻倍保持 effective batch = 128
```

```bash
bash scripts/autodl/launch.sh plus all        # 45-55h 正式
bash scripts/release.sh plus --stage dpo --push-hf you/ClearMind-Plus
```

---

## 🔮 路线图（未来工作）

| Phase | 内容 | 优先级 | 预估工作量 |
|---|---|---|---|
| Phase 3.x | PPO trainer（actor + critic + GAE + value loss） | P2 | ~1 天本地 |
| Phase 3.x | Agentic RL（tool 调用模拟器 + multi-step reward） | P3 | ~3-5 天 |
| Phase E2 | GSM8K-zh / LongBench-zh / Safety-Prompts | P2 | ~1 天 |
| Phase E3 | HumanEval-zh / BBH-zh / 自动 model card 生成器 | P3 | ~1 天 |
| 长上下文 | YaRN scaling 实测验证（LongBench-zh 跑分） | P2 | 几小时 |
| Plus 训练 | 实际 GPU 训练 + 评测 + 发模型卡 | P0 | 30-40h GPU |

---

## 📌 关键技术决策记录

1. **复用 minimind tokenizer**：避免重训 tokenizer 的成本，且数据完全对齐
2. **从训练 GPT 转 Qwen3 兼容**：发布时通过命名映射转 `Qwen3ForCausalLM`，让用户用 `AutoModelForCausalLM.from_pretrained` 即用
3. **Tiny config 缩到 0.5M**：纯流程冒烟，不期待生成质量
4. **NaN 三层防御**：data 截断 + loss 安全聚合 + trainer 守卫，应对 SFT 长 prompt 截断的经典坑
5. **OpenAI 兼容 LLM-Judge**：通过 env var 切换 OpenAI / DeepSeek / 阿里百炼 / 自部署 vLLM
6. **tmux 抗断连**：保证 12-40h 训练不被 SSH 抖动打断
