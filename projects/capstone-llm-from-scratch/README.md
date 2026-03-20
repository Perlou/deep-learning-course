# 🧠 ClearMind — 从零训练你的大语言模型

> **ClearMind** (清心) — 一个从零开始构建的 LLM 训练项目。
> 涵盖 Tokenizer → Pre-training → SFT → DPO 全流程，深入理解大语言模型的每一个环节。

---

## ✨ 项目亮点

- 🔧 **纯 PyTorch 手写** — 不依赖 HuggingFace Transformers，每一行代码都可追溯
- 📐 **现代架构** — RoPE + RMSNorm + SwiGLU + GQA，与 Llama/Gemma 同源
- ⚡ **推理优化** — KV Cache + Flash Attention + Sliding Window Attention
- 🎓 **教育导向** — 中英文详细注释，每个模块都解释"为什么这样做"
- 📊 **四档规模** — Tiny 验证 → Mini 学习 → Medium GPU → Large A100
- 🔄 **完整流水线** — 数据处理 → 预训练 → 指令微调 → 偏好对齐 → 对话推理
- 🧱 **LoRA 微调** — 低秩适配器，少参数高效微调
- 🧪 **单元测试** — 覆盖核心模块，支持边界条件回归测试
- ✅ **Smoke Test** — 一条命令验证数据→分词器→预训练 + SFT + 推理验证
- 🔀 **DDP 训练入口** — 支持 `torchrun` 多 GPU 预训练
- 🚀 **一键部署** — FastAPI / Gradio / Docker / GGUF 导出

## 🏷️ 模型家族

| 模型               | 参数量 | 架构                  | 适合设备        |
| ------------------ | ------ | --------------------- | --------------- |
| **ClearMind-Tiny** | ~1.5M  | MHA, 4层, d=128       | 任意 (流程验证) |
| **ClearMind-Mini** | ~26M   | MHA, 8层, d=512       | MacBook CPU/MPS |
| **ClearMind**      | ~200M  | GQA 2:1, 16层, d=1024 | GPU 24GB+       |
| **ClearMind-Plus** | ~468M  | GQA 4:1, 24层, d=2048 | A100 80GB       |

## 📁 项目结构

```
capstone-llm-from-scratch/
├── configs/
│   ├── tiny.yaml            # ClearMind-Tiny 配置 (流程验证)
│   ├── small.yaml           # ClearMind-Mini 配置
│   ├── medium.yaml          # ClearMind 配置
│   └── large.yaml           # ClearMind-Plus 配置 (A100)
├── src/
│   ├── model/               # 🧠 模型架构 (从零实现)
│   │   ├── config.py        #    ModelConfig 超参数 + 配置校验
│   │   ├── rope.py          #    RoPE 旋转位置编码
│   │   ├── normalization.py #    RMSNorm 层归一化
│   │   ├── activation.py    #    SwiGLU 激活函数
│   │   ├── feedforward.py   #    SwiGLU 前馈网络
│   │   ├── attention.py     #    MHA + GQA + KV Cache + Flash Attention
│   │   ├── transformer.py   #    TransformerBlock (Pre-Norm)
│   │   └── gpt.py           #    完整 GPT 模型
│   ├── data/                # 📦 数据处理
│   │   ├── tokenizer.py     #    BPE 分词器 + OOV 覆盖率检测
│   │   ├── pretrain_dataset.py  # 预训练数据集 (文本→固定长度)
│   │   ├── sft_dataset.py   #    SFT 数据集 (对话模板+Loss Mask)
│   │   └── dpo_dataset.py   #    DPO 数据集 (偏好对)
│   ├── training/            # 🏋️ 训练模块
│   │   ├── base_trainer.py #    Trainer 基类 (封装共享逻辑)
│   │   ├── trainer_utils.py #    LR调度/梯度裁剪/Checkpoint/日志/DDP
│   │   ├── pretrain.py      #    预训练 Trainer
│   │   ├── sft.py           #    SFT 微调 Trainer
│   │   ├── dpo.py           #    DPO 对齐 Trainer
│   │   └── lora.py          #    LoRA 低秩适配微调
│   └── inference/           # 💬 推理模块
│       ├── generate.py      #    文本生成 (KV Cache + Top-k/Top-p)
│       └── chat.py          #    交互式对话
├── scripts/                 # 🚀 入口脚本
│   ├── prepare_data.py      #    数据准备 (样例/HuggingFace)
│   ├── train_tokenizer.py   #    BPE 分词器训练
│   ├── train.py             #    统一训练入口 (--stage pretrain/sft/dpo)
│   ├── chat.py              #    交互式对话
│   ├── launch_ddp.py        #    DDP 多 GPU 预训练入口 (torchrun)
│   ├── smoke_test.py        #    端到端最小链路冒烟测试
│   └── autodl_train.sh      #    AutoDL 一键训练
├── evaluate/                # 📊 评估模块
│   ├── eval_perplexity.py   #    困惑度评估 (支持 --compare 阶段对比)
│   ├── eval_generation.py   #    生成质量评估 (Distinct-N/重复率)
│   ├── eval_instruction.py  #    指令跟随评估 (格式/相关性/安全性)
│   └── eval_benchmark.py    #    综合评估报告 (一键全面评测)
├── deploy/                  # 🚀 部署模块
│   ├── api_server.py        #    FastAPI REST API (兼容 OpenAI 格式)
│   ├── web_demo.py          #    Gradio Web 演示界面
│   ├── export_model.py      #    模型导出 (权重瘦身/量化)
│   ├── export_gguf.py       #    GGUF 格式导出 (llama.cpp 兼容)
│   └── Dockerfile           #    Docker 容器化
├── tests/                   # 🧪 单元测试
│   ├── conftest.py          #    共享 fixture (tiny_config, tiny_model)
│   ├── test_model.py        #    GPT 模型测试
│   ├── test_attention.py    #    Attention + GQA + KV Cache 测试
│   ├── test_config.py       #    配置校验测试
│   ├── test_tokenizer.py    #    Tokenizer 测试
│   ├── test_generate.py     #    文本生成测试
│   ├── test_datasets.py     #    数据集测试 (Pretrain/SFT/DPO)
│   ├── test_trainer_utils.py #    训练工具测试
│   ├── test_training_edge_cases.py # 训练边界条件测试
│   └── test_lora.py         #    LoRA 测试
├── docs/                    # 📚 项目文档
│   ├── PRD.md               #    产品需求文档
│   ├── TECHNICAL_DESIGN.md  #    技术设计文档
│   ├── PROGRESS_TRACKER.md  #    开发进度表
│   ├── DEPLOY.md            #    部署指南 (硬件/API/Docker)
│   └── AUTODL_GUIDE.md      #    AutoDL 租赁与训练攻略
├── requirements.txt
├── requirements-deploy.txt  # 部署专用依赖
└── run.sh                   # 🎯 一键启动脚本 (交互式菜单)
```

## 🚀 快速开始

### 环境准备

```bash
cd capstone-llm-from-scratch

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 一键启动

```bash
bash run.sh
```

交互式菜单引导你完成整个流程：

```
📦 选择模型规模:
  1) Tiny   (~1.5M 参数, 2-5 分钟, 无需联网)
  2) Small  (~26M  参数, MacBook CPU/MPS)
  3) Medium (~200M 参数, GPU 24GB+)
  4) Large  (~468M 参数, A100 80GB)

🔄 选择训练流程:
  1) 全流程  (数据 → 分词器 → 预训练 → SFT → DPO → 对话)
  2) 仅预训练
  3) 从 SFT 继续
  4) 从 DPO 继续
  5) 仅对话
  6) 仅训练分词器
  7) 运行测试  (失败自动回退兼容模式)
  8) 冒烟测试  (数据 → 分词器 → 预训练1步, tiny)
```

<details>
<summary>📝 也可以手动逐步执行</summary>

**快速验证 (Tiny, 无需联网):**

```bash
python scripts/prepare_data.py
python scripts/train_tokenizer.py --config configs/tiny.yaml
python scripts/train.py --stage pretrain --config configs/tiny.yaml
python scripts/train.py --stage sft --config configs/tiny.yaml
python scripts/chat.py --config configs/tiny.yaml
```

**完整训练 (Small/Medium/Large):**

```bash
python scripts/prepare_data.py [--scale medium|large]
python scripts/train_tokenizer.py [--config configs/xxx.yaml]
python scripts/train.py --stage pretrain [--config configs/xxx.yaml]
python scripts/train.py --stage sft [--config configs/xxx.yaml]
python scripts/train.py --stage dpo [--config configs/xxx.yaml]
python scripts/chat.py [--config configs/xxx.yaml]
```

</details>

### 🧪 端到端 Smoke Test（最小链路）

用于快速验证“数据准备 → 分词器 → 预训练”是否可跑通：

```bash
python scripts/smoke_test.py --max_steps 1
```

可选参数：

```bash
python scripts/smoke_test.py --max_steps 2 --work_dir outputs/smoke --clean
```

> [!NOTE]
> `smoke_test.py` 依赖 `sentencepiece`。如提示缺失，请先执行 `pip install -r requirements.txt`。

### 🔀 多 GPU 预训练（DDP）

在多 GPU 机器上使用 `torchrun` 启动：

```bash
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/medium.yaml \
  --data data/pretrain/pretrain_data.jsonl \
  --tokenizer outputs/tokenizer/tokenizer.model
```

常用参数覆盖：

```bash
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/medium.yaml \
  --max_steps 2000 \
  --batch_size 4 \
  --gradient_accumulation 8 \
  --output_dir outputs/pretrain_ddp
```

断点续训：

```bash
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/medium.yaml \
  --resume outputs/pretrain_ddp/checkpoint_step1000.pth
```

### 🧱 LoRA 微调

使用 LoRA 进行参数高效微调，只训练少量低秩适配参数：

```bash
# SFT + LoRA
python scripts/train.py --stage sft --lora --lora_rank 8 --lora_alpha 16

# 自定义配置
python scripts/train.py --stage sft --config configs/medium.yaml --lora --lora_rank 16
```

## 🏗️ 技术架构

```
输入文本
  ↓
[BPE Tokenizer] → Token IDs
  ↓
[Token Embedding] → 向量表示
  ↓
[Transformer Block] × N 层
  ├── RMSNorm → Multi-Head Attention (GQA + RoPE + KV Cache + Flash Attention)
  ├── Residual Connection
  ├── RMSNorm → SwiGLU FeedForward
  └── Residual Connection
  ↓
[Final RMSNorm] → [LM Head] → 词表概率分布
  ↓
Next Token Prediction
```

### 训练流水线

```
Pre-training          →  SFT 指令微调           →  DPO 偏好对齐
（学习语言知识）        （学习按指令回答）        （学习人类偏好）

Loss: Next-token       Loss: 只在 Assistant     Loss: DPO Loss
  CrossEntropy           回复上计算 CE            chosen > rejected
```

## 📊 模型配置详解

| 参数        | ClearMind-Tiny | ClearMind-Mini | ClearMind   | ClearMind-Plus |
| ----------- | -------------- | -------------- | ----------- | -------------- |
| d_model     | 128            | 512            | 1024        | 2048           |
| n_heads     | 4              | 8              | 16          | 32             |
| n_kv_heads  | 4 (MHA)        | 8 (MHA)        | 8 (GQA 2:1) | 8 (GQA 4:1)    |
| n_layers    | 4              | 8              | 16          | 24             |
| d_ff        | 352            | 1408           | 2816        | 5632           |
| vocab_size  | 2,000          | 8,000          | 32,000      | 64,000         |
| max_seq_len | 128            | 512            | 1024        | 2048           |
| 精度        | float32        | float32        | bfloat16    | bfloat16       |

## 📊 模型评估

训练完成后，使用评估工具验证模型效果:

```bash
# 困惑度评估 (PPL 越低越好)
python evaluate/eval_perplexity.py --model outputs/sft/final.pth

# 一键对比各阶段 PPL
python evaluate/eval_perplexity.py --compare

# 生成质量评估 (Distinct-N, 重复率)
python evaluate/eval_generation.py --model outputs/sft/final.pth

# 指令跟随评估 (格式正确率, 安全性)
python evaluate/eval_instruction.py --model outputs/dpo/final.pth

# 综合评估报告 (一键运行所有评估)
python evaluate/eval_benchmark.py --config configs/small.yaml
```

## 🚀 部署上线

详细部署文档请参考 **[DEPLOY.md](docs/DEPLOY.md)**，包含：

- 🖥️ 各模型硬件需求（训练 + 推理）
- 🌐 REST API 服务（兼容 OpenAI 格式）
- 💬 Gradio Web 演示界面
- 🐳 Docker 容器化部署
- 📦 模型导出与 INT8 量化
- ⚡ 性能调优建议

**快速体验：**

```bash
pip install -r requirements-deploy.txt
python deploy/api_server.py --model outputs/dpo/final.pth
```

## 🧪 单元测试

```bash
# 运行全部测试
python -m pytest tests/ -v
```

如果你的环境有第三方 `pytest` 插件冲突，可临时禁用自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
```

测试覆盖全部核心模块:

| 模块       | 测试内容                                  |
| ---------- | ----------------------------------------- |
| 模型 (GPT) | 前向/反向传播, 参数统计, KV Cache         |
| 注意力     | MHA/GQA, KV Cache 增长, Sliding Window    |
| 配置       | 工厂方法, 校验规则, YAML 加载             |
| 分词器     | encode/decode, 特殊 token, 中文, OOV 检测 |
| 文本生成   | 长度控制, Greedy 确定性, EOS 停止         |
| 数据集     | Pretrain/SFT/DPO 加载与 loss mask         |
| 训练工具   | LR 调度, Early Stopping, 梯度裁剪         |
| 训练边界   | zero-step、梯度累积余数、懒加载导入        |
| LoRA       | apply/merge/save/load                     |

## 📖 学习路线

1. **理解架构** — 阅读 `src/model/` 下每个文件的注释
2. **跑通流程** — 用 `small` 配置在 MacBook 上跑完全流程
3. **对比效果** — 用 `eval_benchmark.py` 一键对比各阶段模型
4. **深入评估** — 分析生成质量和指令跟随能力的变化
5. **运行测试** — 用 `pytest` 验证对每个模块的理解
6. **部署上线** — 用 API / Web / Docker 部署到生产环境
7. **扩大规模** — 在 A100 上用 `large` 配置训练更大模型

## 📝 License

本项目为个人学习项目，仅用于教育目的。
