# 🧠 ClearMind — 从零训练你的大语言模型

> **ClearMind** (清心) — 一个从零开始构建的 LLM 训练项目。
> 涵盖 Tokenizer → Pre-training → SFT → DPO 全流程，深入理解大语言模型的每一个环节。

---

## ✨ 项目亮点

- 🔧 **纯 PyTorch 手写** — 不依赖 HuggingFace Transformers，每一行代码都可追溯
- 📐 **现代架构** — RoPE + RMSNorm + SwiGLU + GQA，与 Llama/Gemma 同源
- 🎓 **教育导向** — 中英文详细注释，每个模块都解释"为什么这样做"
- 📊 **三档规模** — 从 MacBook 到 A100，灵活选择训练规模
- 🔄 **完整流水线** — 数据处理 → 预训练 → 指令微调 → 偏好对齐 → 对话推理

## 🏷️ 模型家族

| 模型               | 参数量 | 架构                  | 适合设备        |
| ------------------ | ------ | --------------------- | --------------- |
| **ClearMind-Mini** | ~26M   | MHA, 8层, d=512       | MacBook CPU/MPS |
| **ClearMind**      | ~200M  | GQA 2:1, 16层, d=1024 | GPU 24GB+       |
| **ClearMind-Plus** | ~468M  | GQA 4:1, 24层, d=2048 | A100 80GB       |

## 📁 项目结构

```
capstone-llm-from-scratch/
├── configs/
│   ├── small.yaml           # ClearMind-Mini 配置
│   ├── medium.yaml          # ClearMind 配置
│   └── large.yaml           # ClearMind-Plus 配置 (A100)
├── src/
│   ├── model/               # 🧠 模型架构 (从零实现)
│   │   ├── config.py        #    ModelConfig 超参数
│   │   ├── rope.py          #    RoPE 旋转位置编码
│   │   ├── normalization.py #    RMSNorm 层归一化
│   │   ├── activation.py    #    SwiGLU 激活函数
│   │   ├── feedforward.py   #    SwiGLU 前馈网络
│   │   ├── attention.py     #    Multi-Head Attention + GQA
│   │   ├── transformer.py   #    TransformerBlock (Pre-Norm)
│   │   └── gpt.py           #    完整 GPT 模型
│   ├── data/                # 📦 数据处理
│   │   ├── tokenizer.py     #    BPE 分词器封装
│   │   ├── pretrain_dataset.py  # 预训练数据集 (文本→固定长度)
│   │   ├── sft_dataset.py   #    SFT 数据集 (对话模板+Loss Mask)
│   │   └── dpo_dataset.py   #    DPO 数据集 (偏好对)
│   ├── training/            # 🏋️ 训练模块
│   │   ├── trainer_utils.py #    LR调度/梯度裁剪/Checkpoint/日志
│   │   ├── pretrain.py      #    预训练 Trainer
│   │   ├── sft.py           #    SFT 微调 Trainer
│   │   └── dpo.py           #    DPO 对齐 Trainer
│   └── inference/           # 💬 推理模块
│       ├── generate.py      #    文本生成 (Top-k/Top-p/Temperature)
│       └── chat.py          #    交互式对话
├── scripts/                 # 🚀 入口脚本
│   ├── prepare_data.py      #    数据准备 (样例/HuggingFace)
│   ├── train_tokenizer.py   #    BPE 分词器训练
│   ├── train.py             #    统一训练入口 (--stage pretrain/sft/dpo)
│   ├── chat.py              #    交互式对话
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
│   └── Dockerfile           #    Docker 容器化
├── docs/                    # 📚 项目文档
│   ├── PRD.md               #    产品需求文档
│   ├── TECHNICAL_DESIGN.md  #    技术设计文档
│   └── PROGRESS_TRACKER.md  #    开发进度表
├── requirements.txt
└── requirements-deploy.txt  # 部署专用依赖
```

## 🚀 快速开始

### 环境准备

```bash
cd projects/capstone-llm-from-scratch
pip install -r requirements.txt
```

### MacBook 训练 (ClearMind-Mini, ~26M)

```bash
python scripts/prepare_data.py            # Step 1: 准备样例数据
python scripts/train_tokenizer.py        # Step 2: 训练分词器
python scripts/train.py --stage pretrain # Step 3: 预训练
python scripts/train.py --stage sft      # Step 4: SFT 指令微调
python scripts/train.py --stage dpo      # Step 5: DPO 偏好对齐
python scripts/chat.py                   # Step 6: 开始对话!
```

### A100 训练 (ClearMind-Plus, ~468M)

```bash
# 下载真实大规模数据集
python scripts/prepare_data.py --scale large

# 训练
python scripts/train_tokenizer.py --config configs/large.yaml
python scripts/train.py --stage pretrain --config configs/large.yaml
python scripts/train.py --stage sft --config configs/large.yaml
python scripts/train.py --stage dpo --config configs/large.yaml

# 或使用一键脚本 (AutoDL)
bash scripts/autodl_train.sh large
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
  ├── RMSNorm → Multi-Head Attention (GQA + RoPE + Causal Mask)
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

| 参数        | ClearMind-Mini | ClearMind   | ClearMind-Plus |
| ----------- | -------------- | ----------- | -------------- |
| d_model     | 512            | 1024        | 2048           |
| n_heads     | 8              | 16          | 32             |
| n_kv_heads  | 8 (MHA)        | 8 (GQA 2:1) | 8 (GQA 4:1)    |
| n_layers    | 8              | 16          | 24             |
| d_ff        | 1408           | 2816        | 5632           |
| vocab_size  | 8,000          | 32,000      | 64,000         |
| max_seq_len | 512            | 1024        | 2048           |
| 精度        | float32        | bfloat16    | bfloat16       |

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

### 安装部署依赖

```bash
pip install -r requirements-deploy.txt
```

### 方式一: REST API 服务 (推荐)

```bash
# 启动 API 服务 (兼容 OpenAI 格式)
python deploy/api_server.py --model outputs/dpo/final.pth --port 8000

# 测试 (兼容 OpenAI 客户端)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

### 方式二: Web 演示界面

```bash
# 本地启动 Gradio 界面
python deploy/web_demo.py --model outputs/dpo/final.pth

# 创建公网分享链接
python deploy/web_demo.py --model outputs/dpo/final.pth --share
```

### 方式三: Docker 部署

```bash
# 构建镜像
docker build -t clearmind -f deploy/Dockerfile .

# 运行 API 服务
docker run -p 8000:8000 -v ./outputs:/app/outputs clearmind

# 运行 Web 界面
docker run -p 7860:7860 -v ./outputs:/app/outputs clearmind \
  python deploy/web_demo.py --port 7860
```

### 模型导出与优化

```bash
# 导出精简权重 (去除优化器状态)
python deploy/export_model.py --model outputs/dpo/final.pth --format weights

# INT8 量化 (减少模型大小和推理延迟)
python deploy/export_model.py --model outputs/dpo/final.pth --format quantized

# 导出所有格式
python deploy/export_model.py --model outputs/dpo/final.pth --format all
```

## 📖 学习路线

1. **理解架构** — 阅读 `src/model/` 下每个文件的注释
2. **跑通流程** — 用 `small` 配置在 MacBook 上跑完全流程
3. **对比效果** — 用 `eval_benchmark.py` 一键对比各阶段模型
4. **深入评估** — 分析生成质量和指令跟随能力的变化
5. **部署上线** — 用 API / Web / Docker 部署到生产环境
6. **扩大规模** — 在 A100 上用 `large` 配置训练更大模型

## 📝 License

本项目为个人学习项目，仅用于教育目的。
