# 🖥️ AutoDL 训练完整攻略

> **目的**：把 ClearMind 从本地推到 AutoDL GPU 上完整训练 Tiny → Small → Base → Plus，**且断 SSH / 关电脑 / WiFi 掉线都不影响训练**。
>
> 训练失败的成本很高（一次 plus 训练 30+ 小时、几百块 RMB），本攻略目标就是把每一步都"压到不会出错"。

---

## 📑 目录

- [Step 0 — 上线检查清单](#step-0--上线检查清单)
- [Step 1 — 注册 AutoDL + 选 GPU](#step-1--注册-autodl--选-gpu)
- [Step 2 — SSH 接入 + 部署项目](#step-2--ssh-接入--部署项目)
- [Step 3 — Preflight（强制自检）](#step-3--preflight强制自检)
- [Step 4 — 渐进式训练 Tiny → Small → Base → Plus](#step-4--渐进式训练-tiny--small--base--plus)
- [Step 5 — 断连生存 / 重连 / 续训](#step-5--断连生存--重连--续训)
- [Step 6 — 评估训练效果](#step-6--评估训练效果)
- [Step 7 — 归档下载 + 关机](#step-7--归档下载--关机)
- [常见故障速查](#常见故障速查)

---

## Step 0 — 上线检查清单

**租机之前，先在本地做完这些**：

- [ ] 本地 `pytest tests/ -q` 全部通过（确认代码层面没坑）
- [ ] 本地 `python scripts/smoke_test.py --clean` 通过（确认数据 + 流程跑得通）
- [ ] 项目已 push 到 GitHub / GitLab（AutoDL 是 git clone 拉，不是 scp）
- [ ] 估算预算：
  - **Base** (~12-18h, RTX 4090 / A100 80G ×1): ¥150-300
  - **Plus** (~30-40h, A100/A800 80G ×1): ¥400-700
- [ ] 充值预算的 1.5 倍（避免训到一半余额不足被强制关机）

> 💡 **本攻略已经把代码层面的 NaN / 数据截断 / checkpoint 损坏全部修了**（详见 CLAUDE.md "已知 bug 修复"）。但你必须先在本地跑通 `smoke_test.py`，否则 GPU 上踩雷代价太大。

---

## Step 1 — 注册 AutoDL + 选 GPU

1. [autodl.com](https://www.autodl.com) 注册 + 实名认证
2. 充值（Base ¥300 / Plus ¥800 起步）
3. [算力市场](https://www.autodl.com/market/list) 选机：

| 项 | Tiny / Small | Base | Plus |
|---|---|---|---|
| GPU | 任意 (RTX 3090 24G 起) | RTX 4090 24G ⭐推荐 / A100 80G | A100 80G / A800 80G |
| 内存 | ≥ 16 GB | ≥ 32 GB | ≥ 64 GB |
| 系统盘 | 30 GB | 60 GB | 100 GB |
| 数据盘 | 30 GB | 50 GB | 80 GB |
| 镜像 | **PyTorch 2.1+ / CUDA 12.x** | 同 | 同 |
| 计费 | 按量付费 | 按量付费（**勾"关机不计费"**） | 按量付费 / 包日（看是否能稳定连续 30h+） |

> ⚠️ **务必勾"关机不计费"**。这样调试 / 中场休息时关机不烧钱。
>
> ⚠️ **PyTorch 镜像必须 ≥ 2.1**（项目用了 `torch.amp.autocast` 新 API，旧版会触发兼容路径但不稳定）。

---

## Step 2 — SSH 接入 + 部署项目

### 2.1 接入

AutoDL 控制台 → 复制"SSH 登录指令"和密码 → 本地终端：

```bash
# 假设 SSH 命令是 ssh -p 12345 root@connect.westa.seetacloud.com
ssh -p 12345 root@connect.westa.seetacloud.com
# 输入密码后进入
```

进入后默认在 `/root`。**所有项目和数据都放在 `/root/autodl-tmp/`**（数据盘，关机后保留；`/root` 是系统盘，可能被清）。

### 2.2 部署

```bash
cd /root/autodl-tmp

# 国内 GitHub 慢的话用代理或镜像
git clone https://github.com/<your-username>/clearmind.git
cd clearmind

# 创建 venv（与本地一致）
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 装 tmux（断连生存的关键）
apt-get update && apt-get install -y tmux

# 下载训练数据（base 用 mini，plus 用 full）
python scripts/download_data.py --profile base --source modelscope     # 国内更快
# 或者
python scripts/download_data.py --profile plus --source modelscope
```

> 💡 **首次 clone 完务必跑 `bash scripts/autodl/preflight.sh`**（见下一步），别上来就训。

---

## Step 3 — Preflight（强制自检）

```bash
bash scripts/autodl/preflight.sh --profile base
```

它会一次性检查：

| 检查项 | 失败后果 |
|---|---|
| Python ≥ 3.10 + 核心依赖齐全 | 训练直接 import error |
| GPU 可见 + VRAM ≥ profile 推荐 | OOM / 训不下 |
| 磁盘 ≥ profile 推荐 | 中途写 ckpt 失败损坏 |
| tokenizer/minimind/ + data/*.jsonl 就位 | 数据加载失败 |
| tmux 可用 | 断 SSH 训练被杀 |
| `pytest tests/ -q` 通过 | 代码层 bug |
| `smoke_test.py` 通过（tiny 端到端） | 数据/模型/链路 bug |

**ERROR ≠ 0 时禁止进入 Step 4**。

如果你已经在本地跑过冒烟，可以加 `--skip-smoke` 加速。

---

## Step 4 — 渐进式训练 Tiny → Small → Base → Plus

**核心命令只有一个**：`bash scripts/autodl/launch.sh <规格> <阶段> [--foreground|--kill]`

它做的事：
1. 把训练塞进 `tmux` session（**断 SSH 不影响**）
2. 同步把所有输出 tee 到 `logs/clearmind-<规格>-<阶段>.log`（**tmux 死了 log 还在**）
3. 写 PID + state 文件，方便 `status.sh` 查询
4. `all` 模式时按 pretrain → sft → dpo 自动级联（带自动 resume 链）
5. dpo 完成后自动跑 `eval_perplexity.py --compare`

### 4.1 推荐流程（每一步都要等上一步完全成功）

```bash
# === 4.1 Tiny 冒烟（GPU 上验证整条链路，5-10 分钟）===
bash scripts/autodl/launch.sh tiny all
tmux attach -t clearmind-tiny-all      # 看实时画面，Ctrl+b d 脱离

# 完成后看结果：
bash scripts/autodl/status.sh
ls outputs/dpo/final.pth                # 应该存在

# === 4.2 Small 验证（半小时-1 小时，看是否 GPU 训起来正常）===
bash scripts/autodl/launch.sh small all

# === 4.3 Base 正式训练（10-18 小时）===
# ⚠️ 这一步会跑很久，断 SSH 没事，但记得来看一眼进度
bash scripts/autodl/launch.sh base all

# 中途状态查询（不必接 tmux）：
bash scripts/autodl/status.sh
bash scripts/autodl/status.sh --watch   # 每 5s 刷新，Ctrl+C 退出

# === 4.4 Plus 旗舰（30-40 小时）===
# Base 完成并归档下载之后再跑（避免 Base 训练成果被覆盖）
bash scripts/autodl/launch.sh plus all
```

### 4.2 单阶段控制（细粒度）

```bash
# 只跑预训练
bash scripts/autodl/launch.sh base pretrain

# 只跑 SFT（自动用 outputs/pretrain/final.pth 作为初始权重）
bash scripts/autodl/launch.sh base sft

# 只跑 DPO（自动用 outputs/sft/final.pth）
bash scripts/autodl/launch.sh base dpo
```

### 4.3 调试用的前台模式（不进 tmux）

```bash
bash scripts/autodl/launch.sh tiny pretrain --foreground
# Ctrl+C 直接停；日志只在终端，不存档
```

### 4.4 强制中止训练

```bash
bash scripts/autodl/launch.sh base all --kill
```

---

## Step 5 — 断连生存 / 重连 / 续训

这是整个攻略的核心。**断了别慌**。

### 5.1 SSH 断了，训练还在不在跑？

```bash
# 重新 SSH 进 AutoDL
ssh -p ... root@...
cd /root/autodl-tmp/clearmind
bash scripts/autodl/status.sh
```

输出会告诉你：
- `clearmind-base-all` session 状态（活着 / 已完成 / 已失败）
- 训练进程 PID + 已运行时间 + CPU/MEM
- GPU 利用率（应 > 70%）
- 最近的 Step / Loss 日志
- outputs/ 下各阶段产物

### 5.2 重新看实时画面

```bash
tmux attach -t clearmind-base-all
# Ctrl+b d 脱离（不杀进程）
```

或者只看日志：
```bash
tail -f logs/clearmind-base-all.log
```

### 5.3 进程死了 / 服务器被强关 / OOM 了

不用担心，trainer 每次保存都会写 `outputs/<stage>/_resume.pth`（含 model + optimizer + scheduler + scaler 完整状态）。

```bash
# 直接重新跑同一命令即可
bash scripts/autodl/launch.sh base all
# 日志里会看到：🔄 自动续训：检测到 outputs/pretrain/_resume.pth
#                🔄 自动从 step=10000 续训
```

支持的中断场景：
- ✅ SSH 断开 — 不影响（tmux）
- ✅ tmux 意外死 — 重启服务器后用 `_resume.pth` 续
- ✅ OOM 杀进程 — 调小 batch 后用 `_resume.pth` 续
- ✅ AutoDL 实例被关 — 关机不丢盘，开机后用 `_resume.pth` 续
- ❌ AutoDL 实例被释放（按量付费余额耗尽） — `/root/autodl-tmp/` 也丢；务必充够钱

### 5.4 训练进度量化估计

每个 stage 第一次保存后，log 里会有 `ETA` 字段：
```
Step 1500/100000 | Loss: 4.21 | LR: 3.00e-4 | Grad: 1.34 | ETA: 8h 42m
```

如果 ETA 偏离预期 2 倍以上，先 `nvidia-smi -l 1` 看是不是 GPU 利用率低（数据加载瓶颈、batch 太小）。

---

## Step 6 — 评估训练效果

训练完（dpo 完成后），自动评估已经写到了 `logs/clearmind-<规格>-all.log.eval`。

### 6.1 三档评测策略

| 档 | 用途 | 命令 | 耗时 | 是否需要网络 |
|---|---|---|---|---|
| **快** | 训练时手感、冒烟 | `eval_perplexity.py --compare` | < 1 min | ❌ |
| **中** | 发布前自查（公开基准） | `benchmarks/ceval.py` + `benchmarks/cmmlu.py` | 30 min - 2h | 首次需联网下数据 |
| **重** | 对外宣传"反超 minimind" | `benchmarks/alignbench.py` + `eval_compare.py` | 1-3h（含 LLM-Judge API） | 需 OpenAI/DeepSeek API key |

### 6.2 必跑（发布前 must）

```bash
# (1) PPL 三阶段对比 — 验证 SFT/DPO 真的改善了
python evaluate/eval_perplexity.py --config configs/main.yaml --compare

# (2) C-Eval — 中文综合知识 4 选 1，OpenCompass 标准
python evaluate/benchmarks/ceval.py --config configs/main.yaml \
    --output reports/clearmind_base_ceval.json

# (3) CMMLU — 中文 MMLU 等价
python evaluate/benchmarks/cmmlu.py --config configs/main.yaml \
    --output reports/clearmind_base_cmmlu.json

# (4) AlignBench-zh + LLM-as-Judge（需要 judge API key）
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com/v1   # 国内推荐
export CLEARMIND_JUDGE_MODEL=deepseek-chat
python evaluate/benchmarks/alignbench.py --config configs/main.yaml \
    --output reports/clearmind_base_alignbench.json

# (5) 多模型对照表（贴到 HF 模型卡）
python evaluate/eval_compare.py merge \
    clearmind:reports/clearmind_base_ceval.json \
    minimind:reports/minimind3_official_ceval.json \
    --output reports/vs_minimind.md
```

### 6.3 调试 / 冒烟（每个 runner 都支持 ``--limit``）

```bash
python evaluate/benchmarks/ceval.py --config configs/tiny.yaml --limit 30
python evaluate/benchmarks/alignbench.py --config configs/tiny.yaml --limit 5 --no-judge
```

### 6.4 旧版指令评估（粗糙但快，仅冒烟用）

```bash
python evaluate/eval_instruction.py --config configs/main.yaml
python evaluate/eval_generation.py --config configs/main.yaml --num_prompts 30
```

### 6.5 人工抽测（最关键的最后一步）

```bash
python scripts/chat.py --config configs/main.yaml
# > 你好
# > 介绍一下你自己
# > 写一首关于秋天的诗
```

如果生成结果明显是噪声，对照 `docs/PRD.md` 检查 Phase 1/2 的优化项是否都合上。

> 详细评测体系见 [`evaluate/README.md`](../evaluate/README.md)。

---

## Step 7 — 归档下载 + 关机

**关机前的最后一步**，把训练成果安全带走：

```bash
# 1. 打包归档（含 checkpoints + log + eval 报告 + manifest + sha256）
bash scripts/autodl/save_outputs.sh base
# 产出：releases/clearmind-base-YYYYMMDD-HHMM.tar.gz
#       releases/clearmind-base-YYYYMMDD-HHMM.tar.gz.sha256

# 2. light 模式（只要 final.pth，不带 _resume.pth；包小很多）
bash scripts/autodl/save_outputs.sh base --light

# 3. 顺便 push 到 HuggingFace（需提前 huggingface-cli login）
bash scripts/autodl/save_outputs.sh base --push-hf your-username/clearmind-base

# 4. ModelScope（国内）
bash scripts/autodl/save_outputs.sh base --push-ms your-username/ClearMind-Base
```

### 下载到本地

在 **你的笔记本** 上执行：

```bash
scp -P <端口> root@<autodl-ip>:/root/autodl-tmp/clearmind/releases/clearmind-base-*.tar.gz ./
scp -P <端口> root@<autodl-ip>:/root/autodl-tmp/clearmind/releases/clearmind-base-*.tar.gz.sha256 ./

# 校验
sha256sum -c clearmind-base-*.tar.gz.sha256

# 解压
tar -xzf clearmind-base-*.tar.gz
```

### 关机

确认本地校验通过后，AutoDL 控制台点"关机"。**勾选「关机不计费」就不烧钱**。

---

## Step 8 — Phase 5 发布（HF/ModelScope 兼容）

训练成果归档下载之后，把 ClearMind 转成 **Qwen3ForCausalLM 兼容**的 HuggingFace 标准格式，这样别人 `AutoModelForCausalLM.from_pretrained("you/ClearMind-Base")` 就能直接用。

### 8.1 一键 release 流水线

```bash
# 仅本地转换 + transformers 加载验证（不 push，最稳）
bash scripts/release.sh base --stage dpo

# 转完顺便 push HuggingFace（先 huggingface-cli login 或 export HF_TOKEN）
bash scripts/release.sh base --stage dpo --push-hf your-username/ClearMind-Base

# 同时 push HF + ModelScope
bash scripts/release.sh plus --stage dpo \
    --push-hf you/ClearMind-Plus \
    --push-ms you/ClearMind-Plus

# 调试：演练流程不真发
bash scripts/release.sh base --stage dpo --dry-run
```

流水线 5 步：
1. 检查 `outputs/dpo/final.pth` 存在
2. `convert_to_qwen3.py` → `release/<name>/{config.json, model.safetensors, README.md, tokenizer.*}`
3. **用 transformers 重新加载验证**（防止格式错误，发出去才发现）
4. tar.gz 打包 + sha256 校验
5. 可选 push HF/MS

### 8.2 验证后启动服务

```bash
# OpenAI 兼容 API 服务（被 LangChain / Cherry Studio / OpenWebUI 等直接调用）
python deploy/api_server.py --config configs/main.yaml --port 8000

# Gradio Web Demo
python deploy/web_demo.py --config configs/main.yaml --port 7860 --share

# 测试 API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role":"user","content":"你好"}], "stream": false}'
```

### 8.3 转 GGUF（可选，给 llama.cpp / Ollama 用）

```bash
cd <llama.cpp 目录>
python convert_hf_to_gguf.py /path/to/clearmind/release/clearmind-base \
    --outtype q4_0 \
    --outfile clearmind-base-q4_0.gguf
```

---

## 常见故障速查

| 症状 | 排查 / 解决 |
|---|---|
| `nvidia-smi` 找不到 | 镜像不对，重选 PyTorch + CUDA 镜像 |
| `CUDA out of memory` | 改小 `batch_size`（YAML），加大 `gradient_accumulation` 保持 effective batch 不变 |
| Loss 一直是 NaN | 已修（A 方案三层防御）；若仍出现，看 `_nan_skipped` 计数 + 检查数据 |
| `epoch 平均 loss = nan` 但 step loss 正常 | 已修（同上，gpt.py 安全 loss 聚合） |
| GPU 利用率 < 50% | 数据加载瓶颈：调大 `num_workers` (4-8)，开 `persistent_workers: true` |
| 训练比预期慢 5 倍 | 看是不是跑成 fp32 了（应该是 bf16）；查 YAML 的 `dtype` |
| tmux 找不到 | `apt-get install -y tmux` |
| ssh 重连 prompt 一直转圈 | AutoDL 节点卡住，控制台强制重启实例（不丢 `/autodl-tmp`） |
| 续训 LR 不连续 | 已修，scheduler state 包含在 `_resume.pth` |
| `_resume.pth` 损坏（比如训了一半磁盘满） | 用上一个 epoch 的 `epoch{N}.pth` 手动覆盖：`cp outputs/sft/epoch3.pth outputs/sft/_resume.pth` |
| AutoDL 余额报警 | 立即 `bash scripts/autodl/save_outputs.sh base --light` 救命备份 |

---

## 命令速查表

```bash
# === 一次性完整流程（base 为例）===
bash scripts/autodl/preflight.sh --profile base
bash scripts/autodl/launch.sh   tiny  all              # 5 min 冒烟
bash scripts/autodl/launch.sh   small all              # 30 min 验证
bash scripts/autodl/launch.sh   base  all              # 12-18h 正式
bash scripts/autodl/save_outputs.sh base               # 归档
# scp tarball 到本地，校验 sha256，关机

# === 状态检查（无副作用）===
bash scripts/autodl/status.sh                          # 一次性看
bash scripts/autodl/status.sh --watch                  # 每 5s 刷新
tmux ls                                                # 看活跃 session
tmux attach -t clearmind-base-all                      # 重连 tmux
nvidia-smi -l 1                                        # GPU 实时

# === 故障恢复 ===
bash scripts/autodl/launch.sh base all                 # 自动从 _resume.pth 续
bash scripts/autodl/launch.sh base all --kill          # 中止
bash scripts/autodl/launch.sh base all --foreground    # 前台调试
```

---

## 文件清单

| 文件 | 用途 |
|---|---|
| `scripts/autodl/preflight.sh` | 上线前自检（环境/数据/单元测试/冒烟） |
| `scripts/autodl/launch.sh` | 抗断连训练启动器（tmux + log tee） |
| `scripts/autodl/status.sh` | 训练状态查询 |
| `scripts/autodl/save_outputs.sh` | 归档训练成果（tar.gz + sha256 + manifest） |
| `scripts/autodl_train.sh` | 旧版前台训练脚本（仍可用，无断连保护） |
| `scripts/smoke_test.py` | 端到端冒烟（tiny + 100 行 SFT） |
| `scripts/download_data.py` | 数据下载（按 profile） |
| `evaluate/eval_*.py` | 评估脚本 |
| `scripts/push_to_hub.py` / `push_to_modelscope.py` | 发布脚本 |

---

## 一句话总结

```
preflight → launch tiny → launch small → launch base → save → 下载校验 → 关机
```

**断 SSH 就断，训练在 tmux 里跑；进程死就死，`_resume.pth` 自动续；磁盘满就慢；余额尽就完蛋——所以充够钱、勾关机不计费、装 tmux、跑 preflight，剩下交给脚本。**
