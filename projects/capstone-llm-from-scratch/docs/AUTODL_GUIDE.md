# 🖥️ AutoDL 租赁与训练攻略

> 从零开始在 AutoDL 平台训练 ClearMind 大语言模型的完整指南。

---

## 📋 总览

| 项目         | 说明                                   |
| ------------ | -------------------------------------- |
| **模型**     | ClearMind-Plus, ~468M 参数             |
| **GPU**      | A800-80GB (或 A100-80GB)               |
| **预估费用** | ¥15-30 (约 12-24 小时 × ¥1-2/小时)     |
| **预估时间** | 预训练 ~12-20h, SFT ~1-2h, DPO ~0.5-1h |

---

## Step 1: 注册 AutoDL

1. 打开 [AutoDL 官网](https://www.autodl.com) → 注册账号
2. 充值余额 (建议先充 **¥50**，用不完可退)
3. 实名认证 (部分 GPU 需实名)

---

## Step 2: 租赁 GPU 服务器

### 选择配置

进入 [算力市场](https://www.autodl.com/market/list)：

| 配置项     | 推荐选择                 | 说明                       |
| ---------- | ------------------------ | -------------------------- |
| **GPU**    | A800-80GB-NVLink         | 与 A100 性能相同, 价格更低 |
| **数量**   | 1 卡                     | 468M 模型单卡足够          |
| **镜像**   | PyTorch 2.1+ / CUDA 12.x | 预装好环境                 |
| **数据盘** | 默认免费 50GB            | 够用                       |

> 💡 **省钱技巧**: 选择"竞价实例"可降低约 50% 费用, 但可能被中断。短时间训练建议用按量计费。

### 创建实例

1. 点击 **创建实例**
2. GPU 选择 `A800-80GB` × 1
3. 镜像选择 `PyTorch 2.1.0` + `Python 3.10` + `CUDA 12.1`
4. 点击 **立即创建**

---

## Step 3: 开启网络加速 ⚡

> ⚠️ **重要!** AutoDL 服务器直连 GitHub / HuggingFace 极慢甚至超时。务必先开启加速。

```bash
# SSH 连接到 AutoDL 服务器
ssh -p <端口号> root@<服务器IP>

# ====== 方法 1: AutoDL 学术加速 (推荐 ✅) ======
# 大部分 AutoDL 镜像内置了网络加速脚本, 启用后 GitHub/HuggingFace 都会变快
source /etc/network_turbo

# ====== 方法 2: 手动设置 HuggingFace 镜像 ======
# 如果没有 network_turbo, 手动设置也可以
export HF_ENDPOINT=https://hf-mirror.com

# 建议写入 bashrc, 每次登录自动生效
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 4: 上传代码

### 方式一: Git Clone (推荐 ✅)

```bash
cd /root/autodl-tmp

# 浅克隆 (最快, 只下载最新版本)
git clone --depth 1 https://github.com/Perlou/deep-learning-course.git
cd deep-learning-course/projects/capstone-llm-from-scratch
```

> **Git Clone 很慢?** 试试以下方案:
>
> ```bash
> # 方案 1: 先启用学术加速
> source /etc/network_turbo
> git clone --depth 1 https://github.com/Perlou/deep-learning-course.git
>
> # 方案 2: 强制 HTTP/1.1
> git config --global http.version HTTP/1.1
> git clone --depth 1 https://github.com/Perlou/deep-learning-course.git
>
> # 方案 3: 用 gitclone 镜像
> git clone --depth 1 https://gitclone.com/github.com/Perlou/deep-learning-course.git
> ```

### 方式二: SCP 上传

```bash
# 在本地执行
scp -P <端口号> -r capstone-llm-from-scratch root@<服务器IP>:/root/autodl-tmp/
```

### 方式三: AutoDL 文件管理器

AutoDL 控制台 → 文件管理 → 上传整个项目文件夹到 `/root/autodl-tmp/`

---

## Step 5: 环境配置

```bash
cd /root/autodl-tmp/deep-learning-course/projects/capstone-llm-from-scratch

# 安装依赖
pip install -r requirements.txt

# 验证 GPU 环境
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
print(f'GPU:      {torch.cuda.get_device_name(0)}')
print(f'显存:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')
print(f'BFloat16: {torch.cuda.is_bf16_supported()}')
"
```

预期输出:

```
PyTorch:  2.x.x
CUDA:     True
GPU:      NVIDIA A800-SXM4-80GB
显存:     80 GB
BFloat16: True
```

---

## Step 6: 一键训练

### 方式一: 交互式启动脚本 (推荐 ✅)

```bash
bash run.sh
```

脚本会引导你选择:

1. **模型规模** — Tiny / Small / Medium / Large
2. **训练流程** — 全流程、仅预训练、从 SFT/DPO 继续等

> 💡 建议先选 **Tiny** 跑一遍全流程 (2-5 分钟) 验证环境无问题, 再切到 Large。

### 方式二: 自动化训练脚本

```bash
# 全流程: 数据 → 分词器 → 预训练 → SFT → DPO
bash scripts/autodl_train.sh large

# 只跑某个阶段
bash scripts/autodl_train.sh large pretrain
bash scripts/autodl_train.sh large sft
bash scripts/autodl_train.sh large dpo
```

### 方式三: 手动分步执行

```bash
# 1. 准备数据
python scripts/prepare_data.py --scale large

# 2. 训练分词器
python scripts/train_tokenizer.py --config configs/large.yaml

# 3. 预训练 (最耗时, ~12-20 小时)
python scripts/train.py --stage pretrain --config configs/large.yaml

# 4. SFT 指令微调 (~1-2 小时)
python scripts/train.py --stage sft --config configs/large.yaml

# 5. DPO 偏好对齐 (~0.5-1 小时)
python scripts/train.py --stage dpo --config configs/large.yaml
```

### 方式四: 多卡 DDP 预训练（可选）

当你租用多卡实例（如 2×A100 或 4×A800）时，可用 `torchrun` 提升预训练吞吐：

```bash
# 示例: 4 卡 DDP 预训练
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/large.yaml \
  --data data/pretrain/pretrain_data.jsonl \
  --tokenizer outputs/tokenizer/tokenizer.model \
  --output_dir outputs/pretrain_ddp
```

断点续训：

```bash
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/large.yaml \
  --resume outputs/pretrain_ddp/checkpoint_step1000.pth
```

### ⚠️ 用 tmux 防止 SSH 断连

```bash
# 创建 tmux 会话
tmux new -s train

# 在 tmux 里执行训练
bash run.sh

# 断开会话 (训练继续): Ctrl+B, 然后按 D
# 重新连接: tmux attach -t train
```

> ⚠️ **务必用 tmux!** 否则 SSH 断连会导致训练中断, 之前的进度全部浪费。

---

## Step 7: 监控训练

```bash
# 实时查看 GPU 使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f outputs/pretrain/training.log

# 查看显存占用
nvidia-smi --query-gpu=memory.used --format=csv
```

### 训练进度参考

| 阶段   | 步数          | 预计时长   | 显存占用 |
| ------ | ------------- | ---------- | -------- |
| 预训练 | 100,000 steps | 12-20 小时 | ~55 GB   |
| SFT    | 2 epochs      | 1-2 小时   | ~45 GB   |
| DPO    | 1 epoch       | 0.5-1 小时 | ~60 GB   |

---

## Step 8: 评估模型

```bash
# 困惑度对比各阶段
python evaluate/eval_perplexity.py --compare

# 综合评估
python evaluate/eval_benchmark.py --config configs/large.yaml

# 交互式对话测试
python scripts/chat.py --config configs/large.yaml
```

---

## Step 9: 保存训练成果

### 需要保存的文件

| 文件                                | 大小    | 重要性      | 说明                      |
| ----------------------------------- | ------- | ----------- | ------------------------- |
| `outputs/tokenizer/tokenizer.model` | ~500 KB | ⭐⭐⭐ 必需 | 分词器, 推理必需          |
| `outputs/dpo/final.pth`             | ~1.8 GB | ⭐⭐⭐ 必需 | **最终模型** (DPO 对齐后) |
| `outputs/sft/final.pth`             | ~1.8 GB | ⭐⭐ 推荐   | SFT 阶段模型 (对比用)     |
| `outputs/pretrain/final.pth`        | ~1.8 GB | ⭐ 可选     | 预训练模型 (对比用)       |

> 💡 **最小保存**: 只保留 `tokenizer.model` + `dpo/final.pth` (~1.8 GB) 即可完整运行对话推理。

### 方式一: SCP 下载到本地 (推荐 ✅)

```bash
# 在本地执行 — 只下载最终模型
scp -P <端口号> root@<服务器IP>:/root/autodl-tmp/deep-learning-course/projects/capstone-llm-from-scratch/outputs/tokenizer/tokenizer.model ./outputs/tokenizer/
scp -P <端口号> root@<服务器IP>:/root/autodl-tmp/deep-learning-course/projects/capstone-llm-from-scratch/outputs/dpo/final.pth ./outputs/dpo/

# 或下载全部模型
scp -P <端口号> -r root@<服务器IP>:/root/autodl-tmp/deep-learning-course/projects/capstone-llm-from-scratch/outputs/ ./outputs/
```

### 方式二: AutoDL 文件管理器

AutoDL 控制台 → 文件管理 → 找到 `outputs/` 目录 → 逐个下载

### 方式三: 上传到 HuggingFace Hub (展示 + 分享)

```bash
pip install huggingface_hub

# 登录 (需提前在 https://huggingface.co/settings/tokens 创建 Token)
huggingface-cli login

# 上传模型
huggingface-cli upload Perlou/ClearMind-Plus outputs/dpo/final.pth
huggingface-cli upload Perlou/ClearMind-Plus outputs/tokenizer/tokenizer.model
```

### 下载后在本地使用

```bash
# 确保 outputs/ 目录结构如下:
# outputs/tokenizer/tokenizer.model
# outputs/dpo/final.pth

python scripts/chat.py --config configs/large.yaml
```

---

## Step 10: 关闭实例

> ⚠️ **训练完成后务必关机!** AutoDL 按秒计费, 忘记关机会持续扣费。

1. ✅ 确认模型已下载/上传保存
2. AutoDL 控制台 → **关机** (数据保留) 或 **释放** (彻底删除)

> **关机 vs 释放**: 关机后数据盘保留 (但收少量存储费), 释放后所有数据永久删除。确认模型已保存后建议直接**释放**。

---

## 💡 省钱技巧

| 技巧                             | 效果                    |
| -------------------------------- | ----------------------- |
| 本地先用 Tiny 验证全流程         | 避免在 GPU 上调 bug     |
| `source /etc/network_turbo`      | GitHub/HF 下载提速 10x+ |
| 用 `tmux` 防断连                 | 避免重跑浪费            |
| 不训练时立即关机                 | 停止计费                |
| 选择竞价实例                     | 降低约 50% 费用         |
| 凌晨时段训练                     | GPU 空闲, 不易排队      |
| 只下载 `final.pth` + `tokenizer` | checkpoint 可不保留     |

---

## ❓ 常见问题

**Q: GitHub clone 失败或很慢?**

```bash
# 启用学术加速
source /etc/network_turbo

# 浅克隆
git clone --depth 1 https://github.com/Perlou/deep-learning-course.git

# 如果还不行, 用镜像
git clone --depth 1 https://gitclone.com/github.com/Perlou/deep-learning-course.git
```

**Q: HuggingFace 数据集下载失败?**

```bash
# 设置 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_data.py --scale large
```

**Q: 显存不够 (OOM)?**

```bash
# 减小 batch_size, 增大 gradient_accumulation 保持等效 batch 不变
python scripts/train.py --stage pretrain --config configs/large.yaml \
    --batch_size 12 --gradient_accumulation 16
```

**Q: 训练中断了怎么恢复?**

```bash
# 从已有 checkpoint 恢复
python scripts/train.py --stage pretrain --config configs/large.yaml \
    --resume outputs/pretrain/checkpoint_step1000.pth
```

**Q: `torch.amp.GradScaler` 报错?**

AutoDL 镜像自带的 PyTorch 版本可能较旧 (< 2.1)。项目已内置兼容处理, 如遇到请 `git pull` 更新到最新代码。

**Q: `run.sh` 按 Enter 后没反应?**

确保使用最新代码 (`git pull`)。旧版有一个 `set -e` + 算术运算的兼容性 bug, 已修复。

**Q: MNBVC 中文数据集加载失败?**

这是因为 MNBVC 使用了旧版自定义加载脚本, 新版 `datasets` 库不再支持。不影响训练 — TinyStories 英文数据足够跑通全流程。
