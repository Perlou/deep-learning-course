# 🖥️ AutoDL 租赁与训练攻略

> 从零开始在 AutoDL 平台训练 ClearMind-Plus (~468M) 大语言模型的完整指南。

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

## Step 3: 上传代码

实例启动后, 有 3 种方式上传项目:

### 方式一: Git Clone (推荐 ✅)

```bash
# SSH 连接到 AutoDL 服务器
ssh -p <端口号> root@<服务器IP>

# 进入工作目录
cd /root/autodl-tmp

# Clone 项目
git clone https://github.com/Perlou/deep-learning-course.git
cd deep-learning-course/projects/capstone-llm-from-scratch
```

### 方式二: AutoDL 文件上传

打开 AutoDL 控制台 → 文件管理 → 上传整个项目文件夹到 `/root/autodl-tmp/`

### 方式三: SCP 命令

```bash
# 在本地执行
scp -P <端口号> -r capstone-llm-from-scratch root@<服务器IP>:/root/autodl-tmp/
```

---

## Step 4: 环境配置

```bash
# SSH 连接到服务器后
cd /root/autodl-tmp/deep-learning-course/projects/capstone-llm-from-scratch

# 安装依赖
pip install -r requirements.txt

# 验证 GPU
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
PyTorch:  2.1.x
CUDA:     True
GPU:      NVIDIA A800-SXM4-80GB
显存:     80 GB
BFloat16: True
```

---

## Step 5: 准备数据

```bash
# 下载真实大规模数据集 (AutoDL 服务器网速很快)
python scripts/prepare_data.py --scale large
```

> 如果 HuggingFace 下载慢, AutoDL 支持 HF 镜像加速:
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> python scripts/prepare_data.py --scale large
> ```

---

## Step 6: 一键训练

### 方式一: 一键脚本 (推荐)

```bash
# 全流程: 分词器 → 预训练 → SFT → DPO
bash scripts/autodl_train.sh large
```

### 方式二: 分步执行

```bash
# 1. 训练分词器
python scripts/train_tokenizer.py --config configs/large.yaml

# 2. 预训练 (最耗时, ~12-20 小时)
python scripts/train.py --stage pretrain --config configs/large.yaml

# 3. SFT 指令微调 (~1-2 小时)
python scripts/train.py --stage sft --config configs/large.yaml

# 4. DPO 偏好对齐 (~0.5-1 小时)
python scripts/train.py --stage dpo --config configs/large.yaml
```

### 方式三: 后台运行 (防止 SSH 断连)

```bash
# 使用 tmux 保持会话
tmux new -s train

# 在 tmux 内执行训练
bash scripts/autodl_train.sh large

# 断开 tmux (训练继续): Ctrl+B, 然后按 D
# 重新连接: tmux attach -t train
```

> ⚠️ **强烈建议用 tmux!** SSH 断连不会中断训练。

---

## Step 7: 监控训练

```bash
# 实时查看 GPU 使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f outputs/pretrain/training.log

# 查看显存占用 (预期 ~50-60 GB)
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
# 一键对比各阶段 PPL
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
# 在 AutoDL 服务器上执行
pip install huggingface_hub

# 登录 (需提前在 https://huggingface.co/settings/tokens 创建 Token)
huggingface-cli login

# 上传模型
huggingface-cli upload Perlou/ClearMind-Plus outputs/dpo/final.pth
huggingface-cli upload Perlou/ClearMind-Plus outputs/tokenizer/tokenizer.model
```

上传后任何人都可以在 `https://huggingface.co/Perlou/ClearMind-Plus` 查看和下载你的模型。

### 方式四: GitHub Releases

1. 进入你的 GitHub 仓库 → Releases → Create new release
2. 上传 `final.pth` 和 `tokenizer.model` 作为附件
3. 适合 < 2GB 的单个文件

### 下载后在本地使用

```bash
# 确保 outputs/ 目录结构如下:
# outputs/tokenizer/tokenizer.model
# outputs/dpo/final.pth

# 直接对话
python scripts/chat.py --config configs/large.yaml

# 评估效果
python evaluate/eval_benchmark.py --config configs/large.yaml
```

---

## Step 10: 关闭实例

> ⚠️ **训练完成后务必关机!** AutoDL 按秒计费, 忘记关机会持续扣费。

1. ✅ 确认模型已下载/上传保存
2. AutoDL 控制台 → **关机** (数据保留, 下次可继续用) 或 **释放** (彻底删除, 停止一切费用)

> **关机 vs 释放**: 关机后数据盘保留 (但会收少量存储费), 释放后所有数据永久删除。建议确认模型已保存后直接**释放**。

---

## 💡 省钱技巧

| 技巧                           | 节省                    |
| ------------------------------ | ----------------------- |
| 先用 Tiny/Small 在本地验证流程 | 避免浪费 GPU 时间调 bug |
| 用 `tmux` 防断连               | 避免重跑浪费            |
| 不训练时立即关机               | 停止计费                |
| 选择竞价实例                   | 降低约 50% 费用         |
| 选择凌晨时段训练               | GPU 空闲, 不容易排队    |
| 只下载 `final.pth`             | checkpoint 文件可不下载 |

---

## ❓ 常见问题

**Q: 显存不够 (OOM) 怎么办?**

```bash
# 减小 batch_size, 增大 gradient_accumulation
python scripts/train.py --stage pretrain --config configs/large.yaml --batch_size 12 --gradient_accumulation 16
```

**Q: 训练中断了怎么办?**

```bash
# 从最近的 checkpoint 恢复
python scripts/train.py --stage pretrain --config configs/large.yaml --resume outputs/pretrain/checkpoint_latest.pth
```

**Q: HuggingFace 下载慢?**

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 如何在本地使用训练好的模型?**

```bash
# 下载 outputs/ 后在本地运行
python scripts/chat.py --config configs/large.yaml
```
