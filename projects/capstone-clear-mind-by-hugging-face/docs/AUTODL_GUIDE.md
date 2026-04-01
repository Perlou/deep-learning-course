# AutoDL 训练指南

> ClearMind-HF 在 AutoDL GPU 服务器上的训练指南

## 1. 环境准备

### 1.1 选择实例

| 配置 | 推荐 GPU | 显存 | 适用模型 |
|------|---------|------|---------|
| ClearMind-Tiny | 任意 GPU | 4GB+ | 验证流程 |
| ClearMind-Mini | RTX 3090 / A10 | 8GB+ | 小规模实验 |
| ClearMind | A100-40G | 24GB+ | 正式训练 |
| ClearMind-Plus | A100-80G / 多卡 | 48GB+ | 大规模训练 |

### 1.2 创建环境

```bash
# 克隆项目
git clone <repo-url>
cd capstone-clear-mind-by-hugging-face

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## 2. 数据准备

```bash
# 下载并准备数据 (small/medium/large)
python scripts/prepare_data.py --scale medium --data_dir data/

# 训练 tokenizer
python scripts/train_tokenizer.py --config configs/small.yaml --corpus data/pretrain/pretrain_data.jsonl
```

## 3. 训练流程

### 3.1 预训练

```bash
# 单卡
python scripts/train.py --stage pretrain --config configs/small.yaml

# 多卡 (accelerate)
accelerate launch --multi_gpu --num_processes 2 scripts/train.py --stage pretrain --config configs/small.yaml
```

### 3.2 SFT 微调

```bash
# 全参微调
python scripts/train.py --stage sft --config configs/small.yaml --resume outputs/pretrain

# LoRA 微调 (节省显存)
python scripts/train.py --stage sft --config configs/small.yaml --resume outputs/pretrain --use_lora
```

### 3.3 DPO 对齐

```bash
python scripts/train.py --stage dpo --config configs/small.yaml --resume outputs/sft
```

## 4. 评估

```bash
# 综合评估
python evaluate/eval_benchmark.py --tokenizer outputs/tokenizer

# 困惑度
python evaluate/eval_perplexity.py --model outputs/sft --tokenizer outputs/tokenizer

# 生成质量
python evaluate/eval_generation.py --model outputs/sft --tokenizer outputs/tokenizer
```

## 5. AutoDL 注意事项

### 5.1 数据持久化

AutoDL 实例重启后 `/root/` 数据不丢失，但建议将重要输出保存到 `/root/autodl-fs/`：

```bash
cp -r outputs/ /root/autodl-fs/clearmind-outputs/
```

### 5.2 bf16 训练

AutoDL 上的 A100/A10 支持 bf16，修改配置文件：

```yaml
pretrain:
  bf16: true  # 在 GPU 上启用 bf16
```

### 5.3 显存不足

- 降低 `per_device_train_batch_size`
- 启用 `gradient_checkpointing: true`
- 使用 LoRA 微调 (`--use_lora`)
- 使用 QLoRA (4-bit 量化 + LoRA)

### 5.4 断点续训

```bash
# 从 checkpoint 恢复
python scripts/train.py --stage pretrain --config configs/small.yaml --resume outputs/pretrain/checkpoint-100
```
