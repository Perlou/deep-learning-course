#!/bin/bash
# =========================================================
# AutoDL 一键部署训练脚本
# =========================================================
#
# 在 AutoDL 服务器上执行完整的训练流程
#
# 使用方法:
#   # 上传项目到服务器后:
#   bash scripts/autodl_train.sh small     # 小规模快速验证
#   bash scripts/autodl_train.sh medium    # 中等规模训练
#   bash scripts/autodl_train.sh large     # 大规模 A100 训练
#
#   # 只运行某个阶段:
#   bash scripts/autodl_train.sh medium pretrain
#   bash scripts/autodl_train.sh large sft
#
# 前提:
#   - AutoDL 服务器已启动 (推荐 A100 80GB)
#   - 已上传项目文件到 /root/autodl-tmp/minimind/
#   - Python 环境已安装 PyTorch
# =========================================================

set -e  # 遇错即停

# ===================== 参数解析 =====================
SCALE=${1:-small}            # small / medium / large
STAGE=${2:-all}              # all / pretrain / sft / dpo / chat
PROJECT_DIR="/root/autodl-tmp/minimind"
CONFIG="configs/${SCALE}.yaml"

echo "=========================================="
echo "  ClearMind AutoDL 训练"
echo "=========================================="
echo "  规模:     $SCALE"
echo "  阶段:     $STAGE"
echo "  配置:     $CONFIG"
echo "  项目目录: $PROJECT_DIR"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  VRAM:     $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

cd "$PROJECT_DIR"

# ===================== 环境检查 =====================
echo ""
echo "🔧 Step 0: 环境检查..."

# 安装依赖
pip install -q -r requirements.txt

# 检查 PyTorch + CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:     {torch.cuda.get_device_name(0)}')
    print(f'VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# ===================== 数据准备 =====================
if [ "$STAGE" == "all" ] || [ "$STAGE" == "data" ]; then
    echo ""
    echo "📦 Step 1: 数据准备..."

    if [ "$SCALE" == "small" ]; then
        python scripts/01_prepare_data.py --data_dir data
    else
        python scripts/download_dataset.py --scale "$SCALE" --data_dir data
    fi
fi

# ===================== 分词器 =====================
if [ "$STAGE" == "all" ] || [ "$STAGE" == "tokenizer" ]; then
    echo ""
    echo "📝 Step 2: 训练分词器..."
    python scripts/02_train_tokenizer.py --config "$CONFIG"
fi

# ===================== 预训练 =====================
if [ "$STAGE" == "all" ] || [ "$STAGE" == "pretrain" ]; then
    echo ""
    echo "🚀 Step 3: 预训练..."

    # 根据规模设置训练步数
    case $SCALE in
        small)  MAX_STEPS=5000 ;;
        medium) MAX_STEPS=50000 ;;
        large)  MAX_STEPS=100000 ;;
    esac

    python scripts/03_pretrain.py \
        --config "$CONFIG" \
        --max_steps "$MAX_STEPS" \
        --log_every 50

    echo "✅ 预训练完成!"
fi

# ===================== SFT =====================
if [ "$STAGE" == "all" ] || [ "$STAGE" == "sft" ]; then
    echo ""
    echo "📚 Step 4: SFT 指令微调..."

    python scripts/04_sft.py \
        --config "$CONFIG" \
        --pretrained outputs/pretrain/final.pth \
        --log_every 20

    echo "✅ SFT 完成!"
fi

# ===================== DPO =====================
if [ "$STAGE" == "all" ] || [ "$STAGE" == "dpo" ]; then
    echo ""
    echo "🎯 Step 5: DPO 对齐训练..."

    python scripts/05_dpo.py \
        --config "$CONFIG" \
        --sft_model outputs/sft/final.pth \
        --log_every 10

    echo "✅ DPO 完成!"
fi

# ===================== 评估 =====================
if [ "$STAGE" == "all" ]; then
    echo ""
    echo "📊 Step 6: 评估..."

    echo "--- 预训练模型 ---"
    python evaluate/eval_perplexity.py --model outputs/pretrain/final.pth --config "$CONFIG" 2>/dev/null || true

    echo "--- SFT 模型 ---"
    python evaluate/eval_perplexity.py --model outputs/sft/final.pth --config "$CONFIG" 2>/dev/null || true

    echo "--- DPO 模型 ---"
    python evaluate/eval_perplexity.py --model outputs/dpo/final.pth --config "$CONFIG" 2>/dev/null || true
fi

# ===================== 完成 =====================
echo ""
echo "=========================================="
echo "  ✅ 训练流程完成!"
echo "=========================================="
echo ""
echo "模型文件:"
ls -lh outputs/*/final.pth 2>/dev/null || echo "  (无模型文件)"
echo ""
echo "下一步:"
echo "  对话测试: python scripts/06_chat.py --config $CONFIG"
echo "  下载模型: 从 AutoDL 文件管理器下载 outputs/ 目录"
echo "=========================================="
