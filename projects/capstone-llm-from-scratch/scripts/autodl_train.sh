#!/bin/bash
# =========================================================
# AutoDL 一键训练脚本（ClearMind-Base / ClearMind-Plus）
# =========================================================
#
# 在 AutoDL A100/A800 80GB 上跑完整的 Pretrain → SFT → DPO 流程。
#
# 用法:
#   bash scripts/autodl_train.sh base                    # ClearMind-Base (~64M, ~12-18h)
#   bash scripts/autodl_train.sh plus                    # ClearMind-Plus (~478M, ~30-40h)
#   bash scripts/autodl_train.sh small                   # 单卡小规模测试
#   bash scripts/autodl_train.sh tiny                    # CPU/MPS 冒烟（不推荐 GPU）
#
#   bash scripts/autodl_train.sh plus pretrain           # 仅跑预训练
#   bash scripts/autodl_train.sh plus sft                # 仅跑 SFT
#   bash scripts/autodl_train.sh plus dpo                # 仅跑 DPO
#   bash scripts/autodl_train.sh plus all                # 全流程（默认）
#
# 前置:
#   1. AutoDL 服务器启动（推荐 A100/A800 80GB），项目放在 /root/autodl-tmp/clearmind/
#   2. 已 pip install -r requirements.txt
#   3. 已下载 minimind 数据到 data/（见下方下载命令）
# =========================================================

set -euo pipefail

# ===================== 参数解析 =====================
SCALE="${1:-base}"           # tiny / small / base / plus
STAGE="${2:-all}"            # all / pretrain / sft / dpo
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

case "$SCALE" in
    tiny)  CONFIG="configs/tiny.yaml" ;;
    small) CONFIG="configs/small.yaml" ;;
    base)  CONFIG="configs/main.yaml" ;;
    plus)  CONFIG="configs/plus.yaml" ;;
    *)
        echo "❌ 未知规格: $SCALE（应为 tiny/small/base/plus）"
        exit 1
        ;;
esac

cd "$PROJECT_DIR"

echo "=========================================="
echo "  ClearMind AutoDL 训练"
echo "=========================================="
echo "  规格:     $SCALE → $CONFIG"
echo "  阶段:     $STAGE"
echo "  项目:     $PROJECT_DIR"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"
echo "  VRAM:     $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# ===================== 环境检查 =====================
echo ""
echo "🔧 Step 0: 环境检查"

# 自动安装依赖（首次运行）
if ! python -c "import transformers" 2>/dev/null; then
    echo "  📦 安装依赖..."
    pip install -q -r requirements.txt
fi

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PY

# ===================== 数据检查（不再自动生成）=====================
echo ""
echo "📦 Step 1: 数据检查"

REQUIRED_FILES=()
[[ "$STAGE" == "all" || "$STAGE" == "pretrain" ]] && REQUIRED_FILES+=("data/pretrain_t2t_mini.jsonl")
[[ "$STAGE" == "all" || "$STAGE" == "sft" ]]      && REQUIRED_FILES+=("data/sft_t2t_mini.jsonl")
[[ "$STAGE" == "all" || "$STAGE" == "dpo" ]]      && REQUIRED_FILES+=("data/dpo.jsonl")

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ❌ 缺少: $f"
        MISSING=1
    else
        echo "  ✅ 已就位: $f ($(du -h "$f" | cut -f1))"
    fi
done

if [ "$MISSING" -eq 1 ]; then
    cat <<'EOF'

  快速下载（按训练规格选 profile）：

    # ClearMind-Base 完整训练（~2.9 GB）
    python scripts/download_data.py --profile base

    # ClearMind-Plus 旗舰完整训练（~24 GB）
    python scripts/download_data.py --profile plus

    # 仅快速复现（mini 数据，~2.8 GB）
    python scripts/download_data.py --profile zero

    # 含 RL 数据
    python scripts/download_data.py --profile rl

    # 查看所有 profile
    python scripts/download_data.py --list

EOF
    exit 1
fi

# Tokenizer 检查
if [ ! -f "tokenizer/minimind/tokenizer.json" ]; then
    echo "  ❌ 缺少 tokenizer/minimind/tokenizer.json（应该随仓库自带）"
    exit 1
fi
echo "  ✅ Tokenizer 就位: tokenizer/minimind/"

# ===================== Pretrain =====================
if [[ "$STAGE" == "all" || "$STAGE" == "pretrain" ]]; then
    echo ""
    echo "🚀 Step 2: Pretrain"
    python scripts/train.py --stage pretrain --config "$CONFIG"
    echo "✅ Pretrain 完成 → outputs/pretrain/final.pth"
fi

# ===================== SFT =====================
if [[ "$STAGE" == "all" || "$STAGE" == "sft" ]]; then
    echo ""
    echo "📚 Step 3: SFT"
    if [ ! -f "outputs/pretrain/final.pth" ]; then
        echo "  ❌ 没有 outputs/pretrain/final.pth；先跑 pretrain 阶段"
        exit 1
    fi
    python scripts/train.py --stage sft --config "$CONFIG" \
        --resume outputs/pretrain/final.pth
    echo "✅ SFT 完成 → outputs/sft/final.pth"
fi

# ===================== DPO =====================
if [[ "$STAGE" == "all" || "$STAGE" == "dpo" ]]; then
    echo ""
    echo "🎯 Step 4: DPO"
    if [ ! -f "outputs/sft/final.pth" ]; then
        echo "  ❌ 没有 outputs/sft/final.pth；先跑 sft 阶段"
        exit 1
    fi
    python scripts/train.py --stage dpo --config "$CONFIG" \
        --resume outputs/sft/final.pth
    echo "✅ DPO 完成 → outputs/dpo/final.pth"
fi

# ===================== 完成 =====================
echo ""
echo "=========================================="
echo "  ✅ 训练流程完成"
echo "=========================================="
echo ""
echo "产物:"
ls -lh outputs/*/final.pth 2>/dev/null || echo "  (暂无)"
echo ""
echo "下一步:"
echo "  对话:     python scripts/chat.py --config $CONFIG"
echo "  下载:     从 AutoDL 文件管理器下载 outputs/ 目录"
echo "  发布:     (Phase 5) python scripts/convert_to_qwen3.py + push_to_hub.py"
echo "=========================================="
