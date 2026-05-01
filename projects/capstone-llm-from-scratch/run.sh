#!/bin/bash
# ============================================================
# ClearMind — 一键训练脚本（基于 minimind 数据/tokenizer 的双发布矩阵版）
# ============================================================
# 使用方法: bash run.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ---- 颜色 ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

PYTHON="${PYTHON:-./venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi

# ---- Banner ----
echo ""
echo -e "${BOLD}  ============================================================${NC}"
echo -e "${BOLD}    🧠 ClearMind   ${NC}"
echo -e "${DIM}    双发布矩阵：Base (68.8M) + Plus (486.3M)${NC}"
echo -e "${BOLD}  ============================================================${NC}"

# ---- 前置检查：tokenizer ----
if [ ! -f "tokenizer/minimind/tokenizer.json" ]; then
    echo -e "${RED}  ❌ 找不到 tokenizer/minimind/tokenizer.json${NC}"
    echo -e "${DIM}     仓库自带 tokenizer 应该在 tokenizer/minimind/ 目录${NC}"
    exit 1
fi

# ---- 前置检查：数据（仅训练流程需要）----
check_data() {
    local stage="$1"
    case "$stage" in
        pretrain)
            if [ ! -f "data/pretrain_t2t_mini.jsonl" ]; then
                echo -e "${RED}  ❌ 缺少 data/pretrain_t2t_mini.jsonl${NC}"
                echo -e "${DIM}     从 https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files 下载${NC}"
                echo -e "${DIM}     或: huggingface-cli download jingyaogong/minimind_dataset --repo-type dataset \\${NC}"
                echo -e "${DIM}              --include pretrain_t2t_mini.jsonl --local-dir data/${NC}"
                exit 1
            fi
            ;;
        sft)
            if [ ! -f "data/sft_t2t_mini.jsonl" ]; then
                echo -e "${RED}  ❌ 缺少 data/sft_t2t_mini.jsonl${NC}"
                echo -e "${DIM}     从 https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files 下载${NC}"
                exit 1
            fi
            ;;
        dpo)
            if [ ! -f "data/dpo.jsonl" ]; then
                echo -e "${RED}  ❌ 缺少 data/dpo.jsonl${NC}"
                echo -e "${DIM}     从 https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files 下载${NC}"
                exit 1
            fi
            ;;
    esac
}

# ---- Step 1: 选择配置 ----
echo ""
echo -e "${CYAN}${BOLD}  📦 选择模型规格:${NC}"
echo ""
echo -e "  ${BOLD}1)${NC} Tiny       (~1.5M  参数,  CPU/MPS,  冒烟)"
echo -e "  ${BOLD}2)${NC} Small      (~26M   参数,  单卡,   对齐 minimind2-small)"
echo -e "  ${BOLD}3)${NC} Base       (~68.8M 参数,  A100/A800,  ${BOLD}对标 minimind-3 dense${NC})"
echo -e "  ${BOLD}4)${NC} Plus       (~486M  参数,  A100/A800 80GB,  ${BOLD}对标 minimind-3-moe${NC})"
echo ""
read -p "  请选择 [1-4, 默认 1]: " scale_choice

case "${scale_choice:-1}" in
    1) CONFIG="configs/tiny.yaml";  LABEL="Tiny";;
    2) CONFIG="configs/small.yaml"; LABEL="Small";;
    3) CONFIG="configs/main.yaml";  LABEL="Base (Main)";;
    4) CONFIG="configs/plus.yaml";  LABEL="Plus";;
    *) echo -e "${RED}  ❌ 无效选择${NC}"; exit 1;;
esac

echo -e "\n  ${GREEN}✅ 已选择: ${BOLD}${LABEL}${NC} ${DIM}(${CONFIG})${NC}"

# ---- Step 2: 选择训练流程 ----
echo ""
echo -e "${CYAN}${BOLD}  🔄 选择训练流程:${NC}"
echo ""
echo -e "  ${BOLD}1)${NC} 全流程     预训练 → SFT → DPO → 对话"
echo -e "  ${BOLD}2)${NC} 仅预训练   pretrain"
echo -e "  ${BOLD}3)${NC} 仅 SFT     (需已有预训练 final.pth)"
echo -e "  ${BOLD}4)${NC} 仅 DPO     (需已有 SFT final.pth)"
echo -e "  ${BOLD}5)${NC} 仅对话     (需已有训练好的模型)"
echo -e "  ${BOLD}6)${NC} 运行测试   pytest"
echo -e "  ${BOLD}7)${NC} 端到端冒烟 (tiny + 真实 minimind 数据 + 2 步预训练 + 100 条 SFT)"
echo -e "  ${BOLD}8)${NC} 下载数据   ${DIM}(从 ModelScope/HuggingFace 下载 minimind 数据集)${NC}"
echo ""
read -p "  请选择 [1-8, 默认 1]: " flow_choice
FLOW="${flow_choice:-1}"

echo ""
echo -e "${YELLOW}${BOLD}  即将开始执行, 按 Enter 继续 (Ctrl+C 取消)...${NC}"
read -r

echo ""
echo -e "${BOLD}  ============================================================${NC}"
echo -e "${BOLD}    🚀 开始执行${NC}"
echo -e "${BOLD}  ============================================================${NC}"
echo ""

# ---- 执行步骤 ----
step=0
run_step() {
    step=$((step + 1))
    echo -e "${YELLOW}  [${step}] ▶ $1 ...${NC}"
    eval "$2"
    echo -e "${GREEN}  [${step}] ✅ $1 完成${NC}\n"
}

case $FLOW in
    1)  # 全流程
        check_data pretrain
        check_data sft
        check_data dpo
        run_step "Pretrain"   "$PYTHON scripts/train.py --stage pretrain --config $CONFIG"
        run_step "SFT"        "$PYTHON scripts/train.py --stage sft      --config $CONFIG"
        run_step "DPO"        "$PYTHON scripts/train.py --stage dpo      --config $CONFIG"
        run_step "Chat"       "$PYTHON scripts/chat.py --config $CONFIG"
        ;;
    2)  # 仅预训练
        check_data pretrain
        run_step "Pretrain"   "$PYTHON scripts/train.py --stage pretrain --config $CONFIG"
        ;;
    3)  # 仅 SFT
        check_data sft
        run_step "SFT"        "$PYTHON scripts/train.py --stage sft      --config $CONFIG"
        ;;
    4)  # 仅 DPO
        check_data dpo
        run_step "DPO"        "$PYTHON scripts/train.py --stage dpo      --config $CONFIG"
        ;;
    5)  # 仅对话
        run_step "Chat"       "$PYTHON scripts/chat.py --config $CONFIG"
        ;;
    6)  # 测试
        run_step "Pytest"     "$PYTHON -m pytest tests/ -v"
        ;;
    7)  # 端到端冒烟
        check_data pretrain
        check_data sft
        run_step "Pretrain 2 steps" "rm -rf outputs/smoke_pretrain && $PYTHON scripts/train.py --stage pretrain --config configs/tiny.yaml --output_dir outputs/smoke_pretrain --max_steps 2 --log_every 1"
        run_step "Build SFT subset" "head -100 data/sft_t2t_mini.jsonl > /tmp/sft_smoke.jsonl"
        run_step "SFT 1 epoch"     "rm -rf outputs/smoke_sft && $PYTHON scripts/train.py --stage sft --config configs/tiny.yaml --data /tmp/sft_smoke.jsonl --resume outputs/smoke_pretrain/final.pth --output_dir outputs/smoke_sft --epochs 1 --batch_size 4 --log_every 4"
        ;;
    8)  # 下载数据
        echo ""
        echo -e "${CYAN}${BOLD}  📦 选择数据 profile:${NC}"
        echo ""
        echo -e "  ${BOLD}1)${NC} zero    (~2.8 GB)  快速复现 minimind-zero 对话模型"
        echo -e "  ${BOLD}2)${NC} base    (~2.9 GB)  ${BOLD}ClearMind-Base 完整训练${NC}"
        echo -e "  ${BOLD}3)${NC} plus    (~24 GB)   ${BOLD}ClearMind-Plus 旗舰完整训练${NC}"
        echo -e "  ${BOLD}4)${NC} rl      (~3.0 GB)  含 PPO/GRPO/CISPO/Agent RL 数据"
        echo -e "  ${BOLD}5)${NC} all     (~28 GB)   全部 8 个文件"
        echo ""
        read -p "  请选择 [1-5, 默认 1]: " profile_choice
        case "${profile_choice:-1}" in
            1) DL_PROFILE="zero" ;;
            2) DL_PROFILE="base" ;;
            3) DL_PROFILE="plus" ;;
            4) DL_PROFILE="rl" ;;
            5) DL_PROFILE="all" ;;
            *) echo -e "${RED}  ❌ 无效选择${NC}"; exit 1 ;;
        esac
        echo ""
        echo -e "${CYAN}${BOLD}  🌐 选择下载源:${NC}"
        echo ""
        echo -e "  ${BOLD}1)${NC} auto       (优先 ModelScope，回落 HuggingFace)"
        echo -e "  ${BOLD}2)${NC} modelscope ${DIM}(国内推荐)${NC}"
        echo -e "  ${BOLD}3)${NC} hf         (HuggingFace)"
        echo ""
        read -p "  请选择 [1-3, 默认 1]: " src_choice
        case "${src_choice:-1}" in
            1) DL_SOURCE="auto" ;;
            2) DL_SOURCE="modelscope" ;;
            3) DL_SOURCE="hf" ;;
            *) echo -e "${RED}  ❌ 无效选择${NC}"; exit 1 ;;
        esac
        run_step "下载数据 (profile=$DL_PROFILE, source=$DL_SOURCE)" \
            "python scripts/download_data.py --profile $DL_PROFILE --source $DL_SOURCE"
        ;;
    *)
        echo -e "${RED}  ❌ 无效选择${NC}"; exit 1
        ;;
esac

echo -e "${BOLD}  ============================================================${NC}"
echo -e "${GREEN}${BOLD}    🎉 全部完成!${NC}"
echo -e "${BOLD}  ============================================================${NC}"
echo ""
