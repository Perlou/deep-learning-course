#!/bin/bash
# ============================================================
# ClearMind — 一键训练脚本
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

# ---- Banner ----
echo ""
echo -e "${BOLD}  ============================================================${NC}"
echo -e "${BOLD}    🧠 ClearMind — 从零训练你的大语言模型${NC}"
echo -e "${BOLD}  ============================================================${NC}"

# ---- Step 1: 选择模型规模 ----
echo ""
echo -e "${CYAN}${BOLD}  📦 选择模型规模:${NC}"
echo ""
echo -e "  ${BOLD}1)${NC} Tiny   (~1.5M 参数,  2-5 分钟,  无需联网)"
echo -e "  ${BOLD}2)${NC} Small  (~26M  参数,  MacBook CPU/MPS)"
echo -e "  ${BOLD}3)${NC} Medium (~200M 参数,  GPU 24GB+)"
echo -e "  ${BOLD}4)${NC} Large  (~468M 参数,  A100 80GB)"
echo ""
read -p "  请选择 [1-4, 默认 1]: " scale_choice

case "${scale_choice:-1}" in
    1) CONFIG="configs/tiny.yaml";   LABEL="Tiny";;
    2) CONFIG="configs/small.yaml";  LABEL="Small";;
    3) CONFIG="configs/medium.yaml"; LABEL="Medium";;
    4) CONFIG="configs/large.yaml";  LABEL="Large";;
    *) echo -e "${RED}  ❌ 无效选择${NC}"; exit 1;;
esac

echo -e "\n  ${GREEN}✅ 已选择: ${BOLD}${LABEL}${NC} ${DIM}(${CONFIG})${NC}"

# ---- Step 2: 选择训练流程 ----
echo ""
echo -e "${CYAN}${BOLD}  🔄 选择训练流程:${NC}"
echo ""
echo -e "  ${BOLD}1)${NC} 全流程    数据 → 分词器 → 预训练 → SFT → DPO → 对话"
echo -e "  ${BOLD}2)${NC} 仅预训练  数据 → 分词器 → 预训练"
echo -e "  ${BOLD}3)${NC} 从 SFT 继续  (需已有预训练模型)"
echo -e "  ${BOLD}4)${NC} 从 DPO 继续  (需已有 SFT 模型)"
echo -e "  ${BOLD}5)${NC} 仅对话    (需已有训练好的模型)"
echo -e "  ${BOLD}6)${NC} 仅训练分词器"
echo -e "  ${BOLD}7)${NC} 运行测试  pytest"
echo ""
read -p "  请选择 [1-7, 默认 1]: " flow_choice
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
        run_step "准备数据"          "python scripts/prepare_data.py"
        run_step "训练分词器"        "python scripts/train_tokenizer.py --config $CONFIG"
        run_step "预训练"            "python scripts/train.py --stage pretrain --config $CONFIG"
        run_step "SFT 指令微调"      "python scripts/train.py --stage sft --config $CONFIG"
        run_step "DPO 偏好对齐"      "python scripts/train.py --stage dpo --config $CONFIG"
        run_step "启动对话"          "python scripts/chat.py --config $CONFIG"
        ;;
    2)  # 仅预训练
        run_step "准备数据"          "python scripts/prepare_data.py"
        run_step "训练分词器"        "python scripts/train_tokenizer.py --config $CONFIG"
        run_step "预训练"            "python scripts/train.py --stage pretrain --config $CONFIG"
        ;;
    3)  # 从 SFT 继续
        run_step "SFT 指令微调"      "python scripts/train.py --stage sft --config $CONFIG"
        ;;
    4)  # 从 DPO 继续
        run_step "DPO 偏好对齐"      "python scripts/train.py --stage dpo --config $CONFIG"
        ;;
    5)  # 仅对话
        run_step "启动对话"          "python scripts/chat.py --config $CONFIG"
        ;;
    6)  # 仅训练分词器
        run_step "准备数据"          "python scripts/prepare_data.py"
        run_step "训练分词器"        "python scripts/train_tokenizer.py --config $CONFIG"
        ;;
    7)  # 运行测试
        run_step "运行单元测试"      "python -m pytest tests/ -v"
        ;;
    *)
        echo -e "${RED}  ❌ 无效选择${NC}"; exit 1
        ;;
esac

echo -e "${BOLD}  ============================================================${NC}"
echo -e "${GREEN}${BOLD}    🎉 全部完成!${NC}"
echo -e "${BOLD}  ============================================================${NC}"
echo ""
