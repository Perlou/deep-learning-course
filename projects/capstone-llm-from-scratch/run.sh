#!/bin/bash
# ============================================================
# ClearMind — 一键训练脚本 (交互式箭头选择)
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
BG_CYAN='\033[46m\033[30m'

# ---- 箭头选择菜单 ----
# 用法: select_option "提示标题" option1 option2 ...
# 返回: 选中的索引 (0-based) 存在 SELECTED_INDEX
select_option() {
    local title="$1"
    shift
    local options=("$@")
    local count=${#options[@]}
    local selected=0

    # 隐藏光标
    tput civis 2>/dev/null || true

    # 打印标题
    echo -e "\n${CYAN}${BOLD}${title}${NC}"
    echo -e "${DIM}  ↑/↓ 切换选项, Enter 确认${NC}\n"

    # 渲染菜单
    render_menu() {
        # 移动光标到菜单起始位置
        for ((i = 0; i < count; i++)); do
            tput cuu1 2>/dev/null || printf '\033[1A'
        done

        for ((i = 0; i < count; i++)); do
            tput el 2>/dev/null || printf '\033[2K'
            if [[ $i -eq $selected ]]; then
                echo -e "  ${BG_CYAN} ❯ ${options[$i]} ${NC}"
            else
                echo -e "    ${options[$i]}"
            fi
        done
    }

    # 初始绘制
    for ((i = 0; i < count; i++)); do
        if [[ $i -eq $selected ]]; then
            echo -e "  ${BG_CYAN} ❯ ${options[$i]} ${NC}"
        else
            echo -e "    ${options[$i]}"
        fi
    done

    # 读取按键
    while true; do
        # 读取单个字符 (raw mode)
        IFS= read -rsn1 key

        case "$key" in
            $'\x1b')  # ESC sequence
                read -rsn2 rest
                case "$rest" in
                    '[A')  # 上箭头
                        ((selected > 0)) && ((selected--))
                        ;;
                    '[B')  # 下箭头
                        ((selected < count - 1)) && ((selected++))
                        ;;
                esac
                render_menu
                ;;
            '')  # Enter
                break
                ;;
        esac
    done

    # 恢复光标
    tput cnorm 2>/dev/null || true

    SELECTED_INDEX=$selected
}

# ---- Banner ----
clear
echo ""
echo -e "${BOLD}  ============================================================${NC}"
echo -e "${BOLD}    🧠 ClearMind — 从零训练你的大语言模型${NC}"
echo -e "${BOLD}  ============================================================${NC}"

# ---- Step 1: 选择模型规模 ----
SCALE_OPTIONS=(
    "Tiny   (~1.5M 参数,  2-5 分钟,  无需联网)"
    "Small  (~26M  参数,  MacBook CPU/MPS)"
    "Medium (~200M 参数,  GPU 24GB+)"
    "Large  (~468M 参数,  A100 80GB)"
)

select_option "📦 选择模型规模:" "${SCALE_OPTIONS[@]}"

case $SELECTED_INDEX in
    0) CONFIG="configs/tiny.yaml";   LABEL="Tiny";;
    1) CONFIG="configs/small.yaml";  LABEL="Small";;
    2) CONFIG="configs/medium.yaml"; LABEL="Medium";;
    3) CONFIG="configs/large.yaml";  LABEL="Large";;
esac

echo ""
echo -e "  ${GREEN}✅ 已选择: ${BOLD}${LABEL}${NC} ${DIM}(${CONFIG})${NC}"

# ---- Step 2: 选择训练流程 ----
FLOW_OPTIONS=(
    "全流程    数据 → 分词器 → 预训练 → SFT → DPO → 对话"
    "仅预训练  数据 → 分词器 → 预训练"
    "从 SFT 继续  (需已有预训练模型)"
    "从 DPO 继续  (需已有 SFT 模型)"
    "仅对话    (需已有训练好的模型)"
    "仅训练分词器"
    "运行测试  pytest"
)

select_option "🔄 选择训练流程:" "${FLOW_OPTIONS[@]}"
FLOW=$SELECTED_INDEX

echo ""
echo -e "  ${GREEN}✅ 已选择: ${BOLD}${FLOW_OPTIONS[$FLOW]}${NC}"
echo ""

# ---- 确认 ----
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
    ((step++))
    echo -e "${YELLOW}  [${step}] ▶ $1 ...${NC}"
    eval "$2"
    echo -e "${GREEN}  [${step}] ✅ $1 完成${NC}\n"
}

case $FLOW in
    0)  # 全流程
        run_step "准备数据"          "python scripts/prepare_data.py"
        run_step "训练分词器"        "python scripts/train_tokenizer.py --config $CONFIG"
        run_step "预训练"            "python scripts/train.py --stage pretrain --config $CONFIG"
        run_step "SFT 指令微调"      "python scripts/train.py --stage sft --config $CONFIG"
        run_step "DPO 偏好对齐"      "python scripts/train.py --stage dpo --config $CONFIG"
        run_step "启动对话"          "python scripts/chat.py --config $CONFIG"
        ;;
    1)  # 仅预训练
        run_step "准备数据"          "python scripts/prepare_data.py"
        run_step "训练分词器"        "python scripts/train_tokenizer.py --config $CONFIG"
        run_step "预训练"            "python scripts/train.py --stage pretrain --config $CONFIG"
        ;;
    2)  # 从 SFT 继续
        run_step "SFT 指令微调"      "python scripts/train.py --stage sft --config $CONFIG"
        ;;
    3)  # 从 DPO 继续
        run_step "DPO 偏好对齐"      "python scripts/train.py --stage dpo --config $CONFIG"
        ;;
    4)  # 仅对话
        run_step "启动对话"          "python scripts/chat.py --config $CONFIG"
        ;;
    5)  # 仅训练分词器
        run_step "准备数据"          "python scripts/prepare_data.py"
        run_step "训练分词器"        "python scripts/train_tokenizer.py --config $CONFIG"
        ;;
    6)  # 运行测试
        run_step "运行单元测试"      "python -m pytest tests/ -v"
        ;;
esac

echo -e "${BOLD}  ============================================================${NC}"
echo -e "${GREEN}${BOLD}    🎉 全部完成!${NC}"
echo -e "${BOLD}  ============================================================${NC}"
echo ""
