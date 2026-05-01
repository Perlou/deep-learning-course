#!/usr/bin/env bash
# =========================================================
# status.sh — 查询当前 ClearMind 训练状态
# =========================================================
#
# 不进 tmux 的快速状态检查：
#   - 是否有训练在跑
#   - 跑到第几步 / loss 多少 / 最后一次更新是什么时候
#   - GPU 利用率 / 显存
#   - 各 stage 的产物
#
# 用法:
#   bash scripts/autodl/status.sh
#   bash scripts/autodl/status.sh --watch       # 每 5 秒刷新
# =========================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; NC=$'\033[0m'

WATCH=0
[[ "${1:-}" == "--watch" ]] && WATCH=1

show() {
    [[ "$WATCH" -eq 1 ]] && clear
    echo -e "${BOLD}════════ ClearMind 训练状态 ════════${NC}  $(date '+%F %T')"
    echo ""

    # ---- tmux sessions ----
    echo -e "${BOLD}🖥  tmux sessions:${NC}"
    if command -v tmux &>/dev/null; then
        SESSIONS=$(tmux ls 2>/dev/null | grep '^clearmind-' || true)
        if [[ -n "$SESSIONS" ]]; then
            echo "$SESSIONS" | sed 's/^/  /'
        else
            echo -e "  ${DIM}(无活跃 session)${NC}"
        fi
    else
        echo -e "  ${DIM}tmux 未安装${NC}"
    fi
    echo ""

    # ---- 进程 ----
    echo -e "${BOLD}⚙  训练进程:${NC}"
    if [[ -d "logs" ]]; then
        FOUND=0
        for pf in logs/*.pid; do
            [[ -f "$pf" ]] || continue
            PID=$(cat "$pf")
            NAME=$(basename "$pf" .pid)
            if kill -0 "$PID" 2>/dev/null; then
                CPU=$(ps -o %cpu= -p "$PID" 2>/dev/null | tr -d ' ')
                MEM=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1f", $1/1024}')
                ETIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
                echo -e "  ${GREEN}● $NAME${NC} PID=$PID  CPU=${CPU}%  RSS=${MEM}MB  运行=${ETIME}"
                FOUND=1
            else
                STATE_FILE="logs/${NAME}.state"
                STATE="dead"
                [[ -f "$STATE_FILE" ]] && STATE=$(cat "$STATE_FILE")
                case "$STATE" in
                    done) echo -e "  ${GREEN}✓ $NAME${NC} ${DIM}(已正常完成)${NC}" ;;
                    failed) echo -e "  ${RED}✗ $NAME${NC} ${DIM}(异常退出，看 logs/${NAME}.log)${NC}" ;;
                    *) echo -e "  ${YELLOW}? $NAME${NC} ${DIM}(进程已退出，状态=$STATE)${NC}" ;;
                esac
            fi
        done
        [[ "$FOUND" -eq 0 ]] && [[ -z "$(ls logs/*.pid 2>/dev/null)" ]] && \
            echo -e "  ${DIM}(没有训练记录)${NC}"
    fi
    echo ""

    # ---- GPU ----
    if command -v nvidia-smi &>/dev/null; then
        echo -e "${BOLD}🎮 GPU:${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F',' '{
                gsub(/^ +/, "", $2);
                printf "  GPU%s %s  Util=%s%%  VRAM=%s/%sMB  Temp=%s°C\n",
                       $1, $2, $3, $4, $5, $6
            }'
        echo ""
    fi

    # ---- 最近日志 (从最新的 log 取最后 10 行 train 进度) ----
    LATEST_LOG=$(ls -t logs/clearmind-*.log 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_LOG" ]] && [[ -f "$LATEST_LOG" ]]; then
        echo -e "${BOLD}📜 最近日志 ($LATEST_LOG):${NC}"
        # 优先抓 Step 行；其次抓最后 5 行
        tail -200 "$LATEST_LOG" 2>/dev/null | grep -E '(Step [0-9]+|Epoch|✅|❌|⚠️|Loss:)' | tail -8 | sed 's/^/  /' || \
            tail -5 "$LATEST_LOG" | sed 's/^/  /'
        echo ""
    fi

    # ---- 产物 ----
    echo -e "${BOLD}📦 产物 (outputs/):${NC}"
    if [[ -d "outputs" ]]; then
        for stage in pretrain sft dpo; do
            FINAL="outputs/$stage/final.pth"
            RESUME="outputs/$stage/_resume.pth"
            if [[ -f "$FINAL" ]]; then
                SIZE=$(du -h "$FINAL" | cut -f1)
                MTIME=$(date -r "$FINAL" '+%F %H:%M' 2>/dev/null || stat -c '%y' "$FINAL" 2>/dev/null | cut -d. -f1)
                echo -e "  ${GREEN}✓${NC} $stage/final.pth   $SIZE  $MTIME"
            elif [[ -f "$RESUME" ]]; then
                SIZE=$(du -h "$RESUME" | cut -f1)
                MTIME=$(date -r "$RESUME" '+%F %H:%M' 2>/dev/null || stat -c '%y' "$RESUME" 2>/dev/null | cut -d. -f1)
                echo -e "  ${YELLOW}…${NC} $stage/_resume.pth $SIZE  $MTIME ${DIM}(还在跑或中断)${NC}"
            else
                echo -e "  ${DIM}- $stage/         （无产物）${NC}"
            fi
        done
    else
        echo -e "  ${DIM}(outputs/ 不存在)${NC}"
    fi
    echo ""

    # ---- 磁盘 ----
    echo -e "${BOLD}💾 磁盘:${NC}"
    df -h . | tail -1 | awk '{printf "  当前目录: 已用 %s / 共 %s (%s)\n", $3, $2, $5}'
    if [[ -d "outputs" ]]; then
        OUT_SIZE=$(du -sh outputs 2>/dev/null | cut -f1)
        echo "  outputs/: $OUT_SIZE"
    fi
}

if [[ "$WATCH" -eq 1 ]]; then
    while true; do
        show
        sleep 5
    done
else
    show
fi
