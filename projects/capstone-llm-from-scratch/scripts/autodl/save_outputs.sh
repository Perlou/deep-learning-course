#!/usr/bin/env bash
# =========================================================
# save_outputs.sh — 归档训练成果（防丢失）
# =========================================================
#
# AutoDL 关机后只有 /root/autodl-tmp/ 持久化，但仍有"数据盘满 / 误删 / 实例退掉"风险。
# 本脚本把训练产物整理成可下载的归档包：
#
#   1. 收集 outputs/<stage>/final.pth + 训练 log + 评估报告
#   2. 计算 sha256（防传输损坏）
#   3. 写一份 manifest.txt（记录配置、参数量、训练步数、loss 曲线）
#   4. 打成 tar.gz，名字带规格 + 时间戳
#   5. （可选）push 到 HuggingFace / ModelScope
#
# 用法:
#   bash scripts/autodl/save_outputs.sh base                       # 归档 base 训练产物
#   bash scripts/autodl/save_outputs.sh plus --push-hf my/repo     # 顺便推 HF
#   bash scripts/autodl/save_outputs.sh base --light               # 只打包 final.pth + manifest，不含 _resume.pth（小很多）
# =========================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; NC=$'\033[0m'

SCALE="${1:-}"
shift || true

LIGHT=0
PUSH_HF=""
PUSH_MS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --light) LIGHT=1; shift ;;
        --push-hf) PUSH_HF="$2"; shift 2 ;;
        --push-ms) PUSH_MS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCALE" ]]; then
    head -25 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 1
fi

case "$SCALE" in
    tiny)  CONFIG="configs/tiny.yaml" ;;
    small) CONFIG="configs/small.yaml" ;;
    base)  CONFIG="configs/main.yaml" ;;
    plus)  CONFIG="configs/plus.yaml" ;;
    *) echo -e "${RED}❌ 未知规格: $SCALE${NC}"; exit 1 ;;
esac

PYTHON="${PYTHON:-./venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

TIMESTAMP=$(date '+%Y%m%d-%H%M')
ARCHIVE_NAME="clearmind-${SCALE}-${TIMESTAMP}"
ARCHIVE_DIR="releases/${ARCHIVE_NAME}"
TARBALL="releases/${ARCHIVE_NAME}.tar.gz"

mkdir -p "releases"

echo -e "${BOLD}════════ ClearMind 归档 ════════${NC}"
echo "  规格    : $SCALE"
echo "  归档名  : $ARCHIVE_NAME"
echo "  路径    : $ARCHIVE_DIR"
echo "  Light   : $LIGHT"
echo ""

# ---- 1. 检查产物 ----
MISSING=0
for stage in pretrain sft dpo; do
    if [[ ! -f "outputs/$stage/final.pth" ]]; then
        echo -e "${YELLOW}⚠️  缺少 outputs/$stage/final.pth（跳过该阶段归档）${NC}"
    else
        FOUND=1
    fi
done

if [[ -z "${FOUND:-}" ]]; then
    echo -e "${RED}❌ outputs/ 下没有任何 final.pth，无可归档${NC}"
    exit 1
fi

# ---- 2. 创建归档目录结构 ----
rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"/{checkpoints,logs,eval,configs}

# ---- 3. 复制 checkpoints ----
echo -e "${CYAN}📦 复制 checkpoints...${NC}"
for stage in pretrain sft dpo; do
    if [[ -f "outputs/$stage/final.pth" ]]; then
        mkdir -p "$ARCHIVE_DIR/checkpoints/$stage"
        cp "outputs/$stage/final.pth" "$ARCHIVE_DIR/checkpoints/$stage/"
        if [[ "$LIGHT" -eq 0 ]] && [[ -f "outputs/$stage/_resume.pth" ]]; then
            cp "outputs/$stage/_resume.pth" "$ARCHIVE_DIR/checkpoints/$stage/"
        fi
        # 训练日志
        if [[ -f "outputs/$stage/${stage}_log.jsonl" ]]; then
            cp "outputs/$stage/${stage}_log.jsonl" "$ARCHIVE_DIR/logs/"
        fi
        echo "    ✓ $stage"
    fi
done

# ---- 4. 复制 launch 日志 ----
if ls logs/clearmind-${SCALE}-*.log 2>/dev/null >/dev/null; then
    cp logs/clearmind-${SCALE}-*.log "$ARCHIVE_DIR/logs/" 2>/dev/null || true
fi

# ---- 5. 复制 config ----
cp "$CONFIG" "$ARCHIVE_DIR/configs/"
[[ -f "tokenizer/minimind/tokenizer.json" ]] && \
    mkdir -p "$ARCHIVE_DIR/tokenizer/minimind" && \
    cp tokenizer/minimind/* "$ARCHIVE_DIR/tokenizer/minimind/" 2>/dev/null || true

# ---- 6. 跑评估生成报告 ----
echo -e "${CYAN}📊 生成评估报告...${NC}"
if [[ -f "outputs/dpo/final.pth" ]] || [[ -f "outputs/sft/final.pth" ]]; then
    "$PYTHON" evaluate/eval_perplexity.py --config "$CONFIG" --compare \
        > "$ARCHIVE_DIR/eval/perplexity.txt" 2>&1 || true
    "$PYTHON" evaluate/eval_generation.py --config "$CONFIG" --num_prompts 10 --num_samples 1 \
        > "$ARCHIVE_DIR/eval/generation_samples.txt" 2>&1 || true
    echo "    ✓ perplexity + generation"
fi

# ---- 7. 写 manifest ----
echo -e "${CYAN}📝 写 manifest.txt...${NC}"
{
    echo "ClearMind 训练归档 manifest"
    echo "============================================================"
    echo "归档时间    : $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "规格        : $SCALE"
    echo "Config      : $CONFIG"
    echo "项目根目录  : $PROJECT_DIR"
    echo "Git commit  : $(git rev-parse HEAD 2>/dev/null || echo '（非 git 仓库）')"
    echo "Git branch  : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    echo ""
    echo "模型参数量"
    echo "============================================================"
    "$PYTHON" - <<PY 2>/dev/null || echo "(无法计算参数量)"
import sys
sys.path.insert(0, "src")
from model import ModelConfig
cfg = ModelConfig.from_yaml("$CONFIG")
p = cfg.count_params()
print(f"  d_model     : {cfg.d_model}")
print(f"  n_layers    : {cfg.n_layers}")
print(f"  n_heads     : {cfg.n_heads} (kv={cfg.n_kv_heads})")
print(f"  vocab_size  : {cfg.vocab_size}")
print(f"  max_seq_len : {cfg.max_seq_len}")
print(f"  Total params: {p['total_millions']:.2f} M")
PY
    echo ""
    echo "Checkpoints (sha256)"
    echo "============================================================"
    cd "$ARCHIVE_DIR"
    if command -v sha256sum &>/dev/null; then
        find checkpoints -type f -name "*.pth" -exec sha256sum {} \;
    elif command -v shasum &>/dev/null; then
        find checkpoints -type f -name "*.pth" -exec shasum -a 256 {} \;
    fi
    cd "$PROJECT_DIR"
    echo ""
    echo "归档内容树"
    echo "============================================================"
    cd "$ARCHIVE_DIR" && find . -type f | head -50 | sort && cd "$PROJECT_DIR"
} > "$ARCHIVE_DIR/manifest.txt"

echo "    ✓ $ARCHIVE_DIR/manifest.txt"

# ---- 8. 打 tarball ----
echo -e "${CYAN}🗜  打包 tar.gz...${NC}"
tar -czf "$TARBALL" -C "releases" "$ARCHIVE_NAME"
TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
echo -e "    ✓ ${GREEN}$TARBALL${NC} ($TARBALL_SIZE)"

# 校验和
if command -v sha256sum &>/dev/null; then
    sha256sum "$TARBALL" > "$TARBALL.sha256"
elif command -v shasum &>/dev/null; then
    shasum -a 256 "$TARBALL" > "$TARBALL.sha256"
fi
echo "    ✓ $TARBALL.sha256"

# ---- 9. （可选）推到 HF / ModelScope ----
if [[ -n "$PUSH_HF" ]]; then
    echo -e "${CYAN}🤗 推送到 HuggingFace: $PUSH_HF${NC}"
    "$PYTHON" scripts/push_to_hub.py --config "$CONFIG" --repo "$PUSH_HF" || \
        echo -e "${YELLOW}⚠️  推送失败，可手动 retry${NC}"
fi
if [[ -n "$PUSH_MS" ]]; then
    echo -e "${CYAN}🪐 推送到 ModelScope: $PUSH_MS${NC}"
    "$PYTHON" scripts/push_to_modelscope.py --config "$CONFIG" --repo "$PUSH_MS" || \
        echo -e "${YELLOW}⚠️  推送失败，可手动 retry${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}✅ 归档完成${NC}"
echo ""
echo "下载到本地（在你笔记本里执行）："
echo "  scp -r root@<autodl-ip>:$PROJECT_DIR/$TARBALL ./"
echo "  scp -r root@<autodl-ip>:$PROJECT_DIR/$TARBALL.sha256 ./"
echo ""
echo "解压："
echo "  tar -xzf ${ARCHIVE_NAME}.tar.gz"
echo ""
echo -e "${YELLOW}💡 下载完成并验证 sha256 后，可以放心关掉 AutoDL 实例了。${NC}"
