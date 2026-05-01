#!/usr/bin/env bash
# =========================================================
# preflight.sh — AutoDL 训练前自检（强制在租机后第一步运行）
# =========================================================
#
# 训练失败的成本很高（一次 plus 训练 30+ 小时、几百块），
# 必须在按下"开始"之前确认所有关键依赖都就位：
#
#   ✅ Python / venv / 依赖包齐全
#   ✅ GPU 可见、显存足够
#   ✅ 磁盘容量足够（包括 outputs 增量）
#   ✅ tokenizer / 数据文件就位
#   ✅ tmux 可用（断连生存的关键）
#   ✅ 单元测试通过（避免代码层面 bug）
#   ✅ tiny 冒烟通过（避免数据层面 bug）
#
# 用法:
#   bash scripts/autodl/preflight.sh                      # 全检
#   bash scripts/autodl/preflight.sh --skip-smoke         # 跳过 tiny 冒烟（已经跑过）
#   bash scripts/autodl/preflight.sh --profile base       # 只检查 base 训练所需的数据
#
# 退出码:
#   0  全部通过
#   1  有 ERROR 项，禁止训练
#   2  有 WARN 项但可继续（用户自行决策）
# =========================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# ---- 颜色 ----
RED=$'\033[0;31m'
YELLOW=$'\033[1;33m'
GREEN=$'\033[0;32m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
NC=$'\033[0m'

ERRORS=0
WARNS=0
PROFILE="base"
SKIP_SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --profile) PROFILE="$2"; shift 2 ;;
        -h|--help)
            head -40 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

err()  { echo -e "${RED}❌ $*${NC}"; ERRORS=$((ERRORS+1)); }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; WARNS=$((WARNS+1)); }
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
info() { echo -e "${CYAN}ℹ️  $*${NC}"; }
section() { echo -e "\n${BOLD}=== $* ===${NC}"; }

PYTHON="${PYTHON:-./venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

# =========================================================
# 1. 项目结构
# =========================================================
section "1. 项目结构"

[[ -d "src" ]] && ok "src/ 存在" || err "缺少 src/ 目录（项目根路径不对？）"
[[ -d "configs" ]] && ok "configs/ 存在" || err "缺少 configs/"
[[ -d "scripts" ]] && ok "scripts/ 存在" || err "缺少 scripts/"
[[ -f "scripts/train.py" ]] && ok "scripts/train.py 存在" || err "缺少训练入口脚本"

# =========================================================
# 2. Python 环境
# =========================================================
section "2. Python 环境"

PY_VERSION=$("$PYTHON" -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo "")
if [[ -z "$PY_VERSION" ]]; then
    err "Python 不可用：$PYTHON"
else
    ok "Python: $PY_VERSION ($PYTHON)"
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
        err "Python 版本过低（要求 3.10+），当前 $PY_VERSION"
    fi
fi

# 关键依赖一次性检查
"$PYTHON" - <<'PY' 2>&1 | while IFS= read -r line; do echo "  $line"; done
import importlib, sys
need = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("yaml", "PyYAML"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
    ("jinja2", "jinja2"),
]
optional = [
    ("datasets", "datasets (HF 数据下载用)"),
    ("modelscope", "modelscope (国内镜像)"),
    ("wandb", "wandb (训练监控，可选)"),
]
missing_required = []
for mod, name in need:
    try:
        importlib.import_module(mod)
        print(f"OK    {name}")
    except ImportError:
        print(f"MISS  {name}")
        missing_required.append(name)
for mod, name in optional:
    try:
        importlib.import_module(mod)
        print(f"OK    {name}")
    except ImportError:
        print(f"WARN  {name} (optional)")
sys.exit(1 if missing_required else 0)
PY
if [[ $? -ne 0 ]]; then
    err "缺少必需依赖，运行: pip install -r requirements.txt"
else
    ok "核心依赖齐全"
fi

# =========================================================
# 3. GPU
# =========================================================
section "3. GPU / CUDA"

if ! command -v nvidia-smi &>/dev/null; then
    warn "未检测到 nvidia-smi（可能在 CPU/MPS 上跑，仅适合 tiny 冒烟）"
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_MEM_GB=$((GPU_MEM / 1024))
    ok "GPU: $GPU_NAME (${GPU_MEM_GB} GB)"

    case "$PROFILE" in
        plus)  REQ_VRAM=70 ;;
        base)  REQ_VRAM=20 ;;
        small) REQ_VRAM=12 ;;
        tiny)  REQ_VRAM=4  ;;
        *)     REQ_VRAM=20 ;;
    esac
    if [[ "$GPU_MEM_GB" -lt "$REQ_VRAM" ]]; then
        warn "VRAM ${GPU_MEM_GB}GB < ${PROFILE} 推荐 ${REQ_VRAM}GB；可能需要降低 batch_size 或 grad_accum"
    fi

    "$PYTHON" -c "
import torch
print('  CUDA available:', torch.cuda.is_available())
print('  Device count:  ', torch.cuda.device_count())
if torch.cuda.is_available():
    print('  cuDNN version: ', torch.backends.cudnn.version())
    print('  bf16 support:  ', torch.cuda.is_bf16_supported())
" 2>&1 | sed 's/^/  /'
fi

# =========================================================
# 4. 磁盘
# =========================================================
section "4. 磁盘空间"

# AutoDL 推荐放在 /root/autodl-tmp（按量付费的数据盘）
DISK_FREE=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
ok "当前目录可用空间: ${DISK_FREE} GB"

case "$PROFILE" in
    plus)  REQ_DISK=80 ;;
    base)  REQ_DISK=30 ;;
    small) REQ_DISK=15 ;;
    tiny)  REQ_DISK=5  ;;
    *)     REQ_DISK=30 ;;
esac
if [[ "$DISK_FREE" -lt "$REQ_DISK" ]]; then
    err "磁盘不足：${PROFILE} 训练需要至少 ${REQ_DISK}GB（数据 + outputs + ckpt 多版本）"
fi

# =========================================================
# 5. Tokenizer
# =========================================================
section "5. Tokenizer 资产"

if [[ -f "tokenizer/minimind/tokenizer.json" ]]; then
    SIZE=$(du -h tokenizer/minimind/tokenizer.json | cut -f1)
    ok "tokenizer/minimind/tokenizer.json ($SIZE)"
else
    err "缺少 tokenizer/minimind/tokenizer.json（应随仓库自带）"
fi

# =========================================================
# 6. 数据文件
# =========================================================
section "6. 训练数据 (profile=$PROFILE)"

declare -A DATA_REQ
case "$PROFILE" in
    tiny|small|base)
        DATA_REQ["data/pretrain_t2t_mini.jsonl"]="预训练"
        DATA_REQ["data/sft_t2t_mini.jsonl"]="SFT"
        DATA_REQ["data/dpo.jsonl"]="DPO"
        ;;
    plus)
        DATA_REQ["data/pretrain_t2t.jsonl"]="预训练（plus 全量）"
        DATA_REQ["data/sft_t2t.jsonl"]="SFT（plus 全量）"
        DATA_REQ["data/dpo.jsonl"]="DPO"
        ;;
esac

for f in "${!DATA_REQ[@]}"; do
    if [[ -f "$f" ]]; then
        SIZE=$(du -h "$f" | cut -f1)
        ok "${DATA_REQ[$f]}: $f ($SIZE)"
    else
        err "缺少 ${DATA_REQ[$f]} 数据: $f"
    fi
done

if [[ "$ERRORS" -gt 0 ]] && [[ ${#DATA_REQ[@]} -gt 0 ]]; then
    info "下载命令:"
    echo "    python scripts/download_data.py --profile $PROFILE"
    echo "    python scripts/download_data.py --profile $PROFILE --source modelscope  # 国内更快"
fi

# =========================================================
# 7. tmux（断连生存的关键）
# =========================================================
section "7. tmux（断连生存）"

if command -v tmux &>/dev/null; then
    TMUX_VERSION=$(tmux -V)
    ok "tmux 可用: $TMUX_VERSION"
else
    err "tmux 未安装。AutoDL 上运行：apt-get update && apt-get install -y tmux"
fi

# =========================================================
# 8. 单元测试
# =========================================================
section "8. 单元测试"

if [[ -d "tests" ]]; then
    info "运行 pytest（应 < 10s）..."
    if "$PYTHON" -m pytest tests/ -q --no-header 2>&1 | tail -5; then
        ok "单元测试通过"
    else
        err "单元测试失败 — 训练前必须先修复"
    fi
else
    warn "tests/ 目录不存在，跳过"
fi

# =========================================================
# 9. Tiny 端到端冒烟（最重要！）
# =========================================================
section "9. Tiny 端到端冒烟"

if [[ "$SKIP_SMOKE" -eq 1 ]]; then
    info "已跳过（--skip-smoke）"
elif [[ ! -f "data/pretrain_t2t_mini.jsonl" ]] || [[ ! -f "data/sft_t2t_mini.jsonl" ]]; then
    warn "数据缺失，跳过冒烟"
else
    info "运行 smoke_test.py（CPU/GPU 上 ~2-3 分钟）..."
    if "$PYTHON" scripts/smoke_test.py --clean 2>&1 | tail -10; then
        ok "Tiny 冒烟通过"
    else
        err "Tiny 冒烟失败 — 训练前必须先修复，否则正式训练会浪费 GPU 钱"
    fi
fi

# =========================================================
# 总结
# =========================================================
section "总结"

if [[ "$ERRORS" -gt 0 ]]; then
    echo -e "${RED}${BOLD}❌ ${ERRORS} 项 ERROR + ${WARNS} 项 WARN${NC}"
    echo -e "${RED}严禁开始训练，请先修复 ERROR 项${NC}"
    exit 1
elif [[ "$WARNS" -gt 0 ]]; then
    echo -e "${YELLOW}${BOLD}⚠️  ${WARNS} 项 WARN（可继续，但请确认）${NC}"
    exit 2
else
    echo -e "${GREEN}${BOLD}✅ 全部通过！可以开始训练${NC}"
    echo ""
    echo "推荐下一步："
    echo "  bash scripts/autodl/launch.sh tiny  pretrain    # 先 tiny 跑通整条链路"
    echo "  bash scripts/autodl/launch.sh small all         # 然后 small 验证 GPU 训练"
    echo "  bash scripts/autodl/launch.sh base  all         # 最后 base 正式训练"
    exit 0
fi
