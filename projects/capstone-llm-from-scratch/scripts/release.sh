#!/usr/bin/env bash
# =========================================================
# release.sh — Phase 5 一键发布流水线
# =========================================================
#
# 把 outputs/dpo/final.pth 转成 HF Qwen3 兼容格式 → 验证可加载 →
# 打包归档 → 可选 push HF/ModelScope。
#
# 用法:
#   bash scripts/release.sh <规格> [--push-hf USER/REPO] [--push-ms USER/REPO]
#                                  [--stage dpo|sft|pretrain] [--dtype bf16|fp16|fp32]
#                                  [--dry-run]
#
# 例：
#   # 仅本地转换 + 验证（不 push）
#   bash scripts/release.sh base
#
#   # 转换 + push HF（需先 huggingface-cli login 或设 HF_TOKEN）
#   bash scripts/release.sh base --push-hf my/ClearMind-Base
#
#   # 同时 push HF + ModelScope（需先 modelscope login）
#   bash scripts/release.sh plus --push-hf me/ClearMind-Plus --push-ms me/ClearMind-Plus
#
#   # 只演练流程不真发（看每步会做什么）
#   bash scripts/release.sh base --dry-run
#
# 流水线步骤：
#   1) 选 source ckpt（默认 outputs/dpo/final.pth；可 --stage 切到 sft/pretrain）
#   2) 跑 scripts/convert_to_qwen3.py → release/<name>/{config.json, model.safetensors, ...}
#   3) 用 transformers AutoModel 试加载 release 目录，跑一次推理验证（防止格式错）
#   4) tar.gz 打包 + sha256 校验
#   5) （可选）push_to_hub / push_to_modelscope
#
# =========================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; NC=$'\033[0m'

PYTHON="${PYTHON:-./venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

# ---- 参数 ----
SCALE="${1:-}"
shift 2>/dev/null || true

PUSH_HF=""
PUSH_MS=""
STAGE="dpo"
DTYPE="fp16"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push-hf) PUSH_HF="$2"; shift 2 ;;
        --push-ms) PUSH_MS="$2"; shift 2 ;;
        --stage)   STAGE="$2"; shift 2 ;;
        --dtype)   DTYPE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            head -38 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

if [[ -z "$SCALE" ]]; then
    head -38 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 1
fi

case "$SCALE" in
    tiny)  CONFIG="configs/tiny.yaml";  MODEL_NAME="ClearMind-Tiny" ;;
    small) CONFIG="configs/small.yaml"; MODEL_NAME="ClearMind-Small" ;;
    base)  CONFIG="configs/main.yaml";  MODEL_NAME="ClearMind-Base" ;;
    plus)  CONFIG="configs/plus.yaml";  MODEL_NAME="ClearMind-Plus" ;;
    *) echo "${RED}❌ 未知规格: $SCALE${NC}"; exit 1 ;;
esac

case "$STAGE" in
    pretrain|sft|dpo) ;;
    *) echo "${RED}❌ 未知阶段: $STAGE${NC}"; exit 1 ;;
esac

CKPT="outputs/${STAGE}/final.pth"
TIMESTAMP=$(date '+%Y%m%d-%H%M')
RELEASE_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')   # 兼容 macOS bash 3.2
RELEASE_DIR="release/${RELEASE_NAME}"
TARBALL="release/${RELEASE_NAME}-${TIMESTAMP}.tar.gz"

run() {
    echo -e "${DIM}\$ $*${NC}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo -e "${DIM}  (dry-run，未执行)${NC}"
    else
        "$@"
    fi
}

echo -e "${BOLD}════════ ClearMind Release 流水线 ════════${NC}"
echo "  规格    : $SCALE → $MODEL_NAME"
echo "  Config  : $CONFIG"
echo "  Stage   : $STAGE → $CKPT"
echo "  Dtype   : $DTYPE"
echo "  目标    : $RELEASE_DIR + $TARBALL"
[[ -n "$PUSH_HF" ]] && echo "  Push HF : $PUSH_HF"
[[ -n "$PUSH_MS" ]] && echo "  Push MS : $PUSH_MS"
[[ "$DRY_RUN" -eq 1 ]] && echo -e "  ${YELLOW}模式    : dry-run（仅打印命令）${NC}"
echo ""

# =========================================================
# Step 1: 检查源 checkpoint
# =========================================================
echo -e "${BOLD}1) 检查 checkpoint${NC}"
if [[ ! -f "$CKPT" ]]; then
    echo -e "${RED}❌ 找不到 $CKPT${NC}"
    echo "   请先训完该阶段（bash scripts/autodl/launch.sh $SCALE $STAGE）"
    exit 1
fi
SIZE=$(du -h "$CKPT" | cut -f1)
echo -e "${GREEN}✅ $CKPT ($SIZE)${NC}"
echo ""

# =========================================================
# Step 2: convert_to_qwen3
# =========================================================
echo -e "${BOLD}2) 转 Qwen3ForCausalLM 兼容格式${NC}"
mkdir -p release
run "$PYTHON" scripts/convert_to_qwen3.py \
    --config "$CONFIG" \
    --input "$CKPT" \
    --output "$RELEASE_DIR" \
    --dtype "$DTYPE" \
    --model_name "$MODEL_NAME"

if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -f "$RELEASE_DIR/model.safetensors" ]] || [[ ! -f "$RELEASE_DIR/config.json" ]]; then
        echo -e "${RED}❌ 转换产物缺失${NC}"
        ls -la "$RELEASE_DIR/" 2>/dev/null
        exit 1
    fi
    echo -e "${GREEN}✅ 转换完成${NC}"
    ls -la "$RELEASE_DIR/"
fi
echo ""

# =========================================================
# Step 3: 验证可加载（用 transformers 重新装一次 + 推理）
# =========================================================
echo -e "${BOLD}3) 验证 transformers 可加载${NC}"
if [[ "$DRY_RUN" -eq 0 ]]; then
    "$PYTHON" - <<PY
import sys
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError as e:
    print(f"⚠️  transformers 不可用，跳过验证: {e}")
    sys.exit(0)

try:
    print("  加载模型 ...")
    tok = AutoTokenizer.from_pretrained("$RELEASE_DIR", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "$RELEASE_DIR", trust_remote_code=True, torch_dtype=torch.float32,
    )
    print(f"  ✓ 模型加载成功，参数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 跑一次推理（确认 forward 不挂）
    inputs = tok("你好", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    print(f"  ✓ 推理成功，生成 token: {out.shape}")
except Exception as e:
    print(f"❌ 验证失败: {type(e).__name__}: {e}")
    sys.exit(1)
print("✅ release 目录格式正确")
PY
    [[ $? -ne 0 ]] && exit 1
else
    echo -e "${DIM}  (dry-run 跳过)${NC}"
fi
echo ""

# =========================================================
# Step 4: 打包 + sha256
# =========================================================
echo -e "${BOLD}4) 打包${NC}"
run tar -czf "$TARBALL" -C "release" "$RELEASE_NAME"
if [[ "$DRY_RUN" -eq 0 ]]; then
    if command -v sha256sum &>/dev/null; then
        sha256sum "$TARBALL" > "$TARBALL.sha256"
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$TARBALL" > "$TARBALL.sha256"
    fi
    SIZE=$(du -h "$TARBALL" | cut -f1)
    echo -e "${GREEN}✅ $TARBALL ($SIZE) + .sha256${NC}"
fi
echo ""

# =========================================================
# Step 5: Push HF / ModelScope（可选）
# =========================================================
if [[ -n "$PUSH_HF" ]]; then
    echo -e "${BOLD}5a) Push 到 HuggingFace Hub: $PUSH_HF${NC}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        if "$PYTHON" -c "import huggingface_hub" 2>/dev/null; then
            run "$PYTHON" scripts/push_to_hub.py \
                --model_dir "$RELEASE_DIR" \
                --repo "$PUSH_HF"
        else
            echo -e "${YELLOW}⚠️  pip install huggingface_hub 后重试${NC}"
        fi
    fi
    echo ""
fi

if [[ -n "$PUSH_MS" ]]; then
    echo -e "${BOLD}5b) Push 到 ModelScope: $PUSH_MS${NC}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        if "$PYTHON" -c "import modelscope" 2>/dev/null; then
            run "$PYTHON" scripts/push_to_modelscope.py \
                --model_dir "$RELEASE_DIR" \
                --repo "$PUSH_MS"
        else
            echo -e "${YELLOW}⚠️  pip install modelscope 后重试${NC}"
        fi
    fi
    echo ""
fi

# =========================================================
# 总结
# =========================================================
echo -e "${BOLD}${GREEN}══════ Release 完成 ══════${NC}"
echo ""
echo "产物:"
echo "  $RELEASE_DIR/        (HF 兼容目录，可直接用 AutoModelForCausalLM.from_pretrained)"
echo "  $TARBALL             (tar.gz 归档)"
echo "  $TARBALL.sha256      (校验和)"
echo ""
if [[ -z "$PUSH_HF" && -z "$PUSH_MS" ]]; then
    echo "下一步（push 到 hub）："
    echo "  bash scripts/release.sh $SCALE --push-hf my/$MODEL_NAME"
    echo "  bash scripts/release.sh $SCALE --push-ms my/$MODEL_NAME"
fi
echo ""
echo "本地启动 OpenAI 兼容 API 服务："
echo "  $PYTHON deploy/api_server.py --config $CONFIG --port 8000"
echo ""
echo "本地启动 Web Demo："
echo "  $PYTHON deploy/web_demo.py --config $CONFIG --port 7860 --share"
