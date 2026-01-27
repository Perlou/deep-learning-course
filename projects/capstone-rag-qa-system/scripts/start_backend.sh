#!/bin/bash
# ============================================
# DocuMind AI - 仅启动后端
# ============================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔧 DocuMind AI - 启动后端服务"
echo "=================================="

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 初始化数据库
python scripts/init_db.py 2>/dev/null || true

# 设置环境变量
export USE_MOCK_LLM=${USE_MOCK_LLM:-true}
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "USE_MOCK_LLM=$USE_MOCK_LLM"
echo ""

# 启动服务
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
