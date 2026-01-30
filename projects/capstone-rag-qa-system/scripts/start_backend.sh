#!/bin/bash
# ============================================
# DocuMind AI - 启动后端服务
# 自动停止占用端口的进程
# ============================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PORT=8000

echo "🔧 DocuMind AI - 启动后端服务"
echo "=================================="

# 检查并停止占用端口的进程
kill_port() {
    local pids=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "⚠️  端口 $PORT 被占用，正在停止现有进程..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
        echo "✅ 已停止占用端口的进程"
    fi
}

kill_port

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 初始化数据库
python scripts/init_db.py 2>/dev/null || true

# 设置环境变量
# USE_OLLAMA=true 使用 Ollama LLM（推荐）
# USE_MOCK_LLM=true 使用 Mock 模式（测试用）
export USE_OLLAMA=${USE_OLLAMA:-true}
export USE_MOCK_LLM=${USE_MOCK_LLM:-false}
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo ""
echo "📋 配置:"
echo "   USE_OLLAMA=$USE_OLLAMA"
echo "   USE_MOCK_LLM=$USE_MOCK_LLM"
echo ""

# 启动服务
echo "🚀 启动后端服务 http://localhost:$PORT"
echo ""
uvicorn src.api.main:app --reload --host 0.0.0.0 --port $PORT
