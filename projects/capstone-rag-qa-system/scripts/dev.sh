#!/bin/bash
# ============================================
# DocuMind AI - 开发环境启动脚本
# ============================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 DocuMind AI - 开发环境启动"
echo "=================================="

# 检查 Python 虚拟环境
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "⚠️  未检测到虚拟环境，正在创建..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source .venv/bin/activate
    fi
fi

# 初始化数据库
echo "📦 初始化数据库..."
python scripts/init_db.py 2>/dev/null || echo "数据库已存在"

# 启动后端
echo "🔧 启动后端服务 (端口 8000)..."
export USE_MOCK_LLM=true
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 检查后端是否成功启动
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ 后端启动失败"
    exit 1
fi

echo "✅ 后端已启动: http://localhost:8000"
echo "📚 API 文档: http://localhost:8000/docs"

# 启动前端
echo "🎨 启动前端服务 (端口 5173)..."
cd src/frontend

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

cd "$PROJECT_ROOT"

echo ""
echo "=================================="
echo "✅ 开发环境已启动!"
echo ""
echo "🌐 前端地址: http://localhost:5173"
echo "🔧 后端地址: http://localhost:8000"
echo "📚 API 文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "=================================="

# 捕获退出信号
cleanup() {
    echo ""
    echo "🛑 正在停止服务..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ 服务已停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 等待进程
wait
