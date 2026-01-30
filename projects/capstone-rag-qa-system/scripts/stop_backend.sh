#!/bin/bash
# ============================================
# DocuMind AI - 停止后端服务
# ============================================

PORT=8000

echo "🛑 DocuMind AI - 停止后端服务"
echo "=================================="

# 查找占用端口的进程
pids=$(lsof -ti :$PORT 2>/dev/null || true)

if [ -z "$pids" ]; then
    echo "✅ 端口 $PORT 没有运行的服务"
    exit 0
fi

echo "📋 发现以下进程占用端口 $PORT:"
lsof -i :$PORT 2>/dev/null | head -5

echo ""
echo "⚠️  正在停止进程..."

# 发送 SIGTERM 优雅关闭
echo "$pids" | xargs kill 2>/dev/null || true
sleep 2

# 检查是否还在运行
remaining=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$remaining" ]; then
    echo "⚠️  进程未响应 SIGTERM，强制终止..."
    echo "$remaining" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# 最终确认
final=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -z "$final" ]; then
    echo "✅ 后端服务已停止"
else
    echo "❌ 无法停止部分进程，请手动处理"
    lsof -i :$PORT
    exit 1
fi
