#!/bin/bash
# ============================================
# DocuMind AI - 仅启动前端
# ============================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/src/frontend"

echo "🎨 DocuMind AI - 启动前端服务"
echo "=================================="

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    npm install
fi

# 启动开发服务器
npm run dev
