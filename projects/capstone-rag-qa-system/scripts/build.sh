#!/bin/bash
# ============================================
# DocuMind AI - 构建生产版本
# ============================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🏗️  DocuMind AI - 构建生产版本"
echo "=================================="

# 1. 构建前端
echo "📦 构建前端..."
cd src/frontend

if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    npm install
fi

npm run build

echo "✅ 前端构建完成: src/frontend/dist/"

# 2. 回到项目根目录
cd "$PROJECT_ROOT"

# 3. 创建发布目录
DIST_DIR="$PROJECT_ROOT/dist"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# 4. 复制必要文件
echo "📋 复制文件..."
cp -r src "$DIST_DIR/"
cp -r configs "$DIST_DIR/"
cp -r scripts "$DIST_DIR/"
cp requirements.txt "$DIST_DIR/"
cp README.md "$DIST_DIR/"

# 复制前端构建产物
mkdir -p "$DIST_DIR/static"
cp -r src/frontend/dist/* "$DIST_DIR/static/"

echo ""
echo "=================================="
echo "✅ 构建完成!"
echo ""
echo "发布目录: $DIST_DIR"
echo ""
echo "生产启动命令:"
echo "  cd dist"
echo "  uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo "=================================="
