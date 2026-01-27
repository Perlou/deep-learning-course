#!/bin/bash
# ============================================
# DocuMind AI - 环境配置脚本
# ============================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "⚙️  DocuMind AI - 环境配置"
echo "=================================="

# 1. 创建 Python 虚拟环境
echo "🐍 创建 Python 虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境已创建"
else
    echo "ℹ️  虚拟环境已存在"
fi

source venv/bin/activate

# 2. 安装 Python 依赖
echo "📦 安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. 创建必要目录
echo "📁 创建数据目录..."
mkdir -p data uploads logs models

# 4. 初始化数据库
echo "🗃️  初始化数据库..."
python scripts/init_db.py

# 5. 安装前端依赖
echo "🎨 安装前端依赖..."
cd src/frontend
npm install
cd "$PROJECT_ROOT"

# 6. 创建 .env 文件
if [ ! -f ".env" ]; then
    echo "📝 创建 .env 配置文件..."
    cat > .env << EOF
# DocuMind AI 环境配置

# 应用设置
APP_ENV=development
DEBUG=true

# LLM 设置 (设为 false 需要 GPU)
USE_MOCK_LLM=true

# 模型配置
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# 前端 API 地址
VITE_API_URL=http://localhost:8000/api/v1
EOF
    echo "✅ .env 文件已创建"
fi

echo ""
echo "=================================="
echo "✅ 环境配置完成!"
echo ""
echo "启动开发环境:"
echo "  ./scripts/dev.sh"
echo ""
echo "或分别启动:"
echo "  ./scripts/start_backend.sh"
echo "  ./scripts/start_frontend.sh"
echo "=================================="
