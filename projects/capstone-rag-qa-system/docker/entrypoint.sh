#!/bin/bash
# ============================================
# DocuMind AI - Docker Entrypoint
# ============================================

set -e

echo "=== DocuMind AI Starting ==="

# 初始化数据库
echo "Initializing database..."
python -c "from src.models import init_db; import asyncio; asyncio.run(init_db())" || true

# 启动 Nginx (后台)
echo "Starting Nginx..."
nginx

# 启动 FastAPI 应用
echo "Starting FastAPI application..."
exec uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
