# ============================================
# DocuMind AI - Dockerfile
# 多阶段构建: 前端 + 后端
# ============================================

# ---- Stage 1: 构建前端 ----
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# 复制前端依赖文件
COPY src/frontend/package*.json ./

# 安装依赖
RUN npm ci --silent

# 复制前端源码
COPY src/frontend/ ./

# 构建生产版本
RUN npm run build


# ---- Stage 2: Python 应用 ----
FROM python:3.11-slim AS runtime

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# 复制 Python 依赖文件
COPY requirements.txt ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist /app/static

# 创建必要目录
RUN mkdir -p /app/data /app/uploads /app/models /app/logs

# 复制 Nginx 配置
COPY docker/nginx.conf /etc/nginx/sites-available/default

# 暴露端口
EXPOSE 8000 80

# 启动脚本
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
