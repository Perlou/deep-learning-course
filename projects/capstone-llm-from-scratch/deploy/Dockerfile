# ============================================================
# ClearMind 部署 Dockerfile
# 多阶段构建: 训练环境 → 精简推理镜像
# ============================================================
#
# 构建:
#   docker build -t clearmind -f deploy/Dockerfile .
#
# 运行 API 服务:
#   docker run -p 8000:8000 -v ./outputs:/app/outputs clearmind
#
# 运行 Web 界面:
#   docker run -p 7860:7860 -v ./outputs:/app/outputs clearmind \
#     python deploy/web_demo.py --port 7860
#
# GPU 支持 (需要 nvidia-docker):
#   docker run --gpus all -p 8000:8000 -v ./outputs:/app/outputs clearmind
# ============================================================

FROM python:3.11-slim AS base

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-deploy.txt

# 复制项目代码
COPY src/ src/
COPY configs/ configs/
COPY deploy/ deploy/

# outputs/ 通过 volume 挂载，不打包到镜像
# 这样可以灵活切换不同的 checkpoint
VOLUME ["/app/outputs"]

# 暴露端口
EXPOSE 8000 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 默认启动 API 服务
CMD ["python", "deploy/api_server.py", "--host", "0.0.0.0", "--port", "8000"]
