# ============================================================
# ClearMind 部署 Dockerfile
# ============================================================
#
# 构建:
#   docker build -t clearmind -f deploy/Dockerfile .
#
# 运行 OpenAI 兼容 API:
#   docker run -p 8000:8000 \
#     -v ./outputs:/app/outputs \
#     -v ./tokenizer:/app/tokenizer \
#     -v ./configs:/app/configs \
#     clearmind
#
# 运行 Web 界面:
#   docker run -p 7860:7860 \
#     -v ./outputs:/app/outputs \
#     -v ./tokenizer:/app/tokenizer \
#     clearmind \
#     python deploy/web_demo.py --host 0.0.0.0 --port 7860
#
# GPU 支持 (nvidia-docker):
#   docker run --gpus all -p 8000:8000 \
#     -v ./outputs:/app/outputs -v ./tokenizer:/app/tokenizer \
#     clearmind
# ============================================================

FROM python:3.11-slim AS base

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（合并 requirements + 部署侧）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] gradio

# 复制项目代码（按依赖顺序）
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tokenizer/ tokenizer/
COPY deploy/ deploy/

# outputs/ 与 data/ 通过 volume 挂载（训练数据/checkpoint 不打包到镜像）
VOLUME ["/app/outputs", "/app/data"]

# 暴露端口（API + Web）
EXPOSE 8000 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动 OpenAI 兼容 API（监听 8000）
# 用户可通过 docker run 末尾追加自定义命令覆盖（例如启 web_demo.py）
CMD ["python", "deploy/api_server.py", \
     "--config", "configs/main.yaml", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
