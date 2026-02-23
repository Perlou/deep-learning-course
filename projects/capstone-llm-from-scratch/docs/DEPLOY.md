# 🚀 ClearMind 部署指南

> 详细的部署说明、硬件需求和最佳实践。

---

## 📋 目录

- [硬件需求](#硬件需求)
- [环境准备](#环境准备)
- [部署方式一: REST API](#部署方式一-rest-api)
- [部署方式二: Web 演示界面](#部署方式二-web-演示界面)
- [部署方式三: Docker 容器化](#部署方式三-docker-容器化)
- [模型导出与优化](#模型导出与优化)
- [多 GPU 训练 (DDP)](#多-gpu-训练-ddp)
- [性能调优](#性能调优)
- [常见问题](#常见问题)

---

## 硬件需求

### 训练硬件需求

| 模型           | 参数量 | 配置文件              | **最低显存**  | **推荐设备**     | **预估训练时间** |
| -------------- | ------ | --------------------- | ------------- | ---------------- | ---------------- |
| ClearMind-Tiny | ~1.5M  | `configs/tiny.yaml`   | CPU 即可      | MacBook (任意)   | 2-5 分钟         |
| ClearMind-Mini | ~26M   | `configs/small.yaml`  | CPU / 4GB MPS | MacBook Pro 16GB | 3-6 小时         |
| ClearMind      | ~200M  | `configs/medium.yaml` | **24GB VRAM** | RTX 4090 / A5000 | 6-12 小时        |
| ClearMind-Plus | ~468M  | `configs/large.yaml`  | **48GB VRAM** | A100 80GB        | 12-24 小时       |

> [!TIP]
> **Tiny 模式**只用于快速验证训练流程是否正确，不产出可用模型。

### 推理 (部署) 硬件需求

推理显存需求远低于训练 (无梯度/优化器/激活值)：

| 模型           | 参数量 | **FP32 推理** | **FP16/BF16 推理** | **INT8 量化推理** | 推荐设备           |
| -------------- | ------ | ------------- | ------------------ | ----------------- | ------------------ |
| ClearMind-Tiny | ~1.5M  | ~6 MB         | ~3 MB              | ~1.5 MB           | 任意 CPU           |
| ClearMind-Mini | ~26M   | ~100 MB       | ~50 MB             | ~25 MB            | MacBook / 任意 GPU |
| ClearMind      | ~200M  | ~800 MB       | ~400 MB            | ~200 MB           | 4GB+ VRAM GPU      |
| ClearMind-Plus | ~468M  | ~1.9 GB       | ~950 MB            | ~475 MB           | 8GB+ VRAM GPU      |

> [!NOTE]
> 上表为模型权重占用估算。实际推理时还需约 **0.5-2GB** 额外内存用于 KV Cache、激活值等。

### 各部署方式资源消耗

| 部署方式                   | 额外依赖          | 内存开销     | 适用场景         |
| -------------------------- | ----------------- | ------------ | ---------------- |
| REST API (`api_server.py`) | FastAPI + Uvicorn | ~50 MB       | 后端集成、微服务 |
| Web Demo (`web_demo.py`)   | Gradio            | ~80 MB       | 演示、测试、分享 |
| Docker                     | Docker Engine     | ~500 MB 镜像 | 生产部署、CI/CD  |

---

## 环境准备

### 安装部署依赖

```bash
# 基础训练依赖 (如果尚未安装)
pip install -r requirements.txt

# 部署专用依赖
pip install -r requirements-deploy.txt
```

`requirements-deploy.txt` 包含：

| 依赖                | 用途                          |
| ------------------- | ----------------------------- |
| `fastapi`           | REST API 框架                 |
| `uvicorn[standard]` | ASGI 服务器                   |
| `sse-starlette`     | Server-Sent Events (流式输出) |
| `gradio`            | Web 演示界面                  |
| `pydantic`          | 请求/响应数据校验             |

### 确认模型已训练

部署前需要已训练好的 checkpoint：

```bash
# 检查模型文件是否存在
ls outputs/dpo/final.pth    # DPO 对齐后的最终模型 (推荐)
ls outputs/sft/final.pth    # SFT 微调后的模型
ls outputs/pretrain/final.pth  # 仅预训练的模型
```

---

## 部署方式一: REST API

### 启动服务

```bash
# 默认配置 (small 模型, 8000 端口)
python deploy/api_server.py --model outputs/dpo/final.pth

# 自定义配置
python deploy/api_server.py \
  --model outputs/dpo/final.pth \
  --config configs/small.yaml \
  --tokenizer tokenizer.model \
  --host 0.0.0.0 \
  --port 8000
```

### API 端点

| 端点                   | 方法 | 功能                   |
| ---------------------- | ---- | ---------------------- |
| `/health`              | GET  | 健康检查               |
| `/v1/models`           | GET  | 模型信息               |
| `/v1/chat/completions` | POST | 对话补全 (OpenAI 兼容) |
| `/v1/completions`      | POST | 文本续写               |

### 请求示例

**对话补全 (curl)**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "什么是机器学习？"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

**流式输出 (curl)**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }' --no-buffer
```

**Python OpenAI 客户端**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # ClearMind 不需要 API Key
)

response = client.chat.completions.create(
    model="clearmind",
    messages=[{"role": "user", "content": "解释一下 Transformer 架构"}],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

### 请求参数

| 参数          | 类型  | 默认值 | 说明                 |
| ------------- | ----- | ------ | -------------------- |
| `messages`    | list  | 必填   | 对话消息列表         |
| `temperature` | float | 0.7    | 采样温度 (0.0-2.0)   |
| `top_p`       | float | 0.9    | 核采样概率 (0.0-1.0) |
| `top_k`       | int   | 50     | Top-K 采样           |
| `max_tokens`  | int   | 300    | 最大生成长度         |
| `stream`      | bool  | false  | 是否流式输出         |

---

## 部署方式二: Web 演示界面

基于 Gradio 的交互式 Web 界面，支持多轮对话、参数调节和公网分享。

### 启动界面

```bash
# 本地启动
python deploy/web_demo.py --model outputs/dpo/final.pth

# 指定端口
python deploy/web_demo.py --model outputs/dpo/final.pth --port 7860

# 创建公网分享链接 (72小时有效)
python deploy/web_demo.py --model outputs/dpo/final.pth --share
```

### 功能特性

- 💬 **多轮对话** — 支持上下文连续对话
- 🎛️ **参数调节** — Temperature、Top-K、Top-P 实时调整
- 🔗 **公网分享** — `--share` 一键生成公网链接
- 📱 **响应式设计** — 适配桌面和移动端

### 访问地址

```
http://localhost:7860        # 本地访问
https://xxxxx.gradio.live    # 公网分享 (--share 模式)
```

---

## 部署方式三: Docker 容器化

### 构建镜像

```bash
docker build -t clearmind -f deploy/Dockerfile .
```

### 运行容器

**API 服务 (默认)**

```bash
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/tokenizer.model:/app/tokenizer.model \
  clearmind
```

**Web 演示界面**

```bash
docker run -p 7860:7860 \
  -v $(pwd)/outputs:/app/outputs \
  clearmind python deploy/web_demo.py --host 0.0.0.0 --port 7860
```

**GPU 支持 (需要 nvidia-docker)**

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  clearmind
```

### Docker Compose (可选)

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')",
        ]
      interval: 30s
      timeout: 5s
      retries: 3
```

---

## 模型导出与优化

`deploy/export_model.py` 提供三种导出格式，用于减小模型体积和加速推理。

### 导出格式对比

| 格式            | 命令                   | 大小减少 | 推理加速       | 适用场景     |
| --------------- | ---------------------- | -------- | -------------- | ------------ |
| **权重瘦身**    | `--format weights`     | ~60-70%  | —              | 通用部署     |
| **TorchScript** | `--format torchscript` | —        | ~10-20%        | C++ / 移动端 |
| **INT8 量化**   | `--format quantized`   | ~75%     | ~50-100% (CPU) | CPU 推理优化 |

### 导出命令

```bash
# 权重瘦身 (去除优化器状态, 最常用)
python deploy/export_model.py \
  --model outputs/dpo/final.pth \
  --format weights

# INT8 动态量化 (CPU 推理加速)
python deploy/export_model.py \
  --model outputs/dpo/final.pth \
  --format quantized

# TorchScript (跨平台部署)
python deploy/export_model.py \
  --model outputs/dpo/final.pth \
  --format torchscript

# 一次导出所有格式
python deploy/export_model.py \
  --model outputs/dpo/final.pth \
  --format all
```

### 导出后模型大小参考

| 模型           | 原始 checkpoint | 瘦身后  | INT8 量化 |
| -------------- | --------------- | ------- | --------- |
| ClearMind-Tiny | ~12 MB          | ~6 MB   | ~3 MB     |
| ClearMind-Mini | ~200 MB         | ~100 MB | ~25 MB    |
| ClearMind      | ~1.6 GB         | ~800 MB | ~200 MB   |
| ClearMind-Plus | ~3.7 GB         | ~1.9 GB | ~475 MB   |

---

## 多 GPU 训练 (DDP)

> [!NOTE]
> 该章节用于训练加速，不属于部署阶段，但与生产环境准备常常同时进行。

### 前置条件

- 多张 CUDA GPU
- 已准备好的数据与分词器
- 使用 `torchrun` 启动

### 启动命令

```bash
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/medium.yaml \
  --data data/pretrain/pretrain_data.jsonl \
  --tokenizer outputs/tokenizer/tokenizer.model \
  --output_dir outputs/pretrain_ddp
```

### 常用覆盖参数

```bash
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/medium.yaml \
  --max_steps 2000 \
  --batch_size 4 \
  --gradient_accumulation 8 \
  --save_every 200 \
  --log_every 10
```

### 断点续训

```bash
torchrun --nproc_per_node=4 scripts/launch_ddp.py \
  --config configs/medium.yaml \
  --resume outputs/pretrain_ddp/checkpoint_step1000.pth
```

---

## 性能调优

### API 服务调优

```bash
# 增加 Uvicorn worker 数量 (多进程, 需要多份模型内存)
python deploy/api_server.py --model outputs/dpo/final.pth --workers 4

# 使用量化模型降低内存
python deploy/export_model.py --model outputs/dpo/final.pth --format quantized
python deploy/api_server.py --model outputs/export/clearmind_quantized.pth
```

### 推理速度优化建议

| 优化方法        | 效果                 | 适用设备      |
| --------------- | -------------------- | ------------- |
| KV Cache        | 自回归解码加速       | 全部          |
| INT8 量化       | CPU 推理加速 50-100% | CPU           |
| BFloat16 推理   | 内存减半             | GPU (Ampere+) |
| Flash Attention | 注意力计算加速       | GPU (Ampere+) |
| Batch 推理      | 吞吐量提升           | GPU           |

### MacBook 部署建议

- 使用 `configs/tiny.yaml` 或 `configs/small.yaml`
- 使用 `float32` 精度 (MPS 对 fp16 支持有限)
- INT8 量化可显著降低内存和提升速度
- 建议 `max_tokens ≤ 256` 避免长时间等待

### GPU 服务器部署建议

- 使用 `configs/medium.yaml` 或 `configs/large.yaml`
- 开启 `bfloat16` 推理精度
- 使用 Docker 容器化管理
- 配合 Nginx 做反向代理和负载均衡

---

## 常见问题

### Q: 启动 API 时报 `ModuleNotFoundError`

```bash
# 确保安装了部署依赖
pip install -r requirements-deploy.txt
```

### Q: GPU 显存不足 (OOM)

- 减小 `max_tokens` 参数
- 使用 INT8 量化模型
- 使用更小的模型配置 (small / tiny)

### Q: Docker 如何使用 GPU？

```bash
# 安装 nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
docker run --gpus all -p 8000:8000 -v ./outputs:/app/outputs clearmind
```

### Q: 如何在云服务器上公开访问？

```bash
# 方式一: API 直接绑定 0.0.0.0
python deploy/api_server.py --host 0.0.0.0 --port 8000

# 方式二: Gradio 公网分享 (推荐快速演示)
python deploy/web_demo.py --share

# 方式三: Nginx 反向代理 (生产环境)
# 配置 Nginx 将 80/443 端口转发到 8000
```

### Q: 如何对比不同阶段模型的效果？

```bash
# 预训练模型
python deploy/api_server.py --model outputs/pretrain/final.pth --port 8001

# SFT 模型
python deploy/api_server.py --model outputs/sft/final.pth --port 8002

# DPO 模型
python deploy/api_server.py --model outputs/dpo/final.pth --port 8003
```
