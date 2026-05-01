# 🚀 ClearMind 部署指南

> 训完模型后如何对外提供服务、容器化、上传到 HuggingFace / ModelScope。

---

## 📋 目录

1. [硬件需求](#硬件需求)
2. [部署形态](#部署形态)
3. [OpenAI 兼容 API（推荐）](#openai-兼容-api推荐)
4. [Gradio Web Demo](#gradio-web-demo)
5. [Docker 容器化](#docker-容器化)
6. [模型导出](#模型导出)
7. [发布到 HuggingFace / ModelScope](#发布到-huggingface--modelscope)
8. [性能调优](#性能调优)
9. [常见问题](#常见问题)

---

## 硬件需求

### 训练硬件

| 模型 | 配置 | 最低显存 | 推荐设备 | 训练时长 |
|---|---|---|---|---|
| Tiny | tiny.yaml | CPU 即可 | MacBook | 2-5 分钟（仅冒烟） |
| Small | small.yaml | 8 GB MPS / VRAM | MacBook Pro / RTX 3060 | 6-12 小时 |
| **Base** | main.yaml | **24 GB VRAM** | **RTX 4090 / A100 80GB** | **12-18 小时** |
| **Plus** | plus.yaml | **40 GB VRAM (bf16)** | **A100 / A800 80GB** | **30-40 小时** |

### 推理硬件

| 模型 | 最低显存 (fp16) | 推荐 |
|---|---|---|
| Tiny / Small | 1-2 GB | CPU 也行 |
| Base | 1-2 GB | CPU + 8GB RAM 可跑 |
| Plus | 2-4 GB | RTX 3060 8GB+ / Apple Silicon |

> Plus 量化到 INT8/INT4 后能在 8GB 内存的设备上跑（用 llama.cpp）。

---

## 部署形态

```
训练完 outputs/dpo/final.pth (5-50MB fp16)
       │
       ├──► OpenAI 兼容 API（推荐生产）  → deploy/api_server.py
       │
       ├──► Gradio Web Demo（演示）       → deploy/web_demo.py
       │
       ├──► Docker 容器（云上自管）        → deploy/Dockerfile
       │
       └──► 上传 HF/ModelScope（公开发布） → deploy/export_model.py + push 工具(Phase 5)
                  │
                  └──► ollama / vllm / llama.cpp / Llama-Factory 即用
```

---

## OpenAI 兼容 API（推荐）

```bash
# 启动
python deploy/api_server.py --config configs/main.yaml \
    --model outputs/dpo/final.pth --port 8000

# 或 Plus
python deploy/api_server.py --config configs/plus.yaml --port 8000
```

支持端点：
- `POST /v1/chat/completions`：对话补全（支持 `stream: true` SSE 流式）
- `POST /v1/completions`：纯文本续写
- `GET  /v1/models`：列出模型
- `GET  /health`：健康检查

### 测试示例

```bash
# 非流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"你好"}],
    "max_tokens": 200,
    "temperature": 0.7
  }'

# 流式（SSE）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"讲个笑话"}],
    "stream": true
  }'

# Python（openai client）
pip install openai
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8000/v1', api_key='not-needed')
r = c.chat.completions.create(
    model='ClearMind',
    messages=[{'role':'user','content':'1+1=?'}]
)
print(r.choices[0].message.content)
"
```

### 与生态集成

ClearMind API 可被以下工具直接消费：
- **LangChain** / **LlamaIndex**：`base_url='http://your-host:8000/v1'`
- **Cherry Studio** / **OpenWebUI** / **AnythingLLM**：自定义 OpenAI provider
- **Continue / Cursor IDE 插件**：自定义 endpoint

---

## Gradio Web Demo

```bash
# 本机
python deploy/web_demo.py --config configs/main.yaml --port 7860

# 公网分享链接（gradio.live）
python deploy/web_demo.py --port 7860 --share
```

特性：
- 多轮对话历史
- 自定义 system prompt
- `open_thinking` 自适应思考开关（看模型推理过程）
- 实时调节 temperature / top_p / max_new_tokens

---

## Docker 容器化

```bash
# 构建
docker build -t clearmind -f deploy/Dockerfile .

# 启动 API（默认）
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/tokenizer:/app/tokenizer \
  -v $(pwd)/configs:/app/configs \
  clearmind

# 启动 Web Demo
docker run -p 7860:7860 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/tokenizer:/app/tokenizer \
  clearmind \
  python deploy/web_demo.py --host 0.0.0.0 --port 7860

# GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/tokenizer:/app/tokenizer \
  clearmind
```

---

## 模型导出

`deploy/export_model.py` 把 `.pth` ckpt 转换为 safetensors / fp16 .pth 等格式：

```bash
# 导出 safetensors（推荐发布）
python deploy/export_model.py \
    --input outputs/dpo/final.pth \
    --output release/clearmind-base.safetensors

# 导出 fp16 .pth
python deploy/export_model.py \
    --input outputs/dpo/final.pth \
    --output release/clearmind-base-fp16.pth \
    --dtype fp16

# 仅查看 state_dict 摘要（不导出）
python deploy/export_model.py --input outputs/dpo/final.pth --inspect
```

---

## 发布到 HuggingFace / ModelScope

> 🚧 完整 push 工具计划在 Phase 5 实施（`scripts/push_to_hub.py` / `push_to_modelscope.py`）。
> 在此之前可手动操作。

### 准备模型仓库

```bash
mkdir -p release/clearmind-base
cd release/clearmind-base

# 1. 模型权重
python ../../deploy/export_model.py \
    -i ../../outputs/dpo/final.pth \
    -o model.safetensors

# 2. 复制 tokenizer
cp -r ../../tokenizer/minimind/* .

# 3. 模型卡（README.md）
# 可以参考 minimind 的：https://huggingface.co/jingyaogong/minimind-3
```

### 上传到 HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli upload <你的用户名>/ClearMind-Base ./
```

### 上传到 ModelScope

```bash
pip install modelscope

# 在 modelscope 网站创建模型仓库后
modelscope upload <你的用户名>/ClearMind-Base ./
```

### Phase 5 完整流程预览

```bash
# 1. Qwen3 兼容导出（让权重能被 Qwen3ForCausalLM 加载 → ollama/vllm/llama.cpp 即用）
python scripts/convert_to_qwen3.py \
    --input outputs/dpo/final.pth \
    --config configs/main.yaml \
    --output release/clearmind-base-qwen3-format/

# 2. 一键 push 到 HF
python scripts/push_to_hub.py \
    --repo <your-username>/ClearMind-Base \
    --model-dir release/clearmind-base-qwen3-format/

# 3. 一键 push 到 ModelScope
python scripts/push_to_modelscope.py \
    --repo <your-username>/ClearMind-Base \
    --model-dir release/clearmind-base-qwen3-format/
```

---

## 性能调优

| 措施 | 收益 | 适用 |
|---|---|---|
| `.half()` 落盘 | 磁盘 ½ | 已默认开 |
| safetensors 格式 | 加载快 ×2-3，无 pickle 安全风险 | 推理推荐 |
| `torch.compile(model)`（Phase 4） | 推理 +20-30% | A100 / 3090+ |
| INT8 量化（llama.cpp） | 显存 ½，速度 +50% | 端侧部署 |
| INT4 量化（llama.cpp） | 显存 ¼ | 8GB 设备跑 Plus |
| KV cache（已默认开） | 推理 ×5-10 | 全部 |
| Flash Attention（已通过 SDPA） | attention +30-50% | A100/H100 |

---

## 常见问题

**Q: API server 启动但请求超时？**
A: 检查 `--config` 的 `max_seq_len` 是否过大。默认 1024 通常够用。

**Q: docker 镜像构建很慢？**
A: 本地有 `outputs/` 大文件时 build context 会很慢。`.dockerignore` 已经把 `outputs/`、`data/`、`venv/` 排除。

**Q: HuggingFace 上传失败？**
A: 文件 > 5GB 时 HF 强制走 LFS，需要先 `huggingface-cli lfs-enable-largefiles .`

**Q: ModelScope 国内访问比 HF 稳定吗？**
A: 是。国内推荐先发 ModelScope，再发 HF。两边可同步。

**Q: Plus 模型在端侧（手机/Mac mini）能跑吗？**
A: 直接用 transformers 跑：fp16 需要 ~1GB 内存，应该没问题。要更快用 llama.cpp 量化到 INT4，~150MB 内存就够。
