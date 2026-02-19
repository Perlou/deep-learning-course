"""
api_server.py — FastAPI REST API 服务
======================================

提供 HTTP API 接口，兼容 OpenAI Chat Completions 格式。

支持:
  - /v1/chat/completions  对话补全 (兼容 OpenAI 格式)
  - /v1/completions       文本续写
  - /health               健康检查
  - /model/info           模型信息

启动方式:
  python deploy/api_server.py --model outputs/dpo/final.pth --port 8000

测试:
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages": [{"role": "user", "content": "你好"}]}'
"""

import os
import sys
import time
import uuid
import argparse
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer
from src.inference.generate import generate_text
from src.training.trainer_utils import get_device, load_checkpoint


# ============================================================
# 请求/响应模型 (兼容 OpenAI API 格式)
# ============================================================


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "clearmind"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=300, ge=1, le=2048)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "clearmind"
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=300, ge=1, le=2048)
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "clearmind"
    choices: list[dict]
    usage: dict


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str = "clearmind"
    choices: list[dict]


# ============================================================
# 全局模型实例
# ============================================================

app = FastAPI(
    title="ClearMind API",
    description="ClearMind 大语言模型 API 服务",
    version="1.0.0",
)

# CORS 配置 (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型引用
model = None
tokenizer = None
device = None
model_name = "clearmind"
model_info = {}

# 推理锁 (单模型不支持并发推理)
inference_lock = asyncio.Lock()


# ============================================================
# API 端点
# ============================================================


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.get("/model/info")
async def get_model_info():
    """模型信息"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return model_info


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """对话补全 (兼容 OpenAI 格式)"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    # 构建对话 prompt
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "user":
            prompt_parts.append(f"Human: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
        elif msg.role == "system":
            prompt_parts.append(msg.content)

    prompt = "\n".join(prompt_parts) + "\nAssistant: "

    if request.stream:
        return StreamingResponse(
            _stream_generate(prompt, request),
            media_type="text/event-stream",
        )

    # 非流式生成
    async with inference_lock:
        reply = await asyncio.to_thread(_generate_reply, prompt, request)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    return ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(reply)),
            "total_tokens": len(tokenizer.encode(prompt))
            + len(tokenizer.encode(reply)),
        },
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """文本续写"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    async with inference_lock:
        output = await asyncio.to_thread(_generate_reply, request.prompt, request)

    response_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    return CompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            {
                "index": 0,
                "text": output,
                "finish_reason": "stop",
            }
        ],
    )


# ============================================================
# 生成逻辑
# ============================================================


def _generate_reply(prompt: str, request) -> str:
    """同步生成回复"""
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        device=device,
    )

    # 提取 Assistant 回复
    if "Assistant: " in output:
        reply = output.split("Assistant: ")[-1]
    else:
        reply = output

    # 清理特殊 token
    reply = reply.replace("</s>", "").replace("<s>", "").strip()
    return reply


async def _stream_generate(prompt: str, request):
    """流式生成 (SSE 格式)"""
    import json

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # 编码 prompt
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.eval()
    generated_text = ""
    max_seq_len = getattr(model.config, "max_seq_len", 512)

    with torch.no_grad():
        for _ in range(request.max_tokens):
            cond_ids = input_tensor[:, -max_seq_len:]
            logits, _ = model(cond_ids)
            logits = logits[:, -1, :]

            # Temperature
            if request.temperature > 0:
                logits = logits / request.temperature
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                token_text = tokenizer.decode([next_token.item()])
                if next_token.item() == tokenizer.eos_id:
                    break
                generated_text += token_text
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                continue

            # Top-k
            if request.top_k > 0:
                import torch.nn.functional as F

                top_k_vals, _ = torch.topk(logits, min(request.top_k, logits.size(-1)))
                min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top_k,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            # 采样
            import torch.nn.functional as F

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_tensor = torch.cat([input_tensor, next_token], dim=1)

            if next_token.item() == tokenizer.eos_id:
                break

            # 解码并发送
            token_text = tokenizer.decode([next_token.item()])
            generated_text += token_text

            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)  # 让出事件循环

    # 发送结束信号
    end_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================
# 模型加载与启动
# ============================================================


def load_model(config_path: str, model_path: str, tokenizer_path: str):
    """加载模型到全局变量"""
    global model, tokenizer, device, model_info, model_name

    device = get_device()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])

    # 加载分词器
    tokenizer = ClearMindTokenizer(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 加载模型
    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    param_info = model.count_parameters()

    # 根据模型路径推断阶段
    if "dpo" in model_path:
        stage = "DPO (偏好对齐)"
    elif "sft" in model_path:
        stage = "SFT (指令微调)"
    else:
        stage = "Pretrain (预训练)"

    model_info = {
        "model_name": "ClearMind",
        "stage": stage,
        "parameters": f"{param_info['total_millions']:.1f}M",
        "d_model": model_config.d_model,
        "n_layers": model_config.n_layers,
        "n_heads": model_config.n_heads,
        "vocab_size": model_config.vocab_size,
        "max_seq_len": model_config.max_seq_len,
        "device": str(device),
        "checkpoint": model_path,
    }

    print(f"\n✅ 模型加载完成!")
    print(f"   模型: ClearMind ({param_info['total_millions']:.1f}M)")
    print(f"   阶段: {stage}")
    print(f"   设备: {device}")


def main():
    parser = argparse.ArgumentParser(description="ClearMind API 服务")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/dpo/final.pth",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.tokenizer):
        print(f"❌ 分词器不存在: {args.tokenizer}")
        sys.exit(1)

    if not os.path.exists(args.model):
        # 自动查找
        for path in [
            "outputs/dpo/final.pth",
            "outputs/sft/final.pth",
            "outputs/pretrain/final.pth",
        ]:
            if os.path.exists(path):
                args.model = path
                break
        else:
            print("❌ 未找到任何 checkpoint，请先完成训练")
            sys.exit(1)

    # 加载模型
    load_model(args.config, args.model, args.tokenizer)

    # 启动服务
    import uvicorn

    print(f"\n🚀 启动 API 服务: http://{args.host}:{args.port}")
    print(f"   文档: http://{args.host}:{args.port}/docs")
    print(f"   健康检查: http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
