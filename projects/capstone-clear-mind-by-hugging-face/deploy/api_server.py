"""
api_server.py — FastAPI REST API 服务 (HF 版)
===============================================

提供 HTTP API 接口，兼容 OpenAI Chat Completions 格式。

from-scratch 对比:
  - from-scratch: GPT + load_checkpoint + 手写 generate_text
  - HF 版: from_pretrained + model.generate()

端点:
  - /v1/chat/completions  对话补全 (兼容 OpenAI)
  - /v1/completions       文本续写
  - /health               健康检查
  - /model/info           模型信息

启动:
  pip install fastapi uvicorn
  python deploy/api_server.py --model outputs/sft --tokenizer outputs/tokenizer

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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from inference.generate import generate_text, extract_reply


# ============================================================
# 请求/响应模型 (兼容 OpenAI API)
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


class CompletionRequest(BaseModel):
    model: str = "clearmind"
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=300, ge=1, le=2048)


# ============================================================
# 全局模型
# ============================================================

app = FastAPI(
    title="ClearMind API (HF)",
    description="ClearMind-HF 大语言模型 API 服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tokenizer = None
_model_info = {}
_inference_lock = asyncio.Lock()


# ============================================================
# API 端点
# ============================================================


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info")
async def get_model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return _model_info


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """对话补全 (兼容 OpenAI 格式)"""
    if _model is None:
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

    async with _inference_lock:
        reply = await asyncio.to_thread(_generate, prompt, request)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "clearmind",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """文本续写"""
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    async with _inference_lock:
        output = await asyncio.to_thread(_generate, request.prompt, request)

    response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": response_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": "clearmind",
        "choices": [{"index": 0, "text": output, "finish_reason": "stop"}],
    }


def _generate(prompt: str, request) -> str:
    """同步生成回复 (使用 model.generate)"""
    output = generate_text(
        model=_model,
        tokenizer=_tokenizer,
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
    return extract_reply(output, prompt)


# ============================================================
# 模型加载与启动
# ============================================================


def load_model(model_path: str, tokenizer_path: str):
    global _model, _tokenizer, _model_info

    print(f"加载模型: {model_path}")
    _model = ClearMindForCausalLM.from_pretrained(model_path)
    _model.eval()

    print(f"加载 tokenizer: {tokenizer_path}")
    _tokenizer = ClearMindTokenizer.load(tokenizer_path)

    config = _model.config
    param_count = sum(p.numel() for p in _model.parameters())

    _model_info = {
        "model_name": "ClearMind-HF",
        "parameters": f"{param_count / 1e6:.1f}M",
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
    }

    print(f"模型加载完成: {param_count / 1e6:.1f}M params")


def main():
    parser = argparse.ArgumentParser(description="ClearMind API 服务 (HF 版)")
    parser.add_argument("--model", type=str, help="HF 格式模型目录")
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizer")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.model:
        for path in ["outputs/dpo", "outputs/sft", "outputs/pretrain"]:
            if os.path.exists(path):
                args.model = path
                break
        else:
            print("未找到模型目录")
            sys.exit(1)

    load_model(args.model, args.tokenizer)

    import uvicorn
    print(f"\n启动 API: http://{args.host}:{args.port}")
    print(f"文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
