"""
api_server.py — OpenAI 兼容 REST API（FastAPI + SSE 流式）
=========================================================

将训练好的 ClearMind 模型暴露为 OpenAI 兼容的 HTTP API，可被任意 OpenAI 客户端
（``openai-python``、``LangChain``、``Cherry Studio``、``OpenWebUI`` 等）直接调用。

支持端点:
  POST /v1/chat/completions    对话补全（兼容 OpenAI Chat Completions 协议）
  POST /v1/completions          文本续写（不走 chat_template）
  GET  /v1/models               列出可用模型
  GET  /health                  健康检查

启动:
  python deploy/api_server.py --config configs/main.yaml \\
      --model outputs/dpo/final.pth --port 8000

测试:
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages": [{"role": "user", "content": "你好"}], "stream": false}'

  # 流式
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages": [{"role": "user", "content": "讲个笑话"}], "stream": true}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 模型 / Tokenizer 全局状态（启动时初始化）
# ============================================================


class _ServerState:
    model = None
    tokenizer = None
    device = None
    model_id: str = "ClearMind"
    config_path: str = ""

    @classmethod
    def is_ready(cls) -> bool:
        return cls.model is not None


def _load_runtime(config_path: str, model_path: str, tokenizer_path: str | None) -> None:
    """初始化模型 + tokenizer 到 _ServerState"""
    import torch
    import yaml
    from src.model.config import ModelConfig
    from src.model.gpt import GPT
    from src.training.trainer_utils import get_device, load_checkpoint
    from scripts.train import load_tokenizer

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config = ModelConfig(**config["model"])
    tokenizer = load_tokenizer(config, tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    device = get_device()
    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    _ServerState.model = model
    _ServerState.tokenizer = tokenizer
    _ServerState.device = device
    _ServerState.config_path = config_path
    # release 块中的 model_name 优先；否则文件名
    release = config.get("release", {})
    _ServerState.model_id = release.get("model_name") or Path(model_path).parent.parent.name or "ClearMind"

    print(f"✅ 模型加载完成: {model_path}")
    print(f"   tokenizer: {tokenizer}")
    print(f"   device: {device}")
    print(f"   model_id: {_ServerState.model_id}")


# ============================================================
# 生成核心
# ============================================================


def _build_prompt_ids(messages: list[dict], tools=None, open_thinking=False):
    """messages → token ids"""
    tk = _ServerState.tokenizer
    text = tk.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        open_thinking=open_thinking,
    )
    return tk.encode(text, add_bos=False, add_eos=False)


def _generate_full(messages, *, max_tokens, temperature, top_p, top_k, repetition_penalty, tools, open_thinking) -> tuple[str, int, int]:
    """非流式生成。返回 (text, prompt_token_count, completion_token_count)"""
    import torch
    from src.inference.generate import generate

    tk = _ServerState.tokenizer
    model = _ServerState.model
    device = _ServerState.device

    prompt_ids = _build_prompt_ids(messages, tools, open_thinking)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k or 0,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tk.eos_id,
    )
    new_ids = output_ids[0, prompt_len:].tolist()
    if tk.eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(tk.eos_id)]
    text = tk.decode(new_ids, skip_special_tokens=True)
    return text, prompt_len, len(new_ids)


def _generate_stream_chunks(messages, *, max_tokens, temperature, top_p, top_k, repetition_penalty, tools, open_thinking):
    """逐 token 生成（简化版：用单步前向 + KV cache 一步步出 token）

    实现策略：调用一次 model.generate()，但每步追加一个 token 后 yield 增量字符串。
    这里用最简单的方式：先全量生成，再分块下发（不是真正的 token-by-token 流式，
    但对客户端协议层来说看起来一样）。
    """
    text, prompt_len, completion_len = _generate_full(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        tools=tools,
        open_thinking=open_thinking,
    )
    # 按字符切块（每 4 字符一片）模拟流式
    chunk_size = 4
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
    # 末尾返回 token 计数信息
    yield {"_done": True, "prompt": prompt_len, "completion": completion_len}


# ============================================================
# FastAPI app
# ============================================================


def build_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请运行: pip install fastapi uvicorn pydantic")
        sys.exit(1)

    app = FastAPI(title="ClearMind API", version="1.0")

    class Message(BaseModel):
        role: str
        content: str
        # 可选 minimind / Qwen 风格扩展
        reasoning_content: str | None = None
        tools: str | None = None
        tool_calls: list | None = None

    class ChatCompletionRequest(BaseModel):
        model: str | None = None
        messages: list[Message]
        max_tokens: int = Field(default=512, ge=1, le=8192)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        top_k: int = Field(default=50, ge=0)
        repetition_penalty: float = Field(default=1.1, ge=0.0)
        stream: bool = False
        tools: list | None = None
        open_thinking: bool = False

    class CompletionRequest(BaseModel):
        model: str | None = None
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.1
        stream: bool = False

    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": _ServerState.is_ready()}

    @app.get("/v1/models")
    async def list_models():
        if not _ServerState.is_ready():
            raise HTTPException(503, "Model not loaded")
        return {
            "object": "list",
            "data": [
                {
                    "id": _ServerState.model_id,
                    "object": "model",
                    "owned_by": "clearmind",
                }
            ],
        }

    def _make_response(content: str, prompt_tokens: int, completion_tokens: int) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _ServerState.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not _ServerState.is_ready():
            raise HTTPException(503, "Model not loaded")

        # pydantic Message → dict
        messages = [m.model_dump(exclude_none=True) for m in req.messages]

        # 流式
        if req.stream:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            async def event_stream() -> AsyncGenerator[str, None]:
                # 发送 role 开头
                yield "data: " + json.dumps({
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": _ServerState.model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }, ensure_ascii=False) + "\n\n"

                for chunk in _generate_stream_chunks(
                    messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    tools=req.tools,
                    open_thinking=req.open_thinking,
                ):
                    if isinstance(chunk, dict) and chunk.get("_done"):
                        # 终止 chunk
                        yield "data: " + json.dumps({
                            "id": chat_id, "object": "chat.completion.chunk",
                            "created": created, "model": _ServerState.model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }, ensure_ascii=False) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    yield "data: " + json.dumps({
                        "id": chat_id, "object": "chat.completion.chunk",
                        "created": created, "model": _ServerState.model_id,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                    }, ensure_ascii=False) + "\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # 非流式
        text, prompt_t, completion_t = _generate_full(
            messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            tools=req.tools,
            open_thinking=req.open_thinking,
        )
        return _make_response(text, prompt_t, completion_t)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if not _ServerState.is_ready():
            raise HTTPException(503, "Model not loaded")
        # 走非 chat 路径：直接 encode prompt → generate
        from src.inference.generate import generate_text

        text = generate_text(
            model=_ServerState.model,
            tokenizer=_ServerState.tokenizer,
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            device=_ServerState.device,
            add_bos=True,  # 续写场景用 BOS
            return_only_new=True,
        )
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": _ServerState.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        }

    return app


# ============================================================
# 入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="ClearMind OpenAI 兼容 API")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--model", default=None, help="checkpoint 路径")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # 自动找模型
    model_path = args.model
    if model_path is None:
        for cand in ("outputs/dpo/final.pth", "outputs/sft/final.pth"):
            if os.path.exists(cand):
                model_path = cand
                break
    if model_path is None or not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        sys.exit(1)

    _load_runtime(args.config, model_path, args.tokenizer)

    try:
        import uvicorn
    except ImportError:
        print("❌ 请安装: pip install uvicorn fastapi")
        sys.exit(1)

    app = build_app()
    print(f"\n🚀 启动 API 服务于 http://{args.host}:{args.port}")
    print("    OpenAI 兼容 endpoints:")
    print("      POST /v1/chat/completions")
    print("      POST /v1/completions")
    print("      GET  /v1/models")
    print("      GET  /health")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
