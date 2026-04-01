"""inference 模块 — 推理与生成"""

from .generate import generate_text, extract_reply, create_pipeline, generate_stream
from .chat import chat_loop

__all__ = [
    "generate_text",
    "extract_reply",
    "create_pipeline",
    "generate_stream",
    "chat_loop",
]
