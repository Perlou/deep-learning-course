"""
DocuMind AI - API 路由模块
"""

from .chat import router as chat_router
from .documents import router as documents_router
from .knowledge_base import router as kb_router
from .system import router as system_router

__all__ = [
    "kb_router",
    "documents_router",
    "chat_router",
    "system_router",
]
