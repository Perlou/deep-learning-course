"""
DocuMind AI - Models 模块

数据库模型和实体定义
"""

from .database import (
    Base,
    get_async_db,
    get_async_engine,
    get_db,
    get_engine,
    init_async_db,
    init_db,
)
from .entities import (
    Chunk,
    Conversation,
    Document,
    DocumentStatus,
    Feedback,
    KnowledgeBase,
    Message,
    MessageRole,
)

__all__ = [
    # Database
    "Base",
    "get_engine",
    "get_async_engine",
    "get_db",
    "get_async_db",
    "init_db",
    "init_async_db",
    # Entities
    "KnowledgeBase",
    "Document",
    "DocumentStatus",
    "Chunk",
    "Conversation",
    "Message",
    "MessageRole",
    "Feedback",
]
