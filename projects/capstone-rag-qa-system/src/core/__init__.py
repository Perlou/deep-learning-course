"""
DocuMind AI - Core 模块

核心业务逻辑
"""

from .chat_service import ChatService, ChatResult, StreamEvent, get_chat_service
from .chunker import TextChunk, TextChunker, create_chunker
from .document_processor import (
    DocumentProcessor,
    get_document_processor,
    process_document_task,
)
from .embedder import Embedder, get_embedder
from .llm_engine import LLMEngine, get_llm_engine, init_llm_engine
from .retriever import Retriever, RetrievalResult, get_retriever
from .vector_store import (
    VectorStore,
    VectorStoreManager,
    get_vector_store,
    get_vector_store_manager,
)

__all__ = [
    # Chunker
    "TextChunk",
    "TextChunker",
    "create_chunker",
    # Document Processor
    "DocumentProcessor",
    "get_document_processor",
    "process_document_task",
    # Embedder
    "Embedder",
    "get_embedder",
    # Vector Store
    "VectorStore",
    "VectorStoreManager",
    "get_vector_store",
    "get_vector_store_manager",
    # Retriever
    "Retriever",
    "RetrievalResult",
    "get_retriever",
    # LLM Engine
    "LLMEngine",
    "get_llm_engine",
    "init_llm_engine",
    # Chat Service
    "ChatService",
    "ChatResult",
    "StreamEvent",
    "get_chat_service",
]
