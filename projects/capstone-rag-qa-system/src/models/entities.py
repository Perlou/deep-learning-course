"""
DocuMind AI - 数据库实体模型

定义所有数据库表结构
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .database import Base


class DocumentStatus(str, PyEnum):
    """文档处理状态"""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, PyEnum):
    """消息角色"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class KnowledgeBase(Base):
    """知识库表"""

    __tablename__ = "knowledge_bases"

    id = Column(String(32), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    # 统计信息
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    total_size = Column(Integer, default=0)  # 总文件大小（字节）

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    documents = relationship(
        "Document", back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    conversations = relationship(
        "Conversation", back_populates="knowledge_base", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "total_size": self.total_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Document(Base):
    """文档表"""

    __tablename__ = "documents"

    id = Column(String(32), primary_key=True)
    kb_id = Column(
        String(32), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )

    # 文件信息
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=True)  # 存储路径
    file_hash = Column(String(32), nullable=True)  # 文件 MD5

    # 处理状态
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    error_message = Column(Text, nullable=True)

    # 处理结果
    chunk_count = Column(Integer, default=0)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # 关系
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "kb_id": self.kb_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "status": self.status.value if self.status else None,
            "error_message": self.error_message,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat()
            if self.processed_at
            else None,
        }


class Chunk(Base):
    """文档分块表"""

    __tablename__ = "chunks"

    id = Column(String(32), primary_key=True)
    doc_id = Column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    # 分块内容
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # 在文档中的顺序

    # 元数据
    metadata = Column(JSON, nullable=True)  # 页码、位置等信息

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    document = relationship("Document", back_populates="chunks")

    def to_dict(self):
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


class Conversation(Base):
    """对话表"""

    __tablename__ = "conversations"

    id = Column(String(32), primary_key=True)
    kb_id = Column(
        String(32), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )

    # 对话信息
    title = Column(String(200), nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    knowledge_base = relationship("KnowledgeBase", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "kb_id": self.kb_id,
            "title": self.title,
            "message_count": len(self.messages) if self.messages else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Message(Base):
    """消息表"""

    __tablename__ = "messages"

    id = Column(String(32), primary_key=True)
    conversation_id = Column(
        String(32), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )

    # 消息内容
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)

    # 来源引用（仅 assistant 消息）
    sources = Column(JSON, nullable=True)

    # Token 使用量
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role.value if self.role else None,
            "content": self.content,
            "sources": self.sources,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Feedback(Base):
    """用户反馈表"""

    __tablename__ = "feedbacks"

    id = Column(String(32), primary_key=True)
    message_id = Column(
        String(32), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )

    # 反馈内容
    rating = Column(String(10), nullable=False)  # good, bad
    comment = Column(Text, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "message_id": self.message_id,
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
