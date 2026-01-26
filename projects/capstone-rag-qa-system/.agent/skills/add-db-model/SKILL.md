---
name: add-db-model
description: 为 DocuMind AI 添加新的数据库模型
---

# 添加数据库模型技能

此技能用于为 DocuMind AI 项目添加新的 SQLAlchemy 数据库模型。

## 项目结构

```
projects/capstone-rag-qa-system/src/models/
├── __init__.py      # 模块导出
├── database.py      # 数据库配置
└── entities.py      # 实体定义（所有模型在此文件）
```

## 现有模型

| 模型          | 说明   | 主要字段                           |
| ------------- | ------ | ---------------------------------- |
| KnowledgeBase | 知识库 | id, name, description              |
| Document      | 文档   | id, kb_id, filename, status        |
| Chunk         | 分块   | id, doc_id, content, chunk_index   |
| Conversation  | 对话   | id, kb_id, title                   |
| Message       | 消息   | id, conversation_id, role, content |
| Feedback      | 反馈   | id, message_id, rating             |

## 添加新模型步骤

### 1. 在 `entities.py` 定义模型

```python
class NewEntity(Base):
    """新实体表"""
    __tablename__ = "new_entities"

    # 主键
    id = Column(String(32), primary_key=True)

    # 外键（如果有）
    parent_id = Column(String(32), ForeignKey("parents.id", ondelete="CASCADE"), nullable=False)

    # 字段
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), default="active")
    data = Column(JSON, nullable=True)  # JSON 字段

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    parent = relationship("Parent", back_populates="children")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
```

### 2. 在 `__init__.py` 导出模型

```python
from .entities import (
    # ... 已有模型
    NewEntity,
)

__all__ = [
    # ... 已有导出
    "NewEntity",
]
```

### 3. 重新初始化数据库

```bash
# 删除旧数据库（开发环境）
rm data/db/documind.db

# 重新初始化
python scripts/init_db.py
```

## 常用字段类型

```python
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)

# 字符串
name = Column(String(100), nullable=False)

# 长文本
content = Column(Text, nullable=True)

# 整数
count = Column(Integer, default=0)

# 浮点数
score = Column(Float, nullable=True)

# 布尔值
is_active = Column(Boolean, default=True)

# 日期时间
created_at = Column(DateTime, default=datetime.utcnow)

# JSON
metadata = Column(JSON, nullable=True)

# 枚举
status = Column(Enum(StatusEnum), default=StatusEnum.PENDING)
```

## 定义枚举

```python
from enum import Enum as PyEnum

class StatusEnum(str, PyEnum):
    """状态枚举"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
```

## 定义关系

### 一对多关系

```python
# 父表
class Parent(Base):
    __tablename__ = "parents"
    id = Column(String(32), primary_key=True)
    # 反向关系
    children = relationship("Child", back_populates="parent", cascade="all, delete-orphan")

# 子表
class Child(Base):
    __tablename__ = "children"
    id = Column(String(32), primary_key=True)
    parent_id = Column(String(32), ForeignKey("parents.id", ondelete="CASCADE"))
    # 正向关系
    parent = relationship("Parent", back_populates="children")
```

### 多对多关系

```python
# 关联表
tag_document = Table(
    "tag_document",
    Base.metadata,
    Column("tag_id", ForeignKey("tags.id"), primary_key=True),
    Column("document_id", ForeignKey("documents.id"), primary_key=True),
)

class Tag(Base):
    __tablename__ = "tags"
    id = Column(String(32), primary_key=True)
    documents = relationship("Document", secondary=tag_document, back_populates="tags")

class Document(Base):
    # ... 其他字段
    tags = relationship("Tag", secondary=tag_document, back_populates="documents")
```

## 索引优化

```python
from sqlalchemy import Index

class Document(Base):
    __tablename__ = "documents"
    # ... 字段定义

    # 创建索引
    __table_args__ = (
        Index("idx_documents_kb_id", "kb_id"),
        Index("idx_documents_status", "status"),
        Index("idx_documents_created_at", "created_at"),
    )
```

## 注意事项

1. **主键格式**：使用 `generate_id("prefix")` 生成，如 `kb_abc123`
2. **外键删除**：使用 `ondelete="CASCADE"` 确保级联删除
3. **时间戳**：始终添加 `created_at` 和 `updated_at`
4. **to_dict()**：为每个模型添加 `to_dict()` 方法便于序列化
5. **可空字段**：明确指定 `nullable=True` 或 `nullable=False`
