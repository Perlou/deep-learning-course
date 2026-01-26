"""
DocuMind AI - API Schemas

Pydantic 请求和响应模型
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ============================================
# 通用响应模型
# ============================================


class ResponseModel(BaseModel, Generic[T]):
    """通用响应模型"""

    code: int = 0
    message: str = "success"
    data: Optional[T] = None


class PaginatedData(BaseModel, Generic[T]):
    """分页数据模型"""

    items: List[T]
    total: int
    page: int
    page_size: int


# ============================================
# 知识库模型
# ============================================


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求"""

    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: Optional[str] = Field(None, max_length=500, description="知识库描述")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求"""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""

    id: str
    name: str
    description: Optional[str]
    document_count: int
    chunk_count: int
    total_size: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# 文档模型
# ============================================


class DocumentResponse(BaseModel):
    """文档响应"""

    id: str
    kb_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    error_message: Optional[str]
    chunk_count: int
    created_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True


class DocumentStatusResponse(BaseModel):
    """文档状态响应"""

    id: str
    status: str
    progress: int = 0
    current_step: Optional[str] = None
    steps: List[Dict[str, str]] = []


class ChunkResponse(BaseModel):
    """分块响应"""

    id: str
    index: int
    content: str
    metadata: Optional[Dict[str, Any]]


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""

    id: str
    kb_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: datetime


# ============================================
# 问答模型
# ============================================


class ChatOptions(BaseModel):
    """问答选项"""

    top_k: int = Field(5, ge=1, le=20, description="检索数量")
    temperature: float = Field(0.7, ge=0, le=2, description="生成温度")
    max_tokens: int = Field(512, ge=64, le=4096, description="最大生成长度")


class ChatRequest(BaseModel):
    """问答请求"""

    kb_id: str = Field(..., description="知识库 ID")
    conversation_id: Optional[str] = Field(None, description="对话 ID，用于多轮对话")
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    options: Optional[ChatOptions] = None


class SourceReference(BaseModel):
    """来源引用"""

    doc_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class UsageInfo(BaseModel):
    """Token 使用量"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """问答响应"""

    message_id: str
    conversation_id: str
    answer: str
    sources: List[SourceReference]
    usage: Optional[UsageInfo] = None
    created_at: datetime


# ============================================
# 对话模型
# ============================================


class ConversationResponse(BaseModel):
    """对话响应"""

    id: str
    kb_id: str
    title: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """消息响应"""

    id: str
    role: str
    content: str
    sources: Optional[List[SourceReference]]
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    """对话详情响应"""

    id: str
    kb_id: str
    title: Optional[str]
    messages: List[MessageResponse]
    created_at: datetime
    updated_at: datetime


# ============================================
# 反馈模型
# ============================================


class FeedbackCreate(BaseModel):
    """创建反馈请求"""

    message_id: str
    rating: str = Field(..., pattern="^(good|bad)$", description="评价：good 或 bad")
    comment: Optional[str] = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    """反馈响应"""

    feedback_id: str


# ============================================
# 系统模型
# ============================================


class ComponentHealth(BaseModel):
    """组件健康状态"""

    database: str
    vector_store: str
    embedding_model: str
    llm: str


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    version: str
    components: ComponentHealth
    uptime: int


class ModelsInfo(BaseModel):
    """模型信息"""

    embedding: str
    llm: str


class LimitsInfo(BaseModel):
    """限制信息"""

    max_file_size: int
    supported_formats: List[str]
    max_batch_upload: int


class SystemInfoResponse(BaseModel):
    """系统信息响应"""

    version: str
    models: ModelsInfo
    limits: LimitsInfo


class StatsResponse(BaseModel):
    """统计信息响应"""

    knowledge_bases: int
    documents: int
    chunks: int
    conversations: int
    messages: int
    storage_used: int
