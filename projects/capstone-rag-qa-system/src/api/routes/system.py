"""
DocuMind AI - 系统路由

系统健康检查和信息接口
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_app_settings, get_db_session
from src.api.schemas import (
    ComponentHealth,
    HealthResponse,
    LimitsInfo,
    ModelsInfo,
    ResponseModel,
    StatsResponse,
    SystemInfoResponse,
)
from src.models import Chunk, Conversation, Document, KnowledgeBase, Message
from src.utils import get_settings

router = APIRouter(prefix="/system", tags=["系统"])

# 记录启动时间
_start_time = time.time()


@router.get("/health", response_model=ResponseModel[HealthResponse])
async def health_check(
    db: AsyncSession = Depends(get_db_session),
    settings=Depends(get_app_settings),
):
    """健康检查"""
    # 检查数据库连接
    db_status = "healthy"
    try:
        await db.execute(select(1))
    except Exception:
        db_status = "unhealthy"

    # 其他组件状态（占位，后续实现真正的检查）
    components = ComponentHealth(
        database=db_status,
        vector_store="healthy",  # TODO: 检查 FAISS 索引
        embedding_model="not_loaded",  # TODO: 检查嵌入模型
        llm="not_loaded",  # TODO: 检查 LLM
    )

    # 计算运行时间
    uptime = int(time.time() - _start_time)

    overall_status = "healthy" if db_status == "healthy" else "degraded"

    return ResponseModel(
        data=HealthResponse(
            status=overall_status,
            version=settings.app.version,
            components=components,
            uptime=uptime,
        )
    )


@router.get("/info", response_model=ResponseModel[SystemInfoResponse])
async def system_info(
    settings=Depends(get_app_settings),
):
    """获取系统信息"""
    return ResponseModel(
        data=SystemInfoResponse(
            version=settings.app.version,
            models=ModelsInfo(
                embedding=settings.models.embedding.name,
                llm=settings.models.llm.name,
            ),
            limits=LimitsInfo(
                max_file_size=settings.storage.max_file_size,
                supported_formats=settings.supported_formats,
                max_batch_upload=10,
            ),
        )
    )


@router.get("/stats", response_model=ResponseModel[StatsResponse])
async def system_stats(
    db: AsyncSession = Depends(get_db_session),
):
    """获取统计信息"""
    # 统计各项数据
    kb_count = (
        await db.execute(select(func.count()).select_from(KnowledgeBase))
    ).scalar()
    doc_count = (await db.execute(select(func.count()).select_from(Document))).scalar()
    chunk_count = (await db.execute(select(func.count()).select_from(Chunk))).scalar()
    conv_count = (
        await db.execute(select(func.count()).select_from(Conversation))
    ).scalar()
    msg_count = (await db.execute(select(func.count()).select_from(Message))).scalar()

    # 计算存储使用量
    storage_result = await db.execute(select(func.sum(Document.file_size)))
    storage_used = storage_result.scalar() or 0

    return ResponseModel(
        data=StatsResponse(
            knowledge_bases=kb_count,
            documents=doc_count,
            chunks=chunk_count,
            conversations=conv_count,
            messages=msg_count,
            storage_used=storage_used,
        )
    )
