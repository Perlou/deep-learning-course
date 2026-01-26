"""
DocuMind AI - 知识库路由

知识库 CRUD 接口
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.api.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    PaginatedData,
    ResponseModel,
)
from src.models import KnowledgeBase
from src.utils import generate_id

router = APIRouter(prefix="/kb", tags=["知识库"])


@router.post("", response_model=ResponseModel[KnowledgeBaseResponse])
async def create_knowledge_base(
    data: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """创建知识库"""
    kb = KnowledgeBase(
        id=generate_id("kb"),
        name=data.name,
        description=data.description,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)

    return ResponseModel(data=KnowledgeBaseResponse.model_validate(kb))


@router.get("", response_model=ResponseModel[PaginatedData[KnowledgeBaseResponse]])
async def list_knowledge_bases(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db_session),
):
    """获取知识库列表"""
    # 计算总数
    count_query = select(func.count()).select_from(KnowledgeBase)
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # 分页查询
    offset = (page - 1) * page_size
    query = (
        select(KnowledgeBase)
        .offset(offset)
        .limit(page_size)
        .order_by(KnowledgeBase.updated_at.desc())
    )
    result = await db.execute(query)
    items = result.scalars().all()

    return ResponseModel(
        data=PaginatedData(
            items=[KnowledgeBaseResponse.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get("/{kb_id}", response_model=ResponseModel[KnowledgeBaseResponse])
async def get_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """获取知识库详情"""
    query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    result = await db.execute(query)
    kb = result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    return ResponseModel(data=KnowledgeBaseResponse.model_validate(kb))


@router.put("/{kb_id}", response_model=ResponseModel[KnowledgeBaseResponse])
async def update_knowledge_base(
    kb_id: str,
    data: KnowledgeBaseUpdate,
    db: AsyncSession = Depends(get_db_session),
):
    """更新知识库"""
    query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    result = await db.execute(query)
    kb = result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    if data.name is not None:
        kb.name = data.name
    if data.description is not None:
        kb.description = data.description

    await db.commit()
    await db.refresh(kb)

    return ResponseModel(data=KnowledgeBaseResponse.model_validate(kb))


@router.delete("/{kb_id}", response_model=ResponseModel)
async def delete_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """删除知识库"""
    query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    result = await db.execute(query)
    kb = result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 统计删除的文档和分块数量
    doc_count = kb.document_count
    chunk_count = kb.chunk_count

    await db.delete(kb)
    await db.commit()

    return ResponseModel(
        data={
            "deleted_documents": doc_count,
            "deleted_chunks": chunk_count,
        }
    )
