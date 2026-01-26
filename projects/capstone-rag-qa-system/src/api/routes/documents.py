"""
DocuMind AI - 文档路由

文档上传和管理接口
"""

import shutil
from typing import List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session, get_app_settings
from src.api.schemas import (
    ChunkResponse,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    PaginatedData,
    ResponseModel,
)
from src.models import Chunk, Document, DocumentStatus, KnowledgeBase
from src.utils import (
    format_file_size,
    generate_id,
    get_file_extension,
    get_file_hash,
    get_project_root,
    sanitize_filename,
)

router = APIRouter(prefix="/documents", tags=["文档"])


async def process_document_background(doc_id: str):
    """后台处理文档"""
    from src.core import get_document_processor
    from src.models import get_async_session_local

    AsyncSessionLocal = get_async_session_local()
    async with AsyncSessionLocal() as db:
        processor = get_document_processor()
        await processor.process_document(doc_id, db)


@router.post("/upload", response_model=ResponseModel[DocumentUploadResponse])
async def upload_document(
    background_tasks: BackgroundTasks,
    kb_id: str = Form(..., description="知识库 ID"),
    file: UploadFile = File(..., description="文档文件"),
    db: AsyncSession = Depends(get_db_session),
    settings=Depends(get_app_settings),
):
    """上传文档"""
    # 检查知识库是否存在
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 检查文件格式
    file_ext = get_file_extension(file.filename)
    if file_ext not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file_ext}，支持格式: {', '.join(settings.supported_formats)}",
        )

    # 检查文件大小
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > settings.storage.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制: {format_file_size(settings.storage.max_file_size)}",
        )

    # 生成文档 ID
    doc_id = generate_id("doc")

    # 保存文件
    project_root = get_project_root()
    upload_dir = project_root / settings.storage.upload_dir / kb_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = sanitize_filename(file.filename)
    file_path = upload_dir / f"{doc_id}_{safe_filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 计算文件哈希
    file_hash = get_file_hash(str(file_path))

    # 创建文档记录
    doc = Document(
        id=doc_id,
        kb_id=kb_id,
        filename=file.filename,
        file_type=file_ext,
        file_size=file_size,
        file_path=str(file_path.relative_to(project_root)),
        file_hash=file_hash,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)

    # 更新知识库统计
    kb.document_count += 1
    kb.total_size += file_size

    await db.commit()
    await db.refresh(doc)

    # 触发后台任务处理文档
    background_tasks.add_task(process_document_background, doc_id)

    return ResponseModel(
        data=DocumentUploadResponse(
            id=doc.id,
            kb_id=doc.kb_id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            status=doc.status.value,
            created_at=doc.created_at,
        )
    )


@router.post("/upload/batch", response_model=ResponseModel)
async def upload_documents_batch(
    kb_id: str = Form(..., description="知识库 ID"),
    files: List[UploadFile] = File(..., description="文档文件列表"),
    db: AsyncSession = Depends(get_db_session),
    settings=Depends(get_app_settings),
):
    """批量上传文档"""
    # 检查文件数量
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="一次最多上传 10 个文件")

    # 检查知识库是否存在
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    uploaded = []
    for file in files:
        try:
            # 检查文件格式
            file_ext = get_file_extension(file.filename)
            if file_ext not in settings.supported_formats:
                continue

            # 检查文件大小
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)

            if file_size > settings.storage.max_file_size:
                continue

            # 生成文档 ID 并保存文件
            doc_id = generate_id("doc")
            project_root = get_project_root()
            upload_dir = project_root / settings.storage.upload_dir / kb_id
            upload_dir.mkdir(parents=True, exist_ok=True)

            safe_filename = sanitize_filename(file.filename)
            file_path = upload_dir / f"{doc_id}_{safe_filename}"

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            file_hash = get_file_hash(str(file_path))

            # 创建文档记录
            doc = Document(
                id=doc_id,
                kb_id=kb_id,
                filename=file.filename,
                file_type=file_ext,
                file_size=file_size,
                file_path=str(file_path.relative_to(project_root)),
                file_hash=file_hash,
                status=DocumentStatus.PENDING,
            )
            db.add(doc)

            # 更新知识库统计
            kb.document_count += 1
            kb.total_size += file_size

            uploaded.append(
                {
                    "id": doc_id,
                    "filename": file.filename,
                    "status": "pending",
                }
            )
        except Exception as e:
            continue

    await db.commit()

    return ResponseModel(
        data={
            "uploaded": len(uploaded),
            "documents": uploaded,
        }
    )


@router.get("", response_model=ResponseModel[PaginatedData[DocumentResponse]])
async def list_documents(
    kb_id: str,
    status: str = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db_session),
):
    """获取文档列表"""
    # 构建查询
    query = select(Document).where(Document.kb_id == kb_id)
    count_query = (
        select(func.count()).select_from(Document).where(Document.kb_id == kb_id)
    )

    if status:
        try:
            status_enum = DocumentStatus(status)
            query = query.where(Document.status == status_enum)
            count_query = count_query.where(Document.status == status_enum)
        except ValueError:
            pass

    # 计算总数
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # 分页查询
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Document.created_at.desc())
    result = await db.execute(query)
    items = result.scalars().all()

    return ResponseModel(
        data=PaginatedData(
            items=[
                DocumentResponse(
                    id=item.id,
                    kb_id=item.kb_id,
                    filename=item.filename,
                    file_type=item.file_type,
                    file_size=item.file_size,
                    status=item.status.value,
                    error_message=item.error_message,
                    chunk_count=item.chunk_count,
                    created_at=item.created_at,
                    processed_at=item.processed_at,
                )
                for item in items
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get("/{doc_id}", response_model=ResponseModel[DocumentResponse])
async def get_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """获取文档详情"""
    query = select(Document).where(Document.id == doc_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    return ResponseModel(
        data=DocumentResponse(
            id=doc.id,
            kb_id=doc.kb_id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            status=doc.status.value,
            error_message=doc.error_message,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
            processed_at=doc.processed_at,
        )
    )


@router.get("/{doc_id}/status", response_model=ResponseModel[DocumentStatusResponse])
async def get_document_status(
    doc_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """获取文档处理状态"""
    query = select(Document).where(Document.id == doc_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 构建进度信息
    status = doc.status.value
    progress = 0
    current_step = None
    steps = []

    status_progress = {
        "pending": (0, "等待处理"),
        "parsing": (25, "正在解析文档..."),
        "chunking": (50, "正在分块处理..."),
        "embedding": (75, "正在生成向量嵌入..."),
        "completed": (100, "处理完成"),
        "failed": (0, "处理失败"),
    }

    if status in status_progress:
        progress, current_step = status_progress[status]

    all_steps = ["parsing", "chunking", "embedding"]
    for step in all_steps:
        step_status = "pending"
        if status == "completed" or status == "failed":
            if status == "completed":
                step_status = "completed"
            elif (
                all_steps.index(step) < all_steps.index(status)
                if status in all_steps
                else True
            ):
                step_status = "completed"
            else:
                step_status = "failed" if step == status else "pending"
        elif status in all_steps:
            step_idx = all_steps.index(status)
            current_idx = all_steps.index(step)
            if current_idx < step_idx:
                step_status = "completed"
            elif current_idx == step_idx:
                step_status = "in_progress"

        steps.append({"name": step, "status": step_status})

    return ResponseModel(
        data=DocumentStatusResponse(
            id=doc.id,
            status=status,
            progress=progress,
            current_step=current_step,
            steps=steps,
        )
    )


@router.get(
    "/{doc_id}/chunks", response_model=ResponseModel[PaginatedData[ChunkResponse]]
)
async def get_document_chunks(
    doc_id: str,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db_session),
):
    """获取文档分块"""
    # 检查文档是否存在
    doc_query = select(Document).where(Document.id == doc_id)
    doc_result = await db.execute(doc_query)
    doc = doc_result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 限制每页数量
    page_size = min(page_size, 50)

    # 计算总数
    count_query = select(func.count()).select_from(Chunk).where(Chunk.doc_id == doc_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # 分页查询
    offset = (page - 1) * page_size
    query = (
        select(Chunk)
        .where(Chunk.doc_id == doc_id)
        .offset(offset)
        .limit(page_size)
        .order_by(Chunk.chunk_index)
    )
    result = await db.execute(query)
    items = result.scalars().all()

    return ResponseModel(
        data=PaginatedData(
            items=[
                ChunkResponse(
                    id=item.id,
                    index=item.chunk_index,
                    content=item.content,
                    metadata=item.metadata,
                )
                for item in items
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.delete("/{doc_id}", response_model=ResponseModel)
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """删除文档"""
    query = select(Document).where(Document.id == doc_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 获取知识库
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    # 删除文件
    if doc.file_path:
        project_root = get_project_root()
        file_path = project_root / doc.file_path
        if file_path.exists():
            file_path.unlink()

    # 统计分块数量
    chunk_count = doc.chunk_count

    # 更新知识库统计
    if kb:
        kb.document_count = max(0, kb.document_count - 1)
        kb.chunk_count = max(0, kb.chunk_count - chunk_count)
        kb.total_size = max(0, kb.total_size - doc.file_size)

    # 删除文档（级联删除分块）
    await db.delete(doc)
    await db.commit()

    return ResponseModel(
        data={
            "deleted_chunks": chunk_count,
        }
    )
