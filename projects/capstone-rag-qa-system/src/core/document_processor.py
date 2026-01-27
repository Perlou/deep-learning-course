"""
DocuMind AI - 文档处理服务

协调文档解析、分块、向量化和持久化
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Chunk, Document, DocumentStatus, KnowledgeBase
from src.parsers import ParserFactory
from src.utils import get_project_root, get_settings, log

from .chunker import TextChunker, create_chunker
from .embedder import get_embedder
from .vector_store import get_vector_store_manager


class DocumentProcessor:
    """
    文档处理服务

    负责文档的完整处理流程：解析 → 分块 → 向量化 → 存储
    """

    def __init__(self, chunker: Optional[TextChunker] = None):
        """
        初始化处理器

        Args:
            chunker: 文本分块器，如果不提供则使用默认配置
        """
        self.chunker = chunker or create_chunker()
        self.settings = get_settings()

    async def process_document(
        self,
        doc_id: str,
        db: AsyncSession,
    ) -> bool:
        """
        处理单个文档

        Args:
            doc_id: 文档 ID
            db: 数据库会话

        Returns:
            是否处理成功
        """
        # 获取文档
        query = select(Document).where(Document.id == doc_id)
        result = await db.execute(query)
        doc = result.scalar_one_or_none()

        if not doc:
            log.error(f"文档不存在: {doc_id}")
            return False

        try:
            # 更新状态：解析中
            doc.status = DocumentStatus.PARSING
            await db.commit()
            log.info(f"开始处理文档: {doc.filename}")

            # 1. 解析文档
            project_root = get_project_root()
            file_path = project_root / doc.file_path

            parse_result = ParserFactory.parse(str(file_path))

            if not parse_result.success:
                await self._mark_failed(doc, parse_result.error, db)
                return False

            log.info(
                f"文档解析成功: {doc.filename}, 内容长度: {len(parse_result.content)}"
            )

            # 更新状态：分块中
            doc.status = DocumentStatus.CHUNKING
            await db.commit()

            # 2. 分块
            if parse_result.pages:
                chunks = self.chunker.split_with_pages(
                    parse_result.pages,
                    doc_id=doc.id,
                )
            else:
                chunks = self.chunker.split(
                    parse_result.content,
                    doc_id=doc.id,
                    extra_metadata=parse_result.metadata,
                )

            if not chunks:
                await self._mark_failed(doc, "文档分块结果为空", db)
                return False

            log.info(f"文档分块完成: {doc.filename}, 分块数: {len(chunks)}")

            # 更新状态：嵌入中
            doc.status = DocumentStatus.EMBEDDING
            await db.commit()

            # 3. 生成向量嵌入
            embedder = get_embedder()
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embedder.embed_documents(chunk_texts, show_progress=False)

            log.info(f"向量嵌入完成: {doc.filename}, 向量形状: {embeddings.shape}")

            # 4. 保存到向量存储
            store_manager = get_vector_store_manager()
            vector_store = store_manager.get_store(doc.kb_id)

            # 构建元数据
            vector_metadata = []
            for chunk in chunks:
                meta = {
                    "chunk_id": chunk.id,
                    "doc_id": doc.id,
                    "content": chunk.content,
                    "filename": doc.filename,
                    **chunk.metadata,
                }
                vector_metadata.append(meta)

            vector_store.add(embeddings, vector_metadata)

            # 持久化索引
            store_manager.save_store(doc.kb_id)

            log.info(f"向量索引已更新: kb={doc.kb_id}, 总向量数={vector_store.size}")

            # 5. 保存分块到数据库
            for chunk in chunks:
                db_chunk = Chunk(
                    id=chunk.id,
                    doc_id=doc.id,
                    content=chunk.content,
                    chunk_index=chunk.index,
                    metadata=chunk.metadata,
                )
                db.add(db_chunk)

            # 更新文档状态
            doc.status = DocumentStatus.COMPLETED
            doc.chunk_count = len(chunks)
            doc.processed_at = datetime.utcnow()

            # 更新知识库统计
            kb_query = select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
            kb_result = await db.execute(kb_query)
            kb = kb_result.scalar_one_or_none()
            if kb:
                kb.chunk_count += len(chunks)

            await db.commit()

            log.info(f"文档处理完成: {doc.filename}")
            return True

        except Exception as e:
            log.exception(f"文档处理失败: {doc.filename}")
            await self._mark_failed(doc, str(e), db)
            return False

    async def _mark_failed(
        self,
        doc: Document,
        error: str,
        db: AsyncSession,
    ):
        """标记文档处理失败"""
        doc.status = DocumentStatus.FAILED
        doc.error_message = error
        await db.commit()
        log.error(f"文档处理失败: {doc.filename} - {error}")

    async def process_pending_documents(
        self,
        db: AsyncSession,
        limit: int = 10,
    ) -> int:
        """
        处理所有待处理的文档

        Args:
            db: 数据库会话
            limit: 最大处理数量

        Returns:
            成功处理的数量
        """
        # 查询待处理文档
        query = (
            select(Document)
            .where(Document.status == DocumentStatus.PENDING)
            .limit(limit)
        )
        result = await db.execute(query)
        docs = result.scalars().all()

        if not docs:
            return 0

        log.info(f"找到 {len(docs)} 个待处理文档")

        success_count = 0
        for doc in docs:
            if await self.process_document(doc.id, db):
                success_count += 1

        return success_count


# 全局处理器实例
_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """获取文档处理器单例"""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


async def process_document_task(doc_id: str, db: AsyncSession):
    """
    后台任务：处理文档

    用于在 FastAPI 后台任务中调用
    """
    processor = get_document_processor()
    await processor.process_document(doc_id, db)
