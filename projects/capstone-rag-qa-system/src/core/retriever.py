"""
DocuMind AI - 检索器模块

实现向量检索和结果处理
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_settings, log

from .embedder import Embedder, get_embedder
from .vector_store import VectorStore, get_vector_store


@dataclass
class RetrievalResult:
    """检索结果"""

    chunk_id: str
    """分块 ID"""

    doc_id: str
    """文档 ID"""

    content: str
    """分块内容"""

    score: float
    """相似度分数"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """其他元数据"""

    @property
    def filename(self) -> str:
        """获取文件名"""
        return self.metadata.get("filename", "未知文件")

    @property
    def page(self) -> Optional[int]:
        """获取页码"""
        return self.metadata.get("page")


class Retriever:
    """
    检索器

    负责将查询转换为向量并检索相关文档
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        top_k: int = None,
        score_threshold: float = None,
    ):
        """
        初始化检索器

        Args:
            embedder: 嵌入器实例
            top_k: 返回结果数量
            score_threshold: 最低分数阈值
        """
        settings = get_settings()

        self.embedder = embedder or get_embedder()
        self.top_k = top_k or settings.retrieval.top_k
        self.score_threshold = score_threshold or settings.retrieval.score_threshold

    def retrieve(
        self,
        query: str,
        kb_id: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        检索相关文档分块

        Args:
            query: 查询文本
            kb_id: 知识库 ID
            top_k: 返回数量（覆盖默认值）
            score_threshold: 最低分数（覆盖默认值）

        Returns:
            检索结果列表，按相似度降序排列
        """
        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold

        # 1. 获取向量存储
        vector_store = get_vector_store(kb_id)

        if vector_store.size == 0:
            log.debug(f"知识库 {kb_id} 为空")
            return []

        # 2. 生成查询向量
        query_embedding = self.embedder.embed_query(query)

        # 3. 向量检索
        search_results = vector_store.search(
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # 4. 构建结果
        results = []
        for metadata, score in search_results:
            results.append(
                RetrievalResult(
                    chunk_id=metadata.get("chunk_id", ""),
                    doc_id=metadata.get("doc_id", ""),
                    content=metadata.get("content", ""),
                    score=score,
                    metadata=metadata,
                )
            )

        log.debug(f"检索完成: 查询='{query[:50]}...', 结果数={len(results)}")

        return results

    def retrieve_with_context(
        self,
        query: str,
        kb_id: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        检索并格式化为上下文字符串

        用于直接构建 LLM Prompt

        Args:
            query: 查询文本
            kb_id: 知识库 ID
            top_k: 返回数量

        Returns:
            格式化的上下文字符串
        """
        results = self.retrieve(query, kb_id, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = f"[来源{i}: {result.filename}"
            if result.page:
                source_info += f", 第{result.page}页"
            source_info += "]"

            context_parts.append(f"{source_info}\n{result.content}")

        return "\n\n---\n\n".join(context_parts)

    def multi_retrieve(
        self,
        queries: List[str],
        kb_id: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        多查询检索

        将多个查询的结果合并去重

        Args:
            queries: 查询列表
            kb_id: 知识库 ID
            top_k: 每个查询的返回数量

        Returns:
            去重后的检索结果
        """
        seen_chunks = set()
        all_results = []

        for query in queries:
            results = self.retrieve(query, kb_id, top_k=top_k)

            for result in results:
                if result.chunk_id not in seen_chunks:
                    seen_chunks.add(result.chunk_id)
                    all_results.append(result)

        # 按分数排序
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results


# 全局检索器实例
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """获取检索器单例"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
