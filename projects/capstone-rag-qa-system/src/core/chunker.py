"""
DocuMind AI - 文本分块器

实现递归字符文本分割
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_settings, generate_id


@dataclass
class TextChunk:
    """文本分块"""

    id: str
    """分块 ID"""

    content: str
    """分块文本内容"""

    index: int
    """在文档中的顺序索引"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据（文档ID、位置等）"""

    @property
    def char_count(self) -> int:
        """字符数"""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """词数（简单按空格分割）"""
        return len(self.content.split())


class TextChunker:
    """
    文本分块器

    实现递归字符分割策略，支持按分隔符逐级分割
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: Optional[List[str]] = None,
        min_chunk_size: int = None,
    ):
        """
        初始化分块器

        Args:
            chunk_size: 目标分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
            separators: 分隔符列表，按优先级排序
            min_chunk_size: 最小分块大小，太小的块会被合并
        """
        settings = get_settings()

        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.chunking.min_chunk_size
        self.separators = separators or settings.chunking.separators

        # 确保 overlap 不超过 chunk_size
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 4

    def split(
        self,
        text: str,
        doc_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        将文本分割成块

        Args:
            text: 原始文本
            doc_id: 文档 ID，用于生成分块 ID
            extra_metadata: 额外的元数据

        Returns:
            TextChunk 列表
        """
        if not text or not text.strip():
            return []

        # 递归分割
        raw_chunks = self._recursive_split(text, self.separators)

        # 合并小块
        merged_chunks = self._merge_small_chunks(raw_chunks)

        # 创建 TextChunk 对象
        chunks = []
        for i, content in enumerate(merged_chunks):
            chunk_id = generate_id("chunk") if not doc_id else f"{doc_id}_chunk_{i}"

            metadata = {
                "chunk_index": i,
                "char_count": len(content),
            }
            if doc_id:
                metadata["doc_id"] = doc_id
            if extra_metadata:
                metadata.update(extra_metadata)

            chunks.append(
                TextChunk(
                    id=chunk_id,
                    content=content,
                    index=i,
                    metadata=metadata,
                )
            )

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """
        递归分割文本

        按分隔符优先级逐级分割，直到每个块小于 chunk_size
        """
        # 如果文本已经足够小，直接返回
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 如果没有更多分隔符，强制按字符分割
        if not separators:
            return self._split_by_char(text)

        # 尝试当前分隔符
        separator = separators[0]
        remaining_separators = separators[1:]

        # 分割
        if separator in text:
            parts = text.split(separator)
        else:
            # 当前分隔符不存在，尝试下一个
            return self._recursive_split(text, remaining_separators)

        # 处理每个部分
        result = []
        current_chunk = ""

        for i, part in enumerate(parts):
            # 添加分隔符（除了第一个部分）
            if i > 0:
                part = separator + part

            # 如果当前块加上新部分仍然小于 chunk_size
            if len(current_chunk) + len(part) <= self.chunk_size:
                current_chunk += part
            else:
                # 保存当前块
                if current_chunk.strip():
                    # 如果当前块太大，继续递归分割
                    if len(current_chunk) > self.chunk_size:
                        result.extend(
                            self._recursive_split(current_chunk, remaining_separators)
                        )
                    else:
                        result.append(current_chunk)

                # 开始新块（带重叠）
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + part
                else:
                    current_chunk = part

        # 处理最后一块
        if current_chunk.strip():
            if len(current_chunk) > self.chunk_size:
                result.extend(
                    self._recursive_split(current_chunk, remaining_separators)
                )
            else:
                result.append(current_chunk)

        return result

    def _split_by_char(self, text: str) -> List[str]:
        """按字符强制分割（最后手段）"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            # 下一个起点（考虑重叠）
            start = end - self.chunk_overlap
            if start <= 0 and end >= len(text):
                break

        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并太小的块"""
        if not chunks:
            return []

        merged = []
        current = ""

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if len(current) + len(chunk) + 1 <= self.chunk_size:
                current = current + "\n" + chunk if current else chunk
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        # 合并太小的块到前一个
        final = []
        for chunk in merged:
            if len(chunk) < self.min_chunk_size and final:
                # 合并到前一个
                final[-1] = final[-1] + "\n" + chunk
            else:
                final.append(chunk)

        return final

    def split_with_pages(
        self,
        pages: List[str],
        doc_id: Optional[str] = None,
    ) -> List[TextChunk]:
        """
        按页分割文本

        保留页码信息到元数据

        Args:
            pages: 页面文本列表
            doc_id: 文档 ID

        Returns:
            TextChunk 列表
        """
        all_chunks = []
        chunk_index = 0

        for page_num, page_text in enumerate(pages, start=1):
            chunks = self.split(
                page_text,
                doc_id=doc_id,
                extra_metadata={"page": page_num},
            )

            # 更新全局索引
            for chunk in chunks:
                chunk.index = chunk_index
                chunk.metadata["chunk_index"] = chunk_index
                all_chunks.append(chunk)
                chunk_index += 1

        return all_chunks


def create_chunker(
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> TextChunker:
    """
    创建分块器的便捷函数

    Args:
        chunk_size: 分块大小
        chunk_overlap: 重叠大小

    Returns:
        TextChunker 实例
    """
    return TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
