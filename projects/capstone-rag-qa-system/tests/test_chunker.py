"""
测试文本分块器
"""

import pytest
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import TextChunker, TextChunk, create_chunker


class TestTextChunk:
    """测试 TextChunk 数据类"""

    def test_create_chunk(self):
        chunk = TextChunk(
            id="chunk_001",
            content="这是测试内容",
            index=0,
            metadata={"doc_id": "doc_001"},
        )

        assert chunk.id == "chunk_001"
        assert chunk.content == "这是测试内容"
        assert chunk.index == 0
        assert chunk.char_count == 6

    def test_word_count(self):
        chunk = TextChunk(
            id="chunk_001",
            content="Hello World Test",
            index=0,
        )

        assert chunk.word_count == 3


class TestTextChunker:
    """测试 TextChunker"""

    def test_empty_text(self):
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.split("")

        assert len(chunks) == 0

    def test_small_text(self):
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.split("这是一个很短的文本。")

        assert len(chunks) == 1
        assert "这是一个很短的文本" in chunks[0].content

    def test_split_by_paragraph(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = """第一段落内容，这是测试。

第二段落内容，这也是测试。

第三段落内容，最后一段。"""

        chunks = chunker.split(text)

        assert len(chunks) >= 1
        # 确保所有块都有内容
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_chunk_with_doc_id(self):
        chunker = TextChunker(chunk_size=50)
        chunks = chunker.split("测试内容", doc_id="doc_123")

        assert len(chunks) > 0
        assert "doc_123" in chunks[0].metadata.get("doc_id", "")

    def test_chunk_with_extra_metadata(self):
        chunker = TextChunker(chunk_size=50)
        chunks = chunker.split(
            "测试内容", extra_metadata={"source": "test.pdf", "page": 1}
        )

        assert len(chunks) > 0
        assert chunks[0].metadata.get("source") == "test.pdf"

    def test_large_text_splitting(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        # 创建一个较大的文本
        text = "这是一个测试句子。" * 50

        chunks = chunker.split(text)

        assert len(chunks) > 1
        # 检查每个块的大小
        for chunk in chunks:
            assert len(chunk.content) <= 200  # 允许一些余量

    def test_split_with_pages(self):
        chunker = TextChunker(chunk_size=100)
        pages = [
            "第一页的内容，这是测试数据。",
            "第二页的内容，更多测试数据。",
            "第三页的内容，最后的数据。",
        ]

        chunks = chunker.split_with_pages(pages, doc_id="doc_test")

        assert len(chunks) >= 3
        # 检查页码元数据
        for chunk in chunks:
            assert "page" in chunk.metadata

    def test_chunk_index_continuity(self):
        chunker = TextChunker(chunk_size=50)
        text = "短文本一。\n\n短文本二。\n\n短文本三。"

        chunks = chunker.split(text)

        # 检查索引连续性
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_create_chunker_helper(self):
        chunker = create_chunker(chunk_size=200, chunk_overlap=50)

        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 50


class TestChunkerEdgeCases:
    """测试分块器边界情况"""

    def test_only_whitespace(self):
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.split("   \n\n   \t\t   ")

        assert len(chunks) == 0

    def test_single_long_line(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "这是一个非常长的句子没有任何分隔符它应该被强制分割成多个块因为它超过了分块大小限制"

        chunks = chunker.split(text)

        assert len(chunks) >= 1

    def test_overlap_larger_than_size(self):
        # 重叠不应超过分块大小
        chunker = TextChunker(chunk_size=50, chunk_overlap=100)

        # 应该自动调整重叠大小
        assert chunker.chunk_overlap < chunker.chunk_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
