"""
测试向量检索模块
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEmbedder:
    """测试 Embedder"""

    @pytest.fixture
    def mock_embedder(self, monkeypatch):
        """创建模拟嵌入器（不加载实际模型）"""
        from src.core.embedder import Embedder

        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "mock-model"
        embedder.device = "cpu"
        embedder._dimension = 384
        embedder._loaded = True

        # 模拟 model
        class MockModel:
            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    return np.random.rand(384).astype(np.float32)
                return np.random.rand(len(texts), 384).astype(np.float32)

            def get_sentence_embedding_dimension(self):
                return 384

        embedder.model = MockModel()
        return embedder

    def test_dimension_property(self, mock_embedder):
        assert mock_embedder.dimension == 384

    def test_embed_single_text(self, mock_embedder):
        result = mock_embedder.embed("测试文本")
        assert result.shape == (1, 384)

    def test_embed_multiple_texts(self, mock_embedder):
        texts = ["文本一", "文本二", "文本三"]
        result = mock_embedder.embed(texts)
        assert result.shape == (3, 384)

    def test_embed_empty_list(self, mock_embedder):
        result = mock_embedder.embed([])
        assert len(result) == 0

    def test_embed_query(self, mock_embedder):
        result = mock_embedder.embed_query("查询文本")
        assert result.shape == (384,)

    def test_similarity(self, mock_embedder):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])

        # 对于相同向量，相似度应该是 1
        sim = mock_embedder.similarity(v1, v2)
        assert abs(sim - 1.0) < 0.01

        # 对于正交向量，相似度应该是 0
        v3 = np.array([0.0, 1.0, 0.0])
        sim = mock_embedder.similarity(v1, v3)
        assert abs(sim) < 0.01


class TestVectorStore:
    """测试 VectorStore"""

    @pytest.fixture
    def vector_store(self):
        """创建测试向量存储"""
        from src.core.vector_store import VectorStore

        return VectorStore(dimension=4)

    def test_add_vectors(self, vector_store):
        embeddings = np.random.rand(3, 4).astype(np.float32)
        metadata = [
            {"chunk_id": "c1", "doc_id": "d1", "content": "内容1"},
            {"chunk_id": "c2", "doc_id": "d1", "content": "内容2"},
            {"chunk_id": "c3", "doc_id": "d2", "content": "内容3"},
        ]

        vector_store.add(embeddings, metadata)

        assert vector_store.size == 3

    def test_search(self, vector_store):
        # 添加向量
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        metadata = [
            {"chunk_id": "c1", "content": "第一个"},
            {"chunk_id": "c2", "content": "第二个"},
            {"chunk_id": "c3", "content": "第三个"},
        ]

        vector_store.add(embeddings, metadata)

        # 搜索
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = vector_store.search(query, top_k=2)

        assert len(results) == 2
        # 第一个结果应该是最相似的
        assert results[0][0]["chunk_id"] == "c1"
        assert results[0][1] > 0.9

    def test_search_empty_store(self, vector_store):
        query = np.random.rand(4).astype(np.float32)
        results = vector_store.search(query)

        assert len(results) == 0

    def test_save_and_load(self, vector_store, tmp_path):
        from src.core.vector_store import VectorStore

        # 添加数据
        embeddings = np.random.rand(3, 4).astype(np.float32)
        metadata = [{"id": i} for i in range(3)]
        vector_store.add(embeddings, metadata)

        # 保存
        save_path = tmp_path / "test_index"
        vector_store.save(str(save_path))

        # 加载
        new_store = VectorStore(dimension=4)
        loaded = new_store.load(str(save_path))

        assert loaded
        assert new_store.size == 3

    def test_clear(self, vector_store):
        embeddings = np.random.rand(3, 4).astype(np.float32)
        metadata = [{"id": i} for i in range(3)]
        vector_store.add(embeddings, metadata)

        assert vector_store.size == 3

        vector_store.clear()

        assert vector_store.size == 0


class TestRetriever:
    """测试 Retriever"""

    @pytest.fixture
    def mock_retriever(self, monkeypatch):
        """创建模拟检索器"""
        from src.core.retriever import Retriever

        # 创建检索器
        retriever = Retriever.__new__(Retriever)
        retriever.top_k = 5
        retriever.score_threshold = 0.0

        # 模拟嵌入器
        class MockEmbedder:
            def embed_query(self, query):
                return np.random.rand(4).astype(np.float32)

        retriever.embedder = MockEmbedder()

        return retriever


class TestRetrievalResult:
    """测试 RetrievalResult"""

    def test_create_result(self):
        from src.core.retriever import RetrievalResult

        result = RetrievalResult(
            chunk_id="chunk_001",
            doc_id="doc_001",
            content="这是测试内容",
            score=0.85,
            metadata={"filename": "test.pdf", "page": 1},
        )

        assert result.chunk_id == "chunk_001"
        assert result.filename == "test.pdf"
        assert result.page == 1
        assert result.score == 0.85

    def test_default_filename(self):
        from src.core.retriever import RetrievalResult

        result = RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            content="",
            score=0.5,
        )

        assert result.filename == "未知文件"
        assert result.page is None


class TestVectorStoreManager:
    """测试 VectorStoreManager"""

    def test_get_store(self, tmp_path, monkeypatch):
        from src.core.vector_store import VectorStoreManager

        manager = VectorStoreManager()

        # 模拟 index_dir
        monkeypatch.setattr(manager.settings.storage, "index_dir", str(tmp_path))

        store = manager.get_store("kb_test")

        assert store is not None
        assert "kb_test" in manager.stores

        # 再次获取应该返回同一个实例
        store2 = manager.get_store("kb_test")
        assert store is store2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
