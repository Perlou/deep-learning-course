"""
DocuMind AI - 向量存储模块

使用 FAISS 实现向量索引和检索
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import get_project_root, get_settings, log


class VectorStore:
    """
    向量存储

    使用 FAISS 实现高效向量索引和相似性搜索
    """

    def __init__(
        self,
        dimension: int = None,
        index_type: str = "flat",
    ):
        """
        初始化向量存储

        Args:
            dimension: 向量维度（如 1024）
            index_type: 索引类型 ("flat", "ivf")
        """
        settings = get_settings()
        self.dimension = dimension or settings.models.embedding.dimension
        self.index_type = index_type

        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._initialized = False

    def _create_index(self):
        """创建 FAISS 索引"""
        import faiss

        if self.index_type == "flat":
            # 精确搜索，使用内积（归一化向量等于余弦相似度）
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # IVF 索引，适合大规模数据
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                100,  # nlist
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        self._initialized = True
        log.debug(f"创建 FAISS 索引: 类型={self.index_type}, 维度={self.dimension}")

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ):
        """
        添加向量到索引

        Args:
            embeddings: 向量数组，形状 (n, dimension)
            metadata: 元数据列表，长度与向量数量一致
        """
        if not self._initialized:
            self._create_index()

        # 确保数据类型正确
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings = embeddings.astype(np.float32)

        # 验证维度
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.dimension}, 实际 {embeddings.shape[1]}"
            )

        # 验证元数据数量
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"元数据数量不匹配: 向量 {embeddings.shape[0]}, 元数据 {len(metadata)}"
            )

        # 添加到索引
        self.index.add(embeddings)
        self.metadata.extend(metadata)

        log.debug(f"添加 {embeddings.shape[0]} 个向量，当前总数: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        搜索相似向量

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            score_threshold: 最低分数阈值

        Returns:
            [(metadata, score), ...] 排序后的结果列表
        """
        if not self._initialized or self.index.ntotal == 0:
            return []

        # 确保数据类型和形状
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # 搜索
        scores, indices = self.index.search(
            query_embedding, min(top_k, self.index.ntotal)
        )

        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            if score < score_threshold:
                continue

            results.append((self.metadata[idx], float(score)))

        return results

    def save(self, path: str):
        """
        保存索引到文件

        Args:
            path: 保存目录路径
        """
        import faiss

        if not self._initialized:
            log.warning("索引未初始化，无法保存")
            return

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # 保存元数据
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                },
                f,
            )

        log.info(f"向量索引已保存: {save_path}, 向量数: {self.index.ntotal}")

    def load(self, path: str) -> bool:
        """
        从文件加载索引

        Args:
            path: 索引目录路径

        Returns:
            是否加载成功
        """
        import faiss

        load_path = Path(path)
        index_file = load_path / "index.faiss"
        metadata_file = load_path / "metadata.pkl"

        if not index_file.exists() or not metadata_file.exists():
            log.debug(f"索引文件不存在: {load_path}")
            return False

        try:
            # 加载 FAISS 索引
            self.index = faiss.read_index(str(index_file))

            # 加载元数据
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.dimension = data["dimension"]
                self.index_type = data.get("index_type", "flat")

            self._initialized = True
            log.info(f"向量索引已加载: {load_path}, 向量数: {self.index.ntotal}")
            return True

        except Exception as e:
            log.error(f"加载向量索引失败: {e}")
            return False

    def clear(self):
        """清空索引"""
        self.index = None
        self.metadata = []
        self._initialized = False
        log.debug("向量索引已清空")

    @property
    def size(self) -> int:
        """获取向量数量"""
        if not self._initialized or self.index is None:
            return 0
        return self.index.ntotal

    def remove_by_doc_id(self, doc_id: str):
        """
        移除指定文档的所有向量

        注意：FAISS 不支持直接删除，需要重建索引

        Args:
            doc_id: 文档 ID
        """
        if not self._initialized:
            return

        # 找出需要保留的索引
        keep_indices = []
        for i, meta in enumerate(self.metadata):
            if meta.get("doc_id") != doc_id:
                keep_indices.append(i)

        if len(keep_indices) == len(self.metadata):
            return  # 没有需要删除的

        if not keep_indices:
            self.clear()
            return

        # 重建索引
        import faiss

        # 获取需要保留的向量
        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        keep_vectors = all_vectors[keep_indices]
        keep_metadata = [self.metadata[i] for i in keep_indices]

        # 重建
        self.clear()
        self._create_index()
        self.add(keep_vectors, keep_metadata)

        log.info(f"已移除文档 {doc_id} 的向量，剩余: {self.size}")


class VectorStoreManager:
    """
    向量存储管理器

    管理多个知识库的向量存储
    """

    def __init__(self):
        self.stores: Dict[str, VectorStore] = {}
        self.settings = get_settings()

    def get_store(self, kb_id: str) -> VectorStore:
        """
        获取或创建知识库的向量存储

        Args:
            kb_id: 知识库 ID

        Returns:
            VectorStore 实例
        """
        if kb_id not in self.stores:
            store = VectorStore()

            # 尝试加载已有索引
            index_path = self._get_index_path(kb_id)
            if index_path.exists():
                store.load(str(index_path))

            self.stores[kb_id] = store

        return self.stores[kb_id]

    def save_store(self, kb_id: str):
        """保存知识库的向量存储"""
        if kb_id in self.stores:
            index_path = self._get_index_path(kb_id)
            self.stores[kb_id].save(str(index_path))

    def delete_store(self, kb_id: str):
        """删除知识库的向量存储"""
        if kb_id in self.stores:
            del self.stores[kb_id]

        import shutil

        index_path = self._get_index_path(kb_id)
        if index_path.exists():
            shutil.rmtree(index_path)
            log.info(f"已删除知识库 {kb_id} 的向量索引")

    def _get_index_path(self, kb_id: str) -> Path:
        """获取索引存储路径"""
        project_root = get_project_root()
        return project_root / self.settings.storage.index_dir / kb_id


# 全局管理器实例
_store_manager: Optional[VectorStoreManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """获取向量存储管理器单例"""
    global _store_manager
    if _store_manager is None:
        _store_manager = VectorStoreManager()
    return _store_manager


def get_vector_store(kb_id: str) -> VectorStore:
    """获取指定知识库的向量存储"""
    return get_vector_store_manager().get_store(kb_id)
