"""
DocuMind AI - 向量嵌入模块

使用 Sentence Transformers 生成文本嵌入向量
"""

from typing import List, Optional, Union

import numpy as np

from src.utils import get_settings, log


class Embedder:
    """
    向量嵌入器

    使用预训练的 Sentence Transformer 模型生成文本嵌入
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
    ):
        """
        初始化嵌入器

        Args:
            model_name: 模型名称，默认使用配置中的模型
            device: 设备 ("auto", "cuda", "cpu")
        """
        settings = get_settings()
        self.model_name = model_name or settings.models.embedding.name
        self.device = self._resolve_device(device)
        self.model = None
        self._dimension: Optional[int] = None

        # 延迟加载模型
        self._loaded = False

    def _resolve_device(self, device: str) -> str:
        """解析设备"""
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self):
        """加载模型（延迟加载）"""
        if self._loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            log.info(f"加载嵌入模型: {self.model_name}, 设备: {self.device}")

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            # 获取向量维度
            self._dimension = self.model.get_sentence_embedding_dimension()

            log.info(f"嵌入模型加载完成，维度: {self._dimension}")
            self._loaded = True

        except Exception as e:
            log.error(f"加载嵌入模型失败: {e}")
            raise

    @property
    def dimension(self) -> int:
        """获取嵌入向量维度"""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        生成文本嵌入向量

        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化向量
            show_progress: 是否显示进度条

        Returns:
            嵌入向量数组，形状为 (n, dimension)
        """
        self._load_model()

        # 处理单个文本
        if isinstance(texts, str):
            texts = [texts]

        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return np.array([])

        # 生成嵌入
        embeddings = self.model.encode(
            valid_texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress and len(valid_texts) > 10,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        批量嵌入文档

        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条

        Returns:
            嵌入向量数组
        """
        self._load_model()

        if not texts:
            return np.array([])

        settings = get_settings()
        batch_size = batch_size or settings.models.embedding.batch_size

        log.info(f"开始批量嵌入 {len(texts)} 个文本，批次大小: {batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        log.info(f"批量嵌入完成，向量形状: {embeddings.shape}")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        嵌入查询文本

        对于某些模型（如 BGE），查询需要添加前缀

        Args:
            query: 查询文本

        Returns:
            查询嵌入向量
        """
        self._load_model()

        # BGE 模型查询前缀
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        计算两个嵌入向量的余弦相似度

        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量

        Returns:
            相似度分数 (0-1)
        """
        # 归一化后的向量，内积等于余弦相似度
        return float(np.dot(embedding1, embedding2))


# 全局嵌入器实例
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """获取嵌入器单例"""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
