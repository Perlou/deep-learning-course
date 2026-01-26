---
name: add-core-module
description: 为 DocuMind AI 添加核心业务模块（Embedder、VectorStore、LLM等）
---

# 添加核心模块技能

此技能用于为 DocuMind AI 项目添加核心业务模块。

## 项目结构

```
projects/capstone-rag-qa-system/src/core/
├── __init__.py         # 模块导出
├── chunker.py          # 文本分块器
├── embedder.py         # 向量嵌入
├── vector_store.py     # 向量存储
├── retriever.py        # 检索器
├── llm_engine.py       # LLM 引擎
└── chat_service.py     # 问答服务
```

## 核心模块模板

### 1. 文本分块器 (chunker.py)

```python
"""
文本分块器
"""

from typing import List, Optional
from dataclasses import dataclass

from src.utils import get_settings


@dataclass
class TextChunk:
    """文本分块"""
    content: str
    index: int
    metadata: dict


class TextChunker:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.separators = separators or settings.chunking.separators

    def split(self, text: str) -> List[TextChunk]:
        """
        将文本分割成块

        Args:
            text: 原始文本

        Returns:
            分块列表
        """
        chunks = []
        # 实现分块逻辑
        return chunks
```

### 2. 向量嵌入 (embedder.py)

```python
"""
向量嵌入模块
"""

from typing import List, Union
import numpy as np

from src.utils import get_settings, log


class Embedder:
    """向量嵌入器"""

    def __init__(self, model_name: str = None, device: str = "auto"):
        settings = get_settings()
        self.model_name = model_name or settings.models.embedding.name
        self.device = self._get_device(device)
        self.model = None
        self._load_model()

    def _get_device(self, device: str) -> str:
        """获取设备"""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """加载模型"""
        from sentence_transformers import SentenceTransformer

        log.info(f"加载嵌入模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        log.info(f"嵌入模型加载完成，设备: {self.device}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        生成文本嵌入向量

        Args:
            texts: 单个文本或文本列表

        Returns:
            嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )

        return embeddings

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()
```

### 3. 向量存储 (vector_store.py)

```python
"""
向量存储模块（FAISS）
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class VectorStore:
    """向量存储"""

    def __init__(self, dimension: int, index_path: Optional[str] = None):
        import faiss

        self.dimension = dimension
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
        self.metadata: List[dict] = []

        if index_path and Path(index_path).exists():
            self.load(index_path)

    def add(self, embeddings: np.ndarray, metadata: List[dict]):
        """
        添加向量

        Args:
            embeddings: 向量数组
            metadata: 元数据列表
        """
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """
        搜索相似向量

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            [(metadata, score), ...]
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata) and idx >= 0:
                results.append((self.metadata[idx], float(score)))

        return results

    def save(self, path: str):
        """保存索引"""
        import faiss
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        """加载索引"""
        import faiss
        import pickle

        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
```

### 4. LLM 引擎 (llm_engine.py)

```python
"""
LLM 推理引擎
"""

from typing import Generator, Optional

from src.utils import get_settings, log


class LLMEngine:
    """LLM 推理引擎"""

    def __init__(
        self,
        model_name: str = None,
        device: str = "auto",
        quantization: str = "none",
    ):
        settings = get_settings()
        self.model_name = model_name or settings.models.llm.name
        self.device = device
        self.quantization = quantization or settings.models.llm.quantization

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        log.info(f"加载 LLM 模型: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # 量化配置
        model_kwargs = {"trust_remote_code": True}

        if self.quantization == "int4":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantization == "int8":
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            **model_kwargs,
        )

        log.info("LLM 模型加载完成")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """流式生成"""
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text
```

## 模块注册

在 `__init__.py` 导出：

```python
from .chunker import TextChunker, TextChunk
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .llm_engine import LLMEngine
from .chat_service import ChatService

__all__ = [
    "TextChunker",
    "TextChunk",
    "Embedder",
    "VectorStore",
    "Retriever",
    "LLMEngine",
    "ChatService",
]
```

## 单例模式

对于需要全局共享的模块（如模型），使用单例：

```python
_embedder: Optional[Embedder] = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
```

## 注意事项

1. **延迟加载**：模型在首次使用时加载，避免启动时加载
2. **设备管理**：自动检测 CUDA 可用性
3. **量化支持**：支持 INT4/INT8 量化减少显存
4. **错误处理**：捕获模型加载和推理异常
5. **日志记录**：记录关键操作和性能指标
