"""
DocuMind AI - LangChain 适配层

将现有引擎适配为 LangChain 接口，供 Ragas 评估框架使用。
支持本地 Ollama 和线上 API（OpenAI / DashScope / 兼容接口）双模式。
"""

import os
from typing import Any, Dict, List, Optional

from src.utils import log


# ============================================
# LLM 适配器
# ============================================


class OllamaLangChainLLM:
    """
    将 OllamaEngine 适配为 LangChain BaseLLM

    用于 Ragas 评估时的本地 LLM 调用。
    延迟导入 langchain_core 避免强依赖。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Args:
            model: Ollama 模型名称，如 "qwen2.5:7b"
            base_url: Ollama API 地址
            temperature: 采样温度（评估时建议低温）
        """
        from langchain_community.llms import Ollama

        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.temperature = temperature

        self._llm = Ollama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        log.info(
            f"OllamaLangChainLLM 初始化: model={self.model}, base_url={self.base_url}"
        )

    @property
    def langchain_llm(self):
        """获取 LangChain LLM 实例"""
        return self._llm


class OnlineLangChainLLM:
    """
    线上 API（OpenAI / DashScope / 兼容接口） → LangChain ChatModel 适配

    支持任何 OpenAI 兼容 API：
    - OpenAI (gpt-4o, gpt-4o-mini)
    - 阿里云 DashScope (qwen-plus, qwen-turbo)
    - DeepSeek, Moonshot 等
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 地址
            temperature: 采样温度
        """
        from langchain_openai import ChatOpenAI

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.temperature = temperature

        init_kwargs: Dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
        }
        if self.base_url:
            init_kwargs["base_url"] = self.base_url

        self._llm = ChatOpenAI(**init_kwargs)

        display_url = self.base_url or "https://api.openai.com/v1"
        log.info(
            f"OnlineLangChainLLM 初始化: model={self.model}, base_url={display_url}"
        )

    @property
    def langchain_llm(self):
        """获取 LangChain LLM 实例"""
        return self._llm


# ============================================
# Embeddings 适配器
# ============================================


class DocuMindEmbeddings:
    """
    将 DocuMind Embedder 适配为 LangChain Embeddings 接口

    复用现有的 BGE 嵌入模型，避免重复加载。
    """

    def __init__(self, embedder=None):
        """
        Args:
            embedder: DocuMind Embedder 实例，默认使用全局单例
        """
        from langchain_core.embeddings import Embeddings

        self._embedder = embedder
        self._langchain_embeddings = None
        self._embeddings_class = Embeddings

    def _get_embedder(self):
        """延迟获取 embedder"""
        if self._embedder is None:
            from src.core.embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    @property
    def langchain_embeddings(self):
        """获取 LangChain Embeddings 实例"""
        if self._langchain_embeddings is None:
            self._langchain_embeddings = _DocuMindLangChainEmbeddings(
                self._get_embedder()
            )
        return self._langchain_embeddings


class _DocuMindLangChainEmbeddings:
    """内部类：LangChain Embeddings 接口实现"""

    def __init__(self, embedder):
        self._embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        embeddings = self._embedder.embed_documents(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        embedding = self._embedder.embed_query(text)
        return embedding.tolist()


# ============================================
# 工厂方法
# ============================================


def get_eval_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1,
):
    """
    工厂方法：根据 provider 获取评估用 LLM

    Args:
        provider: "ollama" | "openai" | "dashscope"
        model: 模型名称
        api_key: API 密钥（线上模式）
        base_url: API 地址（线上模式）
        temperature: 采样温度

    Returns:
        LangChain 兼容的 LLM 实例
    """
    if provider == "ollama":
        wrapper = OllamaLangChainLLM(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        return wrapper.langchain_llm

    elif provider in ("openai", "dashscope"):
        # DashScope 使用 OpenAI 兼容接口
        if provider == "dashscope":
            base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
            model = model or "qwen-plus"

        wrapper = OnlineLangChainLLM(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
        return wrapper.langchain_llm

    else:
        raise ValueError(
            f"不支持的 provider: {provider}，支持: ollama, openai, dashscope"
        )


def get_eval_embeddings(embedder=None):
    """
    获取评估用 Embeddings

    Args:
        embedder: DocuMind Embedder 实例

    Returns:
        LangChain 兼容的 Embeddings 实例
    """
    wrapper = DocuMindEmbeddings(embedder=embedder)
    return wrapper.langchain_embeddings
