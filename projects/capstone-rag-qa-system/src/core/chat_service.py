"""
DocuMind AI - 问答服务模块

实现 RAG 问答逻辑，整合检索和 LLM 生成
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, List, Optional

from src.utils import get_settings, log

from .llm_engine import LLMEngine, get_llm_engine
from .retriever import Retriever, RetrievalResult, get_retriever


# ============================================
# RAG Prompt 模板
# ============================================

RAG_SYSTEM_PROMPT = """你是 DocuMind AI，一个专业的文档问答助手。你的任务是基于用户提供的参考文档来回答问题。

## 回答准则

1. **基于事实**：只基于参考内容回答，不要编造或推测
2. **引用来源**：回答时注明信息来源，格式：[来源: 文档名]
3. **承认不足**：如果参考内容无法回答问题，请明确说明
4. **简洁专业**：保持回答简洁、准确、专业
5. **中文回复**：使用中文回复用户问题"""

RAG_PROMPT_TEMPLATE = """{system_prompt}

## 参考内容

{context}

## 用户问题

{question}

## 回答

请基于以上参考内容回答用户问题："""


NO_CONTEXT_PROMPT = """{system_prompt}

## 注意

当前知识库中没有找到与问题相关的内容。

## 用户问题

{question}

## 回答

请告诉用户当前没有找到相关信息："""


# ============================================
# 数据结构
# ============================================


class StreamEventType(str, Enum):
    """流式事件类型"""

    CHUNK = "chunk"
    SOURCES = "sources"
    DONE = "done"
    ERROR = "error"


@dataclass
class SourceInfo:
    """来源信息"""

    doc_id: str
    filename: str
    chunk_index: int
    content: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "score": self.score,
        }


@dataclass
class ChatResult:
    """问答结果"""

    answer: str
    sources: List[SourceInfo]
    usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "usage": self.usage,
        }


@dataclass
class StreamEvent:
    """流式事件"""

    type: StreamEventType
    data: Any

    def to_dict(self) -> Dict[str, Any]:
        if self.type == StreamEventType.CHUNK:
            return {"type": self.type.value, "content": self.data}
        elif self.type == StreamEventType.SOURCES:
            return {"type": self.type.value, "sources": self.data}
        elif self.type == StreamEventType.DONE:
            return {"type": self.type.value, **self.data}
        elif self.type == StreamEventType.ERROR:
            return {"type": self.type.value, "error": self.data}
        return {"type": self.type.value, "data": self.data}


# ============================================
# ChatService 实现
# ============================================


class ChatService:
    """
    RAG 问答服务

    整合检索器和 LLM，实现端到端问答
    """

    def __init__(
        self,
        llm: Optional[LLMEngine] = None,
        retriever: Optional[Retriever] = None,
        use_mock: bool = False,
    ):
        """
        初始化问答服务

        Args:
            llm: LLM 引擎实例
            retriever: 检索器实例
            use_mock: 是否使用 Mock 模式
        """
        self.llm = llm or get_llm_engine(use_mock=use_mock)
        self.retriever = retriever or get_retriever()
        self.use_mock = use_mock

        settings = get_settings()
        self.top_k = settings.retrieval.top_k

        log.info(f"ChatService 初始化: mock={use_mock}")

    def chat(
        self,
        query: str,
        kb_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        **generation_kwargs,
    ) -> ChatResult:
        """
        非流式问答

        Args:
            query: 用户问题
            kb_id: 知识库 ID
            conversation_history: 对话历史
            top_k: 检索数量
            **generation_kwargs: 生成参数

        Returns:
            问答结果
        """
        top_k = top_k or self.top_k

        # 1. 检索相关文档
        retrieval_results = self.retriever.retrieve(query, kb_id, top_k=top_k)

        # 2. 构建 Prompt
        prompt = self._build_prompt(query, retrieval_results, conversation_history)

        # 3. LLM 生成
        answer = self.llm.generate(prompt, **generation_kwargs)

        # 4. 构建来源信息
        sources = self._build_sources(retrieval_results)

        log.debug(f"问答完成: query='{query[:50]}...', sources={len(sources)}")

        return ChatResult(
            answer=answer,
            sources=sources,
        )

    def stream_chat(
        self,
        query: str,
        kb_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        **generation_kwargs,
    ) -> Generator[StreamEvent, None, None]:
        """
        流式问答

        Args:
            query: 用户问题
            kb_id: 知识库 ID
            conversation_history: 对话历史
            top_k: 检索数量
            **generation_kwargs: 生成参数

        Yields:
            流式事件
        """
        top_k = top_k or self.top_k

        try:
            # 1. 检索相关文档
            retrieval_results = self.retriever.retrieve(query, kb_id, top_k=top_k)

            # 2. 构建 Prompt
            prompt = self._build_prompt(query, retrieval_results, conversation_history)

            # 3. 流式生成
            for token in self.llm.stream_generate(prompt, **generation_kwargs):
                yield StreamEvent(type=StreamEventType.CHUNK, data=token)

            # 4. 发送来源信息
            sources = self._build_sources(retrieval_results)
            yield StreamEvent(
                type=StreamEventType.SOURCES,
                data=[s.to_dict() for s in sources],
            )

            # 5. 发送完成信号
            yield StreamEvent(
                type=StreamEventType.DONE,
                data={"message": "ok"},
            )

        except Exception as e:
            log.error(f"流式问答失败: {e}")
            yield StreamEvent(type=StreamEventType.ERROR, data=str(e))

    def _build_prompt(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        构建 RAG Prompt

        Args:
            query: 用户问题
            retrieval_results: 检索结果
            conversation_history: 对话历史

        Returns:
            完整的 Prompt
        """
        # 处理对话历史
        history_context = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-4:]:  # 最多保留最近 4 轮
                role = "用户" if msg.get("role") == "user" else "助手"
                history_parts.append(f"{role}: {msg.get('content', '')}")
            if history_parts:
                history_context = "\n\n## 对话历史\n\n" + "\n\n".join(history_parts)

        # 构建上下文
        if retrieval_results:
            context_parts = []
            for i, result in enumerate(retrieval_results, 1):
                source_info = f"[来源{i}: {result.filename}"
                if result.page:
                    source_info += f", 第{result.page}页"
                source_info += f", 相关度: {result.score:.2f}]"
                context_parts.append(f"{source_info}\n{result.content}")

            context = "\n\n---\n\n".join(context_parts)

            prompt = RAG_PROMPT_TEMPLATE.format(
                system_prompt=RAG_SYSTEM_PROMPT + history_context,
                context=context,
                question=query,
            )
        else:
            # 无检索结果
            prompt = NO_CONTEXT_PROMPT.format(
                system_prompt=RAG_SYSTEM_PROMPT + history_context,
                question=query,
            )

        return prompt

    def _build_sources(
        self, retrieval_results: List[RetrievalResult]
    ) -> List[SourceInfo]:
        """
        构建来源信息列表

        Args:
            retrieval_results: 检索结果

        Returns:
            来源信息列表
        """
        sources = []
        for result in retrieval_results:
            sources.append(
                SourceInfo(
                    doc_id=result.doc_id,
                    filename=result.filename,
                    chunk_index=result.metadata.get("chunk_index", 0),
                    content=result.content[:200] + "..."
                    if len(result.content) > 200
                    else result.content,
                    score=result.score,
                )
            )
        return sources


# ============================================
# 全局单例
# ============================================

_chat_service: Optional[ChatService] = None


def get_chat_service(use_mock: bool = False) -> ChatService:
    """
    获取问答服务单例

    Args:
        use_mock: 是否使用 Mock 模式

    Returns:
        ChatService 实例
    """
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(use_mock=use_mock)
    return _chat_service
