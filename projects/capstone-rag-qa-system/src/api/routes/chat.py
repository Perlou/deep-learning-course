"""
DocuMind AI - 问答路由

对话问答接口
"""

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetailResponse,
    ConversationResponse,
    FeedbackCreate,
    FeedbackResponse,
    MessageResponse,
    PaginatedData,
    ResponseModel,
    SourceReference,
)
from src.core import get_chat_service
from src.models import Conversation, Feedback, KnowledgeBase, Message, MessageRole
from src.utils import generate_id, log

router = APIRouter(prefix="/chat", tags=["问答"])


# ============================================
# 问答配置
# ============================================

import os

# LLM 引擎配置（优先级：Ollama > Mock > 本地模型）
# USE_OLLAMA=true: 使用本地 Ollama 服务 (推荐)
# USE_MOCK_LLM=true: 使用 Mock 模式（测试用）
# 都为 false: 加载本地 transformers 模型（需要 GPU）
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ("true", "1", "yes")
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "true").lower() in ("true", "1", "yes")


def _get_chat_service():
    """获取 ChatService 实例"""
    if USE_OLLAMA:
        return get_chat_service(use_ollama=True)
    return get_chat_service(use_mock=USE_MOCK_LLM)


# ============================================
# 问答接口
# ============================================


@router.post("", response_model=ResponseModel[ChatResponse])
async def chat(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    发起问答请求（非流式）

    集成检索和 LLM 生成，返回完整答案
    """
    # 检查知识库是否存在
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == data.kb_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 创建或获取对话
    conversation_id = data.conversation_id
    conversation_history = []

    if conversation_id:
        conv_query = select(Conversation).where(Conversation.id == conversation_id)
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")

        # 获取历史消息
        msg_query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(8)  # 最多取最近 8 条
        )
        msg_result = await db.execute(msg_query)
        messages = list(reversed(msg_result.scalars().all()))

        conversation_history = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
    else:
        conversation = Conversation(
            id=generate_id("conv"),
            kb_id=data.kb_id,
            title=data.query[:50] + "..." if len(data.query) > 50 else data.query,
        )
        db.add(conversation)
        await db.flush()
        conversation_id = conversation.id

    # 保存用户消息
    user_message = Message(
        id=generate_id("msg"),
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=data.query,
    )
    db.add(user_message)

    # 使用 ChatService 生成回答
    chat_service = _get_chat_service()

    # 获取生成参数
    gen_kwargs = {}
    if data.options:
        if data.options.temperature is not None:
            gen_kwargs["temperature"] = data.options.temperature
        if data.options.max_tokens is not None:
            gen_kwargs["max_tokens"] = data.options.max_tokens

    top_k = data.options.top_k if data.options else None

    try:
        result = chat_service.chat(
            query=data.query,
            kb_id=data.kb_id,
            conversation_history=conversation_history,
            top_k=top_k,
            **gen_kwargs,
        )

        answer = result.answer
        sources = [
            SourceReference(
                doc_id=s.doc_id,
                filename=s.filename,
                chunk_index=s.chunk_index,
                content=s.content,
                score=s.score,
            )
            for s in result.sources
        ]

    except Exception as e:
        log.error(f"问答生成失败: {e}")
        answer = f"抱歉，处理您的问题时出现了错误：{str(e)}"
        sources = []

    # 保存助手消息
    assistant_message = Message(
        id=generate_id("msg"),
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=answer,
        sources=[s.model_dump() for s in sources],
    )
    db.add(assistant_message)

    await db.commit()

    return ResponseModel(
        data=ChatResponse(
            message_id=assistant_message.id,
            conversation_id=conversation_id,
            answer=answer,
            sources=sources,
            usage=None,
            created_at=datetime.utcnow(),
        )
    )


@router.post("/stream")
async def chat_stream(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    发起问答请求（流式）

    使用 SSE 流式返回生成结果
    """
    # 检查知识库是否存在
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == data.kb_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 创建或获取对话
    conversation_id = data.conversation_id
    conversation_history = []

    if conversation_id:
        conv_query = select(Conversation).where(Conversation.id == conversation_id)
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")

        # 获取历史消息
        msg_query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(8)
        )
        msg_result = await db.execute(msg_query)
        messages = list(reversed(msg_result.scalars().all()))

        conversation_history = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
    else:
        conversation = Conversation(
            id=generate_id("conv"),
            kb_id=data.kb_id,
            title=data.query[:50] + "..." if len(data.query) > 50 else data.query,
        )
        db.add(conversation)
        await db.flush()
        conversation_id = conversation.id

    # 保存用户消息
    user_message = Message(
        id=generate_id("msg"),
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=data.query,
    )
    db.add(user_message)
    await db.commit()

    # 生成助手消息 ID
    assistant_message_id = generate_id("msg")

    async def generate():
        """流式生成响应"""
        chat_service = _get_chat_service()
        collected_answer = []
        collected_sources = []

        # 获取生成参数
        gen_kwargs = {}
        if data.options:
            if data.options.temperature is not None:
                gen_kwargs["temperature"] = data.options.temperature
            if data.options.max_tokens is not None:
                gen_kwargs["max_tokens"] = data.options.max_tokens

        top_k = data.options.top_k if data.options else None

        try:
            for event in chat_service.stream_chat(
                query=data.query,
                kb_id=data.kb_id,
                conversation_history=conversation_history,
                top_k=top_k,
                **gen_kwargs,
            ):
                event_dict = event.to_dict()

                if event.type.value == "chunk":
                    collected_answer.append(event_dict.get("content", ""))
                    yield f"event: chunk\ndata: {json.dumps(event_dict, ensure_ascii=False)}\n\n"

                elif event.type.value == "sources":
                    collected_sources = event_dict.get("sources", [])
                    yield f"event: sources\ndata: {json.dumps(event_dict, ensure_ascii=False)}\n\n"

                elif event.type.value == "done":
                    done_data = {
                        "type": "done",
                        "message_id": assistant_message_id,
                        "conversation_id": conversation_id,
                    }
                    yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"

                elif event.type.value == "error":
                    yield f"event: error\ndata: {json.dumps(event_dict, ensure_ascii=False)}\n\n"

        except Exception as e:
            log.error(f"流式生成失败: {e}")
            error_data = {"type": "error", "error": str(e)}
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        # 保存助手消息到数据库
        # 注意：这里需要新建一个数据库会话
        from src.models.database import get_async_session_local

        AsyncSessionLocal = get_async_session_local()
        async with AsyncSessionLocal() as save_db:
            full_answer = "".join(collected_answer)
            assistant_message = Message(
                id=assistant_message_id,
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=full_answer,
                sources=collected_sources,
            )
            save_db.add(assistant_message)
            await save_db.commit()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================
# 对话管理接口
# ============================================


@router.get(
    "/conversations", response_model=ResponseModel[PaginatedData[ConversationResponse]]
)
async def list_conversations(
    kb_id: str,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db_session),
):
    """获取对话列表"""
    # 计算总数
    count_query = (
        select(func.count())
        .select_from(Conversation)
        .where(Conversation.kb_id == kb_id)
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # 分页查询
    offset = (page - 1) * page_size
    query = (
        select(Conversation)
        .where(Conversation.kb_id == kb_id)
        .offset(offset)
        .limit(page_size)
        .order_by(Conversation.updated_at.desc())
    )
    result = await db.execute(query)
    items = result.scalars().all()

    # 获取每个对话的消息数量
    responses = []
    for conv in items:
        msg_count_query = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conv.id)
        )
        msg_count_result = await db.execute(msg_count_query)
        msg_count = msg_count_result.scalar()

        responses.append(
            ConversationResponse(
                id=conv.id,
                kb_id=conv.kb_id,
                title=conv.title,
                message_count=msg_count,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            )
        )

    return ResponseModel(
        data=PaginatedData(
            items=responses,
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ResponseModel[ConversationDetailResponse],
)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """获取对话详情"""
    query = select(Conversation).where(Conversation.id == conversation_id)
    result = await db.execute(query)
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 获取消息列表
    msg_query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    msg_result = await db.execute(msg_query)
    messages = msg_result.scalars().all()

    return ResponseModel(
        data=ConversationDetailResponse(
            id=conversation.id,
            kb_id=conversation.kb_id,
            title=conversation.title,
            messages=[
                MessageResponse(
                    id=msg.id,
                    role=msg.role.value,
                    content=msg.content,
                    sources=[SourceReference(**s) for s in msg.sources]
                    if msg.sources
                    else None,
                    created_at=msg.created_at,
                )
                for msg in messages
            ],
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
    )


@router.delete("/conversations/{conversation_id}", response_model=ResponseModel)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """删除对话"""
    query = select(Conversation).where(Conversation.id == conversation_id)
    result = await db.execute(query)
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 统计消息数量
    msg_count_query = (
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id == conversation_id)
    )
    msg_count_result = await db.execute(msg_count_query)
    msg_count = msg_count_result.scalar()

    await db.delete(conversation)
    await db.commit()

    return ResponseModel(
        data={
            "deleted_messages": msg_count,
        }
    )


@router.post("/feedback", response_model=ResponseModel[FeedbackResponse])
async def submit_feedback(
    data: FeedbackCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """提交反馈"""
    # 检查消息是否存在
    msg_query = select(Message).where(Message.id == data.message_id)
    msg_result = await db.execute(msg_query)
    message = msg_result.scalar_one_or_none()

    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")

    feedback = Feedback(
        id=generate_id("fb"),
        message_id=data.message_id,
        rating=data.rating,
        comment=data.comment,
    )
    db.add(feedback)
    await db.commit()

    return ResponseModel(data=FeedbackResponse(feedback_id=feedback.id))
