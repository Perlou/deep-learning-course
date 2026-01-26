"""
DocuMind AI - 问答路由

对话问答接口（占位实现）
"""

from datetime import datetime
from typing import Optional

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
from src.models import Conversation, Feedback, KnowledgeBase, Message, MessageRole
from src.utils import generate_id

router = APIRouter(prefix="/chat", tags=["问答"])


@router.post("", response_model=ResponseModel[ChatResponse])
async def chat(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    发起问答请求（非流式）

    TODO: 集成检索和 LLM 生成
    """
    # 检查知识库是否存在
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == data.kb_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 创建或获取对话
    conversation_id = data.conversation_id
    if conversation_id:
        conv_query = select(Conversation).where(Conversation.id == conversation_id)
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")
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

    # TODO: 实现真正的检索和生成逻辑
    # 这里先返回占位响应
    answer = f"您好！您的问题是：「{data.query}」\n\n这是一个占位响应。检索和 LLM 生成功能将在后续阶段实现。"

    sources = [
        SourceReference(
            doc_id="doc_placeholder",
            filename="示例文档.pdf",
            chunk_index=0,
            content="这是一个示例来源引用，实际内容将在检索模块完成后显示。",
            score=0.95,
        )
    ]

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

    TODO: 实现真正的流式生成
    """
    # 检查知识库是否存在
    kb_query = select(KnowledgeBase).where(KnowledgeBase.id == data.kb_id)
    kb_result = await db.execute(kb_query)
    kb = kb_result.scalar_one_or_none()

    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    async def generate():
        """生成流式响应"""
        import json

        # 模拟流式输出
        chunks = [
            "您好！",
            "您的问题是：",
            f"「{data.query}」\n\n",
            "这是一个",
            "占位响应。",
            "检索和 LLM ",
            "生成功能将在",
            "后续阶段实现。",
        ]

        for chunk in chunks:
            yield f"event: chunk\ndata: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        # 发送来源引用
        sources = [
            {
                "doc_id": "doc_placeholder",
                "filename": "示例文档.pdf",
                "chunk_index": 0,
                "content": "这是一个示例来源引用。",
                "score": 0.95,
            }
        ]
        yield f"event: sources\ndata: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # 发送完成信号
        yield f"event: done\ndata: {json.dumps({'type': 'done', 'message_id': 'msg_placeholder', 'conversation_id': data.conversation_id or 'conv_new'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


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
