"""
DocuMind AI - API 依赖注入

FastAPI 依赖项
"""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_async_db
from src.utils import get_settings, Settings


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话

    用于 FastAPI 依赖注入
    """
    async for session in get_async_db():
        yield session


def get_app_settings() -> Settings:
    """
    获取应用配置

    用于 FastAPI 依赖注入
    """
    return get_settings()
