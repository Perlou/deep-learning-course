"""
DocuMind AI - 数据库配置

SQLAlchemy 数据库连接和会话管理
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.utils import get_project_root, get_settings

# 创建 Base 类
Base = declarative_base()

# 全局引擎和会话工厂
_engine = None
_async_engine = None
_SessionLocal = None
_AsyncSessionLocal = None


def get_database_url(async_mode: bool = False) -> str:
    """
    获取数据库连接 URL

    Args:
        async_mode: 是否使用异步模式

    Returns:
        数据库 URL
    """
    settings = get_settings()
    project_root = get_project_root()

    db_path = project_root / settings.storage.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if async_mode:
        return f"sqlite+aiosqlite:///{db_path}"
    return f"sqlite:///{db_path}"


def get_engine():
    """获取同步数据库引擎"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(async_mode=False),
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


def get_async_engine():
    """获取异步数据库引擎"""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            get_database_url(async_mode=True),
            echo=False,
        )
    return _async_engine


def get_session_local():
    """获取同步会话工厂"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def get_async_session_local():
    """获取异步会话工厂"""
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _AsyncSessionLocal


def get_db() -> Session:
    """
    获取同步数据库会话（用于依赖注入）

    Yields:
        数据库会话
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取异步数据库会话（用于依赖注入）

    Yields:
        异步数据库会话
    """
    AsyncSessionLocal = get_async_session_local()
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def init_db():
    """初始化数据库（创建所有表）"""
    from . import entities  # noqa: F401 - 导入以注册模型

    engine = get_engine()
    Base.metadata.create_all(bind=engine)


async def init_async_db():
    """异步初始化数据库"""
    from . import entities  # noqa: F401

    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
