"""
DocuMind AI - FastAPI 主应用

API 服务入口
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat_router, documents_router, kb_router, system_router
from src.models import init_async_db
from src.utils import get_settings, init_directories, setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    settings = get_settings()

    # 初始化日志
    setup_logger()

    # 初始化目录
    init_directories()

    # 初始化数据库
    await init_async_db()

    from src.utils import log

    log.info(f"🚀 {settings.app.name} v{settings.app.version} 启动成功")
    log.info(f"📚 API 文档: http://localhost:{settings.server.api_port}/docs")

    yield

    # 关闭时执行
    log.info("👋 应用正在关闭...")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        description=settings.app.description,
        version=settings.app.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(kb_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")

    # 根路由
    @app.get("/", tags=["Root"])
    async def root():
        """根路由"""
        return {
            "name": settings.app.name,
            "version": settings.app.version,
            "description": settings.app.description,
            "docs": "/docs",
        }

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.server.host,
        port=settings.server.api_port,
        reload=settings.app.debug,
    )
