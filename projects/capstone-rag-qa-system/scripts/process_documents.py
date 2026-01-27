#!/usr/bin/env python3
"""
手动处理所有待处理的文档
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import get_document_processor
from src.models.database import get_async_session_local
from src.utils import setup_logger


async def main():
    setup_logger()

    print("=" * 50)
    print("DocuMind AI - 手动文档处理")
    print("=" * 50)

    AsyncSessionLocal = get_async_session_local()
    async with AsyncSessionLocal() as db:
        processor = get_document_processor()

        print("\n正在处理待处理的文档...")
        count = await processor.process_pending_documents(db, limit=50)

        print(f"\n✅ 处理完成，成功处理 {count} 个文档")


if __name__ == "__main__":
    asyncio.run(main())
