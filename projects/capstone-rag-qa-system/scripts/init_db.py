#!/usr/bin/env python
"""
DocuMind AI - 数据库初始化脚本

创建数据库表和初始数据
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import init_db
from src.utils import get_settings, init_directories, setup_logger, log


def main():
    """初始化数据库"""
    # 初始化日志
    setup_logger()

    log.info("开始初始化数据库...")

    # 初始化目录
    init_directories()
    log.info("✓ 目录结构已创建")

    # 创建数据库表
    init_db()

    settings = get_settings()
    log.info(f"✓ 数据库已创建: {settings.storage.db_path}")

    log.info("🎉 初始化完成！")


if __name__ == "__main__":
    main()
