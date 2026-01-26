"""
DocuMind AI - 日志模块

统一的日志管理
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import get_settings, get_project_root


def setup_logger(log_level: Optional[str] = None) -> logger:
    """
    配置日志记录器

    Args:
        log_level: 日志级别，默认从配置文件读取

    Returns:
        配置好的 logger 实例
    """
    settings = get_settings()
    project_root = get_project_root()

    # 移除默认处理器
    logger.remove()

    # 日志级别
    level = log_level or settings.logging.level

    # 日志格式
    log_format = settings.logging.format

    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )

    # 添加文件处理器
    log_dir = project_root / settings.logging.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "documind.log"

    logger.add(
        str(log_file),
        format=log_format,
        level=level,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        encoding="utf-8",
    )

    # 添加错误日志单独记录
    error_log_file = log_dir / "error.log"
    logger.add(
        str(error_log_file),
        format=log_format,
        level="ERROR",
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        encoding="utf-8",
    )

    logger.info(f"日志系统初始化完成，日志目录: {log_dir}")

    return logger


def get_logger(name: str = "documind"):
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logger 实例
    """
    return logger.bind(name=name)


# 默认日志实例
log = logger
