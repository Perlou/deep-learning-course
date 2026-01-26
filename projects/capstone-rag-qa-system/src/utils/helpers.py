"""
DocuMind AI - 工具函数模块

通用辅助函数
"""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_id(prefix: str = "") -> str:
    """
    生成唯一ID

    Args:
        prefix: ID前缀

    Returns:
        唯一ID字符串
    """
    unique_id = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def get_file_hash(file_path: str) -> str:
    """
    计算文件的 MD5 哈希值

    Args:
        file_path: 文件路径

    Returns:
        MD5 哈希值
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_extension(filename: str) -> str:
    """
    获取文件扩展名（小写，不含点号）

    Args:
        filename: 文件名

    Returns:
        文件扩展名
    """
    return Path(filename).suffix.lower().lstrip(".")


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_datetime(
    dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    格式化日期时间

    Args:
        dt: datetime 对象，默认为当前时间
        fmt: 格式字符串

    Returns:
        格式化后的日期时间字符串
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不安全字符

    Args:
        filename: 原始文件名

    Returns:
        安全的文件名
    """
    # 移除路径分隔符和其他不安全字符
    unsafe_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", "\0"]
    for char in unsafe_chars:
        filename = filename.replace(char, "_")

    # 移除前导和尾随空格/点
    filename = filename.strip(". ")

    # 确保文件名不为空
    if not filename:
        filename = "unnamed"

    return filename
