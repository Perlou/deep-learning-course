"""
DocuMind AI - 文档解析器基类

定义解析器接口和通用数据结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParseResult:
    """解析结果"""

    content: str
    """提取的文本内容"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据（页数、标题、作者等）"""

    success: bool = True
    """是否成功"""

    error: Optional[str] = None
    """错误信息"""

    pages: List[str] = field(default_factory=list)
    """按页分割的内容（如果适用）"""

    def __post_init__(self):
        if not self.success and not self.error:
            self.error = "未知错误"


class BaseParser(ABC):
    """
    文档解析器基类

    所有解析器必须继承此类并实现 parse 方法
    """

    # 支持的文件扩展名（子类必须定义）
    supported_extensions: List[str] = []

    # 解析器名称
    name: str = "BaseParser"

    @abstractmethod
    def parse(self, file_path: str) -> ParseResult:
        """
        解析文档

        Args:
            file_path: 文件路径

        Returns:
            ParseResult 解析结果
        """
        pass

    @classmethod
    def can_parse(cls, file_extension: str) -> bool:
        """
        检查是否支持该文件格式

        Args:
            file_extension: 文件扩展名（不含点号）

        Returns:
            是否支持
        """
        ext = file_extension.lower().lstrip(".")
        return ext in cls.supported_extensions

    def _validate_file(self, file_path: str) -> Optional[str]:
        """
        验证文件

        Args:
            file_path: 文件路径

        Returns:
            错误信息，如果验证通过则返回 None
        """
        path = Path(file_path)

        if not path.exists():
            return f"文件不存在: {file_path}"

        if not path.is_file():
            return f"不是有效的文件: {file_path}"

        ext = path.suffix.lower().lstrip(".")
        if not self.can_parse(ext):
            return f"不支持的文件格式: {ext}"

        return None

    def _create_error_result(self, error: str) -> ParseResult:
        """创建错误结果"""
        return ParseResult(
            content="",
            metadata={},
            success=False,
            error=error,
        )

    def _clean_text(self, text: str) -> str:
        """
        清理文本

        - 移除多余空白
        - 规范化换行符
        """
        import re

        # 规范化换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 移除多余空白行
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 移除行尾空白
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return text.strip()
