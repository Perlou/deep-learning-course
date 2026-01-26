"""
DocuMind AI - 解析器工厂

根据文件格式创建对应的解析器
"""

from typing import List, Optional, Type

from .base import BaseParser, ParseResult
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from .txt_parser import TxtParser
from .md_parser import MarkdownParser


class ParserFactory:
    """
    解析器工厂

    根据文件扩展名创建相应的解析器实例
    """

    # 注册的解析器类
    _parsers: List[Type[BaseParser]] = [
        PDFParser,
        DocxParser,
        TxtParser,
        MarkdownParser,
    ]

    @classmethod
    def create(cls, file_extension: str) -> Optional[BaseParser]:
        """
        根据文件扩展名创建解析器

        Args:
            file_extension: 文件扩展名（可带或不带点号）

        Returns:
            解析器实例，如果不支持则返回 None
        """
        ext = file_extension.lower().lstrip(".")

        for parser_class in cls._parsers:
            if parser_class.can_parse(ext):
                return parser_class()

        return None

    @classmethod
    def parse(cls, file_path: str) -> ParseResult:
        """
        便捷方法：直接解析文件

        Args:
            file_path: 文件路径

        Returns:
            ParseResult 解析结果
        """
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            return ParseResult(
                content="",
                metadata={},
                success=False,
                error=f"文件不存在: {file_path}",
            )

        ext = path.suffix.lstrip(".")
        parser = cls.create(ext)

        if parser is None:
            return ParseResult(
                content="",
                metadata={},
                success=False,
                error=f"不支持的文件格式: {ext}",
            )

        return parser.parse(file_path)

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """获取所有支持的文件格式"""
        formats = []
        for parser_class in cls._parsers:
            formats.extend(parser_class.supported_extensions)
        return list(set(formats))

    @classmethod
    def is_supported(cls, file_extension: str) -> bool:
        """检查是否支持该格式"""
        ext = file_extension.lower().lstrip(".")
        return ext in cls.get_supported_formats()

    @classmethod
    def register_parser(cls, parser_class: Type[BaseParser]):
        """
        注册新的解析器

        Args:
            parser_class: 解析器类（必须继承 BaseParser）
        """
        if parser_class not in cls._parsers:
            cls._parsers.append(parser_class)

    @classmethod
    def get_parser_info(cls) -> List[dict]:
        """获取所有解析器信息"""
        return [
            {
                "name": parser_class.name,
                "extensions": parser_class.supported_extensions,
            }
            for parser_class in cls._parsers
        ]
