"""
DocuMind AI - 纯文本文档解析器

解析 TXT 纯文本文件
"""

from pathlib import Path
from typing import Any, Dict

from .base import BaseParser, ParseResult


class TxtParser(BaseParser):
    """纯文本解析器"""

    supported_extensions = ["txt"]
    name = "TxtParser"

    # 常见编码列表，按优先级排序
    ENCODINGS = ["utf-8", "gbk", "gb2312", "gb18030", "utf-16", "latin-1"]

    def parse(self, file_path: str) -> ParseResult:
        """
        解析纯文本文件

        Args:
            file_path: TXT 文件路径

        Returns:
            ParseResult 解析结果
        """
        # 验证文件
        error = self._validate_file(file_path)
        if error:
            return self._create_error_result(error)

        try:
            # 尝试检测编码
            content, encoding = self._read_with_encoding(file_path)

            if content is None:
                return self._create_error_result("无法识别文件编码")

            content = self._clean_text(content)

            # 提取元数据
            metadata = self._extract_metadata(file_path, encoding)

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True,
            )

        except Exception as e:
            return self._create_error_result(f"TXT 解析错误: {str(e)}")

    def _read_with_encoding(self, file_path: str):
        """尝试使用不同编码读取文件"""
        # 首先尝试使用 chardet 检测编码
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw_data = f.read()

            result = chardet.detect(raw_data)
            if result["encoding"]:
                try:
                    content = raw_data.decode(result["encoding"])
                    return content, result["encoding"]
                except (UnicodeDecodeError, LookupError):
                    pass
        except ImportError:
            pass

        # 回退到手动尝试编码
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                return content, encoding
            except (UnicodeDecodeError, LookupError):
                continue

        return None, None

    def _extract_metadata(self, file_path: str, encoding: str) -> Dict[str, Any]:
        """提取文本文件元数据"""
        path = Path(file_path)

        return {
            "filename": path.name,
            "file_size": path.stat().st_size,
            "format": "txt",
            "encoding": encoding,
        }
