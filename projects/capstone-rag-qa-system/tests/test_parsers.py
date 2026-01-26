"""
测试解析器模块
"""

import pytest
from pathlib import Path
import tempfile
import os

import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers import (
    BaseParser,
    ParseResult,
    ParserFactory,
    PDFParser,
    DocxParser,
    TxtParser,
    MarkdownParser,
)


class TestParseResult:
    """测试 ParseResult 数据类"""

    def test_success_result(self):
        result = ParseResult(content="测试内容", metadata={"key": "value"})
        assert result.success == True
        assert result.content == "测试内容"
        assert result.metadata["key"] == "value"

    def test_error_result(self):
        result = ParseResult(content="", metadata={}, success=False, error="测试错误")
        assert result.success == False
        assert result.error == "测试错误"

    def test_auto_error_message(self):
        result = ParseResult(content="", metadata={}, success=False)
        assert result.error == "未知错误"


class TestTxtParser:
    """测试 TXT 解析器"""

    def test_can_parse(self):
        assert TxtParser.can_parse("txt") == True
        assert TxtParser.can_parse("TXT") == True
        assert TxtParser.can_parse("pdf") == False

    def test_parse_utf8_file(self, tmp_path):
        # 创建测试文件
        test_file = tmp_path / "test.txt"
        test_content = "这是一个测试文件\n包含中文内容"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TxtParser()
        result = parser.parse(str(test_file))

        assert result.success == True
        assert "这是一个测试文件" in result.content
        assert result.metadata["format"] == "txt"

    def test_parse_nonexistent_file(self):
        parser = TxtParser()
        result = parser.parse("/nonexistent/file.txt")

        assert result.success == False
        assert "不存在" in result.error


class TestMarkdownParser:
    """测试 Markdown 解析器"""

    def test_can_parse(self):
        assert MarkdownParser.can_parse("md") == True
        assert MarkdownParser.can_parse("markdown") == True
        assert MarkdownParser.can_parse("txt") == False

    def test_parse_markdown_file(self, tmp_path):
        test_file = tmp_path / "test.md"
        test_content = """# 测试标题

这是一个测试段落。

## 二级标题

- 列表项 1
- 列表项 2

```python
print("Hello")
```
"""
        test_file.write_text(test_content, encoding="utf-8")

        parser = MarkdownParser()
        result = parser.parse(str(test_file))

        assert result.success == True
        assert "测试标题" in result.content
        assert result.metadata["format"] == "markdown"
        assert result.metadata.get("title") == "测试标题"
        assert result.metadata["heading_counts"]["h1"] == 1
        assert result.metadata["heading_counts"]["h2"] == 1


class TestParserFactory:
    """测试解析器工厂"""

    def test_get_supported_formats(self):
        formats = ParserFactory.get_supported_formats()
        assert "pdf" in formats
        assert "docx" in formats
        assert "txt" in formats
        assert "md" in formats

    def test_create_parser(self):
        pdf_parser = ParserFactory.create("pdf")
        assert isinstance(pdf_parser, PDFParser)

        txt_parser = ParserFactory.create("txt")
        assert isinstance(txt_parser, TxtParser)

        md_parser = ParserFactory.create("md")
        assert isinstance(md_parser, MarkdownParser)

    def test_create_unsupported_parser(self):
        parser = ParserFactory.create("xyz")
        assert parser is None

    def test_is_supported(self):
        assert ParserFactory.is_supported("pdf") == True
        assert ParserFactory.is_supported("xyz") == False

    def test_parse_txt_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("测试内容", encoding="utf-8")

        result = ParserFactory.parse(str(test_file))

        assert result.success == True
        assert "测试内容" in result.content

    def test_parse_nonexistent_file(self):
        result = ParserFactory.parse("/nonexistent/file.txt")

        assert result.success == False

    def test_parse_unsupported_format(self, tmp_path):
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        result = ParserFactory.parse(str(test_file))

        assert result.success == False
        assert "不支持" in result.error


class TestPDFParser:
    """测试 PDF 解析器"""

    def test_can_parse(self):
        assert PDFParser.can_parse("pdf") == True
        assert PDFParser.can_parse("PDF") == True
        assert PDFParser.can_parse("docx") == False


class TestDocxParser:
    """测试 DOCX 解析器"""

    def test_can_parse(self):
        assert DocxParser.can_parse("docx") == True
        assert DocxParser.can_parse("DOCX") == True
        assert DocxParser.can_parse("doc") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
