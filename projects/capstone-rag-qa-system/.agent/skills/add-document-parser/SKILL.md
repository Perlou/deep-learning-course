---
name: add-document-parser
description: 为 DocuMind AI 添加新的文档解析器
---

# 添加文档解析器技能

此技能用于为 DocuMind AI 项目添加新的文档解析器。

## 项目结构

```
projects/capstone-rag-qa-system/src/parsers/
├── __init__.py         # 模块导出
├── base.py             # 解析器基类
├── pdf_parser.py       # PDF 解析器
├── docx_parser.py      # Word 解析器
├── txt_parser.py       # 纯文本解析器
├── md_parser.py        # Markdown 解析器
└── factory.py          # 解析器工厂
```

## 解析器基类

```python
# base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ParseResult:
    """解析结果"""
    content: str                    # 提取的文本内容
    metadata: Dict[str, Any]        # 元数据（页数、标题等）
    success: bool = True            # 是否成功
    error: Optional[str] = None     # 错误信息


class BaseParser(ABC):
    """解析器基类"""

    # 支持的文件扩展名
    supported_extensions: list[str] = []

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
        """检查是否支持该文件格式"""
        return file_extension.lower() in cls.supported_extensions
```

## 添加新解析器步骤

### 1. 创建解析器类

```python
# new_parser.py
"""
新格式文档解析器
"""

from pathlib import Path
from typing import Dict, Any

from .base import BaseParser, ParseResult


class NewFormatParser(BaseParser):
    """新格式解析器"""

    supported_extensions = ["xyz", "abc"]
    name = "NewFormatParser"

    def parse(self, file_path: str) -> ParseResult:
        """解析文档"""
        try:
            # 验证文件
            error = self._validate_file(file_path)
            if error:
                return self._create_error_result(error)

            path = Path(file_path)

            # 读取和解析文件
            content = self._extract_content(path)
            metadata = self._extract_metadata(path)

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            return ParseResult(
                content="",
                metadata={},
                success=False,
                error=str(e)
            )

    def _extract_content(self, path: Path) -> str:
        """提取文本内容"""
        # 实现具体的内容提取逻辑
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_metadata(self, path: Path) -> Dict[str, Any]:
        """提取元数据"""
        return {
            "filename": path.name,
            "size": path.stat().st_size,
            "format": path.suffix.lstrip("."),
        }
```

### 2. 在工厂中注册解析器

```python
# factory.py
from .new_parser import NewFormatParser

class ParserFactory:
    _parsers: List[Type[BaseParser]] = [
        PDFParser,
        DocxParser,
        TxtParser,
        MarkdownParser,
        NewFormatParser,  # 新增
    ]
```

### 3. 在 `__init__.py` 导出

```python
from .new_parser import NewFormatParser

__all__ = [
    # ... 已有导出
    "NewFormatParser",
]
```

### 4. 更新配置

在 `configs/config.yaml` 中添加支持的格式：

```yaml
supported_formats:
  - pdf
  - docx
  - txt
  - md
  - xyz # 新增
```

## 常用解析库

| 格式     | 推荐库        | 安装命令                     |
| -------- | ------------- | ---------------------------- |
| PDF      | PyMuPDF       | `pip install PyMuPDF`        |
| DOCX     | python-docx   | `pip install python-docx`    |
| Excel    | openpyxl      | `pip install openpyxl`       |
| HTML     | BeautifulSoup | `pip install beautifulsoup4` |
| Markdown | markdown      | `pip install markdown`       |

## 测试解析器

```python
# tests/test_parsers.py
import pytest
from src.parsers import ParserFactory, NewFormatParser


class TestNewFormatParser:
    def test_can_parse(self):
        assert NewFormatParser.can_parse("xyz")
        assert not NewFormatParser.can_parse("pdf")

    def test_parse_success(self, tmp_path):
        test_file = tmp_path / "test.xyz"
        test_file.write_text("测试内容")

        parser = NewFormatParser()
        result = parser.parse(str(test_file))

        assert result.success
        assert "测试内容" in result.content
```

## 注意事项

1. **编码处理**：处理各种文件编码，使用 `chardet` 检测
2. **大文件处理**：使用流式读取避免内存溢出
3. **错误处理**：捕获所有异常，返回有意义的错误信息
4. **元数据提取**：尽可能提取页数、标题、作者等信息
5. **清理输出**：去除多余空白、特殊字符
