"""
DocuMind AI - Markdown 文档解析器

解析 Markdown 文件
"""

from pathlib import Path
from typing import Any, Dict

from .base import BaseParser, ParseResult


class MarkdownParser(BaseParser):
    """Markdown 解析器"""

    supported_extensions = ["md", "markdown"]
    name = "MarkdownParser"

    def parse(self, file_path: str) -> ParseResult:
        """
        解析 Markdown 文件

        Args:
            file_path: Markdown 文件路径

        Returns:
            ParseResult 解析结果
        """
        # 验证文件
        error = self._validate_file(file_path)
        if error:
            return self._create_error_result(error)

        try:
            # 读取文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 选择是否转换为纯文本
            # 对于 RAG，保留 Markdown 格式通常更好
            # 但也可以选择转换为纯文本
            cleaned_content = self._clean_markdown(content)

            # 提取元数据
            metadata = self._extract_metadata(file_path, content)

            return ParseResult(
                content=cleaned_content,
                metadata=metadata,
                success=True,
            )

        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
                cleaned_content = self._clean_markdown(content)
                metadata = self._extract_metadata(file_path, content)
                return ParseResult(
                    content=cleaned_content,
                    metadata=metadata,
                    success=True,
                )
            except Exception as e:
                return self._create_error_result(f"编码错误: {str(e)}")
        except Exception as e:
            return self._create_error_result(f"Markdown 解析错误: {str(e)}")

    def _clean_markdown(self, content: str) -> str:
        """
        清理 Markdown 内容

        保留文本结构，但移除一些格式符号
        """
        import re

        # 移除代码块的语言标识，但保留代码内容
        content = re.sub(r"```\w*\n", "```\n", content)

        # 清理基本文本
        content = self._clean_text(content)

        return content

    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """提取 Markdown 文件元数据"""
        import re

        path = Path(file_path)

        metadata = {
            "filename": path.name,
            "file_size": path.stat().st_size,
            "format": "markdown",
        }

        # 尝试提取标题（第一个 # 标题）
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # 统计标题数量
        h1_count = len(re.findall(r"^#\s+", content, re.MULTILINE))
        h2_count = len(re.findall(r"^##\s+", content, re.MULTILINE))
        h3_count = len(re.findall(r"^###\s+", content, re.MULTILINE))

        metadata["heading_counts"] = {
            "h1": h1_count,
            "h2": h2_count,
            "h3": h3_count,
        }

        # 统计代码块数量
        code_blocks = len(re.findall(r"```", content)) // 2
        metadata["code_block_count"] = code_blocks

        # 检查 YAML front matter
        if content.startswith("---"):
            end_match = re.search(r"\n---\n", content[3:])
            if end_match:
                metadata["has_frontmatter"] = True

        return metadata
