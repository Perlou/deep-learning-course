"""
DocuMind AI - PDF 文档解析器

使用 PyMuPDF (fitz) 解析 PDF 文件
"""

from pathlib import Path
from typing import Any, Dict

from .base import BaseParser, ParseResult


class PDFParser(BaseParser):
    """PDF 文档解析器"""

    supported_extensions = ["pdf"]
    name = "PDFParser"

    def parse(self, file_path: str) -> ParseResult:
        """
        解析 PDF 文档

        Args:
            file_path: PDF 文件路径

        Returns:
            ParseResult 解析结果
        """
        # 验证文件
        error = self._validate_file(file_path)
        if error:
            return self._create_error_result(error)

        try:
            import fitz  # PyMuPDF

            # 打开 PDF
            doc = fitz.open(file_path)

            # 提取文本
            pages = []
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    pages.append(self._clean_text(page_text))

            # 合并所有页面内容
            content = "\n\n".join(pages)

            # 提取元数据
            metadata = self._extract_metadata(doc, file_path)

            doc.close()

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True,
                pages=pages,
            )

        except ImportError:
            return self._create_error_result(
                "缺少 PyMuPDF 依赖，请运行: pip install PyMuPDF"
            )
        except Exception as e:
            return self._create_error_result(f"PDF 解析错误: {str(e)}")

    def _extract_metadata(self, doc, file_path: str) -> Dict[str, Any]:
        """提取 PDF 元数据"""
        path = Path(file_path)

        metadata = {
            "filename": path.name,
            "file_size": path.stat().st_size,
            "format": "pdf",
            "page_count": len(doc),
        }

        # PDF 元信息
        if doc.metadata:
            if doc.metadata.get("title"):
                metadata["title"] = doc.metadata["title"]
            if doc.metadata.get("author"):
                metadata["author"] = doc.metadata["author"]
            if doc.metadata.get("subject"):
                metadata["subject"] = doc.metadata["subject"]
            if doc.metadata.get("creationDate"):
                metadata["creation_date"] = doc.metadata["creationDate"]

        return metadata
