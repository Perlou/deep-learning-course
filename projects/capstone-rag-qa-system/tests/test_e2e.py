"""
DocuMind AI - 端到端测试

完整流程测试：知识库 → 文档上传 → 问答
"""

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest

# API 基础配置
API_BASE = os.getenv("API_URL", "http://localhost:8000/api/v1")
TEST_TIMEOUT = 30.0

# 测试文件路径
TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.fixture(scope="module")
def client():
    """创建 HTTP 客户端"""
    with httpx.Client(base_url=API_BASE, timeout=TEST_TIMEOUT) as client:
        yield client


@pytest.fixture(scope="module")
def test_kb(client):
    """创建测试知识库并在测试后清理"""
    # 创建知识库
    response = client.post(
        "/kb",
        json={
            "name": f"测试知识库_{int(time.time())}",
            "description": "E2E 测试用知识库",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    kb = data["data"]

    yield kb

    # 清理：删除知识库
    client.delete(f"/kb/{kb['id']}")


class TestSystemHealth:
    """系统健康检查测试"""

    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] in ["healthy", "degraded"]

    def test_system_info(self, client):
        """测试系统信息接口"""
        response = client.get("/system/info")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "version" in data["data"]
        assert "models" in data["data"]


class TestKnowledgeBase:
    """知识库管理测试"""

    def test_create_knowledge_base(self, client):
        """测试创建知识库"""
        response = client.post(
            "/kb", json={"name": "临时测试知识库", "description": "用于测试创建功能"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        kb = data["data"]
        assert kb["name"] == "临时测试知识库"

        # 清理
        client.delete(f"/kb/{kb['id']}")

    def test_list_knowledge_bases(self, client, test_kb):
        """测试获取知识库列表"""
        response = client.get("/kb")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "items" in data["data"]
        assert data["data"]["total"] >= 1

    def test_get_knowledge_base(self, client, test_kb):
        """测试获取单个知识库"""
        response = client.get(f"/kb/{test_kb['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["id"] == test_kb["id"]

    def test_update_knowledge_base(self, client, test_kb):
        """测试更新知识库"""
        new_name = f"更新后的名称_{int(time.time())}"
        response = client.put(f"/kb/{test_kb['id']}", json={"name": new_name})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == new_name


class TestDocumentUpload:
    """文档上传测试"""

    def test_upload_text_document(self, client, test_kb):
        """测试上传文本文档"""
        # 创建临时测试文件
        test_content = """
        这是一个测试文档。
        
        DocuMind AI 是一个基于 RAG 的智能问答系统。
        它支持多种文档格式，包括 PDF、DOCX、TXT 和 Markdown。
        
        系统使用向量检索技术，能够快速找到相关文档片段。
        然后利用大语言模型生成准确的回答。
        """

        files = {"file": ("test_document.txt", test_content, "text/plain")}
        data = {"kb_id": test_kb["id"]}

        response = client.post("/documents/upload", files=files, data=data)
        assert response.status_code == 200
        result = response.json()
        assert result["code"] == 0
        doc = result["data"]
        assert doc["filename"] == "test_document.txt"
        assert doc["status"] in [
            "pending",
            "parsing",
            "chunking",
            "embedding",
            "completed",
        ]

        return doc

    def test_list_documents(self, client, test_kb):
        """测试获取文档列表"""
        response = client.get(f"/documents?kb_id={test_kb['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "items" in data["data"]


class TestChatFlow:
    """问答流程测试"""

    def test_send_chat_message(self, client, test_kb):
        """测试发送问答消息"""
        # 先上传一个测试文档
        test_content = """
        DocuMind AI 项目特点：
        1. 支持多种文档格式
        2. 使用 BGE 嵌入模型
        3. 基于 FAISS 向量检索
        4. 采用 Qwen2.5 大语言模型
        """

        files = {"file": ("features.txt", test_content, "text/plain")}
        data = {"kb_id": test_kb["id"]}

        # 上传文档
        upload_response = client.post("/documents/upload", files=files, data=data)
        assert upload_response.status_code == 200

        # 等待文档处理（简化版，实际应轮询状态）
        time.sleep(2)

        # 发送问答请求
        chat_response = client.post(
            "/chat", json={"kb_id": test_kb["id"], "query": "DocuMind AI 有什么特点？"}
        )

        # 即使没有真正的 LLM，也应该返回成功（可能是 mock 响应）
        assert chat_response.status_code == 200
        data = chat_response.json()
        assert data["code"] == 0
        assert "answer" in data["data"]
        assert "conversation_id" in data["data"]

    def test_conversation_history(self, client, test_kb):
        """测试对话历史"""
        response = client.get(f"/chat/conversations?kb_id={test_kb['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "items" in data["data"]


class TestE2EFlow:
    """完整端到端流程测试"""

    def test_complete_workflow(self, client):
        """测试完整工作流程"""
        # 1. 创建知识库
        kb_response = client.post(
            "/kb", json={"name": "E2E 完整测试", "description": "端到端流程验证"}
        )
        assert kb_response.status_code == 200
        kb = kb_response.json()["data"]

        try:
            # 2. 上传文档
            doc_content = """
            人工智能（AI）正在改变我们的生活方式。
            机器学习是 AI 的一个重要分支。
            深度学习使用神经网络进行学习。
            """

            files = {"file": ("ai_intro.txt", doc_content, "text/plain")}
            data = {"kb_id": kb["id"]}

            upload_response = client.post("/documents/upload", files=files, data=data)
            assert upload_response.status_code == 200
            doc = upload_response.json()["data"]

            # 3. 检查文档列表
            docs_response = client.get(f"/documents?kb_id={kb['id']}")
            assert docs_response.status_code == 200
            assert docs_response.json()["data"]["total"] >= 1

            # 4. 等待处理
            time.sleep(2)

            # 5. 发送问答
            chat_response = client.post(
                "/chat", json={"kb_id": kb["id"], "query": "什么是人工智能？"}
            )
            assert chat_response.status_code == 200

            # 6. 检查对话历史
            history_response = client.get(f"/chat/conversations?kb_id={kb['id']}")
            assert history_response.status_code == 200

            print("✅ 端到端测试通过！")

        finally:
            # 清理
            client.delete(f"/kb/{kb['id']}")


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
