"""
测试 API 接口
"""

import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app


client = TestClient(app)


class TestSystemAPI:
    """系统接口测试"""

    def test_root(self):
        """测试根路由"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "DocuMind AI"

    def test_health(self):
        """测试健康检查"""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data
        assert data["data"]["status"] in ["healthy", "degraded"]

    def test_info(self):
        """测试系统信息"""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data
        assert "version" in data["data"]
        assert "models" in data["data"]

    def test_stats(self):
        """测试统计信息"""
        response = client.get("/api/v1/system/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0


class TestKnowledgeBaseAPI:
    """知识库接口测试"""

    def test_create_kb(self):
        """测试创建知识库"""
        response = client.post(
            "/api/v1/kb", json={"name": "测试知识库", "description": "这是一个测试"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == "测试知识库"
        return data["data"]["id"]

    def test_list_kb(self):
        """测试获取知识库列表"""
        response = client.get("/api/v1/kb")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "items" in data["data"]

    def test_get_kb(self):
        """测试获取知识库详情"""
        # 先创建一个知识库
        create_response = client.post(
            "/api/v1/kb", json={"name": "详情测试", "description": "测试详情"}
        )
        kb_id = create_response.json()["data"]["id"]

        # 获取详情
        response = client.get(f"/api/v1/kb/{kb_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["id"] == kb_id

    def test_update_kb(self):
        """测试更新知识库"""
        # 先创建一个知识库
        create_response = client.post(
            "/api/v1/kb", json={"name": "更新前", "description": "待更新"}
        )
        kb_id = create_response.json()["data"]["id"]

        # 更新
        response = client.put(
            f"/api/v1/kb/{kb_id}", json={"name": "更新后", "description": "已更新"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == "更新后"

    def test_delete_kb(self):
        """测试删除知识库"""
        # 先创建一个知识库
        create_response = client.post(
            "/api/v1/kb", json={"name": "待删除", "description": "将被删除"}
        )
        kb_id = create_response.json()["data"]["id"]

        # 删除
        response = client.delete(f"/api/v1/kb/{kb_id}")
        assert response.status_code == 200

        # 确认已删除
        get_response = client.get(f"/api/v1/kb/{kb_id}")
        assert get_response.status_code == 404


class TestChatAPI:
    """问答接口测试"""

    def test_chat_without_kb(self):
        """测试没有知识库时的问答"""
        response = client.post(
            "/api/v1/chat", json={"kb_id": "non_existent", "query": "测试问题"}
        )
        assert response.status_code == 404

    def test_chat_with_kb(self):
        """测试有知识库时的问答"""
        # 先创建知识库
        kb_response = client.post(
            "/api/v1/kb", json={"name": "问答测试库", "description": "测试问答"}
        )
        kb_id = kb_response.json()["data"]["id"]

        # 发起问答
        response = client.post(
            "/api/v1/chat", json={"kb_id": kb_id, "query": "这是一个测试问题？"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "answer" in data["data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
