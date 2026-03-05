"""
DocuMind AI - 评估模块测试

测试 RAGEvaluator、LangChain 适配器和评估 API。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest


# ============================================
# 测试数据准备
# ============================================


SAMPLE_DATASET = {
    "version": "1.0",
    "name": "测试数据集",
    "description": "测试用评估数据集",
    "data": [
        {
            "question": "DocuMind AI 支持哪些文档格式？",
            "ground_truth": "支持 PDF、Word (DOCX)、TXT 和 Markdown 四种格式。",
            "kb_id": "test_kb",
        },
        {
            "question": "系统使用什么嵌入模型？",
            "ground_truth": "使用 BAAI/bge-large-zh-v1.5 嵌入模型。",
            "kb_id": "test_kb",
        },
    ],
}


@pytest.fixture
def sample_dataset_file(tmp_path):
    """创建临时评估数据集文件"""
    dataset_file = tmp_path / "test_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATASET, f, ensure_ascii=False)
    return str(dataset_file)


@pytest.fixture
def invalid_dataset_file(tmp_path):
    """创建无效的评估数据集文件"""
    dataset_file = tmp_path / "invalid_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump({"invalid": True}, f)
    return str(dataset_file)


@pytest.fixture
def mock_chat_result():
    """模拟 ChatService 的返回结果"""

    @dataclass
    class MockSource:
        content: str
        doc_id: str = "doc_1"
        filename: str = "test.pdf"
        chunk_index: int = 0
        score: float = 0.9

    @dataclass
    class MockChatResult:
        answer: str
        sources: List[MockSource]
        message_id: str = "msg_1"
        conversation_id: str = "conv_1"

    return MockChatResult(
        answer="系统支持 PDF、Word、TXT 和 Markdown 格式。",
        sources=[
            MockSource(
                content="DocuMind AI 支持以下文档格式：PDF、DOCX、TXT、Markdown。"
            ),
            MockSource(content="用户可以上传 PDF、Word 等格式的文档。"),
        ],
    )


# ============================================
# EvalDataset 加载测试
# ============================================


class TestLoadDataset:
    """测试评估数据集加载"""

    def test_load_valid_dataset(self, sample_dataset_file):
        """测试加载有效数据集"""
        from src.core.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        dataset = evaluator.load_dataset(sample_dataset_file)

        assert dataset.version == "1.0"
        assert dataset.name == "测试数据集"
        assert dataset.size == 2
        assert dataset.samples[0].question == "DocuMind AI 支持哪些文档格式？"
        assert dataset.samples[0].kb_id == "test_kb"

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        from src.core.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        with pytest.raises(FileNotFoundError):
            evaluator.load_dataset("/nonexistent/path.json")

    def test_load_invalid_format(self, invalid_dataset_file):
        """测试加载格式错误的文件"""
        from src.core.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        with pytest.raises(ValueError, match="缺少 'data' 字段"):
            evaluator.load_dataset(invalid_dataset_file)

    def test_load_missing_fields(self, tmp_path):
        """测试加载缺少必要字段的数据"""
        from src.core.evaluator import RAGEvaluator

        dataset_file = tmp_path / "missing_fields.json"
        with open(dataset_file, "w") as f:
            json.dump({"data": [{"question": "test"}]}, f)

        evaluator = RAGEvaluator()

        with pytest.raises(ValueError, match="ground_truth"):
            evaluator.load_dataset(str(dataset_file))


# ============================================
# 评估结果持久化测试
# ============================================


class TestResultPersistence:
    """测试评估结果的保存和加载"""

    def test_save_and_load_results(self, tmp_path):
        """测试保存和加载评估结果"""
        from src.core.evaluator import EvalResult, RAGEvaluator, SampleResult

        evaluator = RAGEvaluator()

        # 构建模拟结果
        result = EvalResult(
            task_id="eval_test_001",
            summary={
                "faithfulness": 0.85,
                "answer_relevancy": 0.78,
            },
            per_sample=[
                SampleResult(
                    question="测试问题",
                    answer="测试回答",
                    ground_truth="标准答案",
                    contexts=["上下文1", "上下文2"],
                    scores={
                        "faithfulness": 0.90,
                        "answer_relevancy": 0.80,
                    },
                ),
            ],
            metadata={
                "total_samples": 1,
                "duration_seconds": 10.5,
            },
        )

        # 保存
        output_path = str(tmp_path / "test_result.json")
        evaluator.save_results(result, output_path)

        assert Path(output_path).exists()

        # 加载
        loaded = evaluator.load_results(output_path)

        assert loaded.task_id == "eval_test_001"
        assert loaded.summary["faithfulness"] == 0.85
        assert len(loaded.per_sample) == 1
        assert loaded.per_sample[0].question == "测试问题"
        assert loaded.metadata["total_samples"] == 1

    def test_save_creates_directories(self, tmp_path):
        """测试保存时自动创建目录"""
        from src.core.evaluator import EvalResult, RAGEvaluator

        evaluator = RAGEvaluator()
        result = EvalResult(
            task_id="test",
            summary={},
            per_sample=[],
        )

        nested_path = str(tmp_path / "nested" / "dir" / "result.json")
        evaluator.save_results(result, nested_path)

        assert Path(nested_path).exists()


# ============================================
# 报告生成测试
# ============================================


class TestReportGeneration:
    """测试评估报告生成"""

    def test_generate_report(self):
        """测试报告生成"""
        from src.core.evaluator import EvalResult, RAGEvaluator, SampleResult

        evaluator = RAGEvaluator()

        result = EvalResult(
            task_id="eval_report_test",
            summary={
                "faithfulness": 0.85,
                "answer_relevancy": 0.78,
            },
            per_sample=[
                SampleResult(
                    question="测试问题？",
                    answer="这是一个测试回答。",
                    ground_truth="这是标准答案。",
                    contexts=["上下文内容"],
                    scores={
                        "faithfulness": 0.85,
                        "answer_relevancy": 0.78,
                    },
                ),
            ],
            metadata={
                "total_samples": 1,
                "duration_seconds": 5.0,
                "eval_llm": "ollama/qwen2.5:7b",
                "created_at": "2026-03-05T12:00:00",
            },
        )

        report = evaluator.generate_report(result)

        assert "RAG 评估报告" in report
        assert "eval_report_test" in report
        assert "faithfulness" in report.lower() or "忠实度" in report
        assert "0.85" in report


# ============================================
# 评估指标测试
# ============================================


class TestAvailableMetrics:
    """测试可用的评估指标"""

    def test_available_metrics_definition(self):
        """测试指标定义完整性"""
        from src.core.evaluator import AVAILABLE_METRICS

        expected_metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]

        for metric in expected_metrics:
            assert metric in AVAILABLE_METRICS, f"缺少指标: {metric}"


# ============================================
# LangChain 适配器测试
# ============================================


class TestLangChainAdapters:
    """测试 LangChain 适配层"""

    def test_get_eval_llm_ollama(self):
        """测试 Ollama LLM 工厂方法"""
        from src.core.langchain_adapters import OllamaLangChainLLM

        # 测试初始化（不实际连接）
        wrapper = OllamaLangChainLLM(
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
        )
        assert wrapper.model == "qwen2.5:7b"
        assert wrapper.langchain_llm is not None

    def test_get_eval_llm_online(self):
        """测试线上 LLM 工厂方法"""
        from src.core.langchain_adapters import OnlineLangChainLLM

        wrapper = OnlineLangChainLLM(
            model="gpt-4o-mini",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        assert wrapper.model == "gpt-4o-mini"
        assert wrapper.langchain_llm is not None

    def test_get_eval_llm_factory_invalid_provider(self):
        """测试无效 provider"""
        from src.core.langchain_adapters import get_eval_llm

        with pytest.raises(ValueError, match="不支持的 provider"):
            get_eval_llm(provider="invalid_provider")

    def test_get_eval_llm_factory_ollama(self):
        """测试工厂方法 - ollama"""
        from src.core.langchain_adapters import get_eval_llm

        llm = get_eval_llm(provider="ollama", model="qwen2.5:7b")
        assert llm is not None

    def test_get_eval_llm_factory_openai(self):
        """测试工厂方法 - openai"""
        from src.core.langchain_adapters import get_eval_llm

        llm = get_eval_llm(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        assert llm is not None

    def test_get_eval_llm_factory_dashscope(self):
        """测试工厂方法 - dashscope"""
        from src.core.langchain_adapters import get_eval_llm

        llm = get_eval_llm(
            provider="dashscope",
            api_key="test-key",
        )
        assert llm is not None

    def test_documind_embeddings_wrapper(self):
        """测试 DocuMind Embeddings 包装器"""
        from src.core.langchain_adapters import DocuMindEmbeddings

        # 使用 Mock embedder
        mock_embedder = MagicMock()

        import numpy as np

        mock_embedder.embed_documents.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2, 0.3])

        wrapper = DocuMindEmbeddings(embedder=mock_embedder)
        embeddings = wrapper.langchain_embeddings

        # 测试 embed_documents
        result = embeddings.embed_documents(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 3

        # 测试 embed_query
        result = embeddings.embed_query("query text")
        assert len(result) == 3


# ============================================
# 评估 API 端点测试
# ============================================


class TestEvaluationAPI:
    """测试评估 API 端点"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        try:
            from fastapi.testclient import TestClient

            from src.api.main import app

            return TestClient(app)
        except Exception:
            pytest.skip("无法创建测试客户端")

    def test_list_datasets(self, client):
        """测试获取数据集列表"""
        response = client.get("/api/v1/evaluation/datasets")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "items" in data["data"]

    def test_list_tasks_empty(self, client):
        """测试获取空任务列表"""
        response = client.get("/api/v1/evaluation/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    def test_get_nonexistent_task(self, client):
        """测试获取不存在的任务"""
        response = client.get("/api/v1/evaluation/tasks/nonexistent_id")
        assert response.status_code == 404

    def test_delete_nonexistent_task(self, client):
        """测试删除不存在的任务"""
        response = client.delete("/api/v1/evaluation/tasks/nonexistent_id")
        assert response.status_code == 404

    def test_create_dataset(self, client):
        """测试创建评估数据集"""
        response = client.post(
            "/api/v1/evaluation/datasets",
            json={
                "name": "API 测试数据集",
                "data": [
                    {
                        "question": "测试问题？",
                        "ground_truth": "测试答案。",
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == "API 测试数据集"
        assert data["data"]["sample_count"] == 1

    def test_create_dataset_invalid_data(self, client):
        """测试创建无效数据集"""
        response = client.post(
            "/api/v1/evaluation/datasets",
            json={
                "name": "无效数据集",
                "data": [{"question": "只有问题没有答案"}],
            },
        )
        assert response.status_code == 400
