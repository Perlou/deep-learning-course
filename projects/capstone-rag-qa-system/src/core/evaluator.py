"""
DocuMind AI - RAG 评估模块

基于 Ragas 框架对 RAG 管道进行系统化评估，
覆盖检索质量和生成质量两大维度。
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils import get_settings, log


# ============================================
# 数据结构
# ============================================


@dataclass
class EvalSample:
    """评估样本"""

    question: str
    ground_truth: str
    kb_id: str = "default"


@dataclass
class EvalDataset:
    """评估数据集"""

    version: str
    samples: List[EvalSample]
    name: str = "default"
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.samples)


@dataclass
class SampleResult:
    """单样本评估结果"""

    question: str
    answer: str
    ground_truth: str
    contexts: List[str]
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalResult:
    """评估结果"""

    task_id: str
    summary: Dict[str, float]
    per_sample: List[SampleResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================
# 评估指标
# ============================================

AVAILABLE_METRICS = {
    "faithfulness": "Faithfulness - 回答忠实度",
    "answer_relevancy": "Answer Relevancy - 回答相关度",
    "context_precision": "Context Precision - 上下文精确度",
    "context_recall": "Context Recall - 上下文召回率",
    "answer_correctness": "Answer Correctness - 回答正确性",
}


def _get_ragas_metrics(metric_names: List[str]):
    """
    获取 Ragas 指标对象

    Args:
        metric_names: 指标名称列表

    Returns:
        Ragas 指标对象列表
    """
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }

    metrics = []
    for name in metric_names:
        if name in metric_map:
            metrics.append(metric_map[name])
        else:
            log.warning(f"未知的评估指标: {name}，已跳过")

    return metrics


# ============================================
# RAGEvaluator
# ============================================


class RAGEvaluator:
    """
    RAG 系统评估器

    使用 Ragas 框架评估 RAG 管道的检索和生成质量。
    完全复用 ChatService 的真实管道进行评估。
    """

    def __init__(
        self,
        chat_service=None,
        eval_llm=None,
        eval_embeddings=None,
    ):
        """
        初始化评估器

        Args:
            chat_service: ChatService 实例
            eval_llm: LangChain 兼容的评估用 LLM
            eval_embeddings: LangChain 兼容的 Embeddings
        """
        self.chat_service = chat_service
        self.eval_llm = eval_llm
        self.eval_embeddings = eval_embeddings

        log.info("RAGEvaluator 初始化完成")

    def load_dataset(self, path: str) -> EvalDataset:
        """
        从 JSON 文件加载评估数据集

        Args:
            path: JSON 文件路径

        Returns:
            EvalDataset 实例

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 格式错误
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"评估数据集不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 验证格式
        if "data" not in raw:
            raise ValueError("评估数据集格式错误: 缺少 'data' 字段")

        samples = []
        for item in raw["data"]:
            if "question" not in item or "ground_truth" not in item:
                raise ValueError(
                    "评估数据集格式错误: 每条数据需包含 'question' 和 'ground_truth'"
                )
            samples.append(
                EvalSample(
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    kb_id=item.get("kb_id", "default"),
                )
            )

        dataset = EvalDataset(
            version=raw.get("version", "1.0"),
            samples=samples,
            name=raw.get("name", path.stem),
            description=raw.get("description", ""),
        )

        log.info(f"加载评估数据集: {dataset.name}, 样本数: {dataset.size}")
        return dataset

    def _collect_rag_outputs(
        self,
        dataset: EvalDataset,
        kb_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        通过 ChatService 收集 RAG 管道的真实输出

        Args:
            dataset: 评估数据集
            kb_id: 覆盖数据集中的 kb_id
            progress_callback: 进度回调 (completed, total)

        Returns:
            包含 question, answer, contexts, ground_truth 的列表
        """
        outputs = []
        total = dataset.size

        for i, sample in enumerate(dataset.samples):
            target_kb_id = kb_id or sample.kb_id

            try:
                # 调用真实 RAG 管道
                result = self.chat_service.chat(
                    query=sample.question,
                    kb_id=target_kb_id,
                )

                # 提取 contexts（检索到的文档分块内容）
                contexts = [s.content for s in result.sources]

                outputs.append(
                    {
                        "question": sample.question,
                        "answer": result.answer,
                        "contexts": contexts,
                        "ground_truth": sample.ground_truth,
                    }
                )

            except Exception as e:
                log.error(f"评估样本 {i + 1}/{total} 失败: {e}")
                outputs.append(
                    {
                        "question": sample.question,
                        "answer": f"[错误] {str(e)}",
                        "contexts": [],
                        "ground_truth": sample.ground_truth,
                    }
                )

            if progress_callback:
                progress_callback(i + 1, total)

            log.debug(f"收集 RAG 输出: {i + 1}/{total}")

        return outputs

    def run_evaluation(
        self,
        dataset: EvalDataset,
        metrics: Optional[List[str]] = None,
        kb_id: Optional[str] = None,
        task_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> EvalResult:
        """
        执行 Ragas 评估

        Args:
            dataset: 评估数据集
            metrics: 要计算的指标列表
            kb_id: 覆盖数据集中的 kb_id
            task_id: 评估任务 ID
            progress_callback: 进度回调 (completed, total, step)

        Returns:
            EvalResult 评估结果
        """
        from datasets import Dataset
        from ragas import evaluate

        start_time = time.time()

        # 默认指标
        if metrics is None:
            settings = get_settings()
            metrics = getattr(
                getattr(settings, "evaluation", None),
                "default_metrics",
                list(AVAILABLE_METRICS.keys()),
            )

        task_id = task_id or f"eval_{int(time.time())}"

        log.info(f"开始评估: task_id={task_id}, 样本数={dataset.size}, 指标={metrics}")

        # 1. 收集 RAG 输出
        def rag_progress(completed, total):
            if progress_callback:
                progress_callback(completed, total, "collecting")

        rag_outputs = self._collect_rag_outputs(
            dataset, kb_id=kb_id, progress_callback=rag_progress
        )

        # 2. 构建 Ragas Dataset
        ragas_data = {
            "question": [o["question"] for o in rag_outputs],
            "answer": [o["answer"] for o in rag_outputs],
            "contexts": [o["contexts"] for o in rag_outputs],
            "ground_truth": [o["ground_truth"] for o in rag_outputs],
        }
        ragas_dataset = Dataset.from_dict(ragas_data)

        # 3. 获取 Ragas 指标
        ragas_metrics = _get_ragas_metrics(metrics)

        if not ragas_metrics:
            raise ValueError("没有有效的评估指标")

        # 4. 执行 Ragas 评估
        log.info("开始 Ragas 评估计算...")

        eval_kwargs: Dict[str, Any] = {
            "dataset": ragas_dataset,
            "metrics": ragas_metrics,
        }

        if self.eval_llm:
            eval_kwargs["llm"] = self.eval_llm
        if self.eval_embeddings:
            eval_kwargs["embeddings"] = self.eval_embeddings

        ragas_result = evaluate(**eval_kwargs)

        # 5. 构建结果
        duration = time.time() - start_time

        # 汇总分数
        summary = {}
        for metric_name in metrics:
            if metric_name in ragas_result:
                summary[metric_name] = round(ragas_result[metric_name], 4)

        # 单样本结果
        per_sample = []
        result_df = ragas_result.to_pandas()
        for idx, row in result_df.iterrows():
            sample_scores = {}
            for metric_name in metrics:
                if metric_name in row:
                    val = row[metric_name]
                    sample_scores[metric_name] = (
                        round(float(val), 4) if val is not None else None
                    )

            per_sample.append(
                SampleResult(
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    ground_truth=row.get("ground_truth", ""),
                    contexts=row.get("contexts", []),
                    scores=sample_scores,
                )
            )

        eval_result = EvalResult(
            task_id=task_id,
            summary=summary,
            per_sample=per_sample,
            metadata={
                "total_samples": dataset.size,
                "metrics": metrics,
                "eval_llm": str(self.eval_llm) if self.eval_llm else "default",
                "duration_seconds": round(duration, 1),
                "created_at": datetime.now().isoformat(),
            },
        )

        log.info(f"评估完成: task_id={task_id}, 耗时={duration:.1f}s, 结果={summary}")

        return eval_result

    def save_results(self, result: EvalResult, output_path: str):
        """
        保存评估结果到 JSON 文件

        Args:
            result: 评估结果
            output_path: 输出路径
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "task_id": result.task_id,
            "summary": result.summary,
            "per_sample": [
                {
                    "question": s.question,
                    "answer": s.answer,
                    "ground_truth": s.ground_truth,
                    "contexts": s.contexts,
                    "scores": s.scores,
                }
                for s in result.per_sample
            ],
            "metadata": result.metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log.info(f"评估结果已保存: {path}")

    def load_results(self, path: str) -> EvalResult:
        """
        加载评估结果

        Args:
            path: JSON 文件路径

        Returns:
            EvalResult 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_sample = [
            SampleResult(
                question=s["question"],
                answer=s["answer"],
                ground_truth=s["ground_truth"],
                contexts=s.get("contexts", []),
                scores=s.get("scores", {}),
            )
            for s in data.get("per_sample", [])
        ]

        return EvalResult(
            task_id=data["task_id"],
            summary=data["summary"],
            per_sample=per_sample,
            metadata=data.get("metadata", {}),
        )

    def generate_report(self, result: EvalResult) -> str:
        """
        生成可读的评估报告

        Args:
            result: 评估结果

        Returns:
            格式化的报告文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DocuMind AI - RAG 评估报告")
        lines.append("=" * 60)
        lines.append("")

        # 元信息
        meta = result.metadata
        lines.append(f"评估任务 ID: {result.task_id}")
        lines.append(f"样本数量:    {meta.get('total_samples', 'N/A')}")
        lines.append(f"评估耗时:    {meta.get('duration_seconds', 'N/A')}s")
        lines.append(f"评估 LLM:    {meta.get('eval_llm', 'N/A')}")
        lines.append(f"评估时间:    {meta.get('created_at', 'N/A')}")
        lines.append("")

        # 综合得分
        lines.append("-" * 60)
        lines.append("综合评分")
        lines.append("-" * 60)
        for metric, score in result.summary.items():
            display_name = AVAILABLE_METRICS.get(metric, metric)
            bar_len = int(score * 20) if score else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {display_name}")
            lines.append(f"    {bar} {score:.4f}")
        lines.append("")

        # 单样本结果
        lines.append("-" * 60)
        lines.append("单样本详情")
        lines.append("-" * 60)
        for i, sample in enumerate(result.per_sample, 1):
            lines.append(f"\n  [{i}] Q: {sample.question}")
            lines.append(f"      A: {sample.answer[:100]}...")
            lines.append(f"      GT: {sample.ground_truth[:100]}...")
            scores_str = ", ".join(
                f"{k}={v:.2f}" for k, v in sample.scores.items() if v is not None
            )
            lines.append(f"      Scores: {scores_str}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
