"""
DocuMind AI - 评估 API 路由

提供 RESTful 接口管理 RAG 评估任务。
支持启动评估、查询状态、获取报告、管理数据集。
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    EvalDatasetCreate,
    EvalDatasetResponse,
    EvalReportResponse,
    EvalRunRequest,
    EvalSampleScores,
    EvalTaskResponse,
    PaginatedData,
    ResponseModel,
)
from src.utils import generate_id, get_settings, log

router = APIRouter(prefix="/evaluation", tags=["评估"])


# ============================================
# 评估任务内存存储
# ============================================

# 任务状态存储（生产环境应使用数据库）
_eval_tasks: Dict[str, dict] = {}
_eval_lock = threading.Lock()


def _get_task(task_id: str) -> dict:
    """获取任务状态"""
    with _eval_lock:
        task = _eval_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="评估任务不存在")
    return task


def _update_task(task_id: str, **kwargs):
    """更新任务状态"""
    with _eval_lock:
        if task_id in _eval_tasks:
            _eval_tasks[task_id].update(kwargs)


# ============================================
# 评估任务路由
# ============================================


@router.post("/run", response_model=ResponseModel[EvalTaskResponse])
async def run_evaluation(request: EvalRunRequest):
    """
    启动评估任务（后台异步执行）

    评估过程：
    1. 加载评估数据集
    2. 遍历数据集，调用 ChatService 获取 RAG 输出
    3. 使用 Ragas 计算各项指标
    4. 保存评估结果
    """
    settings = get_settings()

    # 生成任务 ID
    task_id = generate_id("eval")

    # 确定评估 LLM 配置
    eval_llm_provider = "ollama"
    if request.eval_llm:
        eval_llm_provider = request.eval_llm.provider

    # 确定评估指标
    metrics = request.metrics
    if not metrics:
        eval_config = getattr(settings, "evaluation", None)
        if eval_config and hasattr(eval_config, "default_metrics"):
            metrics = eval_config.default_metrics
        else:
            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_correctness",
            ]

    # 创建任务记录
    now = datetime.now()
    task_data = {
        "task_id": task_id,
        "kb_id": request.kb_id,
        "dataset_id": request.dataset_id,
        "status": "running",
        "progress": 0,
        "total_samples": 0,
        "completed_samples": 0,
        "current_step": "初始化评估...",
        "metrics": metrics,
        "eval_llm_provider": eval_llm_provider,
        "created_at": now,
        "completed_at": None,
        "result": None,
        "error_message": None,
    }

    with _eval_lock:
        _eval_tasks[task_id] = task_data

    # 启动后台评估线程
    thread = threading.Thread(
        target=_run_evaluation_worker,
        args=(task_id, request),
        daemon=True,
    )
    thread.start()

    log.info(f"评估任务已启动: task_id={task_id}")

    return ResponseModel(
        data=EvalTaskResponse(
            task_id=task_id,
            kb_id=request.kb_id,
            status="running",
            progress=0,
            metrics=metrics,
            eval_llm_provider=eval_llm_provider,
            created_at=now,
        )
    )


def _run_evaluation_worker(task_id: str, request: EvalRunRequest):
    """后台评估工作线程"""
    try:
        from src.core.chat_service import get_chat_service
        from src.core.evaluator import RAGEvaluator
        from src.core.langchain_adapters import get_eval_embeddings, get_eval_llm

        settings = get_settings()

        # 1. 准备评估 LLM
        _update_task(task_id, current_step="初始化评估 LLM...")

        eval_llm_kwargs = {}
        if request.eval_llm:
            eval_llm_kwargs = {
                "provider": request.eval_llm.provider,
                "model": request.eval_llm.model,
                "api_key": request.eval_llm.api_key,
                "base_url": request.eval_llm.base_url,
            }
            # 移除 None 值
            eval_llm_kwargs = {
                k: v for k, v in eval_llm_kwargs.items() if v is not None
            }
        else:
            eval_config = getattr(settings, "evaluation", None)
            if eval_config and hasattr(eval_config, "eval_llm"):
                llm_config = eval_config.eval_llm
                eval_llm_kwargs = {
                    "provider": getattr(llm_config, "provider", "ollama"),
                    "model": getattr(llm_config, "model", None),
                }

        eval_llm = get_eval_llm(**eval_llm_kwargs)
        eval_embeddings = get_eval_embeddings()

        # 2. 初始化评估器
        chat_service = get_chat_service(use_ollama=True)
        evaluator = RAGEvaluator(
            chat_service=chat_service,
            eval_llm=eval_llm,
            eval_embeddings=eval_embeddings,
        )

        # 3. 加载数据集
        _update_task(task_id, current_step="加载评估数据集...")

        dataset_path = _resolve_dataset_path(request.dataset_id)
        dataset = evaluator.load_dataset(dataset_path)
        _update_task(task_id, total_samples=dataset.size)

        # 4. 执行评估
        def progress_callback(completed, total, step):
            progress = int((completed / total) * 100) if total > 0 else 0
            _update_task(
                task_id,
                progress=progress,
                completed_samples=completed,
                current_step=f"正在评估第 {completed}/{total} 条数据...",
            )

        result = evaluator.run_evaluation(
            dataset=dataset,
            metrics=request.metrics,
            kb_id=request.kb_id,
            task_id=task_id,
            progress_callback=progress_callback,
        )

        # 5. 保存结果
        results_dir = Path(
            getattr(
                getattr(settings, "evaluation", None),
                "results_dir",
                "./data/eval/results",
            )
        )
        result_path = results_dir / f"{task_id}.json"
        evaluator.save_results(result, str(result_path))

        # 6. 更新任务状态
        _update_task(
            task_id,
            status="completed",
            progress=100,
            completed_at=datetime.now(),
            current_step="评估完成",
            result={
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
            },
        )

        log.info(f"评估任务完成: task_id={task_id}")

    except Exception as e:
        log.error(f"评估任务失败: task_id={task_id}, error={e}")
        _update_task(
            task_id,
            status="failed",
            error_message=str(e),
            current_step=f"评估失败: {str(e)}",
            completed_at=datetime.now(),
        )


def _resolve_dataset_path(dataset_id: str) -> str:
    """根据 dataset_id 解析数据集文件路径"""
    settings = get_settings()
    eval_config = getattr(settings, "evaluation", None)

    if dataset_id == "default":
        default_path = getattr(
            eval_config, "dataset_path", "./data/eval/eval_dataset.json"
        )
        return str(default_path)

    # 自定义数据集
    datasets_dir = Path("./data/eval/datasets")
    dataset_path = datasets_dir / f"{dataset_id}.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_id}")

    return str(dataset_path)


@router.get("/tasks", response_model=ResponseModel[PaginatedData[EvalTaskResponse]])
async def list_evaluation_tasks(
    kb_id: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """获取评估任务列表"""
    with _eval_lock:
        tasks = list(_eval_tasks.values())

    # 过滤
    if kb_id:
        tasks = [t for t in tasks if t["kb_id"] == kb_id]
    if status:
        tasks = [t for t in tasks if t["status"] == status]

    # 按创建时间倒序
    tasks.sort(key=lambda t: t["created_at"], reverse=True)

    # 分页
    total = len(tasks)
    start = (page - 1) * page_size
    end = start + page_size
    page_tasks = tasks[start:end]

    items = [
        EvalTaskResponse(
            task_id=t["task_id"],
            kb_id=t["kb_id"],
            status=t["status"],
            progress=t["progress"],
            metrics=t["metrics"],
            eval_llm_provider=t["eval_llm_provider"],
            created_at=t["created_at"],
            completed_at=t.get("completed_at"),
        )
        for t in page_tasks
    ]

    return ResponseModel(
        data=PaginatedData(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get("/tasks/{task_id}", response_model=ResponseModel[dict])
async def get_evaluation_task(task_id: str):
    """获取评估任务详细状态"""
    task = _get_task(task_id)

    return ResponseModel(
        data={
            "task_id": task["task_id"],
            "kb_id": task["kb_id"],
            "status": task["status"],
            "progress": task["progress"],
            "total_samples": task["total_samples"],
            "completed_samples": task["completed_samples"],
            "current_step": task["current_step"],
            "metrics": task["metrics"],
            "eval_llm_provider": task["eval_llm_provider"],
            "error_message": task.get("error_message"),
            "created_at": task["created_at"].isoformat(),
            "completed_at": (
                task["completed_at"].isoformat() if task.get("completed_at") else None
            ),
        }
    )


@router.get("/tasks/{task_id}/report", response_model=ResponseModel[EvalReportResponse])
async def get_evaluation_report(task_id: str):
    """获取评估报告详情"""
    task = _get_task(task_id)

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"评估任务尚未完成，当前状态: {task['status']}",
        )

    result = task.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="评估结果不存在")

    per_sample = [
        EvalSampleScores(
            question=s["question"],
            answer=s["answer"],
            ground_truth=s["ground_truth"],
            contexts=s.get("contexts", []),
            scores=s.get("scores", {}),
        )
        for s in result.get("per_sample", [])
    ]

    return ResponseModel(
        data=EvalReportResponse(
            task_id=task_id,
            summary=result["summary"],
            per_sample=per_sample,
            metadata=result.get("metadata", {}),
        )
    )


@router.delete("/tasks/{task_id}", response_model=ResponseModel[dict])
async def delete_evaluation_task(task_id: str):
    """删除评估任务"""
    _get_task(task_id)  # 验证存在

    with _eval_lock:
        del _eval_tasks[task_id]

    # 也删除结果文件
    settings = get_settings()
    results_dir = Path(
        getattr(
            getattr(settings, "evaluation", None),
            "results_dir",
            "./data/eval/results",
        )
    )
    result_file = results_dir / f"{task_id}.json"
    if result_file.exists():
        result_file.unlink()

    return ResponseModel(data={"task_id": task_id, "deleted": True})


# ============================================
# 数据集管理路由
# ============================================


@router.get("/datasets", response_model=ResponseModel[dict])
async def list_datasets():
    """获取可用评估数据集列表"""
    items = []

    # 默认数据集
    settings = get_settings()
    eval_config = getattr(settings, "evaluation", None)
    default_path = getattr(eval_config, "dataset_path", "./data/eval/eval_dataset.json")

    if Path(default_path).exists():
        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items.append(
            {
                "id": "default",
                "name": data.get("name", "默认评估数据集"),
                "sample_count": len(data.get("data", [])),
                "created_at": datetime.fromtimestamp(
                    Path(default_path).stat().st_mtime
                ).isoformat(),
            }
        )

    # 自定义数据集
    custom_dir = Path("./data/eval/datasets")
    if custom_dir.exists():
        for file in custom_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items.append(
                    {
                        "id": file.stem,
                        "name": data.get("name", file.stem),
                        "sample_count": len(data.get("data", [])),
                        "created_at": datetime.fromtimestamp(
                            file.stat().st_mtime
                        ).isoformat(),
                    }
                )
            except Exception:
                continue

    return ResponseModel(data={"items": items})


@router.post("/datasets", response_model=ResponseModel[EvalDatasetResponse])
async def create_dataset(request: EvalDatasetCreate):
    """上传评估数据集"""
    # 验证数据格式
    for item in request.data:
        if "question" not in item or "ground_truth" not in item:
            raise HTTPException(
                status_code=400,
                detail="每条数据需包含 'question' 和 'ground_truth' 字段",
            )

    dataset_id = generate_id("ds")
    datasets_dir = Path("./data/eval/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = datasets_dir / f"{dataset_id}.json"
    dataset_content = {
        "version": "1.0",
        "name": request.name,
        "data": request.data,
    }

    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(dataset_content, f, ensure_ascii=False, indent=2)

    now = datetime.now()

    log.info(f"创建评估数据集: id={dataset_id}, name={request.name}")

    return ResponseModel(
        data=EvalDatasetResponse(
            id=dataset_id,
            name=request.name,
            sample_count=len(request.data),
            created_at=now,
        )
    )
