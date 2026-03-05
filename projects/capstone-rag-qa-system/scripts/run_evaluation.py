#!/usr/bin/env python3
"""
DocuMind AI - RAG 评估脚本

命令行工具，用于运行 Ragas 评估。

用法:
    # 使用本地 Ollama
    python scripts/run_evaluation.py --dataset data/eval/eval_dataset.json --kb-id <KB_ID>

    # 使用线上 API
    python scripts/run_evaluation.py --dataset data/eval/eval_dataset.json --kb-id <KB_ID> \
        --provider openai --model gpt-4o-mini

    # 指定评估指标
    python scripts/run_evaluation.py --dataset data/eval/eval_dataset.json --kb-id <KB_ID> \
        --metrics faithfulness answer_relevancy
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="DocuMind AI RAG 评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/eval_dataset.json",
        help="评估数据集路径 (JSON)",
    )
    parser.add_argument(
        "--kb-id",
        type=str,
        required=True,
        help="要评估的知识库 ID",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="要计算的指标列表（默认全部）",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "openai", "dashscope"],
        help="评估用 LLM 提供方",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="评估用模型名称",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 密钥（线上模式）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API 地址（线上模式）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出路径（默认 data/eval/results/）",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="生成可读评估报告",
    )

    args = parser.parse_args()

    # 延迟导入，避免启动时加载所有依赖
    from src.core.chat_service import get_chat_service
    from src.core.evaluator import RAGEvaluator
    from src.core.langchain_adapters import get_eval_embeddings, get_eval_llm

    print("=" * 60)
    print("DocuMind AI - RAG 评估工具")
    print("=" * 60)
    print()

    # 1. 初始化评估 LLM
    print(f"[1/4] 初始化评估 LLM (provider={args.provider})...")
    eval_llm = get_eval_llm(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # 2. 初始化 Embeddings
    print("[2/4] 初始化 Embeddings...")
    eval_embeddings = get_eval_embeddings()

    # 3. 初始化评估器
    print("[3/4] 初始化 RAG 评估器...")
    chat_service = get_chat_service(use_ollama=True)
    evaluator = RAGEvaluator(
        chat_service=chat_service,
        eval_llm=eval_llm,
        eval_embeddings=eval_embeddings,
    )

    # 4. 加载数据集并运行评估
    print(f"[4/4] 加载数据集: {args.dataset}")
    dataset = evaluator.load_dataset(args.dataset)
    print(f"       样本数: {dataset.size}")
    print()

    # 进度回调
    def progress_callback(completed, total, step):
        bar_len = 30
        filled = int(bar_len * completed / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = (completed / total * 100) if total > 0 else 0
        print(f"\r  进度: {bar} {pct:.0f}% ({completed}/{total})", end="", flush=True)

    print("开始评估...")
    print()

    result = evaluator.run_evaluation(
        dataset=dataset,
        metrics=args.metrics,
        kb_id=args.kb_id,
        progress_callback=progress_callback,
    )

    print()
    print()

    # 保存结果
    output_path = args.output
    if not output_path:
        output_dir = Path("data/eval/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{result.task_id}.json")

    evaluator.save_results(result, output_path)
    print(f"结果已保存: {output_path}")

    # 打印报告
    if args.report:
        report = evaluator.generate_report(result)
        print()
        print(report)
    else:
        # 打印简要结果
        print()
        print("-" * 40)
        print("评估结果摘要:")
        print("-" * 40)
        for metric, score in result.summary.items():
            print(f"  {metric}: {score:.4f}")
        print()
        print(f"评估耗时: {result.metadata.get('duration_seconds', 'N/A')}s")
        print()
        print("使用 --report 参数查看详细报告")


if __name__ == "__main__":
    main()
