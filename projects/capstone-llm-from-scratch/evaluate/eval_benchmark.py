"""
eval_benchmark.py — 综合评估报告
==================================

一键运行所有评估脚本，生成综合对比报告。
对比 Pretrain / SFT / DPO 三阶段的模型表现。

评估维度:
  1. 困惑度 (PPL):        语言建模能力
  2. 生成多样性 (Dist-N):  输出的词汇丰富度
  3. 重复率:               输出的重复程度
  4. 指令跟随:             格式正确率、相关率、完整性
  5. 安全性:               对有害问题的拒绝率

使用方法:
  python evaluate/eval_benchmark.py --config configs/small.yaml
  python evaluate/eval_benchmark.py --config configs/small.yaml --output results/benchmark.json
"""

import os
import sys

import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml


from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer
from src.data.pretrain_dataset import PretrainDataset
from src.training.trainer_utils import get_device, load_checkpoint

# 复用其他评估模块
from eval_perplexity import evaluate_perplexity
from eval_generation import (
    evaluate_model as evaluate_generation,
)
from eval_instruction import (
    evaluate_instructions,
)


# ============================================================
# 综合评估
# ============================================================


def run_full_evaluation(
    config_path: str,
    model_path: str,
    tokenizer_path: str,
    data_path: str,
    device: torch.device,
    stage_name: str = "model",
    max_new_tokens: int = 200,
    ppl_max_batches: int = 50,
) -> dict:
    """对一个模型运行完整评估

    Args:
        config_path:   配置文件路径
        model_path:    checkpoint 路径
        tokenizer_path: 分词器路径
        data_path:     PPL 评估数据路径
        device:        计算设备
        stage_name:    阶段名称
        max_new_tokens: 生成最大 token 数
        ppl_max_batches: PPL 评估最大 batch 数

    Returns:
        完整评估结果字典
    """
    # 加载配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])
    tokenizer = ClearMindTokenizer(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    # 加载模型
    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    param_info = model.count_parameters()
    results = {
        "stage": stage_name,
        "model_path": model_path,
        "parameters": param_info["total_millions"],
    }

    # --- 1. 困惑度评估 ---
    print(f"\n  📊 [1/3] 困惑度评估...")
    try:
        if os.path.exists(data_path):
            dataset = PretrainDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_seq_len=model_config.max_seq_len,
            )
            ppl = evaluate_perplexity(
                model=model,
                dataset=dataset,
                batch_size=8,
                device=device,
                max_batches=ppl_max_batches,
            )
            results["perplexity"] = ppl
            print(f"     PPL = {ppl:.2f}")
        else:
            results["perplexity"] = None
            print(f"     ⚠️  数据文件不存在: {data_path}")
    except Exception as e:
        results["perplexity"] = None
        print(f"     ⚠️  PPL 评估失败: {e}")

    # --- 2. 生成质量评估 ---
    print(f"  📝 [2/3] 生成质量评估...")
    try:
        gen_results = evaluate_generation(
            model,
            tokenizer,
            device,
            model_name=stage_name,
            max_new_tokens=max_new_tokens,
        )
        results["generation"] = {
            task: data["metrics"] for task, data in gen_results.items()
        }
    except Exception as e:
        results["generation"] = None
        print(f"     ⚠️  生成评估失败: {e}")

    # --- 3. 指令跟随评估 ---
    print(f"  🎯 [3/3] 指令跟随评估...")
    try:
        inst_results = evaluate_instructions(
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
        )
        results["instruction"] = inst_results["summary"]
    except Exception as e:
        results["instruction"] = None
        print(f"     ⚠️  指令评估失败: {e}")

    # 清理显存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================
# 报告输出
# ============================================================


def print_comparison_report(all_results: dict):
    """打印综合对比报告"""
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"{'║':^1}{'ClearMind 综合评估报告':^68}{'║':>1}")
    print(f"{'║':^1}{datetime.now().strftime('%Y-%m-%d %H:%M'):^68}{'║':>1}")
    print(f"{'╠' + '═' * 68 + '╣'}")

    stages = list(all_results.keys())

    # === 表格 1: 核心指标总览 ===
    print(f"\n  📊 核心指标总览")
    header = f"  {'阶段':^10} │ {'参数量':^8} │ {'PPL':^8} │ {'Dist-2':^7} │ {'重复率':^7} │ {'格式正确':^8}"
    print(f"  {'─' * 65}")
    print(header)
    print(f"  {'─' * 65}")

    for stage_name, r in all_results.items():
        ppl = f"{r['perplexity']:.1f}" if r.get("perplexity") else "N/A"
        params = f"{r['parameters']:.1f}M"

        # 生成指标
        if r.get("generation") and "总体" in r["generation"]:
            gen = r["generation"]["总体"]
            dist2 = f"{gen['distinct_2']:.3f}"
            rep = f"{gen['avg_repetition_rate']:.1%}"
        else:
            dist2 = "N/A"
            rep = "N/A"

        # 指令指标
        if r.get("instruction"):
            inst = r["instruction"]
            fmt = f"{inst['format_accuracy']:.1%}"
        else:
            fmt = "N/A"

        print(
            f"  {stage_name:^10} │ {params:^8} │ {ppl:^8} │ "
            f"{dist2:^7} │ {rep:^7} │ {fmt:^8}"
        )

    print(f"  {'─' * 65}")

    # === 表格 2: 指令跟随详情 ===
    has_instruction = any(r.get("instruction") for r in all_results.values())
    if has_instruction:
        print(f"\n  🎯 指令跟随详情")
        print(f"  {'─' * 60}")
        print(
            f"  {'阶段':^10} │ {'格式正确':^8} │ {'完整率':^7} │ "
            f"{'相关率':^7} │ {'拒绝率':^7}"
        )
        print(f"  {'─' * 60}")

        for stage_name, r in all_results.items():
            if not r.get("instruction"):
                continue
            inst = r["instruction"]
            print(
                f"  {stage_name:^10} │ "
                f"{inst['format_accuracy']:>7.1%} │ "
                f"{inst['completeness_rate']:>6.1%} │ "
                f"{inst['relevance_rate']:>6.1%} │ "
                f"{inst['refusal_rate']:>6.1%}"
            )

        print(f"  {'─' * 60}")

    # === 表格 3: 各任务生成多样性 ===
    has_generation = any(r.get("generation") for r in all_results.values())
    if has_generation:
        print(f"\n  📝 各任务生成多样性 (Distinct-2)")
        print(f"  {'─' * 55}")
        print(f"  {'阶段':^10} │ {'续写':^8} │ {'指令':^8} │ {'对齐':^8} │ {'总体':^8}")
        print(f"  {'─' * 55}")

        for stage_name, r in all_results.items():
            if not r.get("generation"):
                continue
            gen = r["generation"]

            def get_dist2(task):
                return f"{gen[task]['distinct_2']:.3f}" if task in gen else "N/A"

            print(
                f"  {stage_name:^10} │ "
                f"{get_dist2('续写任务'):^8} │ "
                f"{get_dist2('指令任务'):^8} │ "
                f"{get_dist2('对齐任务'):^8} │ "
                f"{get_dist2('总体'):^8}"
            )

        print(f"  {'─' * 55}")

    # === 结论 ===
    print(f"\n  💡 解读指南:")
    print(f"    • PPL ↓ = 语言建模能力更强 (预训练后应大幅下降)")
    print(f"    • Dist-N ↑ = 输出更多样 (DPO 后应提升)")
    print(f"    • 重复率 ↓ = 输出重复更少 (各阶段应逐步降低)")
    print(f"    • 格式正确 ↑ = 指令跟随更好 (SFT 后应明显提升)")
    print(f"    • 拒绝率 ↑ = 安全对齐更好 (DPO 后应提升)")

    print(f"\n{'╚' + '═' * 68 + '╝'}")


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="ClearMind 综合评估报告")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/pretrain/pretrain_data.jsonl",
        help="困惑度评估数据路径",
    )
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument(
        "--ppl_batches", type=int, default=50, help="PPL 评估最大 batch 数"
    )
    parser.add_argument("--output", type=str, help="结果保存路径 (.json)")
    parser.add_argument(
        "--stages", nargs="+", default=["pretrain", "sft", "dpo"], help="要评估的阶段"
    )
    args = parser.parse_args()

    device = get_device()

    print(f"{'=' * 70}")
    print(f"🧠 ClearMind 综合评估")
    print(f"{'=' * 70}")
    print(f"📋 配置: {args.config}")
    print(f"🔧 设备: {device}")
    print(f"📊 评估阶段: {', '.join(args.stages)}")

    # 定义各阶段 checkpoint 路径
    stage_paths = {
        "pretrain": "outputs/pretrain/final.pth",
        "sft": "outputs/sft/final.pth",
        "dpo": "outputs/dpo/final.pth",
    }

    all_results = {}

    for stage in args.stages:
        model_path = stage_paths.get(stage)
        if not model_path:
            print(f"\n⚠️  未知阶段: {stage}")
            continue

        if not os.path.exists(model_path):
            print(f"\n⚠️  跳过 {stage}: {model_path} 不存在")
            continue

        print(f"\n{'=' * 70}")
        print(f"🔍 评估阶段: {stage.upper()}")
        print(f"   模型: {model_path}")
        print(f"{'=' * 70}")

        results = run_full_evaluation(
            config_path=args.config,
            model_path=model_path,
            tokenizer_path=args.tokenizer,
            data_path=args.data,
            device=device,
            stage_name=stage,
            max_new_tokens=args.max_tokens,
            ppl_max_batches=args.ppl_batches,
        )

        all_results[stage] = results

    # 输出综合报告
    if all_results:
        print_comparison_report(all_results)

    # 保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 完整结果已保存: {args.output}")
    else:
        print(f"\n💡 提示: 使用 --output results/benchmark.json 保存结果")


if __name__ == "__main__":
    main()
