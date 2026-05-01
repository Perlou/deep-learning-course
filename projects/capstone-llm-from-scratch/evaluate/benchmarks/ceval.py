"""
ceval.py — C-Eval 中文综合知识基准
====================================

C-Eval 是中文 LLM 最常用的基准之一（13948 道 4 选 1 题 × 52 学科），覆盖
人文/社科/STEM/医学/法律。C-Eval 也是 OpenCompass 和绝大多数 HF 中文模型卡
的硬通货指标。

评测协议（与 OpenCompass / lm-eval-harness 对齐）：
  - 5-shot：每题前面拼 5 个同学科示例（answer 已知）
  - Loglikelihood scoring：对 4 个 choice 文本各算 logprob，取最大者为预测
  - 报告 per-subject + macro avg + 4 大类（hard/STEM/social/humanities）

数据来源（按优先级）：
  1. ``--data <local_jsonl>``：本地 jsonl（每行 {"subject":..., "question":..., "A":..., "B":..., "C":..., "D":..., "answer":"A"}）
  2. HF datasets ``ceval/ceval-exam``（需联网 + ``pip install datasets``）
  3. ``--limit N``：只跑前 N 题，调试用

用法:
  # 完整 5-shot eval
  python evaluate/benchmarks/ceval.py --config configs/main.yaml --output reports/base_ceval.json

  # 调试：只跑 50 题
  python evaluate/benchmarks/ceval.py --config configs/tiny.yaml --limit 50

  # 自定义模型
  python evaluate/benchmarks/ceval.py --model outputs/sft/final.pth --config configs/main.yaml

输出 JSON:
  {
    "meta": {benchmark, model_path, ...},
    "results": {
      "per_subject": {"high_school_physics": {acc, n}, ...},
      "by_category": {"STEM": ..., "Social Science": ..., "Humanities": ..., "Other": ...},
      "summary": {"macro_acc": ..., "n_total": ...}
    }
  }
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from evaluate._common import (
    load_model_for_eval,
    score_choices_loglikelihood,
    set_seed,
    make_meta,
    dump_json_report,
    load_jsonl,
    try_load_hf_dataset,
)


# C-Eval 52 学科 → 4 大类映射（OpenCompass 标准分组）
SUBJECT_CATEGORIES: dict[str, str] = {
    # STEM
    "computer_network": "STEM", "operating_system": "STEM", "computer_architecture": "STEM",
    "college_programming": "STEM", "college_physics": "STEM", "college_chemistry": "STEM",
    "advanced_mathematics": "STEM", "probability_and_statistics": "STEM",
    "discrete_mathematics": "STEM", "electrical_engineer": "STEM", "metrology_engineer": "STEM",
    "high_school_mathematics": "STEM", "high_school_physics": "STEM",
    "high_school_chemistry": "STEM", "high_school_biology": "STEM",
    "middle_school_mathematics": "STEM", "middle_school_biology": "STEM",
    "middle_school_physics": "STEM", "middle_school_chemistry": "STEM",
    "veterinary_medicine": "STEM",
    # Social Science
    "college_economics": "Social Science", "business_administration": "Social Science",
    "marxism": "Social Science", "mao_zedong_thought": "Social Science",
    "education_science": "Social Science", "teacher_qualification": "Social Science",
    "high_school_politics": "Social Science", "high_school_geography": "Social Science",
    "middle_school_politics": "Social Science", "middle_school_geography": "Social Science",
    # Humanities
    "modern_chinese_history": "Humanities", "ideological_and_moral_cultivation": "Humanities",
    "logic": "Humanities", "law": "Humanities", "chinese_language_and_literature": "Humanities",
    "art_studies": "Humanities", "professional_tour_guide": "Humanities",
    "legal_professional": "Humanities", "high_school_chinese": "Humanities",
    "high_school_history": "Humanities", "middle_school_history": "Humanities",
    # Other (medicine + 工程实务)
    "civil_servant": "Other", "sports_science": "Other",
    "plant_protection": "Other", "basic_medicine": "Other",
    "clinical_medicine": "Other", "urban_and_rural_planner": "Other",
    "accountant": "Other", "fire_engineer": "Other", "environmental_impact_assessment_engineer": "Other",
    "tax_accountant": "Other", "physician": "Other",
}


def _build_5shot_prompt(item: dict, fewshot_pool: list[dict]) -> str:
    """构造 5-shot 提示文本

    格式（C-Eval 官方推荐）：
        以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。

        题目1
        A. ...
        B. ...
        C. ...
        D. ...
        答案：X

        ... (5 examples)

        题目（待答）
        A. ...
        ...
        答案：
    """
    subject = item["subject"]
    subject_zh = subject.replace("_", " ")

    prompt = f"以下是中国关于{subject_zh}考试的单项选择题，请选出其中的正确答案。\n\n"

    # 5 个 few-shot 示例
    for ex in fewshot_pool[:5]:
        prompt += f"{ex['question']}\n"
        prompt += f"A. {ex['A']}\nB. {ex['B']}\nC. {ex['C']}\nD. {ex['D']}\n"
        prompt += f"答案：{ex['answer']}\n\n"

    # 当前题
    prompt += f"{item['question']}\n"
    prompt += f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n"
    prompt += "答案："
    return prompt


def _load_ceval_data(local_path: str | None) -> list[dict]:
    """加载 C-Eval 数据。优先本地，否则 HF。返回扁平 list[dict]"""
    if local_path:
        print(f"📦 从本地加载: {local_path}")
        return load_jsonl(local_path)

    # HF: ceval/ceval-exam 是按 subject 划分的，需要遍历
    print("📦 尝试从 HF datasets 加载 ceval/ceval-exam ...")
    all_samples: list[dict] = []
    for subject in tqdm(SUBJECT_CATEGORIES.keys(), desc="loading subjects"):
        ds = try_load_hf_dataset("ceval/ceval-exam", subset=subject, split="val")
        if ds is None:
            continue
        for item in ds:
            item["subject"] = subject
            all_samples.append(item)

    if not all_samples:
        raise RuntimeError(
            "C-Eval 数据加载失败。请：\n"
            "  1) pip install datasets\n"
            "  2) 网络可访问 huggingface.co；或\n"
            "  3) 用 --data 指定本地 jsonl 路径"
        )
    return all_samples


def _evaluate(
    model, tokenizer, samples: list[dict], device, max_seq_len: int,
    limit: int | None = None,
) -> dict:
    """跑完所有样本，返回 per_subject + by_category + summary"""
    # 按 subject 分组（构 5-shot 池）
    by_subject: dict[str, list[dict]] = collections.defaultdict(list)
    for s in samples:
        by_subject[s["subject"]].append(s)

    per_subject: dict[str, dict] = {}
    n_total = 0
    n_correct = 0
    n_evaluated = 0

    for subject, items in tqdm(by_subject.items(), desc="subjects"):
        if len(items) < 6:  # 不够 5-shot + 1 测试
            continue
        # 用前 5 条做 fewshot pool，从第 6 条开始评测
        fewshot = items[:5]
        test_items = items[5:]
        if limit is not None:
            test_items = test_items[: max(1, limit // len(by_subject))]

        sub_correct = 0
        sub_total = 0
        for item in test_items:
            prompt = _build_5shot_prompt(item, fewshot)
            # 标签是 "A"/"B"/"C"/"D"，loglik 评分
            scores = score_choices_loglikelihood(
                model, tokenizer, prompt, ["A", "B", "C", "D"],
                device=device, max_seq_len=max_seq_len,
            )
            pred = ["A", "B", "C", "D"][int(torch.tensor(scores).argmax().item())]
            gold = item.get("answer", "").strip().upper()
            if pred == gold:
                sub_correct += 1
            sub_total += 1
            n_evaluated += 1

        if sub_total > 0:
            per_subject[subject] = {
                "acc": sub_correct / sub_total,
                "correct": sub_correct,
                "n": sub_total,
                "category": SUBJECT_CATEGORIES.get(subject, "Other"),
            }
            n_total += sub_total
            n_correct += sub_correct

    # 4 大类聚合
    by_cat: dict[str, dict] = collections.defaultdict(lambda: {"correct": 0, "n": 0})
    for sub, st in per_subject.items():
        cat = st["category"]
        by_cat[cat]["correct"] += st["correct"]
        by_cat[cat]["n"] += st["n"]
    by_category = {
        cat: {"acc": v["correct"] / max(v["n"], 1), "correct": v["correct"], "n": v["n"]}
        for cat, v in by_cat.items()
    }

    # macro_acc：按学科平均（avoid 题量大的学科主导）
    macro_acc = (
        sum(s["acc"] for s in per_subject.values()) / max(len(per_subject), 1)
    )
    micro_acc = n_correct / max(n_total, 1)

    return {
        "per_subject": per_subject,
        "by_category": by_category,
        "summary": {
            "macro_acc": macro_acc,
            "micro_acc": micro_acc,
            "n_subjects": len(per_subject),
            "n_total": n_total,
            "n_correct": n_correct,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="C-Eval 中文综合知识基准评测")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--data", default=None,
                        help="本地 jsonl 路径（每行含 subject/question/A/B/C/D/answer）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 题，调试用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="输出 JSON 路径（默认 reports/ceval_<timestamp>.json）")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("ClearMind × C-Eval 评测")
    print("=" * 60)

    # 加载模型
    model, tokenizer, model_config, device = load_model_for_eval(
        args.config, args.model, args.tokenizer,
    )
    print(f"📦 Model:  {args.model or '<auto-detected>'}")
    print(f"🖥  Device: {device}")
    print(f"🎲 Seed:   {args.seed}")

    # 加载数据
    samples = _load_ceval_data(args.data)
    print(f"📊 样本数: {len(samples)}")

    # 跑评测
    results = _evaluate(
        model, tokenizer, samples, device,
        max_seq_len=model_config.max_seq_len,
        limit=args.limit,
    )

    # 打印 summary
    summary = results["summary"]
    print()
    print("=" * 60)
    print("📊 C-Eval 结果")
    print("=" * 60)
    print(f"  Macro Acc: {summary['macro_acc'] * 100:.2f}%")
    print(f"  Micro Acc: {summary['micro_acc'] * 100:.2f}%")
    print(f"  覆盖学科 : {summary['n_subjects']} / {len(SUBJECT_CATEGORIES)}")
    print(f"  总题数   : {summary['n_total']}")
    print()
    print("  各大类:")
    for cat, st in results["by_category"].items():
        print(f"    {cat:20s} {st['acc'] * 100:5.2f}%  ({st['correct']}/{st['n']})")
    print()
    print(f"  随机基线: 25.00%（4 选 1）")

    # 保存 JSON
    out_path = args.output or f"reports/ceval_{model_config.d_model}d.json"
    meta = make_meta(
        benchmark="ceval",
        model_path=args.model or "auto",
        config_path=args.config,
        seed=args.seed,
        device=device,
        n_subjects=summary["n_subjects"],
        n_total=summary["n_total"],
    )
    dump_json_report(out_path, meta, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
