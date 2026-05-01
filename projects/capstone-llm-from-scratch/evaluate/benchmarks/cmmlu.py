"""
cmmlu.py — CMMLU 中文 MMLU 等价基准
====================================

CMMLU 是 MMLU 的中文等价（67 学科 11528 题），与 C-Eval 互补：
  - C-Eval 偏中国国情/政治/语文
  - CMMLU 偏全球通识/科学/医学/工程

评测协议：5-shot loglikelihood scoring，与 ``ceval.py`` 完全相同。

数据来源（按优先级）：
  1. ``--data <local_jsonl>``
  2. HF datasets ``haonan-li/cmmlu``
  3. ``--limit N``：调试用

用法:
  python evaluate/benchmarks/cmmlu.py --config configs/main.yaml --output reports/base_cmmlu.json
  python evaluate/benchmarks/cmmlu.py --config configs/tiny.yaml --limit 50
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


# CMMLU 67 学科 → 5 大类（CMMLU 官方分组）
CMMLU_SUBJECTS = [
    # STEM
    "agronomy", "anatomy", "astronomy", "college_actuarial_science", "college_engineering_hydrology",
    "college_mathematics", "college_medical_statistics", "college_medicine", "computer_science",
    "computer_security", "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "genetics", "high_school_biology", "high_school_chemistry", "high_school_mathematics",
    "high_school_physics", "machine_learning", "nutrition", "virology",
    # Social Science
    "business_ethics", "economics", "education", "elementary_commonsense", "elementary_information_and_technology",
    "global_facts", "high_school_geography", "high_school_politics", "human_sexuality",
    "international_law", "journalism", "jurisprudence", "logical", "management",
    "marketing", "marxist_theory", "professional_psychology", "public_relations",
    "security_study", "sociology", "sports_science",
    # Humanities
    "ancient_chinese", "arts", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "ethnology",
    "philosophy", "professional_law", "world_history", "world_religions",
    # Other
    "construction_project_management", "elementary_chinese", "food_science",
    "high_school_chinese", "high_school_history", "legal_and_moral_basis",
    "modern_chinese", "professional_accounting", "professional_medicine", "traditional_chinese_medicine",
    # China specific
    "clinical_knowledge", "college_law",
]


def _category_of(subject: str) -> str:
    """简化的 5 大类映射"""
    stem = {"agronomy", "anatomy", "astronomy", "college_actuarial_science",
            "college_engineering_hydrology", "college_mathematics",
            "college_medical_statistics", "college_medicine", "computer_science",
            "computer_security", "conceptual_physics", "electrical_engineering",
            "elementary_mathematics", "genetics", "high_school_biology",
            "high_school_chemistry", "high_school_mathematics", "high_school_physics",
            "machine_learning", "nutrition", "virology"}
    if subject in stem:
        return "STEM"
    humanities = {"ancient_chinese", "arts", "chinese_history", "chinese_literature",
                  "ethnology", "philosophy", "professional_law", "world_history",
                  "world_religions"}
    if subject in humanities:
        return "Humanities"
    china = {"chinese_civil_service_exam", "chinese_driving_rule",
             "chinese_food_culture", "chinese_foreign_policy",
             "chinese_teacher_qualification", "elementary_chinese",
             "high_school_chinese", "modern_chinese", "traditional_chinese_medicine",
             "legal_and_moral_basis"}
    if subject in china:
        return "China-specific"
    other = {"construction_project_management", "food_science",
             "high_school_history", "professional_accounting", "professional_medicine",
             "clinical_knowledge", "college_law"}
    if subject in other:
        return "Other"
    return "Social Science"


def _build_prompt(item: dict, fewshot_pool: list[dict]) -> str:
    subject_zh = item["subject"].replace("_", " ")
    prompt = f"以下是关于{subject_zh}的单项选择题，请选出正确答案。\n\n"
    for ex in fewshot_pool[:5]:
        prompt += f"{ex['Question']}\nA. {ex['A']}\nB. {ex['B']}\nC. {ex['C']}\nD. {ex['D']}\n答案：{ex['Answer']}\n\n"
    prompt += f"{item['Question']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n答案："
    return prompt


def _load_data(local: str | None) -> list[dict]:
    if local:
        print(f"📦 从本地加载: {local}")
        return load_jsonl(local)
    print("📦 尝试从 HF datasets 加载 haonan-li/cmmlu ...")
    samples: list[dict] = []
    for subj in tqdm(CMMLU_SUBJECTS, desc="loading"):
        ds = try_load_hf_dataset("haonan-li/cmmlu", subset=subj, split="test")
        if ds is None:
            continue
        for r in ds:
            r["subject"] = subj
            samples.append(r)
    if not samples:
        raise RuntimeError("CMMLU 加载失败；请检查网络或用 --data 指定本地路径")
    return samples


def _evaluate(model, tokenizer, samples, device, max_seq_len, limit=None) -> dict:
    by_subject = collections.defaultdict(list)
    for s in samples:
        by_subject[s["subject"]].append(s)

    per_subject = {}
    n_total = n_correct = 0

    for subject, items in tqdm(by_subject.items(), desc="subjects"):
        if len(items) < 6:
            continue
        fewshot = items[:5]
        test_items = items[5:]
        if limit is not None:
            test_items = test_items[: max(1, limit // len(by_subject))]

        sc = st = 0
        for it in test_items:
            prompt = _build_prompt(it, fewshot)
            scores = score_choices_loglikelihood(
                model, tokenizer, prompt, ["A", "B", "C", "D"],
                device=device, max_seq_len=max_seq_len,
            )
            pred = ["A", "B", "C", "D"][int(torch.tensor(scores).argmax().item())]
            gold = it.get("Answer", "").strip().upper()
            if pred == gold:
                sc += 1
            st += 1

        if st > 0:
            per_subject[subject] = {
                "acc": sc / st, "correct": sc, "n": st,
                "category": _category_of(subject),
            }
            n_correct += sc
            n_total += st

    by_cat = collections.defaultdict(lambda: {"correct": 0, "n": 0})
    for sub, v in per_subject.items():
        c = v["category"]
        by_cat[c]["correct"] += v["correct"]
        by_cat[c]["n"] += v["n"]

    return {
        "per_subject": per_subject,
        "by_category": {
            c: {"acc": v["correct"] / max(v["n"], 1), **v}
            for c, v in by_cat.items()
        },
        "summary": {
            "macro_acc": sum(v["acc"] for v in per_subject.values()) / max(len(per_subject), 1),
            "micro_acc": n_correct / max(n_total, 1),
            "n_subjects": len(per_subject),
            "n_total": n_total,
            "n_correct": n_correct,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CMMLU 中文 MMLU 评测")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print("ClearMind × CMMLU 评测")
    print("=" * 60)

    model, tokenizer, model_config, device = load_model_for_eval(
        args.config, args.model, args.tokenizer,
    )
    samples = _load_data(args.data)
    print(f"📊 样本数: {len(samples)}")

    results = _evaluate(
        model, tokenizer, samples, device,
        max_seq_len=model_config.max_seq_len, limit=args.limit,
    )

    summary = results["summary"]
    print()
    print(f"📊 CMMLU 结果")
    print(f"  Macro Acc: {summary['macro_acc'] * 100:.2f}%")
    print(f"  Micro Acc: {summary['micro_acc'] * 100:.2f}%")
    print(f"  随机基线 : 25.00%")
    print()
    for cat, st in results["by_category"].items():
        print(f"  {cat:18s} {st['acc'] * 100:5.2f}%  ({st['correct']}/{st['n']})")

    out = args.output or f"reports/cmmlu_{model_config.d_model}d.json"
    meta = make_meta(
        benchmark="cmmlu", model_path=args.model or "auto",
        config_path=args.config, seed=args.seed, device=device,
        n_subjects=summary["n_subjects"], n_total=summary["n_total"],
    )
    dump_json_report(out, meta, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
