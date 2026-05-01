# ClearMind 评测体系（evaluate/）

> Phase E1 升级版：从 keyword overlap 升级到 C-Eval / CMMLU / LLM-as-Judge / 多模型对照。

## 设计哲学

| 旧（自用够，发布不够） | 新（达到对外发布门槛） |
|---|---|
| PPL + Distinct-N + 14 题 keyword overlap | C-Eval / CMMLU / AlignBench / LLM-Judge |
| 结果只在 stdout | 统一 JSON 报告（``reports/<bench>_<size>.json``） |
| 同一模型每次跑分不同 | 全部支持 ``--seed`` |
| 无对照基线 | ``eval_compare.py`` 多模型同配置同硬件对照 |
| 无标准化基准 | 与 OpenCompass / lm-eval-harness 对齐 |

## 目录结构

```
evaluate/
├── _common.py                    # 共享：模型加载 / loglikelihood / batch_generate / JSON dump
├── eval_perplexity.py            # PPL（旧，保留）
├── eval_generation.py            # 多样性 / 重复率（旧，保留）
├── eval_instruction.py           # 14 题 keyword overlap（旧，保留 — 用于 tiny 冒烟）
├── eval_benchmark.py             # 一键综合（旧）
├── eval_compare.py               # ⭐ 多模型对照（新）
├── benchmarks/
│   ├── ceval.py                  # ⭐ C-Eval 5-shot loglik（新）
│   ├── cmmlu.py                  # ⭐ CMMLU 5-shot loglik（新）
│   └── alignbench.py             # ⭐ AlignBench-zh + LLM-Judge（新）
└── judge/
    └── llm_judge.py              # ⭐ OpenAI 兼容 LLM 评分客户端（新）
```

## 三档使用方式

### 档 1：本地 / 训练时（不依赖网络、不花钱）

```bash
# PPL 三阶段对比（瞬间出结果，看 SFT/DPO 是否实际有改善）
python evaluate/eval_perplexity.py --config configs/main.yaml --compare

# 14 题 keyword 检查（粗糙但快，用于冒烟）
python evaluate/eval_instruction.py --config configs/main.yaml
```

### 档 2：发布前（公开基准，OpenCompass 可对齐）

```bash
# C-Eval（4 选 1，5-shot loglik；约 30 min on 4090，2h on plus）
python evaluate/benchmarks/ceval.py --config configs/main.yaml \
    --output reports/clearmind_base_ceval.json

# CMMLU（同上协议，覆盖更全球）
python evaluate/benchmarks/cmmlu.py --config configs/main.yaml \
    --output reports/clearmind_base_cmmlu.json

# AlignBench（生成 + LLM 打分；需配 OPENAI_API_KEY）
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com/v1
export CLEARMIND_JUDGE_MODEL=deepseek-chat
python evaluate/benchmarks/alignbench.py --config configs/main.yaml \
    --output reports/clearmind_base_alignbench.json

# 没 API key 时退化（只生成不打分）
python evaluate/benchmarks/alignbench.py --config configs/main.yaml --no-judge
```

### 档 3：对外宣称"反超 minimind" 的硬证据

```bash
# 在同一硬件、同一 sampling 配置下，跑两个本项目 checkpoint 对照
python evaluate/eval_compare.py run \
    --benchmark ceval \
    --models base:configs/main.yaml:outputs/dpo/final.pth \
              plus:configs/plus.yaml:outputs/dpo/final.pth \
    --output reports/compare_ceval.md

# 合并已有报告（含别人发的官方分）→ markdown 对照表
python evaluate/eval_compare.py merge \
    clearmind:reports/clearmind_base_ceval.json \
    minimind3:reports/minimind3_official_ceval.json \
    --output reports/vs_minimind.md
```

## 标准 JSON 报告格式

所有 ``benchmarks/*.py`` 输出统一格式，方便 ``eval_compare.py`` 合并：

```json
{
  "meta": {
    "benchmark": "ceval",
    "model_path": "outputs/dpo/final.pth",
    "config_path": "configs/main.yaml",
    "timestamp": "2026-05-01 23:30:00",
    "seed": 42,
    "device": "mps",
    "extra": {"n_subjects": 52, "n_total": 1346}
  },
  "results": {
    "per_subject": {"high_school_physics": {"acc": 0.31, "n": 19}, ...},
    "by_category": {"STEM": {"acc": 0.27, "n": 380}, ...},
    "summary": {"macro_acc": 0.28, "micro_acc": 0.29, ...}
  }
}
```

## 调试 / 冒烟（每个 runner 都支持 ``--limit``）

```bash
python evaluate/benchmarks/ceval.py --config configs/tiny.yaml --limit 30
python evaluate/benchmarks/cmmlu.py --config configs/tiny.yaml --limit 30
python evaluate/benchmarks/alignbench.py --config configs/tiny.yaml --limit 5 --no-judge
```

## 依赖

- 必需：``torch`` ``transformers`` ``pyyaml`` ``tqdm``（项目本身已含）
- 可选：``datasets``（HF 数据集自动下载） ``httpx``（LLM-Judge HTTP 客户端，无则 fallback urllib）

## 推荐发布前 checklist

- [ ] ``ceval.py`` 跑完，macro_acc 报告
- [ ] ``cmmlu.py`` 跑完，macro_acc 报告
- [ ] ``alignbench.py`` 用 DeepSeek/Qwen-72B 当 judge 跑完
- [ ] ``eval_compare.py merge`` 合并 ClearMind / minimind / Qwen3-0.5B 的 JSON
- [ ] PPL 三阶段对比（pretrain → sft → dpo 是否单调下降）
- [ ] 把 markdown 对照表贴到 HuggingFace 模型卡的 Evaluation 章节
