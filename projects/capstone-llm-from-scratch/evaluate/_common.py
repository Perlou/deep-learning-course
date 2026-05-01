"""
_common.py — Phase E1 评估系统的共享工具
========================================

提供：
  - 模型加载（按 config + checkpoint 路径）
  - Loglikelihood 评分（用于 MCQ 基准）
  - 批量生成（避免逐 prompt 串行）
  - 标准化的 JSON 结果 dump（含 metadata + seed + 模型信息）

所有 ``evaluate/benchmarks/*`` 与 ``evaluate/judge/*`` 都从这里取共用逻辑，
保证不同 benchmark 的运行模式、报告格式、模型加载路径完全一致。
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence, Iterable

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.training.trainer_utils import get_device, load_checkpoint


# ============================================================
# 元信息
# ============================================================


@dataclass
class RunMeta:
    """单次评估运行的元信息（写入 JSON 报告，便于复现）"""

    benchmark: str
    model_path: str
    config_path: str
    timestamp: str
    seed: int
    device: str
    extra: dict = field(default_factory=dict)


def make_meta(benchmark: str, model_path: str, config_path: str, seed: int,
              device: torch.device, **extra) -> RunMeta:
    return RunMeta(
        benchmark=benchmark,
        model_path=model_path,
        config_path=config_path,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=seed,
        device=str(device),
        extra=extra,
    )


def dump_json_report(path: str | Path, meta: RunMeta, results: dict) -> None:
    """统一 JSON 报告格式：{meta:..., results:..., summary:...}"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": asdict(meta), "results": results}
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📝 报告已保存: {p}")


# ============================================================
# 随机种子
# ============================================================


def set_seed(seed: int) -> None:
    """统一种子设置（torch / numpy / python random）"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 模型加载（统一接口）
# ============================================================


def load_model_for_eval(
    config_path: str,
    model_path: str | None = None,
    tokenizer_override: str | None = None,
    device: torch.device | None = None,
) -> tuple[GPT, "object", ModelConfig, torch.device]:
    """加载 GPT 模型 + tokenizer 用于评估

    Args:
        config_path:   YAML 路径（决定模型架构）
        model_path:    checkpoint 路径；None 时按 dpo→sft→pretrain 自动查
        tokenizer_override: 可选 tokenizer 路径
        device:        指定设备；None 自动检测

    Returns:
        (model, tokenizer, model_config, device)
    """
    import yaml

    # 延迟导入，避免循环依赖
    from scripts.train import load_tokenizer

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config = ModelConfig(**config["model"])

    tokenizer = load_tokenizer(config, tokenizer_override)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    if model_path is None:
        for cand in (
            "outputs/dpo/final.pth",
            "outputs/sft/final.pth",
            "outputs/pretrain/final.pth",
        ):
            if os.path.exists(cand):
                model_path = cand
                break
        if model_path is None:
            raise FileNotFoundError(
                "未找到任何 checkpoint，请用 --model 显式指定或先训练"
            )
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    if device is None:
        device = get_device()

    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    return model, tokenizer, model_config, device


# ============================================================
# Loglikelihood scoring（MCQ 基准核心）
# ============================================================


@torch.no_grad()
def score_choices_loglikelihood(
    model: GPT,
    tokenizer,
    context: str,
    choices: Sequence[str],
    device: torch.device,
    max_seq_len: int | None = None,
) -> list[float]:
    """对每个 choice 计算 ``log P(choice | context)``

    用于 C-Eval / CMMLU 等 MCQ 评测：
      - context 是题目（含 5-shot 示例 + 问题 + "Answer: "）
      - choices 是 ["A", "B", "C", "D"]（或更长的选项文本）
      - 返回每个 choice 的 logprob，越大越可能

    实现方式：
      - 拼接 ``context + choice`` → tokenize → forward 一次
      - 取 choice 部分对应位置的 logprob 之和（按 token 平均消除长度偏置）

    Args:
        max_seq_len: 截断长度；None 用 model.config.max_seq_len

    Returns:
        len(choices) 长度的 logprob 列表（已按 choice token 数归一化）
    """
    if max_seq_len is None:
        max_seq_len = model.config.max_seq_len

    ctx_ids = tokenizer.encode(context, add_bos=True, add_eos=False)
    scores: list[float] = []

    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_bos=False, add_eos=False)
        if not choice_ids:
            scores.append(-float("inf"))
            continue

        # 拼接，左侧若超出 max_seq_len，从 context 头部砍（保留题尾 + choice）
        full_ids = ctx_ids + choice_ids
        if len(full_ids) > max_seq_len:
            overflow = len(full_ids) - max_seq_len
            ctx_trim = ctx_ids[overflow:]
            full_ids = ctx_trim + choice_ids

        n_choice = len(choice_ids)
        n_total = len(full_ids)

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        logits, _, _ = model(input_ids)
        # logits[:, t] 预测 input_ids[t+1]
        # 我们要的是预测 choice 部分时的 logprob
        # choice tokens 在 full_ids[n_total - n_choice : n_total]
        # 它们由 logits[n_total - n_choice - 1 : n_total - 1] 预测
        choice_logits = logits[0, n_total - n_choice - 1: n_total - 1, :]
        choice_targets = input_ids[0, n_total - n_choice: n_total]
        log_probs = F.log_softmax(choice_logits.float(), dim=-1)
        chosen_logp = log_probs.gather(-1, choice_targets.unsqueeze(-1)).squeeze(-1)
        # 按 choice token 数归一化（避免长选项被惩罚）
        scores.append(chosen_logp.mean().item())

    return scores


# ============================================================
# 批量生成
# ============================================================


@torch.no_grad()
def batch_generate(
    model: GPT,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device,
    use_chat_template: bool = True,
) -> list[str]:
    """一次跑完一批 prompts（顺序生成，但合并 forward 减少 Python 开销）

    注：当前实现仍是逐 prompt（generate 接口不支持真 batch attention mask），
    但提供了未来切到 padded batch 的占位接口。如果你要真 batch，可基于
    ``F.scaled_dot_product_attention(attn_mask=padding_mask)`` + left-padding 改造。

    返回与 prompts 等长的回复列表。
    """
    from src.inference.generate import generate_text

    replies: list[str] = []
    for i, p in enumerate(prompts):
        try:
            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                reply = generate_text(
                    model=model, tokenizer=tokenizer, prompt=rendered,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=50, top_p=top_p,
                    repetition_penalty=1.1, device=device,
                    add_bos=False, return_only_new=True,
                    skip_special_tokens=True,
                )
            else:
                reply = generate_text(
                    model=model, tokenizer=tokenizer, prompt=p,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=50, top_p=top_p,
                    repetition_penalty=1.1, device=device,
                    add_bos=True, return_only_new=True,
                    skip_special_tokens=True,
                )
        except Exception as e:
            reply = f"[GENERATION_ERROR: {type(e).__name__}: {e}]"
        replies.append(reply)
    return replies


# ============================================================
# 数据加载（jsonl / HF datasets 双路径）
# ============================================================


def load_jsonl(path: str | Path) -> list[dict]:
    """读 jsonl 文件 → list[dict]"""
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def try_load_hf_dataset(
    repo_id: str,
    subset: str | None = None,
    split: str = "test",
) -> list[dict] | None:
    """尝试从 HF datasets 加载；失败返回 None（让上层 fallback 到本地 jsonl）"""
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset(repo_id, subset, split=split, trust_remote_code=True) \
             if subset else load_dataset(repo_id, split=split, trust_remote_code=True)
        return list(ds)
    except Exception as e:
        print(f"⚠️  HF datasets 加载失败 ({repo_id}/{subset}): {e}")
        return None
