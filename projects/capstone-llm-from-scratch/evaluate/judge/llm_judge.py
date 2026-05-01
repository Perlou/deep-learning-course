"""
llm_judge.py — LLM-as-Judge 评分客户端
=========================================

用强 LLM（DeepSeek / GPT-4o / Qwen-72B 等）给 ClearMind 的回复打分，
替代手工 keyword overlap 这种粗糙的指标。

核心设计：
  - **OpenAI 兼容 API**：通过 ``OPENAI_API_BASE`` + ``OPENAI_API_KEY`` 配置任意
    后端（OpenAI / DeepSeek / 阿里云百炼 / 自部署 vLLM 都行）
  - **三选一打分模式**：
      single_rating  — 单 prompt 打 1-10 分（AlignBench 风格）
      pairwise       — A/B 比较（MT-Bench 风格）
      reference      — 与参考答案对比打分
  - **重试 + JSON 解析容错**：API 偶发 5xx 自动重试 3 次；判官回复非标准 JSON 时
    用 regex 兜底抽分
  - **缓存**：相同 (prompt, judge_model) → 命中本地 cache 文件，省钱

环境变量:
  OPENAI_API_BASE   API 端点（默认 https://api.openai.com/v1）
  OPENAI_API_KEY    API key（必需）
  CLEARMIND_JUDGE_MODEL   judge 模型名（默认 ``deepseek-chat``）

注意：本模块不在训练时使用，仅 evaluate/benchmarks/alignbench.py 等离线评测调用。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "judge"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class JudgeConfig:
    api_base: str = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    model: str = os.environ.get("CLEARMIND_JUDGE_MODEL", "deepseek-chat")
    max_retries: int = 3
    timeout: int = 60
    temperature: float = 0.0  # judge 用 0 稳一点

    def assert_ready(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "LLM-Judge 需要 OPENAI_API_KEY 环境变量。\n"
                "  export OPENAI_API_KEY=sk-xxx\n"
                "  export OPENAI_API_BASE=https://api.deepseek.com/v1   # 可选\n"
                "  export CLEARMIND_JUDGE_MODEL=deepseek-chat            # 可选"
            )


# ============================================================
# OpenAI 兼容 chat completions（不依赖 openai SDK，用 httpx 直发）
# ============================================================


def _http_chat_completion(cfg: JudgeConfig, messages: list[dict]) -> str:
    """发起一次 chat completion 请求，返回 message.content"""
    try:
        import httpx
    except ImportError:
        try:
            import urllib.request, urllib.error
            return _urllib_chat_completion(cfg, messages)
        except Exception as e:
            raise RuntimeError("LLM-Judge 需要 httpx 或 urllib") from e

    url = cfg.api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
    }
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            with httpx.Client(timeout=cfg.timeout) as client:
                r = client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries - 1:
                wait = 2 ** attempt
                print(f"  ⚠️  Judge API 失败 (attempt {attempt + 1}/{cfg.max_retries}): "
                      f"{type(e).__name__}: {e}；{wait}s 后重试")
                time.sleep(wait)
    raise RuntimeError(f"Judge API 多次失败: {last_err}")


def _urllib_chat_completion(cfg: JudgeConfig, messages: list[dict]) -> str:
    """httpx 不可用时的 fallback（urllib 版）"""
    import urllib.request
    url = cfg.api_base.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": cfg.model, "messages": messages,
        "temperature": cfg.temperature,
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]


# ============================================================
# Cache（避免重复花钱）
# ============================================================


def _cache_key(payload: dict) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    p = CACHE_DIR / f"{key}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def _cache_put(key: str, value: str) -> None:
    (CACHE_DIR / f"{key}.txt").write_text(value, encoding="utf-8")


# ============================================================
# 评分接口
# ============================================================


SINGLE_RATING_PROMPT = """你是一个严格的中文 AI 评测员。请根据以下用户问题和模型回复，给模型打分。

评分维度（每项 1-10 分）：
1. 准确性：回答内容是否正确、无幻觉
2. 相关性：是否切题、回答了用户的问题
3. 完整性：是否充分、有必要的细节
4. 流畅性：语言是否通顺、有无明显语法错误

【用户问题】
{question}

【模型回复】
{answer}

请严格按以下 JSON 格式输出（不要任何额外文字）：
{{"accuracy": <1-10>, "relevance": <1-10>, "completeness": <1-10>, "fluency": <1-10>, "overall": <1-10>, "reason": "<一句话说明>"}}
"""


def judge_single(
    question: str,
    answer: str,
    *,
    cfg: JudgeConfig | None = None,
    use_cache: bool = True,
) -> dict:
    """单次打分：返回 ``{accuracy, relevance, completeness, fluency, overall, reason}``"""
    cfg = cfg or JudgeConfig()
    cfg.assert_ready()

    payload = {
        "type": "single_rating", "model": cfg.model,
        "question": question, "answer": answer,
    }
    key = _cache_key(payload)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass  # 缓存脏了，重打

    prompt = SINGLE_RATING_PROMPT.format(question=question, answer=answer)
    content = _http_chat_completion(cfg, [{"role": "user", "content": prompt}])
    parsed = _parse_json_or_extract(content)

    if use_cache and parsed:
        _cache_put(key, json.dumps(parsed, ensure_ascii=False))
    return parsed


def _parse_json_or_extract(text: str) -> dict:
    """从 judge 回复抽 JSON。优先严格解析，失败时 regex 兜底"""
    # 优先：直接 json.loads
    text = text.strip()
    # 去掉常见的 markdown code fence
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底：抓最后一个 {} 块
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 再兜底：regex 抓 overall 分
    overall = 0
    m2 = re.search(r"overall[\"']?\s*[:：]\s*(\d+)", text, re.IGNORECASE)
    if m2:
        overall = int(m2.group(1))
    return {
        "accuracy": 0, "relevance": 0, "completeness": 0, "fluency": 0,
        "overall": overall,
        "reason": f"JSON 解析失败，原始回复: {text[:200]}",
    }


# ============================================================
# CLI（手动跑单条 quick test）
# ============================================================


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="LLM-Judge 单条测试")
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    args = parser.parse_args()

    cfg = JudgeConfig()
    print(f"Judge: {cfg.model} @ {cfg.api_base}")
    result = judge_single(args.question, args.answer, cfg=cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
