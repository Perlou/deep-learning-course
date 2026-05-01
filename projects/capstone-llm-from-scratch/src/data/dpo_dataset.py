"""
DPODataset — 偏好对齐数据集
============================

支持两种数据格式：

1. **conversations 列表（minimind 风格，默认）**：
   ``{"chosen": [{"role": ..., "content": ...}, ...],
       "rejected": [{"role": ..., "content": ...}, ...]}``
   chosen / rejected 是完整的多轮对话历史（含 user 提问和 assistant 回复）。
   用 ``apply_chat_template`` 渲染后扫描 ``<|im_start|>assistant\\n`` 与 ``<|im_end|>\\n``
   定位 assistant 段生成 loss mask。

2. **字符串字段（向后兼容）**：
   ``{"prompt": "...", "chosen": "...", "rejected": "..."}``
   自动构造 ``conversations = [{"role":"user","content":prompt},
   {"role":"assistant","content":chosen|rejected}]``。

输出格式与 ClearMind :class:`DPOTrainer` 兼容：
  ``chosen_input_ids`` / ``chosen_labels`` / ``rejected_input_ids`` / ``rejected_labels``，
其中 labels 在 prompt/system/user/tool/padding 区域为 -100，仅 assistant 段保留 token id。
"""

from __future__ import annotations

import json
import os
import random
from typing import Sequence

import torch
from torch.utils.data import Dataset


def _normalize_message(msg: dict) -> dict:
    """与 SFTDataset 共享的消息规范化"""
    if "role" in msg:
        return {
            "role": msg["role"],
            "content": msg.get("content", ""),
            **{
                k: v
                for k, v in msg.items()
                if k in ("reasoning_content", "tools", "tool_calls") and v is not None
            },
        }
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
        "observation": "tool",
    }
    return {
        "role": role_map.get(msg.get("from", "user"), "user"),
        "content": msg.get("value", ""),
    }


def _to_conversations(value, prompt: str | None = None) -> list[dict]:
    """把 chosen/rejected 字段转为 conversations 列表"""
    # 列表形式（minimind 风格）：直接规范化
    if isinstance(value, list):
        return [_normalize_message(m) for m in value]
    # 字符串形式：拼接 prompt + assistant 回复
    if isinstance(value, str):
        convs: list[dict] = []
        if prompt:
            convs.append({"role": "user", "content": prompt})
        convs.append({"role": "assistant", "content": value})
        return convs
    raise ValueError(f"无法识别 chosen/rejected 字段类型: {type(value).__name__}")


class DPODataset(Dataset):
    """DPO 偏好对齐数据集

    Args:
        data_path:    ``.jsonl`` 或 ``.json`` 数据文件路径
        tokenizer:    :class:`HFTokenizer` 实例
        max_seq_len:  最大序列长度
        seed:         shuffle / 增强随机种子
        _samples:     内部使用——直接传入样本列表
    """

    def __init__(
        self,
        data_path: str | None = None,
        tokenizer=None,
        max_seq_len: int = 1024,
        seed: int | None = None,
        _samples: list[dict] | None = None,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("DPODataset 需要 tokenizer（HFTokenizer 实例）")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._rng = random.Random(seed)

        if _samples is not None:
            self.samples: list[dict] = list(_samples)
        else:
            print(f"📦 加载 DPO 数据: {data_path}")
            self.samples = self._load_data(data_path)
        print(f"  样本数: {len(self.samples):,}")

    @staticmethod
    def _load_data(data_path: str) -> list[dict]:
        ext = os.path.splitext(data_path)[1]
        if ext == ".jsonl":
            out: list[dict] = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
            return out
        if ext == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"不支持的文件格式: {ext}")

    @classmethod
    def create_with_split(
        cls,
        data_path: str,
        tokenizer,
        max_seq_len: int = 1024,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> tuple["DPODataset", "DPODataset"]:
        with open(data_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        rng = random.Random(seed)
        rng.shuffle(samples)
        split = int(len(samples) * (1 - val_ratio))
        print(f"📊 DPO 数据划分: 训练 {split:,} / 验证 {len(samples) - split:,}")
        train = cls(
            _samples=samples[:split],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            seed=seed,
        )
        val = cls(
            _samples=samples[split:],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            seed=seed + 1,
        )
        return train, val

    # ----------------------------------------------------------
    # 渲染单侧（chosen 或 rejected）
    # ----------------------------------------------------------

    def _encode_side(self, convs: list[dict]) -> tuple[list[int], list[int]]:
        """渲染 + tokenize + 生成 assistant loss mask + padding

        Returns:
            (input_ids, labels)，长度均为 ``max_seq_len``，labels 中非 assistant 区域为 -100
        """
        text = self.tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=False,
            tokenize=False,
        )
        ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)

        # 截断
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]

        mask = self.tokenizer.generate_assistant_mask(ids)

        # Padding
        pad_id = self.tokenizer.pad_id
        pad_len = self.max_seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            mask = mask + [0] * pad_len

        labels = list(ids)
        for i, m in enumerate(mask):
            if m == 0:
                labels[i] = -100
        return ids, labels

    # ----------------------------------------------------------
    # Dataset 接口
    # ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt = sample.get("prompt")  # 字符串模式下需要

        chosen_convs = _to_conversations(sample["chosen"], prompt)
        rejected_convs = _to_conversations(sample["rejected"], prompt)

        chosen_ids, chosen_labels = self._encode_side(chosen_convs)
        rejected_ids, rejected_labels = self._encode_side(rejected_convs)

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }
