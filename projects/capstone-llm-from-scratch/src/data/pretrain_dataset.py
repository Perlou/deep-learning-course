"""
PretrainDataset — 预训练数据集
==============================

支持两种数据准备策略：

1. **packed**（默认，与原 ClearMind 一致）：
   将所有文档 tokenize 后拼接成一个连续 token 流，再切分为固定长度的 (max_seq_len+1) 块。
   优点：无 padding 浪费，最大化吞吐；缺点：跨文档边界 attention 不重置。
   适合大规模 web 文本（trillion token 级）。

2. **per_sample**（与 minimind 对齐）：
   每条 jsonl 样本独立 ``BOS + tokens + EOS``，再 pad 到 ``max_seq_len``，pad 处
   ``labels=-100``。优点：保留文档边界、与 minimind 训练流程完全一致；缺点：浪费
   padding 计算量。适合中小规模、多样本短文本数据集（如 minimind 的 pretrain_t2t）。

数据格式（两种模式都支持）：
  - ``.jsonl``：每行 ``{"text": "..."}``（minimind 风格，默认 per_sample 模式）
  - ``.jsonl``：每行 ``{"text": "..."}`` 或 ``.txt`` 段落分隔（旧格式，packed 模式）
  - ``.bin``：预先 tokenize 好的 ``uint16/int32`` 二进制流（仅 packed 模式）
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PretrainDataset(Dataset):
    """预训练数据集

    Args:
        data_path:    数据文件路径（.bin / .jsonl / .txt）
        tokenizer:    分词器实例（需暴露 ``encode`` / ``bos_id`` / ``eos_id`` / ``pad_id``）
        max_seq_len:  最大序列长度
        mode:         ``"packed"`` 或 ``"per_sample"``。默认根据文件扩展名与字段自动选择
        _tokens:      内部使用——直接传入已分词的 token 列表（跳过文件加载，仅 packed 模式）
        _samples:     内部使用——直接传入预处理好的样本列表（仅 per_sample 模式）
    """

    def __init__(
        self,
        data_path: str | None = None,
        tokenizer=None,
        max_seq_len: int = 512,
        mode: str = "packed",
        _tokens: list[int] | np.ndarray | None = None,
        _samples: list[dict] | None = None,
    ):
        super().__init__()
        if mode not in ("packed", "per_sample"):
            raise ValueError(f"未知 mode: {mode!r}，应为 'packed' 或 'per_sample'")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode

        if mode == "packed":
            self._init_packed(data_path, _tokens)
        else:
            self._init_per_sample(data_path, _samples)

    # ----------------------------------------------------------
    # packed 模式：拼接所有 token
    # ----------------------------------------------------------

    def _init_packed(
        self,
        data_path: str | None,
        _tokens: list[int] | np.ndarray | None,
    ) -> None:
        if _tokens is not None:
            data = (
                _tokens
                if isinstance(_tokens, np.ndarray)
                else np.array(_tokens, dtype=np.int32)
            )
        else:
            print(f"📦 加载预训练数据 (packed): {data_path}")
            data = self._load_token_stream(data_path)

        # 每条样本占 (max_seq_len + 1) 个 token：前 max_seq_len 作 input，后 max_seq_len 作 target
        self.n_samples = len(data) // (self.max_seq_len + 1)
        self.data = data[: self.n_samples * (self.max_seq_len + 1)]

        print(f"  总 token 数: {len(self.data):,}")
        print(f"  样本数: {self.n_samples:,} (packed)")

    def _load_token_stream(self, data_path: str) -> np.ndarray:
        """加载并 tokenize 数据，返回 int32 token 流"""
        ext = os.path.splitext(data_path)[1]

        if ext == ".bin":
            data = np.fromfile(data_path, dtype=np.uint16)
            return data.astype(np.int32)

        all_tokens: list[int] = []

        if ext == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in tqdm(lines, desc="Tokenizing"):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text:
                    continue
                # BOS + 内容 + EOS（packed 模式下文档之间通过 EOS 分隔）
                tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                all_tokens.extend(tokens)
        elif ext == ".txt":
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            for para in tqdm(text.split("\n\n"), desc="Tokenizing"):
                para = para.strip()
                if not para:
                    continue
                tokens = self.tokenizer.encode(para, add_bos=True, add_eos=True)
                all_tokens.extend(tokens)
        else:
            raise ValueError(f"不支持的文件格式: {ext}（packed 模式仅支持 .bin/.jsonl/.txt）")

        return np.array(all_tokens, dtype=np.int32)

    # ----------------------------------------------------------
    # per_sample 模式：每条样本独立 padding（与 minimind 对齐）
    # ----------------------------------------------------------

    def _init_per_sample(
        self,
        data_path: str | None,
        _samples: list[dict] | None,
    ) -> None:
        if _samples is not None:
            self.samples: list[dict] = list(_samples)
        else:
            print(f"📦 加载预训练数据 (per_sample): {data_path}")
            self.samples = self._load_jsonl_samples(data_path)
        print(f"  样本数: {len(self.samples):,} (per_sample)")

    @staticmethod
    def _load_jsonl_samples(data_path: str) -> list[dict]:
        if not data_path.endswith(".jsonl"):
            raise ValueError(
                f"per_sample 模式仅支持 .jsonl，收到 {data_path!r}"
            )
        out: list[dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    # ----------------------------------------------------------
    # split 工厂方法
    # ----------------------------------------------------------

    @classmethod
    def create_with_split(
        cls,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        val_ratio: float = 0.05,
        mode: str = "per_sample",
    ) -> tuple["PretrainDataset", "PretrainDataset"]:
        """创建训练集与验证集

        per_sample 模式下按样本数切分；packed 模式下按 token 数切分。

        Returns:
            (train_dataset, val_dataset)
        """
        if mode == "per_sample":
            samples = cls._load_jsonl_samples(data_path)
            split_idx = int(len(samples) * (1 - val_ratio))
            print(
                f"📊 数据划分: 训练 {split_idx:,} 样本, 验证 "
                f"{len(samples) - split_idx:,} 样本 (per_sample)"
            )
            train = cls(
                _samples=samples[:split_idx],
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                mode="per_sample",
            )
            val = cls(
                _samples=samples[split_idx:],
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                mode="per_sample",
            )
            return train, val

        # packed
        print(f"📦 加载预训练数据 (packed split): {data_path}")
        temp = cls(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_len=1,
            mode="packed",
        )
        all_tokens = temp.data
        split_idx = int(len(all_tokens) * (1 - val_ratio))
        print(
            f"📊 数据划分: 训练 {split_idx:,} tokens, 验证 "
            f"{len(all_tokens) - split_idx:,} tokens (packed)"
        )
        train = cls(
            _tokens=all_tokens[:split_idx],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode="packed",
        )
        val = cls(
            _tokens=all_tokens[split_idx:],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode="packed",
        )
        return train, val

    # ----------------------------------------------------------
    # Dataset 接口
    # ----------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples if self.mode == "packed" else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "packed":
            return self._get_packed(idx)
        return self._get_per_sample(idx)

    def _get_packed(self, idx: int) -> dict:
        start = idx * (self.max_seq_len + 1)
        chunk = self.data[start : start + self.max_seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    def _get_per_sample(self, idx: int) -> dict:
        """与 minimind PretrainDataset 对齐的样本生成

        每条 jsonl 样本：
          tokens = BOS + tokenize(text)[: max_seq_len-2] + EOS
          input_ids = pad(tokens, pad_id, max_seq_len)
          labels = input_ids，pad 位置改为 -100

        训练时模型预测 input_ids[i+1]，labels 的 -100 区域被 cross_entropy 忽略。
        """
        sample = self.samples[idx]
        text = str(sample.get("text", ""))
        body = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        # 截断到 max_seq_len - 2，给 BOS / EOS 留位置
        body = body[: max(0, self.max_seq_len - 2)]
        tokens = [self.tokenizer.bos_id] + body + [self.tokenizer.eos_id]

        pad_id = self.tokenizer.pad_id
        pad_len = max(0, self.max_seq_len - len(tokens))
        if pad_len > 0:
            tokens = tokens + [pad_id] * pad_len
        else:
            tokens = tokens[: self.max_seq_len]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        # pad 位置不计算 loss
        labels[input_ids == pad_id] = -100
        return {"input_ids": input_ids, "labels": labels}


def save_pretrain_bin(
    input_path: str,
    output_path: str,
    tokenizer,
    dtype: str | None = None,
) -> None:
    """将文本数据预 tokenize 并保存为二进制（加速后续加载，仅 packed 模式使用）

    Args:
        input_path:  输入 .jsonl 或 .txt
        output_path: 输出 .bin
        tokenizer:   分词器
        dtype:       存储 dtype，``None`` 时按 vocab_size 自动选择 ``uint16`` (≤65535)
                     或 ``uint32`` (>65535)
    """
    dataset = PretrainDataset(
        data_path=input_path,
        tokenizer=tokenizer,
        max_seq_len=1,
        mode="packed",
    )
    tokens = dataset.data

    if dtype is None:
        vocab_size = getattr(tokenizer, "vocab_size", 0)
        dtype = "uint16" if vocab_size and vocab_size <= 65535 else "uint32"
    arr = np.array(tokens, dtype=np.dtype(dtype))
    arr.tofile(output_path)
    print(f"✅ 保存二进制数据: {output_path}")
    print(f"   Token 数: {len(tokens):,}")
    print(f"   dtype: {dtype}")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
