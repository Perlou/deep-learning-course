"""
DPODataset — 偏好对齐数据集
============================

处理 Direct Preference Optimization 训练数据。
每个样本包含一个 prompt 和两个回复:
  - chosen:   人类偏好的高质量回复
  - rejected: 人类不偏好的低质量回复

DPO 训练目标:
  让模型学会: P(chosen | prompt) > P(rejected | prompt)

数据格式:
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):
    """DPO 偏好对齐数据集

    Args:
        data_path:   数据文件路径 (.jsonl 或 .json)
        tokenizer:   分词器实例
        max_seq_len: 最大序列长度
        _samples:    内部使用 — 直接传入样本列表 (跳过文件加载)
    """

    def __init__(
        self,
        data_path: str = None,
        tokenizer=None,
        max_seq_len: int = 512,
        _samples: list[dict] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 加载数据
        if _samples is not None:
            self.samples = _samples
        else:
            print(f"📦 加载 DPO 数据: {data_path}")
            self.samples = self._load_data(data_path)
        print(f"  样本数: {len(self.samples):,}")

    @classmethod
    def create_with_split(
        cls,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple["DPODataset", "DPODataset"]:
        """创建训练集和验证集

        加载全部样本后 shuffle → 按索引划分。

        Returns:
            (train_dataset, val_dataset)
        """
        print(f"📦 加载 DPO 数据: {data_path}")
        temp = cls(data_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
        all_samples = temp.samples.copy()

        rng = random.Random(seed)
        rng.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1 - val_ratio))
        print(f"📊 数据划分: 训练 {split_idx} 样本, 验证 {len(all_samples) - split_idx} 样本")

        train_ds = cls(_samples=all_samples[:split_idx], tokenizer=tokenizer, max_seq_len=max_seq_len)
        val_ds = cls(_samples=all_samples[split_idx:], tokenizer=tokenizer, max_seq_len=max_seq_len)

        return train_ds, val_ds

    def _load_data(self, data_path: str) -> list[dict]:
        """加载数据文件"""
        ext = os.path.splitext(data_path)[1]

        if ext == ".jsonl":
            samples = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            return samples
        elif ext == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _tokenize_pair(
        self,
        prompt: str,
        response: str,
    ) -> tuple[list[int], list[int]]:
        """将 prompt + response 编码并构建 labels

        Returns:
            (input_ids, labels) - 其中 prompt 部分的 labels 为 -100
        """
        prompt_text = f"Human: {prompt}\nAssistant: "
        full_text = f"Human: {prompt}\nAssistant: {response}"

        prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        full_ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)

        # 截断
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]
            full_ids[-1] = self.tokenizer.eos_id

        # Labels: prompt 部分为 -100
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = [-100] * prompt_len

        # Padding
        pad_len = self.max_seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return full_ids, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """获取一个 DPO 训练样本

        Returns:
            dict with:
              chosen_input_ids:   [seq_len] chosen 回复的输入
              chosen_labels:      [seq_len] chosen 回复的标签
              rejected_input_ids: [seq_len] rejected 回复的输入
              rejected_labels:    [seq_len] rejected 回复的标签
        """
        sample = self.samples[idx]

        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_ids, chosen_labels = self._tokenize_pair(prompt, chosen)
        rejected_ids, rejected_labels = self._tokenize_pair(prompt, rejected)

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }
