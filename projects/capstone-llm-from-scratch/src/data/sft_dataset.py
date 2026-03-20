"""
SFTDataset — 指令微调数据集
===========================

将指令-回复对话数据格式化为模型输入，用于 Supervised Fine-Tuning。

核心设计要点:
  1. 对话模板: 将原始指令/回复格式化为统一的对话格式
  2. Loss Mask: 只对 Assistant 回复部分计算 loss，
     忽略 Human 提问部分 (避免模型学习"提问"而不是"回答")

对话模板:
  <s>Human: {instruction}
  Assistant: {response}</s>

Loss Mask 示例:
  tokens:  <s> Human : ... \n Assistant : 回 答 内 容 </s>
  mask:     0   0    0  0  0    0       0   1  1  1  1   1
  labels:  -100 ...                    ... 答 内 容 </s> ...
  (mask=0 的位置 label 设为 -100, cross_entropy 会忽略)

数据格式 (支持):
  1. Alpaca 格式: {"instruction": "...", "input": "...", "output": "..."}
  2. ShareGPT 格式: {"conversations": [{"from": "human", "value": "..."}, ...]}
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SFTDataset(Dataset):
    """SFT 指令微调数据集

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
            print(f"📦 加载 SFT 数据: {data_path}")
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
    ) -> tuple["SFTDataset", "SFTDataset"]:
        """创建训练集和验证集

        加载全部样本后 shuffle → 按索引划分。

        Returns:
            (train_dataset, val_dataset)
        """
        print(f"📦 加载 SFT 数据: {data_path}")
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

    def _format_conversation(self, sample: dict) -> tuple[str, str]:
        """将数据样本格式化为 (prompt, response) 对

        支持 Alpaca 和 ShareGPT 格式。

        Returns:
            (prompt_text, response_text) 元组
        """
        # Alpaca 格式
        if "instruction" in sample:
            instruction = sample["instruction"]
            inp = sample.get("input", "")
            output = sample.get("output", "")

            if inp:
                prompt = f"{instruction}\n{inp}"
            else:
                prompt = instruction

            return prompt, output

        # ShareGPT 格式
        if "conversations" in sample:
            convs = sample["conversations"]
            prompt = ""
            response = ""
            for conv in convs:
                role = conv.get("from", conv.get("role", ""))
                value = conv.get("value", conv.get("content", ""))
                if role in ("human", "user"):
                    prompt = value
                elif role in ("gpt", "assistant"):
                    response = value
            return prompt, response

        raise ValueError(f"无法识别的数据格式: {list(sample.keys())}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """获取一个 SFT 训练样本

        Returns:
            dict with:
              input_ids: [seq_len] 输入 token
              labels:    [seq_len] 目标 token (Human 部分为 -100)
        """
        sample = self.samples[idx]
        prompt, response = self._format_conversation(sample)

        # 构建对话文本
        # 格式: <s>Human: {prompt}\nAssistant: {response}</s>
        prompt_text = f"Human: {prompt}\nAssistant: "
        full_text = f"Human: {prompt}\nAssistant: {response}"

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        full_ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)

        # 截断到 max_seq_len
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]
            # 确保以 EOS 结尾
            full_ids[-1] = self.tokenizer.eos_id

        # 构建 labels
        # prompt 部分 → -100 (不计算 loss)
        # response 部分 → 实际 token id
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = [-100] * prompt_len

        # Padding
        pad_len = self.max_seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
