"""
PretrainDataset — 预训练数据集
==============================

将大量文本数据处理为固定长度的 token 序列，用于 next-token prediction 预训练。

数据处理流程:
  1. 加载所有文本文件
  2. 使用 BPE 分词器 tokenize 每篇文本
  3. 将所有 token 拼接为一个超长序列
  4. 切分为固定长度 (max_seq_len + 1) 的块
     - 前 max_seq_len 个 token 作为 input
     - 后 max_seq_len 个 token 作为 target (右移一位)

示例:
  原始序列: [a, b, c, d, e, f, g, h, ...]
  max_seq_len = 4:
    sample 0: input=[a,b,c,d], target=[b,c,d,e]
    sample 1: input=[e,f,g,h], target=[f,g,h,i]
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PretrainDataset(Dataset):
    """预训练数据集

    将 tokenized 文本切分为固定长度的序列。

    Args:
        data_path:   预处理后的数据文件 (.bin 或 .jsonl)
        tokenizer:   分词器实例
        max_seq_len: 最大序列长度
        _tokens:     内部使用 — 直接传入已分词的 token 列表 (跳过文件加载)
    """

    def __init__(
        self,
        data_path: str = None,
        tokenizer=None,
        max_seq_len: int = 512,
        _tokens: list[int] = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        # 加载并处理数据
        if _tokens is not None:
            self.data = np.array(_tokens, dtype=np.int32) if not isinstance(_tokens, np.ndarray) else _tokens
        else:
            print(f"📦 加载预训练数据: {data_path}")
            self.data = self._load_and_tokenize(data_path)

        # 计算样本数
        # 每个样本需要 max_seq_len + 1 个 token (input + 最后一个 target)
        self.n_samples = len(self.data) // (max_seq_len + 1)
        self.data = self.data[: self.n_samples * (max_seq_len + 1)]

        print(f"  总 token 数: {len(self.data):,}")
        print(f"  样本数: {self.n_samples:,}")

    @classmethod
    def create_with_split(
        cls,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        val_ratio: float = 0.1,
    ) -> tuple["PretrainDataset", "PretrainDataset"]:
        """创建训练集和验证集

        在 token 流 concat 后按比例截取（前 90% 训练，后 10% 验证）。

        Returns:
            (train_dataset, val_dataset)
        """
        print(f"📦 加载预训练数据: {data_path}")
        temp = cls(data_path=data_path, tokenizer=tokenizer, max_seq_len=1)
        all_tokens = temp.data

        split_idx = int(len(all_tokens) * (1 - val_ratio))

        print(f"📊 数据划分: 训练 {split_idx:,} tokens, 验证 {len(all_tokens) - split_idx:,} tokens")
        train_ds = cls(_tokens=all_tokens[:split_idx], tokenizer=tokenizer, max_seq_len=max_seq_len)
        val_ds = cls(_tokens=all_tokens[split_idx:], tokenizer=tokenizer, max_seq_len=max_seq_len)

        return train_ds, val_ds

    def _load_and_tokenize(self, data_path: str) -> np.ndarray:
        """加载数据文件并 tokenize

        支持格式:
          - .bin: 预先 tokenize 好的二进制文件
          - .jsonl: 每行一个 JSON, 有 "text" 字段
          - .txt: 纯文本文件
        """
        ext = os.path.splitext(data_path)[1]

        if ext == ".bin":
            # 直接加载预处理好的 token 序列
            data = np.fromfile(data_path, dtype=np.uint16)
            return data.astype(np.int32)

        # 从文本文件加载并 tokenize
        all_tokens = []

        if ext == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in tqdm(lines, desc="Tokenizing"):
                obj = json.loads(line.strip())
                text = obj.get("text", "")
                if text:
                    tokens = self.tokenizer.encode(text, add_bos=False, add_eos=True)
                    all_tokens.extend(tokens)
        elif ext == ".txt":
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            # 按段落切分
            paragraphs = text.split("\n\n")
            for para in tqdm(paragraphs, desc="Tokenizing"):
                para = para.strip()
                if para:
                    tokens = self.tokenizer.encode(para, add_bos=False, add_eos=True)
                    all_tokens.extend(tokens)
        else:
            raise ValueError(f"不支持的文件格式: {ext}, 请使用 .bin, .jsonl 或 .txt")

        return np.array(all_tokens, dtype=np.int32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """获取一个训练样本

        Returns:
            dict with:
              input_ids: [max_seq_len] 输入 token
              labels:    [max_seq_len] 目标 token (右移一位)
        """
        start = idx * (self.max_seq_len + 1)
        chunk = self.data[start : start + self.max_seq_len + 1]

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def save_pretrain_bin(
    input_path: str,
    output_path: str,
    tokenizer,
):
    """将文本数据预处理并保存为二进制格式 (加速后续加载)

    Args:
        input_path:  输入文件路径 (.jsonl 或 .txt)
        output_path: 输出文件路径 (.bin)
        tokenizer:   分词器实例
    """
    import numpy as np

    dataset = PretrainDataset(input_path, tokenizer, max_seq_len=1)  # 仅用于 tokenize
    tokens = dataset.data

    # 保存为 uint16 (节省空间, 支持 vocab_size ≤ 65535)
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"✅ 保存二进制数据: {output_path}")
    print(f"   Token 数: {len(tokens):,}")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
