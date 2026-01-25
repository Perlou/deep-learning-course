"""
数据集处理
==========

包含指令数据集的加载、预处理和格式化。

学习要点：
    1. ChatML 对话模板格式
    2. 指令数据的 tokenization
    3. 标签掩码（只计算助手回复的损失）
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig


# ==================== 简单分词器 ====================


class SimpleTokenizer:
    """简单的字符级分词器（用于演示）

    注意：实际应用中应使用 BPE/SentencePiece 等专业分词器
    """

    def __init__(self, vocab_size: int = 32000):
        # 特殊token
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"

        # ChatML 特殊标记
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"

        # 构建基础词表
        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.im_start,
            self.im_end,
            "system",
            "user",
            "assistant",
            "\n",
        ]

        # 添加常见字符
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")

        # 添加中文常用字（简化版）
        chinese_chars = list(
            "的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事"
        )
        chinese_chars += list(
            "成方多经么当转机工与好看学进将还如已正新想前但因理问意此其题间高本次什体从现开者无情已点面但问"
        )

        all_tokens = self.special_tokens + chars + chinese_chars

        # 限制词表大小
        all_tokens = all_tokens[:vocab_size]

        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # 特殊token ID
        self.pad_token_id = self.vocab[self.pad_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.unk_token_id = self.vocab[self.unk_token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本为token ID列表"""
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        # 处理特殊标记
        text = text.replace(self.im_start, f" {self.im_start} ")
        text = text.replace(self.im_end, f" {self.im_end} ")

        # 逐字符编码
        i = 0
        while i < len(text):
            # 检查是否是特殊token
            matched = False
            for special in self.special_tokens:
                if text[i:].startswith(special):
                    if special.strip():  # 非空白
                        tokens.append(self.vocab.get(special, self.unk_token_id))
                    i += len(special)
                    matched = True
                    break

            if not matched:
                char = text[i]
                tokens.append(self.vocab.get(char, self.unk_token_id))
                i += 1

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码token ID列表为文本"""
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, self.unk_token)
            if skip_special_tokens and token in [
                self.pad_token,
                self.bos_token,
                self.eos_token,
            ]:
                continue
            tokens.append(token)
        return "".join(tokens)


# ==================== ChatML 模板 ====================


def apply_chatml_template(
    messages: List[Dict[str, str]], data_config: DataConfig
) -> str:
    """应用 ChatML 对话模板

    Args:
        messages: 对话消息列表，每条消息包含 role 和 content
        data_config: 数据配置

    Returns:
        格式化后的对话字符串

    示例输出:
        <|im_start|>system
        你是AI助手<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！有什么可以帮助你的吗？<|im_end|>
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"{data_config.im_start}{role}\n{content}{data_config.im_end}\n"
    return formatted.strip()


def parse_chatml(text: str, data_config: DataConfig) -> List[Dict[str, str]]:
    """解析 ChatML 格式文本为消息列表"""
    messages = []
    parts = text.split(data_config.im_start)

    for part in parts:
        if not part.strip():
            continue
        if data_config.im_end in part:
            content = part.split(data_config.im_end)[0]
            lines = content.strip().split("\n", 1)
            if len(lines) >= 2:
                role = lines[0].strip()
                msg_content = lines[1].strip()
                messages.append({"role": role, "content": msg_content})

    return messages


# ==================== 指令数据集 ====================


# 内置示例数据
SAMPLE_DATA = [
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "什么是机器学习？"},
            {
                "role": "assistant",
                "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需被明确编程。",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "Python是什么？"},
            {
                "role": "assistant",
                "content": "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "深度学习和机器学习有什么区别？"},
            {
                "role": "assistant",
                "content": "深度学习是机器学习的子集，使用多层神经网络来学习数据的复杂表示。",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "什么是神经网络？"},
            {
                "role": "assistant",
                "content": "神经网络是一种受大脑结构启发的计算模型，由相互连接的节点层组成。",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "解释一下什么是GPT。"},
            {
                "role": "assistant",
                "content": "GPT是生成式预训练变换器，是一种基于Transformer架构的大型语言模型。",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "什么是LoRA？"},
            {
                "role": "assistant",
                "content": "LoRA是一种高效的模型微调方法，通过低秩分解来减少可训练参数的数量。",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "什么是Transformer？"},
            {
                "role": "assistant",
                "content": "Transformer是一种神经网络架构，使用自注意力机制来处理序列数据。",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "什么是注意力机制？"},
            {
                "role": "assistant",
                "content": "注意力机制让模型能够关注输入中最相关的部分，提高处理效率和效果。",
            },
        ]
    },
]


class InstructionDataset(Dataset):
    """指令微调数据集

    特点：
        1. 支持 ChatML 格式
        2. 只对助手回复计算损失
        3. 自动填充和截断
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: SimpleTokenizer,
        data_config: DataConfig,
        max_length: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        messages = item["messages"]

        # 应用ChatML模板
        text = apply_chatml_template(messages, self.data_config)

        # 编码
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        # 创建标签（用于语言模型训练）
        labels = input_ids.copy()

        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)

        # 填充
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length  # -100 是 CrossEntropy 的忽略索引
            attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def create_dataloaders(
    data: Optional[List[Dict]] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
    data_config: Optional[DataConfig] = None,
    batch_size: int = 4,
    max_length: int = 256,
    test_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器

    Args:
        data: 数据列表，如果为None则使用内置示例
        tokenizer: 分词器
        data_config: 数据配置
        batch_size: 批次大小
        max_length: 最大序列长度
        test_split: 测试集比例

    Returns:
        训练和验证的DataLoader
    """
    from config import get_config

    if data is None:
        data = SAMPLE_DATA

    if data_config is None:
        data_config = get_config().data

    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    # 划分数据集
    split_idx = int(len(data) * (1 - test_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # 确保验证集不为空
    if len(val_data) == 0:
        val_data = train_data[:1]

    train_dataset = InstructionDataset(train_data, tokenizer, data_config, max_length)
    val_dataset = InstructionDataset(val_data, tokenizer, data_config, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    tokenizer = SimpleTokenizer(config.model.vocab_size)

    print("=" * 60)
    print("数据集演示")
    print("=" * 60)

    # 测试 ChatML 模板
    sample = SAMPLE_DATA[0]
    formatted = apply_chatml_template(sample["messages"], config.data)
    print(f"\n原始消息:\n{json.dumps(sample, ensure_ascii=False, indent=2)}")
    print(f"\nChatML 格式:\n{formatted}")

    # 测试分词
    tokens = tokenizer.encode(formatted)
    print(f"\nToken IDs (前20个): {tokens[:20]}...")
    print(f"Token 数量: {len(tokens)}")

    # 测试数据加载器
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer, data_config=config.data, batch_size=2
    )

    batch = next(iter(train_loader))
    print(f"\n批次形状:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
