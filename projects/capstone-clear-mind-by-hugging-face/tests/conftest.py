"""共享测试 fixtures"""

import json
import sys
from pathlib import Path

import torch
import pytest

# 将 src/ 加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import ClearMindConfig, ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer


@pytest.fixture
def tiny_config():
    """Tiny 配置 fixture"""
    return ClearMindConfig.tiny()


@pytest.fixture
def tiny_model(tiny_config):
    """Tiny 模型 fixture (eval 模式)"""
    model = ClearMindForCausalLM(tiny_config)
    model.eval()
    return model


@pytest.fixture
def sample_input_ids(tiny_config):
    """随机 token ids [batch=2, seq=16]"""
    return torch.randint(0, tiny_config.vocab_size, (2, 16))


# ============================================================
# 数据处理相关 fixtures
# ============================================================

@pytest.fixture(scope="session")
def tmp_tokenizer(tmp_path_factory):
    """训练一个 tiny tokenizer 用于数据处理测试

    scope=session: 整个测试会话只训练一次，所有测试共享
    """
    tmp_dir = tmp_path_factory.mktemp("tokenizer")
    corpus_path = tmp_dir / "corpus.txt"

    # 构建训练语料 — 足够覆盖中英文和特殊 token
    corpus_lines = [
        "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。",
        "自然语言处理是人工智能的重要领域，涵盖文本分类、机器翻译等任务。",
        "Transformer 架构使用自注意力机制来捕获序列中的长距离依赖关系。",
        "预训练语言模型通过在大规模文本上学习，获得丰富的语言理解能力。",
        "指令微调让模型学会遵循人类的指令，生成有帮助的回复。",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing is a subfield of artificial intelligence.",
        "The Transformer architecture uses self-attention mechanisms.",
        "Pre-trained language models learn from large-scale text data.",
        "Instruction tuning teaches models to follow human instructions.",
        "人工智能正在改变我们的生活方式和工作方式。",
        "强化学习从人类反馈中学习，提升模型的对齐性和安全性。",
    ]
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

    # 训练 tiny tokenizer (vocab_size=500)
    tokenizer = ClearMindTokenizer.train(
        corpus_path=str(corpus_path),
        vocab_size=500,
    )

    # 保存并重新加载 — 确保与正式流程一致
    save_dir = tmp_dir / "saved"
    tokenizer.save_pretrained(str(save_dir))
    return ClearMindTokenizer.load(str(save_dir))


@pytest.fixture
def sample_pretrain_data(tmp_path):
    """临时预训练数据 JSONL 文件"""
    data_path = tmp_path / "pretrain_data.jsonl"
    samples = [
        {"text": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。"},
        {"text": "自然语言处理是人工智能的重要领域，涵盖文本分类、机器翻译等任务。"},
        {"text": "Transformer 架构使用自注意力机制来捕获序列中的长距离依赖关系。"},
        {"text": "预训练语言模型通过在大规模文本上学习，获得丰富的语言理解能力。"},
        {"text": "Deep learning uses neural networks with multiple layers."},
        {"text": "The Transformer architecture uses self-attention mechanisms."},
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return str(data_path)


@pytest.fixture
def sample_sft_data(tmp_path):
    """临时 SFT 数据 JSONL 文件 (Alpaca 格式)"""
    data_path = tmp_path / "sft_data.jsonl"
    samples = [
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。",
        },
        {
            "instruction": "请解释 Transformer 架构",
            "input": "",
            "output": "Transformer 使用自注意力机制来捕获序列中的长距离依赖关系。",
        },
        {
            "instruction": "翻译以下文本为英文",
            "input": "深度学习很有趣",
            "output": "Deep learning is very interesting.",
        },
        {
            "instruction": "什么是自然语言处理？",
            "input": "",
            "output": "自然语言处理是人工智能的重要领域。",
        },
        {
            "instruction": "解释预训练的概念",
            "input": "",
            "output": "预训练是在大规模文本上学习语言表示的过程。",
        },
        {
            "instruction": "什么是注意力机制？",
            "input": "",
            "output": "注意力机制让模型关注输入序列中最相关的部分。",
        },
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return str(data_path)


@pytest.fixture
def sample_dpo_data(tmp_path):
    """临时 DPO 数据 JSONL 文件"""
    data_path = tmp_path / "dpo_data.jsonl"
    samples = [
        {
            "prompt": "什么是人工智能？",
            "chosen": "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。",
            "rejected": "人工智能就是让电脑变聪明。",
        },
        {
            "prompt": "解释深度学习",
            "chosen": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次表示。",
            "rejected": "深度学习就是很深的学习。",
        },
        {
            "prompt": "什么是 NLP？",
            "chosen": "NLP（自然语言处理）是人工智能的一个重要领域，专注于让计算机理解和生成人类语言。",
            "rejected": "NLP 是一种编程语言。",
        },
        {
            "prompt": "什么是强化学习？",
            "chosen": "强化学习是一种通过与环境交互来学习最优策略的机器学习方法。",
            "rejected": "强化学习就是反复练习。",
        },
        {
            "prompt": "什么是迁移学习？",
            "chosen": "迁移学习是将在一个任务上学到的知识应用到另一个相关任务的方法。",
            "rejected": "迁移学习是搬家学习。",
        },
        {
            "prompt": "什么是注意力机制？",
            "chosen": "注意力机制是一种让模型动态关注输入序列不同部分的技术，是Transformer的核心。",
            "rejected": "注意力机制就是集中注意力。",
        },
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return str(data_path)
