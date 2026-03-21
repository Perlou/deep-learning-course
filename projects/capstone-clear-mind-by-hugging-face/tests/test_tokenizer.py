"""
test_tokenizer.py — ClearMind Tokenizer 单元测试
=================================================

测试 HF tokenizers BPE 训练、encode/decode、特殊 token、
chat template、save/load 往返等功能。
"""

import os
import tempfile

import pytest
from transformers import PreTrainedTokenizerFast

from src.data.tokenizer import ClearMindTokenizer, CHAT_TEMPLATE


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory):
    """创建临时训练语料"""
    corpus_dir = tmp_path_factory.mktemp("corpus")
    corpus_file = corpus_dir / "corpus.txt"

    texts = [
        "深度学习是机器学习的一个子领域，它利用多层神经网络来学习数据的层次化表示。",
        "Transformer架构于2017年被提出，彻底改变了自然语言处理领域的研究方向。",
        "注意力机制允许模型在处理序列数据时，关注输入中最相关的部分。",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "The Transformer architecture has revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Pre-trained language models acquire powerful language understanding capabilities.",
        "Backpropagation uses the chain rule to compute gradients of the loss function.",
        "Convolutional neural networks extract spatial features from images.",
        "Recurrent neural networks can process variable-length sequences.",
        "预训练语言模型通过在大量文本数据上进行无监督学习，获得了强大的语言理解能力。",
        "反向传播算法通过链式法则计算损失函数对每个参数的梯度。",
        "批归一化技术通过对每个小批量数据进行归一化，加速了网络训练过程。",
        "Dropout是一种正则化技术，通过在训练过程中随机丢弃部分神经元来防止过拟合。",
        "强化学习是一种通过与环境交互来学习最优策略的机器学习方法。",
    ]
    # 重复多次以提供足够的训练数据
    with open(corpus_file, "w", encoding="utf-8") as f:
        for _ in range(100):
            for text in texts:
                f.write(text + "\n")

    return str(corpus_file)


@pytest.fixture(scope="module")
def tokenizer(corpus_path):
    """训练一个 vocab_size=500 的 tokenizer (module scope，只训练一次)"""
    return ClearMindTokenizer.train(
        corpus_path=corpus_path,
        vocab_size=500,
    )


# ============================================================
# Tests
# ============================================================


class TestTokenizerType:
    """测试 tokenizer 类型和基本属性"""

    def test_returns_pretrained_tokenizer_fast(self, tokenizer):
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

    def test_vocab_size(self, tokenizer):
        # vocab_size 应该接近请求的 500（可能略有差异）
        assert 400 <= tokenizer.vocab_size <= 600


class TestSpecialTokens:
    """测试特殊 token 配置"""

    def test_unk_token(self, tokenizer):
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.unk_token_id is not None

    def test_bos_token(self, tokenizer):
        assert tokenizer.bos_token == "<s>"
        assert tokenizer.bos_token_id is not None

    def test_eos_token(self, tokenizer):
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.eos_token_id is not None

    def test_pad_token(self, tokenizer):
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.pad_token_id is not None

    def test_special_token_ids_are_distinct(self, tokenizer):
        ids = {
            tokenizer.unk_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        }
        assert len(ids) == 4


class TestEncodeDecode:
    """测试编码和解码"""

    def test_encode_english(self, tokenizer):
        text = "Hello, world!"
        ids = tokenizer.encode(text)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_chinese(self, tokenizer):
        text = "深度学习很有趣"
        ids = tokenizer.encode(text)
        assert len(ids) > 0

    def test_decode_roundtrip_english(self, tokenizer):
        text = "Deep learning is powerful"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text

    def test_decode_roundtrip_chinese(self, tokenizer):
        text = "注意力机制是核心"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text

    def test_encode_contains_bos_eos(self, tokenizer):
        """post_processor 应自动添加 BOS 和 EOS"""
        text = "test"
        ids = tokenizer.encode(text)
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id

    def test_encode_empty_string(self, tokenizer):
        ids = tokenizer.encode("")
        # 至少应该有 BOS 和 EOS
        assert tokenizer.bos_token_id in ids
        assert tokenizer.eos_token_id in ids


class TestChatTemplate:
    """测试 chat template"""

    def test_has_chat_template(self, tokenizer):
        assert tokenizer.chat_template is not None
        assert len(tokenizer.chat_template) > 0

    def test_apply_chat_template_single_turn(self, tokenizer):
        messages = [{"role": "user", "content": "What is AI?"}]
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "Human: What is AI?" in result
        assert "Assistant: " in result

    def test_apply_chat_template_multi_turn(self, tokenizer):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert "Human: Hello" in result
        assert "Assistant: Hi there!" in result
        assert "Human: How are you?" in result
        # add_generation_prompt 应在末尾添加 "Assistant: "
        assert result.rstrip().endswith("Assistant:")

    def test_apply_chat_template_tokenized(self, tokenizer):
        messages = [{"role": "user", "content": "Test"}]
        result = tokenizer.apply_chat_template(messages, tokenize=True)
        # 提取 token ids (可能是 list, dict, 或 BatchEncoding)
        if hasattr(result, "input_ids"):
            ids = result.input_ids
        elif hasattr(result, "__getitem__") and not isinstance(result, list):
            ids = result["input_ids"]
        else:
            ids = result
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)


class TestSaveLoad:
    """测试保存和加载"""

    def test_save_load_roundtrip(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存
            tokenizer.save_pretrained(tmpdir)

            # 检查核心文件存在
            assert os.path.exists(os.path.join(tmpdir, "tokenizer.json"))
            assert os.path.exists(os.path.join(tmpdir, "tokenizer_config.json"))

            # 加载
            loaded = ClearMindTokenizer.load(tmpdir)

            # 验证功能一致
            assert loaded.vocab_size == tokenizer.vocab_size
            assert loaded.unk_token == tokenizer.unk_token
            assert loaded.bos_token == tokenizer.bos_token
            assert loaded.eos_token == tokenizer.eos_token
            assert loaded.pad_token == tokenizer.pad_token

    def test_save_load_encode_consistency(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            loaded = ClearMindTokenizer.load(tmpdir)

            text = "Deep learning and Transformer"
            orig_ids = tokenizer.encode(text)
            loaded_ids = loaded.encode(text)
            assert orig_ids == loaded_ids

    def test_save_load_chat_template_preserved(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            loaded = ClearMindTokenizer.load(tmpdir)

            messages = [{"role": "user", "content": "Hi"}]
            orig = tokenizer.apply_chat_template(messages, tokenize=False)
            loaded_result = loaded.apply_chat_template(messages, tokenize=False)
            assert orig == loaded_result
