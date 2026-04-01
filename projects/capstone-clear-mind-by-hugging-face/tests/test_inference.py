"""推理模块测试 (HF 版)"""

import torch
import pytest

from model import ClearMindConfig, ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from inference.generate import generate_text, extract_reply, create_pipeline
from inference.chat import _build_prompt


@pytest.fixture
def infer_tokenizer(tmp_path):
    """推理测试用 tokenizer"""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "深度学习是机器学习的一个分支\n"
        "Human Assistant 你好 什么是 机器学习\n"
        "hello world test deep learning\n" * 5,
        encoding="utf-8",
    )
    tokenizer = ClearMindTokenizer.train(str(corpus_path), vocab_size=500)
    save_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(str(save_dir))
    return ClearMindTokenizer.load(str(save_dir))


@pytest.fixture
def infer_config():
    return ClearMindConfig(
        hidden_size=64, num_attention_heads=2, num_key_value_heads=2,
        num_hidden_layers=2, intermediate_size=176, vocab_size=500,
        max_position_embeddings=64, hidden_dropout_prob=0.0, rms_norm_eps=1e-6,
    )


# ============================================================
# TestGenerateText
# ============================================================


class TestGenerateText:
    """测试 generate_text"""

    def test_basic_generation(self, infer_config, infer_tokenizer):
        """基本生成不报错"""
        model = ClearMindForCausalLM(infer_config)
        model.eval()

        text = generate_text(
            model, infer_tokenizer, "hello", max_new_tokens=10,
        )
        assert isinstance(text, str)
        assert len(text) > 0

    def test_greedy_generation(self, infer_config, infer_tokenizer):
        """Greedy 模式可用"""
        model = ClearMindForCausalLM(infer_config)
        model.eval()

        text = generate_text(
            model, infer_tokenizer, "hello",
            max_new_tokens=10, do_sample=False,
        )
        assert isinstance(text, str)

    def test_generation_length(self, infer_config, infer_tokenizer):
        """生成长度受 max_new_tokens 控制"""
        model = ClearMindForCausalLM(infer_config)
        model.eval()

        short = generate_text(model, infer_tokenizer, "hi", max_new_tokens=5)
        long = generate_text(model, infer_tokenizer, "hi", max_new_tokens=30)

        # 长生成应不短于短生成 (大概率成立)
        assert len(long) >= len(short)


# ============================================================
# TestExtractReply
# ============================================================


class TestExtractReply:
    """测试 extract_reply"""

    def test_with_assistant_prefix(self):
        """从 Assistant: 后提取"""
        text = "Human: 你好\nAssistant: 你好，我是 AI"
        reply = extract_reply(text)
        assert "你好" in reply
        assert "Human:" not in reply

    def test_with_prompt_prefix(self):
        """通过 prompt 前缀提取"""
        prompt = "hello"
        text = "hello world"
        reply = extract_reply(text, prompt)
        assert reply == "world"

    def test_truncate_at_human(self):
        """在 Human: 处截断"""
        text = "Assistant: 回复内容\nHuman: 下一个问题"
        reply = extract_reply(text)
        assert "下一个问题" not in reply
        assert "回复内容" in reply


# ============================================================
# TestCreatePipeline
# ============================================================


class TestCreatePipeline:
    """测试 pipeline 创建"""

    def test_pipeline_runs(self, infer_config, infer_tokenizer):
        """pipeline 可以正常运行"""
        model = ClearMindForCausalLM(infer_config)
        model.eval()

        pipe = create_pipeline(model, infer_tokenizer, max_new_tokens=10)
        result = pipe("hello")
        assert len(result) > 0
        assert "generated_text" in result[0]


# ============================================================
# TestBuildPrompt
# ============================================================


class TestBuildPrompt:
    """测试对话 prompt 构建"""

    def test_empty_history(self):
        """无历史时的 prompt"""
        prompt = _build_prompt([], "你好")
        assert prompt == "Human: 你好\nAssistant: "

    def test_with_history(self):
        """有历史时的 prompt"""
        history = [("你好", "你好！我是 AI")]
        prompt = _build_prompt(history, "什么是深度学习")
        assert "Human: 你好\nAssistant: 你好！我是 AI\n" in prompt
        assert prompt.endswith("Human: 什么是深度学习\nAssistant: ")

    def test_multi_turn_history(self):
        """多轮历史"""
        history = [("A", "B"), ("C", "D")]
        prompt = _build_prompt(history, "E")
        assert prompt.count("Human:") == 3
        assert prompt.count("Assistant:") == 3
