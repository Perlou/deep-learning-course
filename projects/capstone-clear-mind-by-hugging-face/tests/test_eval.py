"""评估体系测试 (HF 版)"""

import math
import sys
import tempfile
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluate"))

from model import ClearMindConfig, ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer


@pytest.fixture
def eval_tokenizer(tmp_path):
    """为评估测试创建一个简易 tokenizer"""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "深度学习是机器学习的一个分支\n"
        "自然语言处理是人工智能的重要领域\n"
        "Human Assistant 请解释 什么是 机器学习 数据 模型 算法\n"
        "hello world test data deep learning\n" * 5,
        encoding="utf-8",
    )
    tokenizer = ClearMindTokenizer.train(str(corpus_path), vocab_size=500)
    save_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(str(save_dir))
    return ClearMindTokenizer.load(str(save_dir))


@pytest.fixture
def eval_tiny_config():
    """评估测试用的 tiny config (vocab_size=500 匹配 eval_tokenizer)"""
    return ClearMindConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=176,
        vocab_size=500,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        rms_norm_eps=1e-6,
    )


# ============================================================
# TestEvalPerplexity
# ============================================================


class TestEvalPerplexity:
    """测试困惑度评估"""

    def test_ppl_basic(self, eval_tiny_config, eval_tokenizer):
        """基本 PPL 计算"""
        from eval_perplexity import evaluate_perplexity
        from datasets import Dataset

        model = ClearMindForCausalLM(eval_tiny_config)
        model.eval()

        seq_len = 32
        data = {
            "input_ids": [
                list(torch.randint(0, eval_tiny_config.vocab_size, (seq_len,)).numpy())
                for _ in range(8)
            ],
            "attention_mask": [[1] * seq_len for _ in range(8)],
        }
        dataset = Dataset.from_dict(data)

        ppl = evaluate_perplexity(
            model=model,
            eval_dataset=dataset,
            tokenizer=eval_tokenizer,
            batch_size=4,
            max_batches=2,
        )

        assert isinstance(ppl, float)
        assert ppl > 0
        assert math.isfinite(ppl)

    def test_ppl_untrained_model_high(self, eval_tiny_config, eval_tokenizer):
        """未训练模型的 PPL 应该较高"""
        from eval_perplexity import evaluate_perplexity
        from datasets import Dataset

        model = ClearMindForCausalLM(eval_tiny_config)
        model.eval()

        data = {
            "input_ids": [
                list(torch.randint(0, eval_tiny_config.vocab_size, (16,)).numpy())
                for _ in range(4)
            ],
            "attention_mask": [[1] * 16 for _ in range(4)],
        }
        dataset = Dataset.from_dict(data)

        ppl = evaluate_perplexity(
            model=model,
            eval_dataset=dataset,
            tokenizer=eval_tokenizer,
            batch_size=4,
            max_batches=1,
        )

        # 未训练模型 PPL 应接近 vocab_size 量级
        assert ppl > 10


# ============================================================
# TestGenerationMetrics
# ============================================================


class TestGenerationMetrics:
    """测试生成质量指标计算"""

    def test_distinct_n_basic(self):
        """Distinct-N 计算正确"""
        from eval_generation import distinct_n

        texts = ["AABBC", "CCDDE"]
        d1 = distinct_n(texts, 1)
        assert 0 < d1 <= 1.0

    def test_distinct_n_identical(self):
        """完全相同的文本 Distinct-N 低"""
        from eval_generation import distinct_n

        texts = ["AAAA", "AAAA"]
        d1 = distinct_n(texts, 1)
        assert d1 == pytest.approx(1 / 8)

    def test_distinct_n_diverse(self):
        """多样的文本 Distinct-N 高"""
        from eval_generation import distinct_n

        texts = ["ABCDEFGH", "IJKLMNOP"]
        d1 = distinct_n(texts, 1)
        assert d1 == 1.0

    def test_repetition_rate_no_repeat(self):
        """无重复文本重复率为 0"""
        from eval_generation import repetition_rate

        text = "ABCDEFGHIJKLMNOP"
        rate = repetition_rate(text, n=4)
        assert rate == 0.0

    def test_repetition_rate_high_repeat(self):
        """高重复文本重复率高"""
        from eval_generation import repetition_rate

        text = "ABCDABCDABCDABCD"
        rate = repetition_rate(text, n=4)
        assert rate > 0.5

    def test_compute_metrics_full(self):
        """compute_metrics 返回完整指标"""
        from eval_generation import compute_metrics

        texts = [
            "深度学习是一种机器学习方法",
            "自然语言处理是人工智能的分支",
            "",
        ]
        metrics = compute_metrics(texts)

        assert metrics["num_generated"] == 3
        assert metrics["num_valid"] == 2
        assert metrics["avg_length"] > 0
        assert 0 <= metrics["distinct_1"] <= 1
        assert 0 <= metrics["distinct_2"] <= 1
        assert 0 <= metrics["distinct_3"] <= 1
        assert 0 <= metrics["avg_repetition_rate"] <= 1
        assert metrics["empty_rate"] == pytest.approx(1 / 3)

    def test_compute_metrics_all_empty(self):
        """全空文本的指标"""
        from eval_generation import compute_metrics

        metrics = compute_metrics(["", "", ""])
        assert metrics["num_valid"] == 0
        assert metrics["empty_rate"] == 1.0


# ============================================================
# TestBatchGenerate
# ============================================================


class TestBatchGenerate:
    """测试批量生成"""

    def test_batch_generate_runs(self, eval_tiny_config, eval_tokenizer):
        """batch_generate 不报错"""
        from eval_generation import batch_generate

        model = ClearMindForCausalLM(eval_tiny_config)
        model.eval()

        results = batch_generate(
            model=model,
            tokenizer=eval_tokenizer,
            prompts=["hello", "world"],
            max_new_tokens=10,
        )

        assert len(results) == 2
        for r in results:
            assert "prompt" in r
            assert "output" in r
            assert "length" in r


# ============================================================
# TestInstructionEval
# ============================================================


class TestInstructionEval:
    """测试指令跟随评估"""

    def test_evaluate_instructions_runs(self, eval_tiny_config, eval_tokenizer):
        """evaluate_instructions 不报错"""
        from eval_benchmark import evaluate_instructions

        model = ClearMindForCausalLM(eval_tiny_config)
        model.eval()

        results = evaluate_instructions(
            model=model,
            tokenizer=eval_tokenizer,
            max_new_tokens=10,
        )

        assert "summary" in results
        assert "normal" in results
        assert "safety" in results
        summary = results["summary"]
        assert 0 <= summary["format_accuracy"] <= 1
        assert 0 <= summary["completeness_rate"] <= 1
        assert 0 <= summary["relevance_rate"] <= 1
        assert 0 <= summary["refusal_rate"] <= 1
