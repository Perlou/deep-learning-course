"""共享 pytest fixtures"""

import os
import sys

import pytest
import torch

# 将 src 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GPT  # noqa: E402


# ============================================================
# 模型 fixtures
# ============================================================


@pytest.fixture
def tiny_config():
    """极小模型配置 (~1.5M 参数), 与 configs/tiny.yaml 一致"""
    return ModelConfig.tiny()


@pytest.fixture
def tiny_model(tiny_config):
    """基于 tiny_config 的 GPT 模型"""
    model = GPT(tiny_config)
    model.eval()
    return model


@pytest.fixture
def device():
    """测试使用 CPU"""
    return torch.device("cpu")


@pytest.fixture
def sample_input_ids(tiny_config):
    """样本 input_ids [batch=2, seq_len=16]"""
    return torch.randint(0, tiny_config.vocab_size, (2, 16))


# ============================================================
# Tokenizer fixtures
# ============================================================


def _minimind_tokenizer_dir() -> str:
    """获取仓库自带的 minimind tokenizer 目录"""
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "tokenizer",
            "minimind",
        )
    )


@pytest.fixture(scope="session")
def hf_tokenizer():
    """加载 tokenizer/minimind 的 HF tokenizer

    依赖 ``transformers`` 与 tokenizer 文件，缺失时整个 fixture skip。
    session 作用域避免重复加载。
    """
    pytest.importorskip("transformers", reason="HF tokenizer 测试需要 transformers")
    tk_dir = _minimind_tokenizer_dir()
    if not os.path.exists(os.path.join(tk_dir, "tokenizer.json")):
        pytest.skip(f"minimind tokenizer 不存在: {tk_dir}")
    from data.hf_tokenizer import HFTokenizer

    return HFTokenizer(tk_dir)
