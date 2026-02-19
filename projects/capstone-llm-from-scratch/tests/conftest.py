"""共享 pytest fixtures"""

import sys
import os

import pytest
import torch

# 将 src 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GPT  # noqa: E402


@pytest.fixture
def tiny_config():
    """极小模型配置 (~1.5M 参数), 用于 CPU 上快速测试"""
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
