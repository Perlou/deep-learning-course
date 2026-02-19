"""Attention 模块单元测试"""

import torch

from model import ModelConfig
from model.attention import Attention


class TestAttentionShape:
    """测试 Attention 输出形状"""

    def test_output_shape(self, tiny_config):
        """输出应为 [batch, seq_len, d_model]"""
        attn = Attention(tiny_config)
        x = torch.randn(2, 16, tiny_config.d_model)
        output, _ = attn(x)
        assert output.shape == (2, 16, tiny_config.d_model)

    def test_single_token(self, tiny_config):
        """单 token 输入应正常工作"""
        attn = Attention(tiny_config)
        x = torch.randn(1, 1, tiny_config.d_model)
        output, _ = attn(x)
        assert output.shape == (1, 1, tiny_config.d_model)


class TestGQA:
    """测试 Grouped Query Attention"""

    def test_mha_mode(self):
        """n_kv_heads == n_heads 时为标准 MHA"""
        config = ModelConfig(d_model=128, n_heads=4, n_kv_heads=4, d_ff=352)
        attn = Attention(config)
        assert attn.n_kv_groups == 1  # 每个 Q head 独享一个 KV head

    def test_gqa_mode(self):
        """n_kv_heads < n_heads 时为 GQA"""
        config = ModelConfig(d_model=128, n_heads=4, n_kv_heads=2, d_ff=352)
        attn = Attention(config)
        assert attn.n_kv_groups == 2  # 每 2 个 Q head 共享一个 KV head

    def test_expand_kv(self):
        """_expand_kv 应正确复制 KV heads"""
        config = ModelConfig(d_model=128, n_heads=4, n_kv_heads=2, d_ff=352)
        attn = Attention(config)
        x = torch.randn(1, 2, 8, 32)  # [batch, n_kv_heads, seq, head_dim]
        expanded = attn._expand_kv(x)
        assert expanded.shape == (1, 4, 8, 32)  # → [batch, n_heads, seq, head_dim]

    def test_gqa_output_shape(self):
        """GQA 模式输出形状应正确"""
        config = ModelConfig(d_model=128, n_heads=4, n_kv_heads=2, d_ff=352)
        attn = Attention(config)
        x = torch.randn(2, 8, 128)
        output, _ = attn(x)
        assert output.shape == (2, 8, 128)


class TestKVCache:
    """测试 KV Cache"""

    def test_kv_cache_returned(self, tiny_config):
        """use_cache=True 时应返回 KV Cache"""
        attn = Attention(tiny_config)
        x = torch.randn(1, 8, tiny_config.d_model)
        _, kv_cache = attn(x, use_cache=True)
        assert kv_cache is not None
        k, v = kv_cache
        assert k.shape[2] == 8  # seq_len
        assert v.shape[2] == 8

    def test_kv_cache_none_when_not_used(self, tiny_config):
        """use_cache=False 时 KV Cache 应为 None"""
        attn = Attention(tiny_config)
        x = torch.randn(1, 8, tiny_config.d_model)
        _, kv_cache = attn(x, use_cache=False)
        assert kv_cache is None

    def test_kv_cache_extends(self, tiny_config):
        """带 cache 调用时 cache 应增长"""
        attn = Attention(tiny_config)

        # Prefill 8 tokens
        x1 = torch.randn(1, 8, tiny_config.d_model)
        _, cache1 = attn(x1, use_cache=True)

        # Decode 1 token with cache
        x2 = torch.randn(1, 1, tiny_config.d_model)
        _, cache2 = attn(x2, kv_cache=cache1, use_cache=True)

        k2, v2 = cache2
        assert k2.shape[2] == 9  # 8 + 1
        assert v2.shape[2] == 9


class TestSlidingWindow:
    """测试 Sliding Window Attention"""

    def test_sliding_window_no_error(self):
        """启用 sliding_window 时不应报错"""
        config = ModelConfig(
            d_model=128, n_heads=4, n_kv_heads=4, d_ff=352, sliding_window=64
        )
        attn = Attention(config)
        x = torch.randn(1, 16, 128)
        output, _ = attn(x)
        assert output.shape == (1, 16, 128)

    def test_sliding_window_output_shape(self):
        """sliding window 输出形状应与全局注意力一致"""
        config = ModelConfig(
            d_model=128, n_heads=4, n_kv_heads=4, d_ff=352, sliding_window=64
        )
        attn = Attention(config)
        x = torch.randn(1, 16, 128)
        output, _ = attn(x)
        assert output.shape == (1, 16, 128)
