"""ModelConfig 单元测试"""

import warnings

import pytest

from model import ModelConfig


class TestModelConfig:
    """测试 ModelConfig 数据类"""

    def test_tiny_factory(self):
        """tiny() 应返回有效配置"""
        config = ModelConfig.tiny()
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 4
        assert config.vocab_size == 2000

    def test_small_factory(self):
        """small() 应返回有效配置"""
        config = ModelConfig.small()
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 8

    def test_medium_factory(self):
        """medium() 应使用 GQA (n_kv_heads < n_heads)"""
        config = ModelConfig.medium()
        assert config.n_kv_heads < config.n_heads
        assert config.n_kv_groups == 2

    def test_head_dim(self, tiny_config):
        """head_dim = d_model / n_heads"""
        assert tiny_config.head_dim == tiny_config.d_model // tiny_config.n_heads

    def test_n_kv_groups(self, tiny_config):
        """n_kv_groups = n_heads / n_kv_heads"""
        assert tiny_config.n_kv_groups == tiny_config.n_heads // tiny_config.n_kv_heads

    def test_count_params(self, tiny_config):
        """参数计数应返回正确的 key 和正数"""
        params = tiny_config.count_params()
        expected_keys = [
            "embedding",
            "attention_per_layer",
            "ffn_per_layer",
            "per_layer_total",
            "all_layers",
            "total",
            "total_millions",
        ]
        for key in expected_keys:
            assert key in params
            assert params[key] > 0

    def test_count_params_consistency(self, tiny_config):
        """total = embedding + all_layers + final"""
        params = tiny_config.count_params()
        reconstructed = (
            params["embedding"]
            + params["all_layers"]
            + params["final_norm_and_lm_head"]
        )
        assert params["total"] == reconstructed

    def test_invalid_d_model_n_heads(self):
        """d_model 不能被 n_heads 整除时应报错"""
        with pytest.raises(AssertionError, match="d_model"):
            ModelConfig(d_model=100, n_heads=3, n_kv_heads=3, d_ff=270)

    def test_invalid_n_heads_n_kv_heads(self):
        """n_heads 不能被 n_kv_heads 整除时应报错"""
        with pytest.raises(AssertionError, match="n_heads"):
            ModelConfig(d_model=128, n_heads=8, n_kv_heads=3, d_ff=352)

    def test_max_seq_len_too_small(self):
        """max_seq_len < 64 应报错"""
        with pytest.raises(AssertionError, match="max_seq_len"):
            ModelConfig(d_model=128, n_heads=4, n_kv_heads=4, d_ff=352, max_seq_len=32)

    def test_d_ff_warning(self):
        """d_ff 偏离推荐值超过 20% 应发出 warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ModelConfig(d_model=128, n_heads=4, n_kv_heads=4, d_ff=1000)
            assert len(w) == 1
            assert "d_ff" in str(w[0].message)

    def test_from_yaml(self, tmp_path):
        """从 YAML 文件加载配置"""
        yaml_content = """
model:
  d_model: 128
  n_heads: 4
  n_kv_heads: 4
  n_layers: 4
  d_ff: 352
  vocab_size: 8000
  max_seq_len: 128
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = ModelConfig.from_yaml(str(yaml_file))
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.vocab_size == 8000
