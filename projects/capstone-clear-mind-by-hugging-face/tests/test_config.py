"""ClearMindConfig 单元测试"""

import tempfile
from pathlib import Path

import pytest

from model import ClearMindConfig


class TestClearMindConfig:
    """测试 ClearMindConfig"""

    def test_factory_tiny(self):
        """tiny() 工厂方法应返回正确配置"""
        config = ClearMindConfig.tiny()
        assert config.hidden_size == 128
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 4
        assert config.num_hidden_layers == 4
        assert config.intermediate_size == 352
        assert config.vocab_size == 2000
        assert config.max_position_embeddings == 128

    def test_factory_small(self):
        """small() 工厂方法应返回正确配置"""
        config = ClearMindConfig.small()
        assert config.hidden_size == 512
        assert config.num_attention_heads == 8
        assert config.num_hidden_layers == 8
        assert config.vocab_size == 8000

    def test_factory_medium(self):
        """medium() 工厂方法应返回正确配置"""
        config = ClearMindConfig.medium()
        assert config.hidden_size == 1024
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8  # GQA
        assert config.num_hidden_layers == 16
        assert config.vocab_size == 32000

    def test_head_dim(self):
        """head_dim 属性应正确计算"""
        config = ClearMindConfig.tiny()
        assert config.head_dim == 128 // 4  # hidden_size / num_attention_heads

    def test_num_key_value_groups_mha(self):
        """MHA 配置下 num_key_value_groups 应为 1"""
        config = ClearMindConfig.tiny()
        assert config.num_key_value_groups == 1

    def test_num_key_value_groups_gqa(self):
        """GQA 配置下 num_key_value_groups 应正确"""
        config = ClearMindConfig.medium()
        assert config.num_key_value_groups == 2  # 16 / 8

    def test_from_yaml(self):
        """从 YAML 文件加载应正确"""
        yaml_path = Path(__file__).parent.parent / "configs" / "tiny.yaml"
        config = ClearMindConfig.from_yaml(str(yaml_path))
        assert config.hidden_size == 128
        assert config.vocab_size == 2000
        assert config.num_hidden_layers == 4

    def test_save_load_pretrained(self):
        """save_pretrained / from_pretrained 往返应一致"""
        config = ClearMindConfig.tiny()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = ClearMindConfig.from_pretrained(tmpdir)

        assert loaded.hidden_size == config.hidden_size
        assert loaded.num_attention_heads == config.num_attention_heads
        assert loaded.num_key_value_heads == config.num_key_value_heads
        assert loaded.num_hidden_layers == config.num_hidden_layers
        assert loaded.intermediate_size == config.intermediate_size
        assert loaded.vocab_size == config.vocab_size
        assert loaded.model_type == "clearmind"

    def test_count_params(self):
        """count_params 应返回合理的参数量估算"""
        config = ClearMindConfig.tiny()
        params = config.count_params()

        assert params["total"] > 0
        assert params["total_millions"] > 0
        assert params["embedding"] > 0
        assert params["all_layers"] > 0
        assert params["attention_per_layer"] > 0
        assert params["ffn_per_layer"] > 0

    def test_model_type(self):
        """model_type 应为 clearmind"""
        config = ClearMindConfig()
        assert config.model_type == "clearmind"
