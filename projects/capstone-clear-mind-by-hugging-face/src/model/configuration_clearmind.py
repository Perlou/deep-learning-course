"""
ClearMindConfig — HuggingFace 风格的模型配置
============================================

继承 PretrainedConfig，将 from-scratch 版 ModelConfig 迁移为 HF 标准命名。

字段映射 (from-scratch → HF):
  d_model     → hidden_size
  n_heads     → num_attention_heads
  n_kv_heads  → num_key_value_heads
  n_layers    → num_hidden_layers
  d_ff        → intermediate_size
  max_seq_len → max_position_embeddings
  dropout     → hidden_dropout_prob
  norm_eps    → rms_norm_eps
"""

import yaml
from transformers import PretrainedConfig


class ClearMindConfig(PretrainedConfig):
    """ClearMind 模型配置

    Attributes:
        hidden_size:              隐藏层维度
        num_attention_heads:      Query 注意力头数
        num_key_value_heads:      KV 头数 (GQA)
        num_hidden_layers:        Transformer 层数
        intermediate_size:        FFN 中间维度 (SwiGLU)
        vocab_size:               词表大小
        max_position_embeddings:  最大序列长度
        hidden_dropout_prob:      Dropout 比率
        rms_norm_eps:             RMSNorm epsilon
        rope_theta:               RoPE 频率底数
        use_cache:                是否使用 KV Cache
        tie_word_embeddings:      是否共享输入/输出 embedding
    """

    model_type = "clearmind"

    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        num_hidden_layers: int = 8,
        intermediate_size: int = 1408,
        vocab_size: int = 8000,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_cache = use_cache

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        """GQA 中每组 Q heads 共享一组 KV 的数量"""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def tiny(cls) -> "ClearMindConfig":
        """Tiny 配置 (~0.6M 参数), 用于快速验证"""
        return cls(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=4,
            intermediate_size=352,
            vocab_size=2000,
            max_position_embeddings=128,
            hidden_dropout_prob=0.1,
        )

    @classmethod
    def small(cls) -> "ClearMindConfig":
        """Small 配置 (~26M 参数), 适合 MacBook CPU/MPS"""
        return cls(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=8,
            intermediate_size=1408,
            vocab_size=8000,
            max_position_embeddings=512,
        )

    @classmethod
    def medium(cls) -> "ClearMindConfig":
        """Medium 配置 (~200M 参数), 需要 GPU"""
        return cls(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=16,
            intermediate_size=2816,
            vocab_size=32000,
            max_position_embeddings=1024,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ClearMindConfig":
        """从 YAML 配置文件加载

        Args:
            path: YAML 文件路径 (读取 model 段)
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw["model"])

    def count_params(self) -> dict[str, int | float]:
        """估算模型参数量 (不含 weight tying 重复)

        Returns:
            各部分参数量的字典
        """
        d = self.hidden_size
        h = self.num_attention_heads
        kv_h = self.num_key_value_heads
        hd = self.head_dim
        ff = self.intermediate_size
        L = self.num_hidden_layers
        V = self.vocab_size

        embedding = V * d
        # Attention: q_proj + k_proj + v_proj + o_proj
        attn_per_layer = d * (h * hd) + d * (kv_h * hd) * 2 + (h * hd) * d
        # FFN (SwiGLU): gate_proj + up_proj + down_proj
        ffn_per_layer = d * ff * 3
        # RMSNorm: 2 per layer
        norm_per_layer = d * 2
        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        # Final norm + LM head
        final = d + V * d

        total = embedding + L * per_layer + final

        return {
            "embedding": embedding,
            "attention_per_layer": attn_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "norm_per_layer": norm_per_layer,
            "per_layer_total": per_layer,
            "all_layers": L * per_layer,
            "final_norm_and_lm_head": final,
            "total": total,
            "total_millions": total / 1e6,
        }
