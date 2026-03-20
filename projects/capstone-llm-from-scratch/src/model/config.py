"""
ModelConfig — 模型配置
====================

定义 GPT 模型的所有超参数。
支持从 YAML 配置文件加载，也支持直接实例化。

对比 GPT-4 / Llama / Gemini:
  - 这些模型使用完全相同的配置结构
  - 区别仅在于规模 (d_model, n_layers 等参数更大)
"""

import warnings

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """GPT 模型配置

    Attributes:
        d_model:     隐藏层维度 (embedding dimension)
        n_heads:     注意力头数 (query heads)
        n_kv_heads:  KV 头数 (key/value heads, 用于 GQA)
                     等于 n_heads 时为标准 MHA
                     小于 n_heads 时为 GQA (Grouped Query Attention)
        n_layers:    Transformer 层数
        d_ff:        FFN 中间维度 (SwiGLU 的隐藏层大小)
        vocab_size:  词表大小
        max_seq_len: 最大序列长度
        dropout:     Dropout 比率
        norm_eps:    RMSNorm 的 epsilon
    """

    # --- 模型维度 ---
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1408
    vocab_size: int = 8000
    max_seq_len: int = 512
    dropout: float = 0.1
    norm_eps: float = 1e-6
    sliding_window: int | None = None  # None = 全局注意力, >0 = 滑动窗口大小

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """GQA 中每组 Q heads 共享一组 KV 的数量

        例: n_heads=16, n_kv_heads=8 → 每 2 个 Q head 共享一组 KV
        """
        return self.n_heads // self.n_kv_heads

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny 配置 (~1.5M 参数), 用于快速验证训练流程"""
        return cls(
            d_model=128,
            n_heads=4,
            n_kv_heads=4,
            n_layers=4,
            d_ff=352,
            vocab_size=2000,
            max_seq_len=128,
            dropout=0.1,
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small 配置 (~26M 参数), 适合 MacBook CPU/MPS"""
        return cls(
            d_model=512,
            n_heads=8,
            n_kv_heads=8,  # 标准 MHA
            n_layers=8,
            d_ff=1408,
            vocab_size=8000,
            max_seq_len=512,
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium 配置 (~200M 参数), 需要 GPU"""
        return cls(
            d_model=1024,
            n_heads=16,
            n_kv_heads=8,  # GQA (2:1)
            n_layers=16,
            d_ff=2816,
            vocab_size=32000,
            max_seq_len=1024,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """从 YAML 配置文件加载模型配置

        Args:
            path: YAML 文件路径

        Returns:
            ModelConfig 实例
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config["model"])

    def count_params(self) -> dict[str, int | float]:
        """估算模型参数量 (不含共享权重)

        Returns:
            各部分参数量的字典
        """
        d = self.d_model
        h = self.n_heads
        kv_h = self.n_kv_heads
        hd = self.head_dim
        ff = self.d_ff
        L = self.n_layers
        V = self.vocab_size

        embedding = V * d
        # Attention: W_q + W_k + W_v + W_o
        attn_per_layer = d * (h * hd) + d * (kv_h * hd) * 2 + (h * hd) * d
        # FFN (SwiGLU): W_gate + W_up + W_down
        ffn_per_layer = d * ff * 3
        # RMSNorm: 2 per layer (attn + ffn)
        norm_per_layer = d * 2
        # Per layer total
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

    def __post_init__(self):
        """验证配置的合法性"""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) 必须能被 n_kv_heads ({self.n_kv_heads}) 整除"
        )
        assert self.max_seq_len >= 64, f"max_seq_len ({self.max_seq_len}) 至少为 64"
        assert self.vocab_size >= 256, f"vocab_size ({self.vocab_size}) 至少为 256"
        assert self.n_layers >= 1, f"n_layers ({self.n_layers}) 至少为 1"
        assert 0.0 <= self.dropout <= 0.5, (
            f"dropout ({self.dropout}) 应在 [0, 0.5] 范围内"
        )
        if self.sliding_window is not None:
            assert self.sliding_window >= 64, (
                f"sliding_window ({self.sliding_window}) 至少为 64"
            )

        # d_ff 通常约为 8/3 × d_model (SwiGLU 最佳)
        expected_ff = int(8 / 3 * self.d_model)
        if abs(self.d_ff - expected_ff) > expected_ff * 0.2:
            warnings.warn(
                f"d_ff={self.d_ff} 偏离推荐值 ≈{expected_ff} (8/3 × d_model) 超过 20%",
                UserWarning,
                stacklevel=2,
            )


if __name__ == "__main__":
    # 快速测试
    for name, config in [
        ("Small", ModelConfig.small()),
        ("Medium", ModelConfig.medium()),
    ]:
        params = config.count_params()
        print(f"\n{'=' * 50}")
        print(f"{name} 配置:")
        print(
            f"  d_model={config.d_model}, n_heads={config.n_heads}, "
            f"n_kv_heads={config.n_kv_heads}, n_layers={config.n_layers}"
        )
        print(f"  head_dim={config.head_dim}, GQA groups={config.n_kv_groups}")
        print(f"  总参数量: {params['total_millions']:.1f}M")
        print(f"  各部分:")
        print(f"    Embedding:     {params['embedding'] / 1e6:.2f}M")
        print(f"    All Layers:    {params['all_layers'] / 1e6:.2f}M")
        print(f"    Final + LMHead:{params['final_norm_and_lm_head'] / 1e6:.2f}M")
