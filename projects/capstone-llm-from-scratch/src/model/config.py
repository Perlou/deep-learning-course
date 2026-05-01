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
        d_model:        隐藏层维度 (embedding dimension)
        n_heads:        注意力头数 (query heads)
        n_kv_heads:     KV 头数 (key/value heads, 用于 GQA)
                        等于 n_heads 时为标准 MHA
                        小于 n_heads 时为 GQA (Grouped Query Attention)
        n_layers:       Transformer 层数
        d_ff:           FFN 中间维度 (SwiGLU 的隐藏层大小)
        vocab_size:     词表大小
        max_seq_len:    最大序列长度
        dropout:        Dropout 比率
        norm_eps:       RMSNorm 的 epsilon
        sliding_window: Sliding Window Attention 窗口大小（None 关闭）
        rope_theta:     RoPE 频率底数。默认 1e6（与 minimind / Qwen3 对齐，
                        长上下文友好）。传统 GPT-2/Llama-2 用 10000
        use_qk_norm:    是否在 attention 中对 Q/K 做 RMSNorm（Llama-3/Gemma2 同款）
                        长训 loss 更稳，参数开销极小（仅 2 * n_layers * head_dim）
        rope_scaling:   YaRN 等长上下文外推配置（None 关闭）。示例：
                        ``{"type": "yarn", "factor": 4, "original_max_pos": 1024,
                          "beta_fast": 32, "beta_slow": 1, "attention_factor": 1.0}``
                        启用后推理时可外推到 ``factor × original_max_pos`` 长度
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

    # --- 现代 LLM 架构升级（Phase 2）---
    rope_theta: float = 1.0e6           # 与 minimind / Qwen3 对齐
    use_qk_norm: bool = True            # Llama-3 / Gemma2 同款
    rope_scaling: dict | None = None    # YaRN 长上下文外推（推理时启用）

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
        """Tiny 配置 (~0.5M 参数), 用于 CPU/MPS 冒烟测试

        与 ``configs/tiny.yaml`` 一致。使用 minimind tokenizer (vocab=6400)。
        设计目标：本地最快路径跑通 pretrain → sft → dpo 全流程；不追求生成质量。
        """
        return cls(
            d_model=64,
            n_heads=4,        # head_dim = 16
            n_kv_heads=2,     # GQA 2:1
            n_layers=2,
            d_ff=256,         # ⌈64·π/64⌉·64
            vocab_size=6400,
            max_seq_len=128,
            dropout=0.1,
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small 配置 (~26M 参数), 对齐 minimind2-small

        与 ``configs/small.yaml`` 一致。
        """
        return cls(
            d_model=512,
            n_heads=8,
            n_kv_heads=2,    # GQA 4:1
            n_layers=8,
            d_ff=1664,       # ⌈512·π/64⌉·64
            vocab_size=6400,
            max_seq_len=1024,
            dropout=0.0,
        )

    @classmethod
    def main(cls) -> "ModelConfig":
        """Main 配置 (~64M 参数), 对齐 minimind-3 dense

        与 ``configs/main.yaml`` 一致。这是 ClearMind-Base 发布版本的训练规格。
        """
        return cls(
            d_model=768,
            n_heads=8,
            n_kv_heads=4,    # GQA 2:1
            n_layers=8,
            d_ff=2432,       # ⌈768·π/64⌉·64
            vocab_size=6400,
            max_seq_len=1024,
            dropout=0.0,
        )

    @classmethod
    def plus(cls) -> "ModelConfig":
        """Plus 配置 (~478M 参数), ClearMind-Plus 旗舰版

        与 ``configs/plus.yaml`` 一致。dense 路线，对标 minimind-3-moe 198M-A64M
        但用单 token 计算量 ~7× 的 dense 模型超越其效果。
        单卡 A100/A800 80GB bf16 训练，两天内完成 Pretrain + SFT + DPO。
        """
        return cls(
            d_model=1280,
            n_heads=16,      # head_dim = 80
            n_kv_heads=4,    # GQA 4:1
            n_layers=24,
            d_ff=4032,       # ⌈1280·π/64⌉·64
            vocab_size=6400,
            max_seq_len=1024,
            dropout=0.0,
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

        # d_ff 推荐两种公式之一：
        #   - 经典 SwiGLU：约 8/3 · d_model
        #   - minimind / Qwen3 风格：⌈d_model·π/64⌉·64（TensorCore/SIMD 对齐）
        # 任一公式偏差超过 30% 才报警（避免对小尺寸误报）
        expected_classic = int(8 / 3 * self.d_model)
        expected_aligned = -(-int(self.d_model * 3.141592653589793) // 64) * 64
        diff_classic = abs(self.d_ff - expected_classic) / max(expected_classic, 1)
        diff_aligned = abs(self.d_ff - expected_aligned) / max(expected_aligned, 1)
        if min(diff_classic, diff_aligned) > 0.3:
            warnings.warn(
                f"d_ff={self.d_ff} 偏离推荐值 (经典 ≈{expected_classic} 或 "
                f"对齐 ≈{expected_aligned}) 超过 30%",
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
