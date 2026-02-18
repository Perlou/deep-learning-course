"""
Multi-Head Attention + GQA (Grouped Query Attention)
=====================================================

实现 Transformer 的核心组件 — 自注意力机制。
支持标准 MHA 和 GQA 两种模式:
  - MHA (Multi-Head Attention): n_kv_heads == n_heads, 每个 Q head 有独立的 K/V
  - GQA (Grouped Query Attention): n_kv_heads < n_heads, 多个 Q heads 共享一组 K/V

GQA 的优势:
  - 减少 KV Cache 内存占用 (推理时每个 token 需要缓存 K/V)
  - n_kv_heads = n_heads → 标准 MHA
  - n_kv_heads = 1       → MQA (Multi-Query Attention)
  - 1 < n_kv_heads < n_heads → GQA (效果介于两者之间)

参考:
  - Attention Is All You Need (https://arxiv.org/abs/1706.03762)
  - GQA: Training Generalized Multi-Query Transformer (https://arxiv.org/abs/2305.13245)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import precompute_rope_frequencies, apply_rope


class Attention(nn.Module):
    """Multi-Head Self-Attention with GQA and RoPE

    内部流程:
      1. 线性投影: x → Q, K, V
      2. 分头: reshape 为 [batch, n_heads, seq, head_dim]
      3. 应用 RoPE: 对 Q, K 施加旋转位置编码
      4. GQA 扩展: 将 KV heads 复制以匹配 Q heads 数量
      5. 注意力计算: softmax(Q·K^T / √d) · V
      6. 合并头: reshape 回 [batch, seq, d_model]
      7. 输出投影: W_o

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_kv_groups

        d_model = config.d_model

        # Query 投影: d_model → n_heads × head_dim
        self.w_q = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        # Key 投影: d_model → n_kv_heads × head_dim (GQA 时更少)
        self.w_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        # Value 投影: d_model → n_kv_heads × head_dim
        self.w_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        # 输出投影: n_heads × head_dim → d_model
        self.w_o = nn.Linear(self.n_heads * self.head_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 预计算 RoPE 频率
        cos, sin = precompute_rope_frequencies(self.head_dim, config.max_seq_len)
        # 注册为 buffer (不参与梯度计算, 但会跟随模型移动到 GPU)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    输入张量 [batch, seq_len, d_model]
            mask: 注意力掩码 [seq_len, seq_len] (causal mask)

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # ========== Step 1: 线性投影 ==========
        q = self.w_q(x)  # [batch, seq, n_heads × head_dim]
        k = self.w_k(x)  # [batch, seq, n_kv_heads × head_dim]
        v = self.w_v(x)  # [batch, seq, n_kv_heads × head_dim]

        # ========== Step 2: 分头 (reshape) ==========
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # 现在形状: [batch, n_heads/n_kv_heads, seq_len, head_dim]

        # ========== Step 3: 应用 RoPE ==========
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # ========== Step 4: GQA 扩展 ==========
        # 如果使用 GQA, 需要将 KV heads 复制到和 Q heads 相同数量
        if self.n_kv_groups > 1:
            k = self._expand_kv(k)  # [batch, n_heads, seq, head_dim]
            v = self._expand_kv(v)

        # ========== Step 5: 注意力计算 ==========
        # Q·K^T / √d
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        # [batch, n_heads, seq, seq]

        # 应用 causal mask (上三角为 -inf)
        if mask is not None:
            attn_weights = attn_weights + mask[:seq_len, :seq_len]

        # softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, v)
        # [batch, n_heads, seq, head_dim]

        # ========== Step 6: 合并头 ==========
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        # [batch, seq, n_heads × head_dim]

        # ========== Step 7: 输出投影 ==========
        output = self.w_o(output)
        output = self.resid_dropout(output)

        return output

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """将 KV heads 扩展到与 Q heads 相同数量 (GQA)

        例: n_kv_heads=8, n_heads=16, n_kv_groups=2
        [batch, 8, seq, head_dim] → [batch, 16, seq, head_dim]
        每个 KV head 复制 2 次
        """
        batch, n_kv_heads, seq_len, head_dim = x.shape
        # 在 head 维度后插入新维度并展开
        x = x.unsqueeze(2)  # [batch, n_kv, 1, seq, head_dim]
        x = x.expand(
            -1, -1, self.n_kv_groups, -1, -1
        )  # [batch, n_kv, groups, seq, head_dim]
        x = x.reshape(batch, self.n_heads, seq_len, head_dim)
        return x


if __name__ == "__main__":
    # 测试 MHA 和 GQA
    print("=" * 50)
    print("标准 MHA (Small 配置)")
    print("=" * 50)
    config_mha = ModelConfig.small()  # n_heads=8, n_kv_heads=8
    attn_mha = Attention(config_mha)

    x = torch.randn(2, 10, config_mha.d_model)
    # 创建 causal mask
    mask = torch.full((config_mha.max_seq_len, config_mha.max_seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    y = attn_mha(x, mask)
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  Q params: {attn_mha.w_q.weight.shape}")
    print(f"  K params: {attn_mha.w_k.weight.shape}")
    print(f"  参数量: {sum(p.numel() for p in attn_mha.parameters()) / 1e6:.2f}M")

    print(f"\n{'=' * 50}")
    print("GQA (Medium 配置)")
    print("=" * 50)
    config_gqa = ModelConfig.medium()  # n_heads=16, n_kv_heads=8
    attn_gqa = Attention(config_gqa)

    x2 = torch.randn(2, 10, config_gqa.d_model)
    mask2 = torch.full((config_gqa.max_seq_len, config_gqa.max_seq_len), float("-inf"))
    mask2 = torch.triu(mask2, diagonal=1)

    y2 = attn_gqa(x2, mask2)
    print(f"  输入: {x2.shape}")
    print(f"  输出: {y2.shape}")
    print(f"  Q params: {attn_gqa.w_q.weight.shape} (16 heads)")
    print(f"  K params: {attn_gqa.w_k.weight.shape} (8 KV heads)")
    print(f"  GQA 节省 KV 参数: {1 - config_gqa.n_kv_heads / config_gqa.n_heads:.0%}")
