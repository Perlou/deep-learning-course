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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import precompute_rope_frequencies, apply_rope


class Attention(nn.Module):
    """Multi-Head Self-Attention with GQA, RoPE, KV Cache & Flash Attention

    内部流程:
      1. 线性投影: x → Q, K, V
      2. 分头: reshape 为 [batch, n_heads, seq, head_dim]
      3. 应用 RoPE: 对 Q, K 施加旋转位置编码
      4. KV Cache: 推理时缓存 K/V，避免重复计算 (可选)
      5. GQA 扩展: 将 KV heads 复制以匹配 Q heads 数量
      6. Flash Attention: F.scaled_dot_product_attention() (PyTorch 2.0+)
      7. 合并头: reshape 回 [batch, seq, d_model]
      8. 输出投影: W_o

    KV Cache 工作原理:
      - Prefill 阶段: 输入完整 prompt, 计算并缓存所有 K/V
      - Decode 阶段: 每步只输入 1 个新 token, 用新 K/V 拼接 cache
      - 效果: 避免每步重算整个序列, 推理速度提升 5-10x

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

        # Sliding Window Attention (可选)
        # 限制每个 token 只关注最近 window_size 个 token
        # None = 全局注意力, >0 = 滑动窗口
        self.sliding_window = config.sliding_window

        # 记录 max_seq_len：当 rope_cos/sin 没传入时（如独立单元测试），
        # forward 内 fallback 即时构造（仅用于测试 / 调试，生产路径走 GPT 顶层共享）
        self.max_seq_len = config.max_seq_len

        # ========== QK-Norm（Llama-3 / Gemma2 同款）==========
        # 在 Q 和 K 上各做一次 RMSNorm（在 head_dim 上），让 attention 分数更稳定，
        # 长训过程不容易出现 attention logit 爆炸。
        # 参数开销：仅 2 × head_dim 个可学习 scale（极小）。
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            from .normalization import RMSNorm

            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        # RoPE 频率：不再每层注册 buffer（n_layers 份重复），改由 GPT 顶层共享
        # （传入 forward 时通过参数 ``rope_cos`` / ``rope_sin``）
        # 这是 minimind / Qwen3 等主流实现的标准做法。

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x:         输入张量 [batch, seq_len, d_model]
            mask:      注意力掩码 [max_seq, max_seq] (causal mask)
            kv_cache:  缓存的 (K, V) 张量, 用于推理加速
            use_cache: 是否返回更新后的 KV cache

        Returns:
            output:       输出张量 [batch, seq_len, d_model]
            new_kv_cache: 更新后的 (K, V) cache (仅 use_cache=True 时)
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

        # ========== Step 2.5: QK-Norm（如启用）==========
        # 在 head_dim 上对 Q/K 做 RMSNorm，提高 attention 分数稳定性
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ========== Step 3: 应用 RoPE ==========
        # RoPE 的 position offset: 有 cache 时从 cache 长度开始编号
        if kv_cache is not None:
            # decode 阶段: 当前 token 的位置 = 已缓存的长度
            offset = kv_cache[0].shape[2]
        else:
            offset = 0
        # rope_cos/sin 由 GPT 顶层传入（顶层共享 buffer，避免 n_layers 份重复）
        # 当独立调用 Attention（单元测试）时 fallback 到即时构造
        if rope_cos is None or rope_sin is None:
            cos, sin = precompute_rope_frequencies(
                self.head_dim, self.max_seq_len, device=q.device
            )
            rope_cos, rope_sin = cos.to(q.dtype), sin.to(q.dtype)
        q = apply_rope(q, rope_cos, rope_sin, offset=offset)
        k = apply_rope(k, rope_cos, rope_sin, offset=offset)

        # ========== Step 4: KV Cache ==========
        if kv_cache is not None:
            # 将新的 K/V 拼接到缓存
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # 保存 cache (在 GQA 扩展之前, 节省内存)
        new_kv_cache = (k, v) if use_cache else None

        # ========== Step 5: GQA 扩展 ==========
        # 如果使用 GQA, 需要将 KV heads 复制到和 Q heads 相同数量
        if self.n_kv_groups > 1:
            k_expanded = self._expand_kv(k)  # [batch, n_heads, kv_len, head_dim]
            v_expanded = self._expand_kv(v)
        else:
            k_expanded = k
            v_expanded = v

        # ========== Step 6: Flash Attention ==========
        # 使用 PyTorch 2.0+ 的 F.scaled_dot_product_attention
        # 自动选择最优 kernel: FlashAttention-2 / Memory-Efficient / Math
        #
        # 训练时: is_causal=True 自动生成 causal mask
        # 推理 decode 时: Q 只有 1 个 token, 不需要 causal mask
        # 当传入 4D combined mask (causal + padding) 时使用 attn_mask 模式
        kv_len = k_expanded.shape[2]
        has_explicit_mask = mask is not None and mask.dim() == 4

        if seq_len == kv_len:
            # Prefill / 训练: 完整序列
            if has_explicit_mask:
                # 有显式 4D mask (含 padding mask) — 使用 attn_mask 模式
                attn_mask = mask[:, :, :seq_len, :kv_len] if mask.shape[-1] > kv_len else mask
                output = F.scaled_dot_product_attention(
                    q,
                    k_expanded,
                    v_expanded,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,
                )
            elif self.sliding_window is not None and self.sliding_window < seq_len:
                # Sliding Window Attention:
                # 每个 token 只看最近 window_size 个 token
                # 构造 mask: causal + sliding window
                row_idx = torch.arange(seq_len, device=q.device).unsqueeze(1)
                col_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)
                # causal: col <= row; window: col >= row - window + 1
                causal = col_idx <= row_idx
                window = col_idx >= (row_idx - self.sliding_window + 1)
                valid = causal & window
                # 转为 additive mask: True→0, False→-inf
                attn_mask = torch.where(
                    valid,
                    torch.tensor(0.0, device=q.device),
                    torch.tensor(float("-inf"), device=q.device),
                )
                output = F.scaled_dot_product_attention(
                    q,
                    k_expanded,
                    v_expanded,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,  # 已通过 attn_mask 处理
                )
            else:
                # 全局 causal attention
                output = F.scaled_dot_product_attention(
                    q,
                    k_expanded,
                    v_expanded,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True,
                )
        else:
            # Decode: Q 只有 1 个 token, 不需要 causal mask
            # SWA decode: 通过 KV Cache 自然限制窗口
            # (可选: 在 KV Cache 拼接后截断到 window_size)
            if self.sliding_window is not None and kv_len > self.sliding_window:
                # 截断 KV Cache 到最近的 window_size 个 token
                k_expanded = k_expanded[:, :, -self.sliding_window :, :]
                v_expanded = v_expanded[:, :, -self.sliding_window :, :]
            output = F.scaled_dot_product_attention(
                q,
                k_expanded,
                v_expanded,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        # output: [batch, n_heads, seq_len, head_dim]

        # ========== Step 7: 合并头 ==========
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        # [batch, seq, n_heads × head_dim]

        # ========== Step 8: 输出投影 ==========
        output = self.w_o(output)
        output = self.resid_dropout(output)

        return output, new_kv_cache

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
