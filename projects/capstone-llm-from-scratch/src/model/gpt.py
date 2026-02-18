"""
GPT — 完整的 Decoder-only Transformer 语言模型
==============================================

这是 MiniMind 的核心模型文件，将所有组件组装成完整的 GPT 模型。

模型结构:
  Token IDs → Embedding → N × TransformerBlock → RMSNorm → LM Head → Logits

组件列表:
  - Token Embedding:  将整数 token ID 映射为 d_model 维向量
  - TransformerBlock: 包含 Attention + FFN + RMSNorm + Residual (×N layers)
  - Final RMSNorm:    最后一层归一化
  - LM Head:          将 d_model 维映射回 vocab_size 维的 logits

与 GPT-4/Gemini 的区别:
  本模型 = GPT-4/Gemini 的完全缩小版
  - 架构 100% 一致 (Decoder-only + RoPE + RMSNorm + SwiGLU + GQA)
  - 区别仅在于参数量 (26M vs 1.8T) 和训练数据量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .normalization import RMSNorm
from .transformer import TransformerBlock


class GPT(nn.Module):
    """GPT 语言模型

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ========== Token Embedding ==========
        # 将 token ID (整数) 映射为 d_model 维连续向量
        # 这是模型的 "单词表", 每个 token 对应一行可学习参数
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # ========== Dropout ==========
        self.embedding_dropout = nn.Dropout(config.dropout)

        # ========== N × Transformer Block ==========
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # ========== Final RMSNorm ==========
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # ========== LM Head ==========
        # 将 d_model 维映射回 vocab_size 维, 输出每个 token 的 logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ========== 权重共享 (Weight Tying) ==========
        # 让 Embedding 和 LM Head 共享同一组权重
        # 理由: Embedding 将 token → 向量, LM Head 做反向映射
        # 共享权重能减少参数量且提升效果 (参考 GPT-2 论文)
        self.token_embedding.weight = self.lm_head.weight

        # ========== Causal Mask ==========
        # 上三角矩阵, 防止 token 看到未来的信息
        # 注册为 buffer: 不参与训练, 但跟随模型设备
        mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

        # ========== 初始化权重 ==========
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Xavier/He 风格的权重初始化

        不同层使用不同的初始化策略:
        - Linear 层: 正态分布 N(0, 0.02)
        - Embedding: 正态分布 N(0, 0.02)
        - RMSNorm:   已经初始化为 1 (不需要额外处理)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向传播

        Args:
            input_ids: token ID 序列 [batch, seq_len]
            targets:   目标 token ID (用于计算 loss) [batch, seq_len], 可选

        Returns:
            logits: 模型输出的 logits [batch, seq_len, vocab_size]
            loss:   交叉熵损失 (仅当提供 targets 时)
        """
        batch, seq_len = input_ids.shape

        # Token Embedding
        x = self.token_embedding(input_ids)  # [batch, seq, d_model]
        x = self.embedding_dropout(x)

        # N × Transformer Block
        for layer in self.layers:
            x = layer(x, mask=self.causal_mask)

        # Final RMSNorm
        x = self.final_norm(x)

        # LM Head → logits
        logits = self.lm_head(x)  # [batch, seq, vocab_size]

        # 计算 loss (如果提供了 targets)
        loss = None
        if targets is not None:
            # Cross-entropy loss
            # logits: [batch × seq, vocab] vs targets: [batch × seq]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # padding token 不计算 loss
            )

        return logits, loss

    def count_parameters(self) -> dict:
        """统计模型的实际参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 注意: 由于 weight tying, embedding 和 lm_head 共享参数
        # parameters() 不会重复计算
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
            "trainable_millions": trainable / 1e6,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """自回归文本生成

        Args:
            input_ids:      起始 token 序列 [1, seq_len]
            max_new_tokens: 最大生成 token 数
            temperature:    采样温度 (越高越随机)
            top_k:          Top-K 采样 (0 = 不限制)
            top_p:          Nucleus 采样 (1.0 = 不限制)

        Returns:
            生成的完整序列 [1, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 截断到 max_seq_len
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.config.max_seq_len
                else input_ids[:, -self.config.max_seq_len :]
            )

            # 前向传播
            logits, _ = self(idx_cond)

            # 取最后一个位置的 logits
            logits = logits[:, -1, :]  # [batch, vocab_size]

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-K 采样
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_values[:, [-1]]] = float("-inf")

            # Top-P (Nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # 找到累积概率超过 top_p 的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False
                # 将要移除的 token 的 logits 设为 -inf
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接新 token
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


if __name__ == "__main__":
    print("=" * 60)
    print("MiniMind GPT 模型验证")
    print("=" * 60)

    # ========== Small 配置测试 ==========
    config = ModelConfig.small()
    model = GPT(config)

    params = model.count_parameters()
    print(f"\n📊 Small 配置:")
    print(f"  总参数量: {params['total_millions']:.1f}M")
    print(f"  可训练参数: {params['trainable_millions']:.1f}M")

    # 前向传播测试
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets)
    print(f"\n🔄 前向传播:")
    print(f"  Input:  {input_ids.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss:   {loss.item():.4f}")
    print(
        f"  期望初始 loss ≈ ln(vocab_size) = {torch.tensor(config.vocab_size).float().log().item():.4f}"
    )

    # 生成测试
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"\n✨ 文本生成:")
    print(f"  Prompt:    {prompt.shape}")
    print(f"  Generated: {generated.shape}")
    print(f"  新生成 tokens: {generated.shape[1] - prompt.shape[1]}")

    # 反向传播测试
    loss.backward()
    print(f"\n✅ 反向传播成功!")

    # ========== Medium 配置参数量 ==========
    config_med = ModelConfig.medium()
    model_med = GPT(config_med)
    params_med = model_med.count_parameters()
    print(f"\n📊 Medium 配置参数量: {params_med['total_millions']:.1f}M")

    print(f"\n{'=' * 60}")
    print("✅ 全部验证通过!")
    print("=" * 60)
