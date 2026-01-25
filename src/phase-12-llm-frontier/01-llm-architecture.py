"""
LLM 架构理解
============

学习目标：
    1. 理解现代LLM的核心架构组件
    2. 掌握Decoder-Only架构的工作原理
    3. 了解RMSNorm、SwiGLU等关键改进
    4. 实现简化版的LLM架构

核心概念：
    - Decoder-Only: 自回归生成架构
    - RMSNorm: 更高效的归一化
    - SwiGLU: 门控前馈网络
    - Causal Attention: 因果注意力掩码

前置知识：
    - Phase 7: Transformer基础
    - Phase 11: NLP基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 第一部分：LLM架构概述 ====================


def introduction():
    """LLM架构介绍"""
    print("=" * 60)
    print("第一部分：大语言模型架构概述")
    print("=" * 60)

    print("""
现代大语言模型（LLM）的核心架构：

┌─────────────────────────────────────────────────────────────┐
│                  Decoder-Only Architecture                   │
│                                                              │
│  输入: [BOS, token1, token2, ..., tokenN]                   │
│                     ↓                                        │
│  ┌─────────────────────────────────────────┐                │
│  │         Embedding Layer                  │                │
│  │    Token Embedding + Position Info       │                │
│  └─────────────────────────────────────────┘                │
│                     ↓                                        │
│  ┌─────────────────────────────────────────┐                │
│  │      Transformer Block × N              │                │
│  │  ┌──────────────────────────────────┐   │                │
│  │  │  RMSNorm                          │   │                │
│  │  │  Causal Self-Attention (+ RoPE)   │   │                │
│  │  │  Residual Connection              │   │                │
│  │  │  RMSNorm                          │   │                │
│  │  │  SwiGLU FFN                       │   │                │
│  │  │  Residual Connection              │   │                │
│  │  └──────────────────────────────────┘   │                │
│  └─────────────────────────────────────────┘                │
│                     ↓                                        │
│  ┌─────────────────────────────────────────┐                │
│  │      RMSNorm + LM Head                   │                │
│  │      → Vocabulary Logits                 │                │
│  └─────────────────────────────────────────┘                │
│                     ↓                                        │
│  输出: P(next_token | previous_tokens)                      │
└─────────────────────────────────────────────────────────────┘

为什么选择Decoder-Only架构？
1. 统一的自回归训练目标，简单高效
2. 强大的生成能力
3. 涌现能力通常需要大规模参数
4. 实践证明效果最好
    """)


# ==================== 第二部分：RMSNorm ====================


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    相比LayerNorm：
    - 不计算均值，只计算RMS（均方根）
    - 计算量更小，效果相当
    - LLaMA、Qwen等模型使用
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return x / rms * self.weight


def rmsnorm_examples():
    """RMSNorm示例"""
    print("\n" + "=" * 60)
    print("第二部分：RMSNorm vs LayerNorm")
    print("=" * 60)
    
    # 创建测试数据
    batch_size, seq_len, hidden_size = 2, 4, 8
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # RMSNorm
    rmsnorm = RMSNorm(hidden_size)
    rms_out = rmsnorm(x)
    
    # LayerNorm
    layernorm = nn.LayerNorm(hidden_size)
    ln_out = layernorm(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"RMSNorm 输出形状: {rms_out.shape}")
    print(f"LayerNorm 输出形状: {ln_out.shape}")
    
    print("\nRMSNorm 公式:")
    print("  RMS(x) = √(mean(x²) + ε)")
    print("  output = x / RMS(x) × γ")
    
    print("\nLayerNorm 公式:")
    print("  output = (x - μ) / σ × γ + β")
    print("  需要计算均值μ和标准差σ")
    
    print("\nRMSNorm优势:")
    print("  - 计算量减少约7-15%")
    print("  - 没有偏置项，参数更少")
    print("  - 实验表明效果相当")


# ==================== 第三部分：SwiGLU FFN ====================


class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit
    
    原始FFN: GELU(xW₁)W₂
    SwiGLU: (Swish(xW₁) ⊙ xV)W₂
    
    引入门控机制，效果更好
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # 门控投影
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 上投影
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 下投影
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish激活 + 门控
        gate = F.silu(self.gate_proj(x))  # Swish = x * sigmoid(x)
        up = self.up_proj(x)
        # 门控乘法
        hidden = gate * up
        # 下投影
        return self.down_proj(hidden)


def swiglu_examples():
    """SwiGLU示例"""
    print("\n" + "=" * 60)
    print("第三部分：SwiGLU 前馈网络")
    print("=" * 60)
    
    hidden_size = 256
    # LLaMA中intermediate_size通常是hidden_size的8/3倍
    intermediate_size = int(hidden_size * 8 / 3)
    intermediate_size = ((intermediate_size + 63) // 64) * 64  # 对齐到64
    
    print(f"\nhidden_size: {hidden_size}")
    print(f"intermediate_size: {intermediate_size}")
    
    # 创建SwiGLU
    swiglu = SwiGLU(hidden_size, intermediate_size)
    
    # 计算参数量
    params = sum(p.numel() for p in swiglu.parameters())
    print(f"SwiGLU参数量: {params:,}")
    
    # 对比传统FFN
    traditional_ffn = nn.Sequential(
        nn.Linear(hidden_size, intermediate_size * 2 // 3),
        nn.GELU(),
        nn.Linear(intermediate_size * 2 // 3, hidden_size)
    )
    trad_params = sum(p.numel() for p in traditional_ffn.parameters())
    print(f"传统FFN参数量: {trad_params:,}")
    
    print("\nSwiGLU公式:")
    print("  gate = Swish(x × Wgate)")
    print("  up = x × Wup")
    print("  out = (gate ⊙ up) × Wdown")
    
    print("\nSwish激活函数:")
    print("  Swish(x) = x × sigmoid(x)")
    print("  也叫SiLU (Sigmoid Linear Unit)")


# ==================== 第四部分：因果注意力 ====================


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）
    
    使用下三角掩码，确保token只能看到之前的token
    """
    
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # QKV投影
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 注册因果掩码
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer("causal_mask", mask.bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式: (batch, seq, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # 应用因果掩码
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attn_weights, v)
        
        # 重塑并投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        return output


def causal_attention_examples():
    """因果注意力示例"""
    print("\n" + "=" * 60)
    print("第四部分：因果自注意力")
    print("=" * 60)
    
    print("""
因果掩码（Causal Mask）：

Token:    [A] [B] [C] [D]
           ↓   ↓   ↓   ↓
A能看到:  [A] [×] [×] [×]
B能看到:  [A] [B] [×] [×]
C能看到:  [A] [B] [C] [×]
D能看到:  [A] [B] [C] [D]

掩码矩阵（上三角为-inf）：
     A    B    C    D
A  [ 0, -∞, -∞, -∞]
B  [ 0,  0, -∞, -∞]
C  [ 0,  0,  0, -∞]
D  [ 0,  0,  0,  0]
    """)
    
    # 演示
    hidden_size, num_heads = 64, 4
    attention = CausalSelfAttention(hidden_size, num_heads)
    
    x = torch.randn(2, 8, hidden_size)
    output = attention(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力头数: {num_heads}")
    print(f"每头维度: {hidden_size // num_heads}")


# ==================== 第五部分：完整LLM Block ====================


class LLMBlock(nn.Module):
    """
    单个Transformer Block（LLaMA风格）
    
    Pre-Norm架构：
    1. RMSNorm → Attention → Residual
    2. RMSNorm → SwiGLU → Residual
    """
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        
        # 注意力相关
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size, num_heads)
        
        # FFN相关
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, intermediate_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + Attention + Residual
        h = x + self.attention(self.attention_norm(x))
        # Pre-Norm + FFN + Residual
        out = h + self.ffn(self.ffn_norm(h))
        return out


class SimpleLLM(nn.Module):
    """
    简化版LLM模型
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Token嵌入
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            LLMBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        # 最终归一化
        self.norm = RMSNorm(hidden_size)
        
        # 语言模型头
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 通过所有Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 计算logits
        logits = self.lm_head(hidden_states)
        
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0):
        """简单的自回归生成"""
        for _ in range(max_new_tokens):
            # 获取logits
            logits = self(input_ids)
            # 只看最后一个token
            next_token_logits = logits[:, -1, :] / temperature
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def llm_architecture_demo():
    """LLM架构演示"""
    print("\n" + "=" * 60)
    print("第五部分：完整LLM架构演示")
    print("=" * 60)
    
    # 模型配置（迷你版）
    config = {
        "vocab_size": 32000,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "intermediate_size": 512,
        "max_seq_len": 512
    }
    
    print("\n模型配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 创建模型
    model = SimpleLLM(**config)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 分解参数
    embed_params = model.embed_tokens.weight.numel()
    lm_head_params = model.lm_head.weight.numel()
    layer_params = sum(p.numel() for p in model.layers.parameters())
    
    print(f"嵌入层参数: {embed_params:,}")
    print(f"Transformer层参数: {layer_params:,}")
    print(f"LM Head参数: {lm_head_params:,}")
    
    # 测试前向传播
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"\n输入形状: {input_ids.shape}")
    print(f"输出logits形状: {logits.shape}")
    
    # 测试生成
    print("\n测试自回归生成...")
    prompt = torch.randint(0, config["vocab_size"], (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"Prompt长度: {prompt.shape[1]}")
    print(f"生成后长度: {generated.shape[1]}")


# ==================== 第六部分：练习 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：实现Grouped-Query Attention
    任务：修改CausalSelfAttention，实现GQA
    提示：多个Q head共享一对K,V

练习 1 答案：
    class GroupedQueryAttention(nn.Module):
        def __init__(self, hidden_size, num_heads, num_kv_heads):
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.num_groups = num_heads // num_kv_heads
            self.head_dim = hidden_size // num_heads
            
            self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
            
        def forward(self, x):
            B, L, _ = x.shape
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
            
            # 扩展K,V以匹配Q的head数量
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)
            # 继续计算attention...

练习 2：比较模型配置
    任务：计算不同规模LLM的参数量
    7B: hidden=4096, layers=32, heads=32
    13B: hidden=5120, layers=40, heads=40
    
    参数量 ≈ 12 × hidden² × layers (简化估计)

练习 2 答案：
    def estimate_params(hidden, layers):
        # 嵌入: vocab * hidden ≈ 32000 * hidden
        # 每层: 12 * hidden^2 (attention + ffn)
        return 32000 * hidden + 12 * hidden**2 * layers
    
    print(f"7B估计: {estimate_params(4096, 32)/1e9:.1f}B")
    print(f"13B估计: {estimate_params(5120, 40)/1e9:.1f}B")

思考题：为什么使用Pre-Norm而不是Post-Norm？
    答案：
    - Pre-Norm训练更稳定，梯度流动更顺畅
    - 可以训练更深的网络
    - Post-Norm在深层网络中容易梯度消失
    """)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    rmsnorm_examples()
    swiglu_examples()
    causal_attention_examples()
    llm_architecture_demo()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 02-tokenization-advanced.py: 分词器深入解析
    
关键要点回顾：
    ✓ Decoder-Only是主流LLM架构
    ✓ RMSNorm比LayerNorm更高效
    ✓ SwiGLU引入门控机制提升效果
    ✓ 因果掩码确保自回归特性
    ✓ Pre-Norm架构训练更稳定
    """)


if __name__ == "__main__":
    main()
