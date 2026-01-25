"""
Flash Attention 原理
===================

学习目标：
    1. 理解标准Attention的显存问题
    2. 了解Flash Attention的核心思想
    3. 理解分块计算和在线softmax
    4. 学会使用Flash Attention库

核心概念：
    - IO感知: 减少GPU显存访问
    - Tiling: 分块计算
    - Recomputation: 反向传播时重计算
    - Online Softmax: 分块softmax算法

前置知识：
    - Phase 7: Transformer注意力机制
    - 基本的GPU内存层次理解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ==================== 第一部分：注意力机制回顾 ====================


def introduction():
    """Flash Attention介绍"""
    print("=" * 60)
    print("第一部分：标准Attention的问题")
    print("=" * 60)

    print("""
标准Self-Attention计算：

    Attention(Q, K, V) = softmax(QK^T / √d) × V
    
    步骤：
    1. 计算 S = QK^T          O(n²d) 时间, O(n²) 空间
    2. 计算 P = softmax(S)    O(n²) 时间
    3. 计算 O = PV            O(n²d) 时间

显存问题：
┌─────────────────────────────────────────────────────────────┐
│  序列长度n    注意力矩阵大小    显存占用(FP16)                │
├─────────────────────────────────────────────────────────────┤
│    512         512×512          512 KB                      │
│   2048        2048×2048          8 MB                       │
│   4096        4096×4096         32 MB                       │
│   8192        8192×8192        128 MB                       │
│  32768      32768×32768          2 GB                       │
│ 131072     131072×131072        32 GB                       │
└─────────────────────────────────────────────────────────────┘

注意：这只是单个注意力头、单层的开销！
多头、多层会成倍增加。

GPU内存层次：
┌─────────────────────────────────────────────────────────────┐
│  层次          容量            带宽                         │
├─────────────────────────────────────────────────────────────┤
│  SRAM (片上)   ~20 MB          ~19 TB/s                     │
│  HBM (显存)    ~80 GB          ~2 TB/s                      │
│  主存          ~1 TB           ~50 GB/s                     │
└─────────────────────────────────────────────────────────────┘

瓶颈不是计算，而是内存带宽！
标准Attention需要反复读写HBM，成为瓶颈。
    """)


# ==================== 第二部分：Flash Attention核心思想 ====================


def flash_attention_concept():
    """Flash Attention核心概念"""
    print("\n" + "=" * 60)
    print("第二部分：Flash Attention核心思想")
    print("=" * 60)

    print("""
Flash Attention的核心思想：

1. 分块计算 (Tiling)
   - 将Q, K, V分成小块
   - 每个块可以放入SRAM
   - 在SRAM中完成大部分计算
   
2. 避免存储完整注意力矩阵
   - 不需要把n×n矩阵写入HBM
   - 直接计算输出，节省显存
   
3. 在线Softmax算法
   - 分块计算softmax
   - 动态维护最大值和归一化因子
   
┌─────────────────────────────────────────────────────────────┐
│                Flash Attention 工作流程                      │
│                                                              │
│   Q     K     V                                              │
│   │     │     │                                              │
│   ↓     ↓     ↓                                              │
│  [Q1]  [K1]  [V1]   ←── 分块                                 │
│  [Q2]  [K2]  [V2]                                            │
│  ...   ...   ...                                             │
│                                                              │
│   对于每个Q块：                                               │
│     遍历所有K, V块：                                          │
│       在SRAM中计算局部注意力                                  │
│       更新运行最大值和输出                                    │
│     写回最终输出到HBM                                        │
│                                                              │
│   显存: O(n²) → O(n)                                         │
│   速度: 2-4x 加速                                            │
└─────────────────────────────────────────────────────────────┘
    """)


# ==================== 第三部分：在线Softmax算法 ====================


def online_softmax():
    """在线Softmax算法详解"""
    print("\n" + "=" * 60)
    print("第三部分：在线Softmax算法")
    print("=" * 60)

    print("""
标准Softmax需要两遍扫描：
    遍1: 找最大值 m = max(x)
    遍2: 计算 exp(x - m) / sum(exp(x - m))
    
在线Softmax只需一遍扫描：
    维护运行最大值和运行和
    遇到新块时动态更新
    """)

    # 演示在线softmax
    print("\n演示在线Softmax:")

    # 模拟分块数据
    blocks = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0]),
        torch.tensor([2.0, 1.0]),
    ]

    # 标准方法
    full_data = torch.cat(blocks)
    standard_result = F.softmax(full_data, dim=0)
    print(f"\n完整数据: {full_data.tolist()}")
    print(f"标准Softmax: {standard_result.tolist()}")

    # 在线方法
    running_max = float("-inf")
    running_sum = 0.0

    for i, block in enumerate(blocks):
        block_max = block.max().item()
        if block_max > running_max:
            # 更新之前的sum
            running_sum = running_sum * math.exp(running_max - block_max)
            running_max = block_max

        # 累加当前块
        block_sum = torch.sum(torch.exp(block - running_max)).item()
        running_sum += block_sum

        print(f"Block {i + 1}: max={running_max:.1f}, sum={running_sum:.4f}")

    # 最终计算
    online_result = torch.exp(full_data - running_max) / running_sum
    print(f"\n在线Softmax: {online_result.tolist()}")
    print(f"结果一致: {torch.allclose(standard_result, online_result)}")


# ==================== 第四部分：简化版Flash Attention实现 ====================


def simple_flash_attention(Q, K, V, block_size=64):
    """
    简化版Flash Attention实现（仅用于理解原理）

    注意：这不是高效实现，真实的Flash Attention需要CUDA kernel
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape

    # 输出和中间变量
    O = torch.zeros_like(Q)
    L = torch.zeros(batch_size, num_heads, seq_len, 1, device=Q.device)  # 归一化因子
    M = torch.full(
        (batch_size, num_heads, seq_len, 1), float("-inf"), device=Q.device
    )  # 最大值

    # 分块数量
    num_blocks = (seq_len + block_size - 1) // block_size

    for i in range(num_blocks):
        # Q块
        q_start = i * block_size
        q_end = min((i + 1) * block_size, seq_len)
        Qi = Q[:, :, q_start:q_end, :]

        for j in range(num_blocks):
            # K, V块
            kv_start = j * block_size
            kv_end = min((j + 1) * block_size, seq_len)
            Kj = K[:, :, kv_start:kv_end, :]
            Vj = V[:, :, kv_start:kv_end, :]

            # 计算局部注意力分数
            scale = math.sqrt(head_dim)
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) / scale

            # 在线softmax更新
            Mi = M[:, :, q_start:q_end, :]
            Li = L[:, :, q_start:q_end, :]
            Oi = O[:, :, q_start:q_end, :]

            # 新的最大值
            Mij = Sij.max(dim=-1, keepdim=True).values
            Mnew = torch.maximum(Mi, Mij)

            # 更新归一化因子
            exp_Mi = torch.exp(Mi - Mnew)
            exp_Mij = torch.exp(Sij - Mnew)
            Lnew = exp_Mi * Li + exp_Mij.sum(dim=-1, keepdim=True)

            # 更新输出
            Pij = exp_Mij  # 未归一化的attention
            Onew = exp_Mi * Oi + torch.matmul(Pij, Vj)

            # 保存
            M[:, :, q_start:q_end, :] = Mnew
            L[:, :, q_start:q_end, :] = Lnew
            O[:, :, q_start:q_end, :] = Onew

    # 最终归一化
    O = O / L
    return O


def flash_attention_demo():
    """Flash Attention演示"""
    print("\n" + "=" * 60)
    print("第四部分：Flash Attention实现演示")
    print("=" * 60)

    # 小规模测试
    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 32

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"输入形状: ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

    # 标准attention
    def standard_attention(Q, K, V):
        scale = math.sqrt(Q.shape[-1])
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    standard_out = standard_attention(Q, K, V)
    flash_out = simple_flash_attention(Q, K, V, block_size=32)

    # 比较结果
    max_diff = (standard_out - flash_out).abs().max().item()
    print(f"\n标准Attention输出形状: {standard_out.shape}")
    print(f"Flash Attention输出形状: {flash_out.shape}")
    print(f"最大差异: {max_diff:.6f}")
    print(f"结果一致: {max_diff < 1e-5}")


# ==================== 第五部分：使用Flash Attention库 ====================


def flash_attention_library():
    """Flash Attention库使用"""
    print("\n" + "=" * 60)
    print("第五部分：Flash Attention库使用")
    print("=" * 60)

    print("""
Flash Attention库安装和使用：

1. 官方Flash Attention库
   pip install flash-attn --no-build-isolation
   
   # 需要CUDA 11.6+, PyTorch 1.12+
   # 编译可能需要较长时间
    """)

    code_example = """
# 使用flash-attn库
from flash_attn import flash_attn_func

# 输入形状: (batch, seqlen, nheads, headdim)
q = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)

# 调用Flash Attention
output = flash_attn_func(q, k, v, causal=True)
print(f"Output shape: {output.shape}")
"""
    print(code_example)

    print("""
2. PyTorch内置（2.0+）
    """)

    pytorch_example = """
import torch.nn.functional as F

# PyTorch 2.0+ 自动使用Flash Attention
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,       # Flash Attention
    enable_math=False,       # 禁用标准实现
    enable_mem_efficient=True # 内存高效实现
):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
"""
    print(pytorch_example)

    print("""
3. Hugging Face Transformers
    """)

    hf_example = """
from transformers import AutoModelForCausalLM

# 自动使用Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # 显式指定
)
"""
    print(hf_example)


# ==================== 第六部分：性能对比 ====================


def performance_comparison():
    """性能对比"""
    print("\n" + "=" * 60)
    print("第六部分：性能对比")
    print("=" * 60)

    print("""
Flash Attention性能提升：

┌─────────────────────────────────────────────────────────────┐
│  序列长度   标准Attention   Flash Attention   加速比        │
├─────────────────────────────────────────────────────────────┤
│    512         1.0x            1.0x           ~1x          │
│   1024         1.0x            0.5x           ~2x          │
│   2048         1.0x            0.35x          ~3x          │
│   4096         OOM             0.25x          ∞            │
│   8192         OOM             0.5x           ∞            │
└─────────────────────────────────────────────────────────────┘

显存对比：

┌─────────────────────────────────────────────────────────────┐
│  序列长度   标准Attention   Flash Attention   节省          │
├─────────────────────────────────────────────────────────────┤
│   2048         8 MB           ~0.5 MB        16x           │
│   4096        32 MB           ~1 MB          32x           │
│   8192       128 MB           ~2 MB          64x           │
│  16384       512 MB           ~4 MB         128x           │
└─────────────────────────────────────────────────────────────┘

实际影响：
- 可以训练更长的上下文
- 可以使用更大的batch size
- 推理速度显著提升
    """)


# ==================== 第七部分：练习 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：计算注意力矩阵的显存需求
    任务：计算给定配置下attention矩阵的显存
    配置：seq_len=8192, batch=4, heads=32, FP16

练习 1 答案：
    # 每个注意力头的矩阵: 8192 × 8192 = 67M 元素
    # 32个头: 67M × 32 = 2.1B 元素
    # batch=4: 2.1B × 4 = 8.6B 元素
    # FP16 (2字节): 8.6B × 2 = 17.2 GB
    
    def calc_attention_memory(seq_len, batch, heads, dtype_bytes=2):
        elements = seq_len * seq_len * heads * batch
        bytes_needed = elements * dtype_bytes
        return bytes_needed / (1024**3)  # GB
    
    print(f"{calc_attention_memory(8192, 4, 32):.1f} GB")

练习 2：实现因果掩码的分块处理
    任务：在分块attention中正确处理因果掩码
    提示：只需在对角线块和上三角块做特殊处理

练习 2 答案：
    def causal_block_attention(Qi, Kj, i, j, block_size):
        '''
        i, j: 块索引
        因果掩码：位置p只能看到位置<=p的token
        '''
        if j > i:
            # 上三角块：全部mask掉
            return None
        elif j == i:
            # 对角线块：部分mask
            q_len = Qi.shape[-2]
            k_len = Kj.shape[-2]
            mask = torch.triu(torch.ones(q_len, k_len), diagonal=1)
            scores = Qi @ Kj.T
            scores.masked_fill_(mask.bool(), float('-inf'))
            return scores
        else:
            # 下三角块：无需mask
            return Qi @ Kj.T

思考题：Flash Attention的局限性？
    答案：
    1. 需要CUDA支持，CPU/其他硬件不可用
    2. 实现复杂，需要精细的CUDA编程
    3. 某些变体（如相对位置编码）需要特殊处理
    4. 调试困难，不如标准实现直观
    """)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    flash_attention_concept()
    online_softmax()
    flash_attention_demo()
    flash_attention_library()
    performance_comparison()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 04-pre-training-basics.py: 预训练基础
    
关键要点回顾：
    ✓ 标准Attention的显存复杂度是O(n²)
    ✓ Flash Attention通过分块计算降至O(n)
    ✓ 在线Softmax算法是关键技术
    ✓ PyTorch 2.0+内置Flash Attention支持
    ✓ 实际可实现2-4倍加速
    """)


if __name__ == "__main__":
    main()
