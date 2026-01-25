"""
推理优化
========

学习目标：
    1. 理解KV Cache原理
    2. 掌握推理优化技术
    3. 学会使用vLLM等推理框架
"""

import torch
import time


# ==================== 第一部分：KV Cache ====================


def kv_cache_explained():
    """KV Cache原理"""
    print("=" * 60)
    print("第一部分：KV Cache")
    print("=" * 60)

    print("""
自回归生成的问题：
    生成第n个token时，需要计算前n-1个token的K,V
    每次生成都重复计算，效率低

KV Cache解决方案：
    缓存之前计算过的K,V，避免重复计算

┌────────────────────────────────────────────────────────┐
│  Step 1: 输入 [A, B, C]                                │
│          计算并缓存 K=[kA,kB,kC], V=[vA,vB,vC]         │
│                                                         │
│  Step 2: 生成 D                                        │
│          只需计算 kD, vD                               │
│          使用缓存 K=[kA,kB,kC,kD]                      │
│                                                         │
│  Step 3: 生成 E                                        │
│          只需计算 kE, vE                               │
│          K=[kA,kB,kC,kD,kE]                            │
└────────────────────────────────────────────────────────┘

KV Cache显存：
    = batch × layers × 2 × seq × head_dim × heads × dtype
    7B模型, 2K序列: ~8GB
    """)


# ==================== 第二部分：推理优化技术 ====================


def optimization_techniques():
    """推理优化技术"""
    print("\n" + "=" * 60)
    print("第二部分：推理优化技术")
    print("=" * 60)

    print("""
主要优化技术：

1. Continuous Batching (动态批处理)
   - 请求完成立即释放，新请求立即加入
   - 避免等待最长序列完成
   - 提升吞吐量2-5倍

2. PagedAttention (vLLM)
   - 分页管理KV Cache
   - 减少显存碎片
   - 支持更多并发请求

3. Speculative Decoding (投机解码)
   - 小模型快速生成候选
   - 大模型批量验证
   - 加速2-3倍

4. Flash Attention
   - 减少显存访问
   - 加速2-4倍

5. 量化
   - INT8/INT4推理
   - 减少显存和计算
    """)


# ==================== 第三部分：vLLM使用 ====================


def vllm_usage():
    """vLLM使用"""
    print("\n" + "=" * 60)
    print("第三部分：vLLM推理框架")
    print("=" * 60)

    print("""
vLLM: 高性能LLM推理框架

安装: pip install vllm

1. 基础使用:

from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

outputs = llm.generate(
    ["你好，请介绍一下自己"],
    sampling_params
)

for output in outputs:
    print(output.outputs[0].text)

2. OpenAI兼容服务器:

# 启动服务
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 调用API
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct",
       "messages": [{"role": "user", "content": "你好"}]}'

3. 量化模型:

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq"
)
    """)


# ==================== 第四部分：性能对比 ====================


def performance_comparison():
    """性能对比"""
    print("\n" + "=" * 60)
    print("第四部分：推理框架性能对比")
    print("=" * 60)

    print("""
7B模型推理性能对比 (A100 40GB):

┌────────────────────────────────────────────────────────┐
│  框架              吞吐量(tokens/s)    延迟(首token)   │
├────────────────────────────────────────────────────────┤
│  HuggingFace       ~50                 ~500ms         │
│  vLLM              ~300                ~100ms         │
│  TensorRT-LLM      ~400                ~80ms          │
│  llama.cpp (CPU)   ~20                 ~1000ms        │
└────────────────────────────────────────────────────────┘

选择建议：
    - 快速原型: HuggingFace
    - 生产部署: vLLM / TensorRT-LLM
    - CPU推理: llama.cpp
    - 极致性能: TensorRT-LLM
    """)


def main():
    kv_cache_explained()
    optimization_techniques()
    vllm_usage()
    performance_comparison()

    print("\n" + "=" * 60)
    print("课程完成！下一步: 09-multimodal.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
