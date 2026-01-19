"""
扩散模型理论 (DDPM)
===================

学习目标：
    1. 理解扩散模型的核心思想
    2. 掌握前向扩散过程和反向去噪过程
    3. 理解 DDPM 的数学原理
    4. 了解噪声预测网络的训练目标

核心概念：
    - 前向过程: 逐步添加高斯噪声
    - 反向过程: 逐步去噪恢复数据
    - 噪声调度: 控制加噪强度的 β 序列
    - 噪声预测: 预测当前时刻的噪声

前置知识：
    - 05-vae.py: VAE 基础
    - 概率论（高斯分布、马尔可夫链）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


# ==================== 第一部分：扩散模型概述 ====================


def introduction():
    """
    扩散模型概述
    """
    print("=" * 60)
    print("第一部分：扩散模型概述")
    print("=" * 60)

    print("""
什么是扩散模型？

    扩散模型通过两个过程生成数据：
    1. 前向过程：逐步向数据添加噪声，直到变成纯噪声
    2. 反向过程：学习如何逐步去噪，从纯噪声恢复数据

可视化：
    ┌─────────────────────────────────────────────────────────┐
    │                    前向过程 (加噪)                       │
    │                                                         │
    │   x₀ ──→ x₁ ──→ x₂ ──→ ... ──→ xₜ ──→ ... ──→ xₜ       │
    │   🖼️     🖼️+    🖼️++        📊        📊📊    📊📊📊    │
    │  清晰   轻微   更多          混合              纯噪声    │
    │         噪声   噪声                                     │
    │                                                         │
    │   每一步：x_t = √(1-βₜ)x_{t-1} + √βₜ ε                  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │                    反向过程 (去噪)                       │
    │                                                         │
    │   xₜ ──→ xₜ₋₁ ──→ xₜ₋₂ ──→ ... ──→ x₁ ──→ x₀           │
    │  📊📊📊   📊📊     📊      🖼️++   🖼️+    🖼️            │
    │  纯噪声                               轻微   清晰        │
    │                                       噪声               │
    │                                                         │
    │   每一步：神经网络学习 p(x_{t-1}|xₜ)                     │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

为什么扩散模型有效？

    1. 将复杂的生成问题分解为简单的去噪步骤
       - 每一步只需要去除一点点噪声
       - 比直接从噪声生成图像容易得多

    2. 理论基础扎实
       - 基于热力学和概率论
       - 有明确的似然函数

    3. 训练稳定
       - 没有 GAN 的对抗训练问题
       - 没有 VAE 的后验崩塌问题

扩散模型的发展：
    ┌─────────────────────────────────────────────────────────┐
    │ 2015: 扩散概率模型 (DPM) - Sohl-Dickstein et al.        │
    │ 2020: DDPM - Ho et al. ← 本课重点                       │
    │ 2021: Score-based models - Song et al.                  │
    │ 2022: DALL-E 2, Stable Diffusion, Midjourney            │
    │ 2024: Sora（视频生成）                                   │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：前向扩散过程 ====================


def forward_process_demo():
    """前向扩散过程演示"""
    print("\n" + "=" * 60)
    print("第二部分：前向扩散过程")
    print("=" * 60)

    print("""
前向过程（加噪过程）：

    定义：给定初始数据 x₀，逐步添加高斯噪声

    q(xₜ|x_{t-1}) = N(xₜ; √(1-βₜ)x_{t-1}, βₜI)

    其中 βₜ 是噪声调度（0 < β₁ < β₂ < ... < βₜ < 1）

重要性质：可以直接从 x₀ 计算任意 xₜ！

    定义：αₜ = 1 - βₜ
          ᾱₜ = α₁ × α₂ × ... × αₜ (累积乘积)

    那么：q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)

    即：xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε,  ε ~ N(0, I)

    这允许我们在训练时直接采样任意时间步！
    """)

    # 演示前向过程
    print("\n示例: 一维数据的前向扩散\n")

    # 创建简单的 1D 数据
    x_0 = 3.0
    T = 1000  # 总时间步

    # 线性噪声调度
    beta = np.linspace(0.0001, 0.02, T)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)

    # 计算不同时间步的噪声版本
    timesteps = [0, 100, 300, 500, 700, 900, 999]

    print(f"初始值 x₀ = {x_0}")
    print(f"\n时间步 | √ᾱₜ  | √(1-ᾱₜ) | 噪声后的值 (期望)")
    print("-" * 50)

    for t in timesteps:
        sqrt_alpha_bar = np.sqrt(alpha_bar[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar[t])
        expected_mean = sqrt_alpha_bar * x_0

        print(
            f"{t:5d}   | {sqrt_alpha_bar:.4f} | {sqrt_one_minus_alpha_bar:.4f}  | {expected_mean:.4f}"
        )

    # 可视化
    plt.figure(figsize=(12, 4))

    # 1. αbar 曲线
    plt.subplot(1, 2, 1)
    plt.plot(alpha_bar)
    plt.xlabel("时间步 t")
    plt.ylabel("ᾱₜ")
    plt.title("累积系数 ᾱₜ")
    plt.grid(True)

    # 2. 前向扩散过程可视化
    plt.subplot(1, 2, 2)

    # 生成一些样本
    np.random.seed(42)
    num_samples = 100
    x_0_samples = np.random.normal(3, 0.5, num_samples)

    for t in [0, 250, 500, 750, 999]:
        sqrt_alpha_bar = np.sqrt(alpha_bar[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar[t])
        noise = np.random.randn(num_samples)
        x_t = sqrt_alpha_bar * x_0_samples + sqrt_one_minus_alpha_bar * noise
        plt.hist(x_t, bins=30, alpha=0.5, label=f"t={t}", density=True)

    plt.xlabel("值")
    plt.ylabel("密度")
    plt.title("前向扩散：分布变化")
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "diffusion_forward.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 前向过程可视化已保存到: {output_path}")

    print("""
观察：
    - 随着 t 增加，ᾱₜ 趋近于 0
    - xₜ 逐渐失去原始信息，趋近于标准正态分布
    - 在 t=T 时，数据几乎变成纯噪声
    """)


# ==================== 第三部分：反向去噪过程 ====================


def reverse_process_demo():
    """反向去噪过程演示"""
    print("\n" + "=" * 60)
    print("第三部分：反向去噪过程")
    print("=" * 60)

    print("""
反向过程（去噪过程）：

    目标：学习反向转移概率 p_θ(x_{t-1}|xₜ)

    根据贝叶斯公式：
    p(x_{t-1}|xₜ, x₀) = N(x_{t-1}; μ̃ₜ, σ̃ₜ²I)

    其中后验均值：
    μ̃ₜ = (√ᾱ_{t-1} β_t)/(1-ᾱₜ) x₀ + (√αₜ(1-ᾱ_{t-1}))/(1-ᾱₜ) xₜ

    后验方差：
    σ̃ₜ² = (1-ᾱ_{t-1})/(1-ᾱₜ) βₜ

关键洞察：如果我们知道 x₀，就可以计算出 μ̃ₜ！

    问题：在反向过程中，我们不知道真实的 x₀

    解决：用神经网络预测 x₀（或等价地，预测噪声 ε）

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   因为 xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε                          │
    │                                                         │
    │   所以 x₀ = (xₜ - √(1-ᾱₜ) ε) / √ᾱₜ                      │
    │                                                         │
    │   我们训练网络 ε_θ(xₜ, t) 来预测噪声 ε                   │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

训练目标简化：

    原始目标：最大化对数似然 log p_θ(x₀)
    
    通过变分下界，简化为简单的MSE损失：

    L = E_{t,x₀,ε}[||ε - ε_θ(xₜ, t)||²]

    这就是 DDPM 的核心！
    """)


# ==================== 第四部分：噪声调度 ====================


def noise_schedule_demo():
    """噪声调度演示"""
    print("\n" + "=" * 60)
    print("第四部分：噪声调度")
    print("=" * 60)

    print("""
噪声调度（Noise Schedule）：

    定义 βₜ 如何随时间变化

常见调度类型：

    1. 线性调度 (Linear)
       βₜ = β_min + (β_max - β_min) × t/T

    2. 余弦调度 (Cosine) - 推荐
       更平滑的过渡，生成质量更好

    3. 平方根调度 (Sqrt)
       在早期步骤添加更少噪声
    """)

    T = 1000

    # 不同的噪声调度
    def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
        return np.linspace(beta_start, beta_end, T)

    def cosine_schedule(T, s=0.008):
        steps = np.arange(T + 1)
        f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(beta, 0.0001, 0.9999)

    def sqrt_schedule(T, beta_start=1e-4, beta_end=0.02):
        return np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), T) ** 2

    # 计算不同调度
    beta_linear = linear_schedule(T)
    beta_cosine = cosine_schedule(T)
    beta_sqrt = sqrt_schedule(T)

    # 计算 alpha_bar
    def compute_alpha_bar(beta):
        alpha = 1 - beta
        return np.cumprod(alpha)

    alpha_bar_linear = compute_alpha_bar(beta_linear)
    alpha_bar_cosine = compute_alpha_bar(beta_cosine)
    alpha_bar_sqrt = compute_alpha_bar(beta_sqrt)

    # 可视化
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(beta_linear, label="Linear")
    plt.plot(beta_cosine, label="Cosine")
    plt.plot(beta_sqrt, label="Sqrt")
    plt.xlabel("时间步 t")
    plt.ylabel("βₜ")
    plt.title("噪声调度 βₜ")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(alpha_bar_linear, label="Linear")
    plt.plot(alpha_bar_cosine, label="Cosine")
    plt.plot(alpha_bar_sqrt, label="Sqrt")
    plt.xlabel("时间步 t")
    plt.ylabel("ᾱₜ")
    plt.title("累积系数 ᾱₜ")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "noise_schedules.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 噪声调度对比已保存到: {output_path}")

    print("""
观察：
    - 线性调度：ᾱₜ 下降较快，早期噪声较大
    - 余弦调度：更平滑的过渡，推荐用于图像生成
    - 平方根调度：早期变化更慢
    """)


# ==================== 第五部分：DDPM 训练算法 ====================


def ddpm_algorithm():
    """DDPM 训练算法"""
    print("\n" + "=" * 60)
    print("第五部分：DDPM 训练算法")
    print("=" * 60)

    print("""
DDPM 训练算法：

    ┌─────────────────────────────────────────────────────────┐
    │ Algorithm: DDPM Training                                 │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ 1. repeat:                                              │
    │                                                         │
    │     // 采样数据和时间步                                  │
    │     x₀ ~ p_data(x)        # 从数据集采样                │
    │     t ~ Uniform({1,...,T}) # 随机选择时间步              │
    │     ε ~ N(0, I)            # 采样噪声                   │
    │                                                         │
    │     // 构造加噪数据                                      │
    │     xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε                            │
    │                                                         │
    │     // 计算损失                                          │
    │     L = ||ε - ε_θ(xₜ, t)||²                             │
    │                                                         │
    │     // 梯度下降                                          │
    │     θ ← θ - η∇_θ L                                      │
    │                                                         │
    │ 2. until converged                                      │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

DDPM 采样算法：

    ┌─────────────────────────────────────────────────────────┐
    │ Algorithm: DDPM Sampling                                 │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ 1. xₜ ~ N(0, I)  # 从纯噪声开始                         │
    │                                                         │
    │ 2. for t = T, T-1, ..., 1:                             │
    │                                                         │
    │     // 预测噪声                                          │
    │     ε̂ = ε_θ(xₜ, t)                                      │
    │                                                         │
    │     // 计算均值                                          │
    │     μ_θ = (1/√αₜ)(xₜ - (βₜ/√(1-ᾱₜ))ε̂)                  │
    │                                                         │
    │     // 采样 (t>1时添加噪声)                              │
    │     z ~ N(0, I) if t > 1 else 0                        │
    │     x_{t-1} = μ_θ + σₜ z                                │
    │                                                         │
    │ 3. return x₀                                            │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

关键公式汇总：

    前向过程：
    xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε

    反向过程均值：
    μ_θ(xₜ, t) = (1/√αₜ)(xₜ - (βₜ/√(1-ᾱₜ)) ε_θ(xₜ, t))

    训练损失：
    L = E[||ε - ε_θ(√ᾱₜ x₀ + √(1-ᾱₜ) ε, t)||²]
    """)


# ==================== 第六部分：神经网络架构 ====================


def unet_architecture():
    """UNet 架构介绍"""
    print("\n" + "=" * 60)
    print("第六部分：噪声预测网络 (UNet)")
    print("=" * 60)

    print("""
噪声预测网络 ε_θ(xₜ, t)：

    通常使用 UNet 架构，接受两个输入：
    1. 加噪图像 xₜ
    2. 时间步 t（通过 sinusoidal embedding 编码）

UNet 结构：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   输入 xₜ                                               │
    │      │                                                  │
    │      ▼                                                  │
    │   ┌─────┐                                               │
    │   │Conv │──────────────────────────────────┐            │
    │   └──┬──┘                                  │            │
    │      │ 下采样                               │            │
    │      ▼                                     │            │
    │   ┌─────┐                                  │            │
    │   │Conv │──────────────────────┐           │            │
    │   └──┬──┘                      │           │            │
    │      │                         │           │            │
    │      ▼                         │           │            │
    │   ┌─────┐                      │           │            │
    │   │中间层│ ← 时间嵌入 + 注意力   │           │            │
    │   └──┬──┘                      │           │            │
    │      │                         │           │            │
    │      ▼ 上采样                   │           │            │
    │   ┌─────┐ ←───────────────────┘           │            │
    │   │Conv │                                  │            │
    │   └──┬──┘                                  │            │
    │      │                                     │            │
    │      ▼                                     │            │
    │   ┌─────┐ ←────────────────────────────────┘            │
    │   │Conv │                                               │
    │   └──┬──┘                                               │
    │      │                                                  │
    │      ▼                                                  │
    │   预测噪声 ε̂                                            │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

时间嵌入 (Time Embedding)：

    使用 sinusoidal positional encoding（类似 Transformer）

    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))

    然后通过 MLP 映射到每一层
    """)


class SinusoidalTimeEmbedding(nn.Module):
    """时间步的正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


def time_embedding_demo():
    """时间嵌入演示"""
    print("\n示例: 时间嵌入\n")

    time_embed = SinusoidalTimeEmbedding(dim=64)

    # 不同时间步
    t = torch.tensor([0.0, 250.0, 500.0, 750.0, 999.0])
    embeddings = time_embed(t)

    print(f"时间步: {t.numpy()}")
    print(f"嵌入形状: {embeddings.shape}")

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.imshow(embeddings.detach().numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="值")
    plt.xlabel("嵌入维度")
    plt.ylabel("时间步")
    plt.yticks(range(5), [f"t={int(t[i])}" for i in range(5)])
    plt.title("时间嵌入可视化")

    output_path = os.path.join(os.path.dirname(__file__), "time_embedding.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ 时间嵌入可视化已保存到: {output_path}")


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：实现前向扩散
    任务: 对一张 MNIST 图像应用前向扩散
    可视化不同时间步的加噪结果

练习 2：实现不同噪声调度
    任务: 实现线性、余弦、sigmoid 调度
    对比它们的 ᾱₜ 曲线

练习 3：实现时间嵌入
    任务: 实现 sinusoidal time embedding
    测试不同时间步的嵌入是否正交

练习 4：推导 DDPM 损失
    任务: 从 ELBO 推导出简化的训练目标
    理解为什么预测噪声等价于预测 x₀

思考题 1：为什么使用 1000 步？
    步数越多越好吗？
    步数与生成质量的关系是什么？

思考题 2：扩散模型 vs VAE
    两者的隐空间有什么区别？
    两者的生成过程有什么不同？

思考题 3：条件生成
    如何在扩散模型中加入条件（如文本）？
    Classifier-Free Guidance 是什么？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    forward_process_demo()
    reverse_process_demo()
    noise_schedule_demo()
    ddpm_algorithm()
    unet_architecture()
    time_embedding_demo()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 07-diffusion-implementation.py: 扩散模型实现

关键要点回顾：
    ✓ 前向过程逐步添加噪声
    ✓ 反向过程学习去噪
    ✓ xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε
    ✓ 训练目标是预测噪声 ε
    ✓ 使用 UNet 架构配合时间嵌入
    ✓ 余弦噪声调度效果更好
    """)


if __name__ == "__main__":
    main()
