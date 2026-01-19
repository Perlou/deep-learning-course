"""
Stable Diffusion 架构介绍
==========================

学习目标：
    1. 理解 Stable Diffusion 的整体架构
    2. 掌握 Latent Diffusion 的核心思想
    3. 了解文本条件生成 (Text Conditioning) 的原理
    4. 理解 CLIPText Encoder 和 Cross-Attention 的作用

核心概念：
    - Latent Diffusion: 在隐空间进行扩散，大幅降低计算成本
    - VAE: 图像与隐空间的编解码器
    - CLIP: 文本编码器，将文本转为向量
    - Cross-Attention: 将文本条件注入 UNet

前置知识：
    - 05-vae.py: VAE 基础
    - 06-diffusion-theory.py: 扩散模型理论
    - 07-diffusion-implementation.py: 扩散模型实现
    - Phase 7: Transformer 和注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


# ==================== 第一部分：Stable Diffusion 概述 ====================


def introduction():
    """
    Stable Diffusion 概述
    """
    print("=" * 60)
    print("第一部分：Stable Diffusion 概述")
    print("=" * 60)

    print("""
什么是 Stable Diffusion？

    Stable Diffusion 是一个开源的文本到图像生成模型，
    能够根据文本描述生成高质量的图像。

核心创新：Latent Diffusion

    传统扩散模型的问题：
    ┌─────────────────────────────────────────────────────────┐
    │ - 直接在像素空间操作                                     │
    │ - 512×512×3 = 786,432 维度                              │
    │ - 计算成本极高                                           │
    │ - 需要大量 GPU 内存                                      │
    └─────────────────────────────────────────────────────────┘

    Latent Diffusion 的解决方案：
    ┌─────────────────────────────────────────────────────────┐
    │ - 先用 VAE 压缩到隐空间                                  │
    │ - 隐空间维度: 64×64×4 = 16,384                          │
    │ - 压缩比: 48 倍！                                        │
    │ - 在隐空间进行扩散                                       │
    │ - 最后用 VAE 解码回像素空间                              │
    └─────────────────────────────────────────────────────────┘

Stable Diffusion 架构图：

    ┌─────────────────────────────────────────────────────────────┐
    │                  Stable Diffusion 架构                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   "A cat sitting     ┌───────────────┐                      │
    │    on a chair"  ──→  │ CLIP Text     │ ──→ 文本嵌入         │
    │                      │ Encoder       │     (77, 768)        │
    │                      └───────────────┘         │            │
    │                                                │            │
    │                                                ↓            │
    │   随机噪声 ──→  ┌───────────────────────────────────┐       │
    │   (64×64×4)    │                                   │       │
    │                │         UNet + Scheduler          │       │
    │                │    (Time Emb + Cross-Attention)   │       │
    │                │                                   │       │
    │                └───────────────────────────────────┘       │
    │                            │                               │
    │                            ↓                               │
    │                    去噪后的隐向量                            │
    │                      (64×64×4)                             │
    │                            │                               │
    │                            ↓                               │
    │                   ┌───────────────┐                        │
    │                   │ VAE Decoder   │                        │
    │                   └───────────────┘                        │
    │                            │                               │
    │                            ↓                               │
    │                     生成的图像                              │
    │                    (512×512×3)                             │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

主要组件：

    1. VAE (Variational Autoencoder)
       - 将图像压缩到隐空间
       - 将隐向量解码为图像

    2. CLIP Text Encoder
       - 将文本 prompt 转换为嵌入向量
       - 使用 Transformer 架构

    3. UNet + Scheduler
       - 在隐空间进行去噪
       - 结合时间嵌入和文本条件
    """)


# ==================== 第二部分：Latent Diffusion ====================


def latent_diffusion_demo():
    """Latent Diffusion 详解"""
    print("\n" + "=" * 60)
    print("第二部分：Latent Diffusion")
    print("=" * 60)

    print("""
为什么在隐空间进行扩散？

    像素空间 vs 隐空间：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   像素空间                    隐空间                     │
    │   512 × 512 × 3              64 × 64 × 4                │
    │   = 786,432 维               = 16,384 维                │
    │                                                         │
    │   注意力矩阵:                 注意力矩阵:                 │
    │   262,144 × 262,144          4,096 × 4,096              │
    │   (275 GB!)                  (67 MB)                    │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    压缩后的隐空间保留了图像的语义信息，
    同时大幅降低了计算成本。

VAE 的作用：

    Encoder (编码器):
    ┌─────────────┐     ┌─────────────┐
    │ 图像        │     │ 隐向量      │
    │ 512×512×3   │ ──→ │ 64×64×4     │
    │ (像素空间)  │     │ (隐空间)    │
    └─────────────┘     └─────────────┘

    Decoder (解码器):
    ┌─────────────┐     ┌─────────────┐
    │ 隐向量      │     │ 图像        │
    │ 64×64×4     │ ──→ │ 512×512×3   │
    │ (隐空间)    │     │ (像素空间)  │
    └─────────────┘     └─────────────┘

    注意: Stable Diffusion 的 VAE 是预训练的，
    扩散模型只在隐空间进行训练。
    """)


class SimpleLatentVAE(nn.Module):
    """简化的隐空间 VAE (演示用)"""

    def __init__(self, in_channels=3, latent_channels=4, scale_factor=8):
        super().__init__()

        # 编码器: 下采样 scale_factor 倍
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # /8
            nn.ReLU(),
            nn.Conv2d(256, latent_channels * 2, 3, padding=1),  # μ 和 logvar
        )

        # 解码器: 上采样 scale_factor 倍
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # x2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # x4
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, stride=2, padding=1),  # x8
            nn.Tanh(),
        )

        self.latent_channels = latent_channels

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_demo():
    """VAE 编解码演示"""
    print("\n示例: VAE 编解码\n")

    vae = SimpleLatentVAE()

    # 模拟一张图像
    image = torch.randn(1, 3, 512, 512)
    print(f"原始图像形状: {image.shape}")

    # 编码
    mu, logvar = vae.encode(image)
    print(f"隐向量形状: {mu.shape}")
    print(f"压缩比: {512 * 512 * 3 / (64 * 64 * 4):.1f}x")

    # 解码
    z = vae.reparameterize(mu, logvar)
    recon = vae.decode(z)
    print(f"重构图像形状: {recon.shape}")


# ==================== 第三部分：文本条件生成 ====================


def text_conditioning_demo():
    """文本条件生成"""
    print("\n" + "=" * 60)
    print("第三部分：文本条件生成")
    print("=" * 60)

    print("""
CLIP Text Encoder：

    CLIP (Contrastive Language-Image Pre-training) 是一个
    学习图像和文本对齐的模型。

    Stable Diffusion 使用 CLIP 的文本编码器将 prompt 转换为嵌入:

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   "A beautiful sunset    ┌──────────────┐               │
    │    over the ocean"  ──→  │ Tokenizer    │ ──→ Token IDs │
    │                          └──────────────┘       ↓       │
    │                                                 ↓       │
    │                          ┌──────────────┐               │
    │                          │ CLIP Text    │ ──→ 文本嵌入  │
    │                          │ Transformer  │    (77, 768)  │
    │                          └──────────────┘               │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    - 最大 77 个 tokens
    - 每个 token 嵌入维度 768
    - 输出形状: (batch, 77, 768)

Cross-Attention 机制：

    将文本条件注入 UNet 的每一层:

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   图像特征 h ──→ Query (Q)                               │
    │                     ↓                                   │
    │   文本嵌入 ──→ Key (K), Value (V)                        │
    │                     ↓                                   │
    │   Attention(Q, K, V) = softmax(QK^T/√d) V             │
    │                     ↓                                   │
    │   条件化的图像特征                                       │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    通过 Cross-Attention，模型可以"关注"与当前位置
    最相关的文本信息。
    """)


class CrossAttention(nn.Module):
    """Cross-Attention 层"""

    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """
        Args:
            x: 图像特征 (batch, seq_len, query_dim)
            context: 文本嵌入 (batch, context_len, context_dim)

        Returns:
            条件化的特征 (batch, seq_len, query_dim)
        """
        batch_size = x.shape[0]

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 重塑为多头
        q = q.view(batch_size, -1, self.heads, q.shape[-1] // self.heads).transpose(
            1, 2
        )
        k = k.view(batch_size, -1, self.heads, k.shape[-1] // self.heads).transpose(
            1, 2
        )
        v = v.view(batch_size, -1, self.heads, v.shape[-1] // self.heads).transpose(
            1, 2
        )

        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, out.shape[-1] * self.heads)

        return self.to_out(out)


def cross_attention_demo():
    """Cross-Attention 演示"""
    print("\n示例: Cross-Attention\n")

    cross_attn = CrossAttention(query_dim=320, context_dim=768, heads=8)

    # 模拟输入
    image_features = torch.randn(2, 64 * 64, 320)  # 图像特征
    text_embeddings = torch.randn(2, 77, 768)  # 文本嵌入

    print(f"图像特征形状: {image_features.shape}")
    print(f"文本嵌入形状: {text_embeddings.shape}")

    # Cross-Attention
    conditioned_features = cross_attn(image_features, text_embeddings)
    print(f"条件化特征形状: {conditioned_features.shape}")


# ==================== 第四部分：Classifier-Free Guidance ====================


def cfg_demo():
    """Classifier-Free Guidance"""
    print("\n" + "=" * 60)
    print("第四部分：Classifier-Free Guidance")
    print("=" * 60)

    print("""
Classifier-Free Guidance (CFG)：

    问题: 如何让模型更"听话"，生成更符合 prompt 的图像？

    解决方案: Classifier-Free Guidance

    原理:
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │ 1. 训练时随机丢弃文本条件 (用空文本替换)                  │
    │    - 10-20% 的概率使用空文本                             │
    │    - 模型学会条件生成和无条件生成                         │
    │                                                         │
    │ 2. 采样时使用引导:                                       │
    │                                                         │
    │    ε_pred = ε_uncond + w * (ε_cond - ε_uncond)          │
    │                                                         │
    │    其中 w 是 guidance scale (通常 7-15)                  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    效果:
    - w = 1: 原始条件生成
    - w > 1: 更强的文本遵循
    - w 太大: 图像过饱和、失真

引导强度对比:

    w = 1          w = 7.5         w = 15
    ┌────────┐    ┌────────┐     ┌────────┐
    │ 模糊    │    │ 清晰    │    │ 过曝光  │
    │ 多样    │    │ 忠实    │    │ 失真    │
    └────────┘    └────────┘     └────────┘
    """)


# ==================== 第五部分：Stable Diffusion 训练流程 ====================


def training_overview():
    """训练流程概述"""
    print("\n" + "=" * 60)
    print("第五部分：Stable Diffusion 训练流程")
    print("=" * 60)

    print("""
Stable Diffusion 训练流程:

    ┌─────────────────────────────────────────────────────────┐
    │ 1. 数据准备                                              │
    │    - LAION-5B 数据集 (50亿图文对)                        │
    │    - 过滤、清洗、NSFW 检测                               │
    │                                                         │
    │ 2. 预训练 VAE                                           │
    │    - 在大规模图像上训练                                  │
    │    - 学习图像的隐空间表示                                │
    │                                                         │
    │ 3. 训练 UNet                                            │
    │    - 使用预训练 VAE 编码图像                             │
    │    - 在隐空间训练扩散模型                                │
    │    - 使用 CLIP 文本编码器                                │
    │                                                         │
    │ 4. 微调 (可选)                                          │
    │    - DreamBooth: 个性化主题                              │
    │    - LoRA: 参数高效微调                                  │
    │    - ControlNet: 添加控制条件                            │
    └─────────────────────────────────────────────────────────┘

训练损失:

    L = E_{t,z,ε,c}[||ε - ε_θ(z_t, t, c)||²]

    其中:
    - z_t: 加噪的隐向量
    - t: 时间步
    - c: 文本条件 (CLIP 嵌入)
    - ε: 噪声
    - ε_θ: UNet 预测的噪声

计算资源需求:

    ┌─────────────────────────────────────────────────────────┐
    │ Stable Diffusion v1.5:                                  │
    │ - 参数量: ~860M (UNet) + ~123M (VAE) + ~123M (CLIP)    │
    │ - 训练: ~200,000 A100 GPU 小时                          │
    │ - 数据: LAION-5B 子集                                   │
    └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第六部分：使用 Stable Diffusion ====================


def usage_demo():
    """使用演示"""
    print("\n" + "=" * 60)
    print("第六部分：使用 Stable Diffusion")
    print("=" * 60)

    print("""
使用 Hugging Face Diffusers 库:

    ```python
    from diffusers import StableDiffusionPipeline
    import torch

    # 加载模型
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # 生成图像
    prompt = "A beautiful sunset over the ocean, digital art"
    image = pipe(prompt).images[0]
    image.save("sunset.png")
    ```

高级参数:

    ```python
    image = pipe(
        prompt="...",
        negative_prompt="blurry, low quality",  # 负向提示
        num_inference_steps=50,  # 采样步数
        guidance_scale=7.5,      # CFG 强度
        height=512,              # 图像高度
        width=512,               # 图像宽度
        generator=torch.Generator().manual_seed(42)  # 随机种子
    ).images[0]
    ```

常用变体:

    1. Stable Diffusion v1.5
       - 基础版本，512×512

    2. Stable Diffusion v2.0/2.1
       - 更大 CLIP，768×768 支持

    3. SDXL (Stable Diffusion XL)
       - 两阶段生成，1024×1024
       - 更好的图像质量

    4. SD Turbo / SDXL Turbo
       - 1-4 步生成
       - 实时速度
    """)


# ==================== 第七部分：扩展技术 ====================


def extensions_demo():
    """扩展技术介绍"""
    print("\n" + "=" * 60)
    print("第七部分：扩展技术")
    print("=" * 60)

    print("""
1. ControlNet
   ┌─────────────────────────────────────────────────────────┐
   │ 添加空间控制条件:                                        │
   │ - 边缘检测 (Canny)                                       │
   │ - 深度图                                                 │
   │ - 姿态骨架                                               │
   │ - 语义分割                                               │
   └─────────────────────────────────────────────────────────┘

2. LoRA (Low-Rank Adaptation)
   ┌─────────────────────────────────────────────────────────┐
   │ 参数高效微调:                                            │
   │ - 只训练少量参数 (~4MB vs ~2GB)                         │
   │ - 可以叠加多个 LoRA                                     │
   │ - 支持风格、角色、概念定制                               │
   └─────────────────────────────────────────────────────────┘

3. DreamBooth
   ┌─────────────────────────────────────────────────────────┐
   │ 个性化微调:                                              │
   │ - 用 3-5 张图片学习新概念                                │
   │ - "A photo of [V] dog on the beach"                     │
   └─────────────────────────────────────────────────────────┘

4. Textual Inversion
   ┌─────────────────────────────────────────────────────────┐
   │ 学习新的文本嵌入:                                        │
   │ - 用少量图片学习一个新"词"                               │
   │ - 不修改模型参数                                         │
   └─────────────────────────────────────────────────────────┘

5. IP-Adapter
   ┌─────────────────────────────────────────────────────────┐
   │ 图像 Prompt:                                             │
   │ - 使用图像作为条件                                       │
   │ - 结合文本和图像控制                                     │
   └─────────────────────────────────────────────────────────┘
    """)


# ==================== 第八部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    exercises_text = """
练习 1：使用 Diffusers 生成图像
    安装 diffusers:
    pip install diffusers transformers accelerate

    任务: 使用不同的 prompt 和参数生成图像
    观察 guidance_scale 的影响

练习 2：探索不同采样器
    Diffusers 支持多种采样器:
    - DDPMScheduler
    - DDIMScheduler
    - EulerAncestralDiscreteScheduler

    任务: 对比不同采样器的效果和速度

练习 3：使用 ControlNet
    任务: 使用边缘检测 ControlNet 进行图像生成
    控制生成图像的结构

练习 4：理解 Cross-Attention
    任务: 可视化 Cross-Attention 的权重
    观察模型如何"关注"不同的文本 token

思考题 1：为什么 Latent Diffusion 更高效？
    从信息论角度分析图像的冗余性

思考题 2：CLIP 的局限性
    CLIP 文本编码器有什么限制？
    如何改进？

思考题 3：视频生成
    如何将 Stable Diffusion 扩展到视频生成？
    Sora 等模型的创新在哪里？
    """
    print(exercises_text)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    latent_diffusion_demo()
    vae_demo()
    text_conditioning_demo()
    cross_attention_demo()
    cfg_demo()
    training_overview()
    usage_demo()
    extensions_demo()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
🎉 Phase 8: 生成模型 全部完成！

学习总结：

    GAN 系列:
    ✓ 01-gan-basics: 对抗训练原理
    ✓ 02-dcgan: 卷积 GAN 架构
    ✓ 03-wgan: Wasserstein 距离

    VAE 系列:
    ✓ 04-autoencoder: 自编码器基础
    ✓ 05-vae: 变分推断和重参数化

    扩散模型系列:
    ✓ 06-diffusion-theory: DDPM 理论
    ✓ 07-diffusion-implementation: 扩散模型实现
    ✓ 08-stable-diffusion-intro: Stable Diffusion 架构

下一步：
    - Phase 9: 训练技巧与优化
    - 项目: GAN 生成人脸/动漫图像
    """)


if __name__ == "__main__":
    main()
