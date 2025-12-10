# Phase 8: 生成模型

> **目标**：掌握主流生成模型  
> **预计时长**：2 周  
> **前置条件**：Phase 1-7 完成

---

## 🎯 学习目标

完成本阶段后，你将能够：

1. 理解 GAN 的对抗训练原理
2. 掌握 VAE 的变分推断基础
3. 理解扩散模型 (Diffusion) 的去噪原理
4. 能够训练简单的图像生成模型
5. 了解 Stable Diffusion 的架构

---

## 📚 核心概念

### GAN (生成对抗网络)

两个网络的博弈：

- **生成器 (Generator)**: 生成假样本
- **判别器 (Discriminator)**: 区分真假

```python
# GAN 目标函数
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

### VAE (变分自编码器)

学习数据的潜在分布：

- **编码器**: 输入 → 潜在分布 (μ, σ)
- **解码器**: 潜在采样 → 重构
- **重参数化技巧**: z = μ + σ × ε

### Diffusion Models (扩散模型)

```
正向过程：逐步加噪声 x_0 → x_1 → ... → x_T
反向过程：逐步去噪声 x_T → x_{T-1} → ... → x_0
```

---

## 📁 文件列表

| 文件                             | 描述                  | 状态 |
| -------------------------------- | --------------------- | ---- |
| `01-gan-basics.py`               | 原始 GAN              | ⏳   |
| `02-dcgan.py`                    | 深度卷积 GAN          | ⏳   |
| `03-wgan.py`                     | Wasserstein GAN       | ⏳   |
| `04-autoencoder.py`              | 自编码器基础          | ⏳   |
| `05-vae.py`                      | VAE 原理与实现        | ⏳   |
| `06-diffusion-theory.py`         | DDPM 理论             | ⏳   |
| `07-diffusion-implementation.py` | 简易实现              | ⏳   |
| `08-stable-diffusion-intro.py`   | Stable Diffusion 架构 | ⏳   |

---

## 🚀 运行方式

```bash
python src/phase-8-generative/01-gan-basics.py
python src/phase-8-generative/05-vae.py
```

---

## 📖 推荐资源

- [GAN Lab](https://poloclub.github.io/ganlab/)
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- 论文：GAN, VAE, DDPM

---

## ✅ 完成检查

- [ ] 理解 GAN 的对抗训练原理
- [ ] 能够识别 GAN 训练中的常见问题
- [ ] 理解 VAE 的 ELBO 损失
- [ ] 理解重参数化技巧
- [ ] 理解扩散模型的正向和反向过程
- [ ] 能够训练简单的图像生成模型
- [ ] 了解 Stable Diffusion 的组成部分
