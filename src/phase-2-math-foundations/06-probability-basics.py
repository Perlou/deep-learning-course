"""
06-probability-basics.py
Phase 2: 深度学习数学基础

概率论基础 - 深度学习的不确定性建模

学习目标：
1. 理解概率分布的基本概念
2. 掌握常见概率分布及其应用
3. 理解贝叶斯定理
4. 了解采样和蒙特卡洛方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("深度学习数学基础 - 概率论基础")
print("=" * 60)

# =============================================================================
# 1. 概率基础概念
# =============================================================================
print("\n【1. 概率基础概念】")

print("""
基本概念：
- 样本空间 Ω：所有可能结果的集合
- 事件 A：样本空间的子集
- 概率 P(A)：事件发生的可能性，0 ≤ P(A) ≤ 1

概率公理：
1. 非负性：P(A) ≥ 0
2. 规范性：P(Ω) = 1
3. 可加性：P(A ∪ B) = P(A) + P(B)，若 A ∩ B = ∅

条件概率：
    P(A|B) = P(A ∩ B) / P(B)

贝叶斯定理：
    P(A|B) = P(B|A) × P(A) / P(B)
""")

# 简单的概率计算示例
print("\n掷骰子示例：")
n_rolls = 10000
rolls = np.random.randint(1, 7, n_rolls)

for i in range(1, 7):
    p = np.mean(rolls == i)
    print(f"  P(点数={i}) = {p:.4f} (理论值: {1/6:.4f})")

# =============================================================================
# 2. 离散概率分布
# =============================================================================
print("\n" + "=" * 60)
print("【2. 离散概率分布】")

# 2.1 伯努利分布
print("\n2.1 伯努利分布 Bernoulli(p)")
print("    只有两个结果：成功(1)和失败(0)")
print("    P(X=1) = p, P(X=0) = 1-p")

p = 0.7
bernoulli_samples = np.random.binomial(1, p, 1000)
print(f"    样本均值: {bernoulli_samples.mean():.4f} (理论期望: {p})")

# 2.2 二项分布
print("\n2.2 二项分布 Binomial(n, p)")
print("    n 次独立伯努利试验中成功的次数")

n, p = 20, 0.5
binomial_samples = np.random.binomial(n, p, 1000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 二项分布
ax = axes[0, 0]
values = np.arange(0, n+1)
pmf = stats.binom.pmf(values, n, p)
ax.bar(values, pmf, color='steelblue', edgecolor='black', alpha=0.7)
ax.hist(binomial_samples, bins=values, density=True, alpha=0.5, color='coral', label='采样')
ax.set_xlabel('成功次数')
ax.set_ylabel('概率')
ax.set_title(f'二项分布 Binomial(n={n}, p={p})', fontsize=12, fontweight='bold')
ax.legend()

# 2.3 泊松分布
print("\n2.3 泊松分布 Poisson(λ)")
print("    单位时间/空间内随机事件发生的次数")

lam = 5
poisson_samples = np.random.poisson(lam, 1000)

ax = axes[0, 1]
values = np.arange(0, 20)
pmf = stats.poisson.pmf(values, lam)
ax.bar(values, pmf, color='steelblue', edgecolor='black', alpha=0.7)
ax.hist(poisson_samples, bins=values, density=True, alpha=0.5, color='coral', label='采样')
ax.set_xlabel('事件次数')
ax.set_ylabel('概率')
ax.set_title(f'泊松分布 Poisson(λ={lam})', fontsize=12, fontweight='bold')
ax.legend()

# 2.4 Categorical 分布（深度学习中常用）
print("\n2.4 Categorical 分布（分类分布）")
print("    多类别分类问题的输出分布")

probs = [0.1, 0.2, 0.3, 0.25, 0.15]  # 5 个类别的概率
n_samples = 1000
categorical_samples = np.random.choice(5, n_samples, p=probs)

ax = axes[1, 0]
ax.bar(range(5), probs, color='steelblue', edgecolor='black', alpha=0.7, label='真实概率')
hist, _ = np.histogram(categorical_samples, bins=np.arange(6))
ax.bar(range(5), hist/n_samples, color='coral', alpha=0.5, label='采样频率')
ax.set_xlabel('类别')
ax.set_ylabel('概率')
ax.set_title('Categorical 分布', fontsize=12, fontweight='bold')
ax.set_xticks(range(5))
ax.set_xticklabels([f'类别 {i}' for i in range(5)])
ax.legend()

# 2.5 Softmax 与 Categorical
print("\n深度学习中的应用：")
print("    Softmax 输出 → Categorical 分布的参数")

logits = np.array([2.0, 1.0, 0.1, 0.5, -0.5])
probs_softmax = np.exp(logits) / np.sum(np.exp(logits))
print(f"    Logits: {logits}")
print(f"    Softmax: {probs_softmax}")
print(f"    预测类别: {np.argmax(probs_softmax)}")

ax = axes[1, 1]
ax.bar(range(5), probs_softmax, color='steelblue', edgecolor='black')
ax.set_xlabel('类别')
ax.set_ylabel('概率')
ax.set_title('Softmax 输出（分类器概率分布）', fontsize=12, fontweight='bold')
ax.set_xticks(range(5))

plt.tight_layout()
plt.savefig('outputs/06_discrete_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: outputs/06_discrete_distributions.png")

# =============================================================================
# 3. 连续概率分布
# =============================================================================
print("\n" + "=" * 60)
print("【3. 连续概率分布】")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 均匀分布
print("\n3.1 均匀分布 Uniform(a, b)")
a, b = 0, 5
x = np.linspace(-1, 6, 1000)
ax = axes[0, 0]
ax.fill_between(x, stats.uniform.pdf(x, a, b-a), alpha=0.5, color='steelblue')
ax.hist(np.random.uniform(a, b, 5000), bins=50, density=True, alpha=0.5, color='coral')
ax.set_title(f'均匀分布 Uniform({a}, {b})', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('概率密度')

# 3.2 正态分布
print("\n3.2 正态分布（高斯分布）N(μ, σ²)")
print("    深度学习中最重要的分布！")
print("    权重初始化、噪声建模、VAE 的隐变量分布")

mus = [0, 0, 2]
sigmas = [1, 2, 1]
x = np.linspace(-6, 6, 1000)
ax = axes[0, 1]
for mu, sigma in zip(mus, sigmas):
    ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2, 
            label=f'μ={mu}, σ={sigma}')
ax.set_title('正态分布 N(μ, σ²)', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('概率密度')
ax.legend()

# 3.3 指数分布
print("\n3.3 指数分布 Exponential(λ)")
lam = 1
x = np.linspace(0, 5, 1000)
ax = axes[1, 0]
for l in [0.5, 1, 2]:
    ax.plot(x, stats.expon.pdf(x, scale=1/l), linewidth=2, label=f'λ={l}')
ax.set_title('指数分布 Exponential(λ)', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('概率密度')
ax.legend()

# 3.4 Beta 分布
print("\n3.4 Beta 分布 Beta(α, β)")
print("    定义在 [0, 1] 上，常用于概率的概率")

ax = axes[1, 1]
x = np.linspace(0, 1, 1000)
for a, b in [(0.5, 0.5), (1, 1), (2, 5), (5, 2), (5, 5)]:
    ax.plot(x, stats.beta.pdf(x, a, b), linewidth=2, label=f'α={a}, β={b}')
ax.set_title('Beta 分布', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('概率密度')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/06_continuous_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/06_continuous_distributions.png")

# =============================================================================
# 4. 期望、方差和协方差
# =============================================================================
print("\n" + "=" * 60)
print("【4. 期望、方差和协方差】")

print("""
期望（均值）：
    E[X] = ∫ x × p(x) dx    (连续)
    E[X] = Σ x × P(x)       (离散)

方差：
    Var(X) = E[(X - E[X])²] = E[X²] - E[X]²

协方差：
    Cov(X, Y) = E[(X - E[X])(Y - E[Y])]

相关系数：
    ρ = Cov(X, Y) / (σ_X × σ_Y)
""")

# 示例
np.random.seed(42)
n = 1000

# 生成相关的随机变量
mean = [0, 0]
cov_matrix = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov_matrix, n)
x, y = data[:, 0], data[:, 1]

print(f"样本均值: E[X]={x.mean():.4f}, E[Y]={y.mean():.4f}")
print(f"样本方差: Var(X)={x.var():.4f}, Var(Y)={y.var():.4f}")
print(f"样本协方差: Cov(X,Y)={np.cov(x, y)[0,1]:.4f}")
print(f"样本相关系数: ρ={np.corrcoef(x, y)[0,1]:.4f}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 散点图
axes[0].scatter(x, y, alpha=0.3, s=10)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title(f'相关随机变量 (ρ={np.corrcoef(x,y)[0,1]:.2f})', fontsize=12, fontweight='bold')

# 不相关变量
x2, y2 = np.random.randn(n), np.random.randn(n)
axes[1].scatter(x2, y2, alpha=0.3, s=10)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title(f'独立随机变量 (ρ={np.corrcoef(x2,y2)[0,1]:.2f})', fontsize=12, fontweight='bold')

# 负相关变量
mean = [0, 0]
cov_neg = [[1, -0.8], [-0.8, 1]]
data_neg = np.random.multivariate_normal(mean, cov_neg, n)
axes[2].scatter(data_neg[:, 0], data_neg[:, 1], alpha=0.3, s=10)
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].set_title(f'负相关随机变量 (ρ={np.corrcoef(data_neg[:,0], data_neg[:,1])[0,1]:.2f})', 
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/06_covariance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: outputs/06_covariance.png")

# =============================================================================
# 5. 贝叶斯定理
# =============================================================================
print("\n" + "=" * 60)
print("【5. 贝叶斯定理】")

print("""
贝叶斯定理：
    P(A|B) = P(B|A) × P(A) / P(B)

术语：
- P(A): 先验概率 (Prior)
- P(B|A): 似然 (Likelihood)
- P(A|B): 后验概率 (Posterior)
- P(B): 证据 (Evidence)

深度学习应用：
- 贝叶斯神经网络
- 变分推断
- 不确定性估计
""")

# 经典例子：疾病诊断
print("\n医学诊断示例：")
print("  - 疾病发病率 P(病) = 0.001")
print("  - 患病阳性率 P(阳性|病) = 0.99")
print("  - 健康假阳性 P(阳性|健康) = 0.01")

p_disease = 0.001
p_positive_given_disease = 0.99
p_positive_given_healthy = 0.01

# P(阳性) = P(阳性|病)P(病) + P(阳性|健康)P(健康)
p_positive = p_positive_given_disease * p_disease + p_positive_given_healthy * (1 - p_disease)

# P(病|阳性) = P(阳性|病)P(病) / P(阳性)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"\n  P(阳性) = {p_positive:.4f}")
print(f"  P(患病|阳性检测) = {p_disease_given_positive:.4f}")
print(f"  即：阳性结果只有约 {p_disease_given_positive*100:.1f}% 的概率真的患病")
print("  这就是为什么需要多次检测的原因！")

# =============================================================================
# 6. 多维正态分布
# =============================================================================
print("\n" + "=" * 60)
print("【6. 多维正态分布】")

print("""
多维正态分布 N(μ, Σ)：
- μ: 均值向量
- Σ: 协方差矩阵

概率密度函数：
    p(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-1/2 (x-μ)^T Σ^(-1) (x-μ))

深度学习应用：
- VAE 的隐变量分布
- 高斯混合模型
- 扩散模型的噪声
""")

# 可视化二维高斯
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

mean = [0, 0]
covs = [
    [[1, 0], [0, 1]],          # 独立
    [[1, 0.8], [0.8, 1]],      # 正相关
    [[2, 0], [0, 0.5]]         # 各向异性
]
titles = ['独立 (Σ = I)', '正相关', '各向异性']

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

for ax, cov, title in zip(axes, covs, titles):
    rv = stats.multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    
    ax.contour(X, Y, Z, levels=10, cmap='viridis')
    ax.contourf(X, Y, Z, levels=10, alpha=0.5, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/06_multivariate_gaussian.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/06_multivariate_gaussian.png")

# =============================================================================
# 7. 采样方法
# =============================================================================
print("\n" + "=" * 60)
print("【7. 采样方法】")

print("""
深度学习中的采样：
1. 直接采样：np.random.normal(), np.random.uniform()
2. 重参数化技巧 (VAE)：z = μ + σ × ε, ε ~ N(0,1)
3. Gumbel-Softmax：使离散采样可微
4. MCMC：马尔可夫链蒙特卡洛（贝叶斯推断）
""")

# 重参数化技巧演示
print("\n重参数化技巧 (VAE 核心)：")
mu = np.array([1.0, 2.0])
sigma = np.array([0.5, 1.0])

# 方法1：直接采样（不可微）
z_direct = np.random.normal(mu, sigma, size=(1000, 2))

# 方法2：重参数化采样（可微）
epsilon = np.random.normal(0, 1, size=(1000, 2))
z_reparam = mu + sigma * epsilon

print(f"  直接采样均值: {z_direct.mean(axis=0)}")
print(f"  重参数化采样均值: {z_reparam.mean(axis=0)}")
print(f"  理论均值: {mu}")

# =============================================================================
# 8. 深度学习中的概率应用
# =============================================================================
print("\n" + "=" * 60)
print("【8. 深度学习中的概率应用】")

print("""
╔═══════════════════════════════════════════════════════════════╗
║                深度学习中的概率应用                            ║
╠═══════════════╦═══════════════════════════════════════════════╣
║  Dropout      ║  每个神经元以概率 p 被"丢弃"                  ║
║  VAE          ║  学习数据的概率分布 p(z), p(x|z)             ║
║  GAN          ║  学习从噪声到数据的映射                       ║
║  Diffusion    ║  逐步添加和去除高斯噪声                       ║
║  Softmax      ║  logits → 概率分布                           ║
║  交叉熵损失   ║  衡量两个分布的差异                           ║
║  权重初始化   ║  He/Xavier 初始化用正态/均匀分布              ║
╚═══════════════╩═══════════════════════════════════════════════╝
""")

# =============================================================================
# 9. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 用 numpy 采样 10000 个标准正态分布样本，验证其均值和方差

2. 实现重参数化技巧：从 N(μ=3, σ=2) 中采样

3. 给定 Softmax 输出 [0.7, 0.2, 0.1]，从对应的 Categorical 分布采样

4. 贝叶斯定理应用：
   - 初始相信某邮件是垃圾邮件的概率 P(垃圾) = 0.3
   - 如果邮件包含"促销"，垃圾邮件有此词的概率 P(促销|垃圾) = 0.8
   - 正常邮件有此词的概率 P(促销|正常) = 0.1
   - 计算 P(垃圾|促销)
""")

# === 练习代码 ===
# 练习 1: 标准正态分布验证
# samples = np.random.randn(10000)
# print(f"均值: {samples.mean():.4f} (理论: 0)")
# print(f"方差: {samples.var():.4f} (理论: 1)")

# 练习 2: 重参数化技巧
# mu, sigma = 3.0, 2.0
# epsilon = np.random.randn(10000)
# z = mu + sigma * epsilon
# print(f"采样均值: {z.mean():.4f} (应接近 {mu})")
# print(f"采样标准差: {z.std():.4f} (应接近 {sigma})")

# 练习 3: Categorical 采样
# probs = np.array([0.7, 0.2, 0.1])
# samples = np.random.choice(len(probs), size=1000, p=probs)
# print(f"采样频率: {np.bincount(samples, minlength=3) / 1000}")

# 练习 4: 贝叶斯垃圾邮件
# P_spam = 0.3
# P_promo_spam = 0.8
# P_promo_normal = 0.1
# P_promo = P_promo_spam * P_spam + P_promo_normal * (1 - P_spam)
# P_spam_promo = (P_promo_spam * P_spam) / P_promo
# print(f"P(垃圾|促销) = {P_spam_promo:.4f} = {P_spam_promo*100:.2f}%")

print("\n✅ 概率论基础完成！")
print("下一步：07-entropy-kl-divergence.py - 熵与KL散度")
