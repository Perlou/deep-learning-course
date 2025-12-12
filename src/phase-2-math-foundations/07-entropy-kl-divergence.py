"""
07-entropy-kl-divergence.py
Phase 2: 深度学习数学基础

熵与KL散度 - 信息论与损失函数

学习目标：
1. 理解信息熵的概念
2. 掌握交叉熵损失的原理
3. 理解KL散度及其应用
4. 了解这些概念在深度学习中的应用
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("深度学习数学基础 - 熵与 KL 散度")
print("=" * 60)

# =============================================================================
# 1. 信息论基础
# =============================================================================
print("\n【1. 信息论基础】")

print("""
信息量：
    I(x) = -log₂(p(x))
    
- 概率越小的事件，信息量越大
- 确定事件（p=1）信息量为 0
- 不可能事件（p→0）信息量为 ∞

例如：
- "太阳从东边升起"：概率高，信息量低
- "今天彩票中奖"：概率低，信息量高
""")

# 信息量示例
probs = [0.9999, 0.5, 0.1, 0.01, 0.001]
for p in probs:
    info = -np.log2(p)
    print(f"  P(x)={p:.4f}, 信息量 I(x) = {info:.4f} bits")

# =============================================================================
# 2. 熵 (Entropy)
# =============================================================================
print("\n" + "=" * 60)
print("【2. 熵 (Entropy)】")

print("""
熵：随机变量的平均信息量（不确定性度量）

    H(X) = -Σ p(x) × log₂(p(x))

性质：
- 熵越高，不确定性越大
- 均匀分布熵最大
- 确定性分布（某类别概率为1）熵为 0
""")

def entropy(probs):
    """计算离散分布的熵"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # 避免 log(0)
    return -np.sum(probs * np.log2(probs))

# 不同分布的熵
distributions = [
    ([1.0], "确定性 [1.0]"),
    ([0.5, 0.5], "均匀 [0.5, 0.5]"),
    ([0.9, 0.1], "偏斜 [0.9, 0.1]"),
    ([0.25, 0.25, 0.25, 0.25], "均匀 4类 [0.25×4]"),
    ([0.7, 0.1, 0.1, 0.1], "偏斜 4类 [0.7, 0.1×3]"),
]

print("\n不同分布的熵：")
for probs, name in distributions:
    h = entropy(probs)
    print(f"  {name}: H = {h:.4f} bits")

# 可视化熵与概率的关系（二元情况）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 二元熵
p = np.linspace(0.001, 0.999, 100)
h = -p * np.log2(p) - (1-p) * np.log2(1-p)
axes[0].plot(p, h, 'b-', linewidth=2)
axes[0].set_xlabel('P(X=1)')
axes[0].set_ylabel('熵 H(X)')
axes[0].set_title('二元熵函数', fontsize=12, fontweight='bold')
axes[0].axhline(y=1, color='r', linestyle='--', label='最大熵=1 @ p=0.5')
axes[0].axvline(x=0.5, color='r', linestyle='--')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 不同类别数的最大熵
n_classes = np.arange(2, 20)
max_entropy = np.log2(n_classes)
axes[1].plot(n_classes, max_entropy, 'bo-', linewidth=2, markersize=8)
axes[1].set_xlabel('类别数')
axes[1].set_ylabel('最大熵')
axes[1].set_title('均匀分布的最大熵 = log₂(n)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/07_entropy.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: outputs/07_entropy.png")

# =============================================================================
# 3. 交叉熵 (Cross Entropy)
# =============================================================================
print("\n" + "=" * 60)
print("【3. 交叉熵 (Cross Entropy)】")

print("""
交叉熵：用分布 q 来编码来自分布 p 的数据的平均比特数

    H(p, q) = -Σ p(x) × log(q(x))

深度学习中：
- p: 真实标签分布（one-hot）
- q: 模型预测的概率分布（softmax输出）

对于分类问题：
    L = -Σ y_true × log(y_pred)
    
对于二分类：
    L = -[y×log(p) + (1-y)×log(1-p)]  (Binary Cross Entropy)
""")

def cross_entropy(p_true, q_pred):
    """交叉熵"""
    q_pred = np.clip(q_pred, 1e-15, 1)  # 避免 log(0)
    return -np.sum(p_true * np.log(q_pred))

# 示例
y_true = np.array([1, 0, 0])  # 真实标签（类别0）

predictions = [
    [0.9, 0.05, 0.05],   # 好的预测
    [0.7, 0.2, 0.1],     # 一般的预测
    [0.4, 0.3, 0.3],     # 差的预测
    [0.1, 0.8, 0.1],     # 错误的预测
]

print("\n真实标签: [1, 0, 0] (类别0)")
print("不同预测的交叉熵损失：")
for pred in predictions:
    ce = cross_entropy(y_true, pred)
    print(f"  预测 {pred} → 交叉熵 = {ce:.4f}")

# =============================================================================
# 4. KL 散度 (KL Divergence)
# =============================================================================
print("\n" + "=" * 60)
print("【4. KL 散度 (KL Divergence)】")

print("""
KL散度：衡量两个概率分布的"距离"

    D_KL(p || q) = Σ p(x) × log(p(x) / q(x))
                 = H(p, q) - H(p)
                 = 交叉熵 - 熵

性质：
1. D_KL(p || q) ≥ 0，当且仅当 p = q 时等于 0
2. 不对称：D_KL(p || q) ≠ D_KL(q || p)
3. 不满足三角不等式，所以不是真正的"距离"

深度学习应用：
- VAE 的 ELBO 损失中的正则项
- 知识蒸馏
- 策略梯度方法（PPO）
""")

def kl_divergence(p, q):
    """KL散度 D_KL(p || q)"""
    p = np.array(p)
    q = np.array(q)
    p = np.clip(p, 1e-15, 1)
    q = np.clip(q, 1e-15, 1)
    return np.sum(p * np.log(p / q))

# 示例
p = np.array([0.4, 0.3, 0.3])
distributions_q = [
    ([0.4, 0.3, 0.3], "q = p"),
    ([0.35, 0.35, 0.3], "q 略有不同"),
    ([0.5, 0.25, 0.25], "q 有差异"),
    ([0.8, 0.1, 0.1], "q 差异较大"),
]

print(f"\n真实分布 p = {p}")
print("不同 q 分布的 KL 散度：")
for q, name in distributions_q:
    kl = kl_divergence(p, q)
    print(f"  {name}: D_KL(p||q) = {kl:.6f}")

# KL 散度的不对称性
q = np.array([0.6, 0.2, 0.2])
print(f"\np = {list(p)}, q = {list(q)}")
print(f"D_KL(p || q) = {kl_divergence(p, q):.4f}")
print(f"D_KL(q || p) = {kl_divergence(q, p):.4f}")
print("注意：KL 散度是不对称的！")

# =============================================================================
# 5. 交叉熵 vs KL 散度
# =============================================================================
print("\n" + "=" * 60)
print("【5. 交叉熵 vs KL 散度】")

print("""
关系：
    H(p, q) = H(p) + D_KL(p || q)
    交叉熵 = 熵 + KL散度

为什么深度学习用交叉熵而不是 KL 散度？
- 当 p 是真实标签（固定的），H(p) 是常数
- 最小化交叉熵 = 最小化 KL 散度
- 交叉熵计算更简单（不需要计算 H(p)）
""")

# 验证关系
p = np.array([0.7, 0.2, 0.1])
q = np.array([0.6, 0.3, 0.1])

h_p = entropy(p)
h_p_q = cross_entropy(p, q)
kl_p_q = kl_divergence(p, q)

print(f"\np = {list(p)}, q = {list(q)}")
print(f"H(p) = {h_p:.4f}")
print(f"H(p, q) = {h_p_q:.4f}")
print(f"D_KL(p || q) = {kl_p_q:.4f}")
print(f"H(p) + D_KL(p||q) = {h_p + kl_p_q:.4f}")

# =============================================================================
# 6. 可视化 KL 散度
# =============================================================================
print("\n" + "=" * 60)
print("【6. 可视化 KL 散度】")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 两个高斯分布的 KL 散度
from scipy import stats

mu_p, sigma_p = 0, 1
x = np.linspace(-5, 5, 1000)
p_pdf = stats.norm.pdf(x, mu_p, sigma_p)

mus_q = [0, 0.5, 1, 2]
ax = axes[0]
ax.plot(x, p_pdf, 'k-', linewidth=2, label='p ~ N(0, 1)')

for mu_q in mus_q:
    sigma_q = 1
    q_pdf = stats.norm.pdf(x, mu_q, sigma_q)
    # 高斯分布的 KL 散度有解析解
    kl = 0.5 * ((sigma_p/sigma_q)**2 + (mu_q - mu_p)**2 / sigma_q**2 - 1 + 2*np.log(sigma_q/sigma_p))
    ax.plot(x, q_pdf, '--', linewidth=2, label=f'q ~ N({mu_q}, 1), KL={kl:.4f}')

ax.set_xlabel('x')
ax.set_ylabel('概率密度')
ax.set_title('KL散度：均值变化的影响', fontsize=12, fontweight='bold')
ax.legend()

sigmas_q = [0.5, 1, 1.5, 2]
ax = axes[1]
ax.plot(x, p_pdf, 'k-', linewidth=2, label='p ~ N(0, 1)')

for sigma_q in sigmas_q:
    mu_q = 0
    q_pdf = stats.norm.pdf(x, mu_q, sigma_q)
    kl = 0.5 * ((sigma_p/sigma_q)**2 + (mu_q - mu_p)**2 / sigma_q**2 - 1 + 2*np.log(sigma_q/sigma_p))
    ax.plot(x, q_pdf, '--', linewidth=2, label=f'q ~ N(0, {sigma_q}²), KL={kl:.4f}')

ax.set_xlabel('x')
ax.set_ylabel('概率密度')
ax.set_title('KL散度：方差变化的影响', fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/07_kl_divergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/07_kl_divergence.png")

# =============================================================================
# 7. 深度学习中的应用
# =============================================================================
print("\n" + "=" * 60)
print("【7. 深度学习中的应用】")

print("""
╔════════════════════════════════════════════════════════════════╗
║                   信息论在深度学习中的应用                      ║
╠════════════════╦═══════════════════════════════════════════════╣
║  交叉熵损失    ║  分类任务的标准损失函数                        ║
║  BCELoss       ║  二分类：-[y×log(p) + (1-y)×log(1-p)]         ║
║  CELoss        ║  多分类：-Σy_i×log(p_i)                       ║
╠════════════════╬═══════════════════════════════════════════════╣
║  VAE           ║  ELBO = E[log p(x|z)] - D_KL(q(z|x) || p(z)) ║
║  知识蒸馏      ║  用 KL 散度让学生模型模仿教师模型              ║
║  PPO           ║  限制策略更新：D_KL(π_new || π_old) < ε       ║
║  Focal Loss    ║  修改交叉熵解决类别不平衡                      ║
╚════════════════╩═══════════════════════════════════════════════╝
""")

# 实现交叉熵损失
print("\n实现分类损失：")

def binary_cross_entropy(y_true, y_pred):
    """二分类交叉熵"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    """多分类交叉熵（y_true 是 one-hot）"""
    y_pred = np.clip(y_pred, 1e-15, 1)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 二分类示例
y_true_binary = np.array([1, 0, 1, 1, 0])
y_pred_binary = np.array([0.9, 0.2, 0.8, 0.7, 0.3])
bce = binary_cross_entropy(y_true_binary, y_pred_binary)
print(f"二分类: y_true={y_true_binary}, y_pred={y_pred_binary}")
print(f"BCELoss = {bce:.4f}")

# 多分类示例
y_true_multi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_multi = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
cce = categorical_cross_entropy(y_true_multi, y_pred_multi)
print(f"\n多分类 y_true (one-hot):\n{y_true_multi}")
print(f"y_pred:\n{y_pred_multi}")
print(f"CELoss = {cce:.4f}")

# =============================================================================
# 8. VAE 中的 KL 散度
# =============================================================================
print("\n" + "=" * 60)
print("【8. VAE 中的 KL 散度】")

print("""
VAE 损失 = 重构损失 + KL 正则项

KL 正则项（假设 p(z) = N(0, I)）：
    D_KL(q(z|x) || p(z)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)

作用：使编码器输出的分布接近标准正态分布
""")

def vae_kl_loss(mu, log_var):
    """VAE 的 KL 损失项
    
    mu: 编码器输出的均值
    log_var: 编码器输出的对数方差
    """
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

# 示例
mu = np.array([0.5, -0.3, 0.2])
log_var = np.array([-0.5, 0.1, -0.2])  # log(σ²)

kl = vae_kl_loss(mu, log_var)
print(f"μ = {mu}")
print(f"log(σ²) = {log_var}")
print(f"KL Loss = {kl:.4f}")

# 当 μ=0, σ=1 时 KL 应该接近 0
mu_zero = np.zeros(3)
log_var_zero = np.zeros(3)  # log(1) = 0
kl_zero = vae_kl_loss(mu_zero, log_var_zero)
print(f"\n当 μ=0, σ=1 时: KL Loss = {kl_zero:.4f} (应该接近 0)")

# =============================================================================
# 9. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 计算分布 [0.5, 0.3, 0.2] 的熵

2. 给定真实标签 [0, 1, 0] 和预测 [0.1, 0.7, 0.2]，计算交叉熵

3. 计算以下两个分布的 KL 散度：
   p = [0.3, 0.4, 0.3]
   q = [0.25, 0.5, 0.25]

4. 实现 Label Smoothing：将 one-hot 标签 [1, 0, 0] 平滑为 [0.9, 0.05, 0.05]

5. 实现 Focal Loss：FL = -(1-p)^γ × log(p)，其中 γ=2
""")

# === 练习代码 ===
# 练习 1: 计算熵
# p = np.array([0.5, 0.3, 0.2])
# H = -np.sum(p * np.log2(p))
# print(f"H([0.5, 0.3, 0.2]) = {H:.4f} bits")

# 练习 2: 交叉熵
# y_true = np.array([0, 1, 0])
# y_pred = np.array([0.1, 0.7, 0.2])
# CE = -np.sum(y_true * np.log(y_pred))
# print(f"交叉熵 = {CE:.4f}")

# 练习 3: KL 散度
# p = np.array([0.3, 0.4, 0.3])
# q = np.array([0.25, 0.5, 0.25])
# KL = np.sum(p * np.log(p / q))
# print(f"D_KL(p||q) = {KL:.6f}")

# 练习 4: Label Smoothing
# def label_smoothing(one_hot, epsilon=0.1):
#     n_classes = len(one_hot)
#     return one_hot * (1 - epsilon) + epsilon / n_classes
# y = np.array([1, 0, 0])
# y_smooth = label_smoothing(y, epsilon=0.1)
# print(f"平滑后: {y_smooth}")  # [0.9, 0.05, 0.05]

# 练习 5: Focal Loss
# def focal_loss(p, gamma=2):
#     return -((1 - p)**gamma) * np.log(p)
# for p in [0.1, 0.5, 0.9]:
#     ce = -np.log(p)
#     fl = focal_loss(p)
#     print(f"p={p}: CE={ce:.4f}, Focal={fl:.4f}")

print("\n✅ 熵与KL散度完成！")
print("🎉 Phase 2 全部模块已完成！")
print("\n下一步：进入 Phase 3 - PyTorch 核心技能")
