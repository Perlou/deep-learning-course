"""
03-vanishing-gradient.py - 梯度消失/爆炸问题

本节学习:
1. 梯度消失的原因
2. 梯度爆炸的原因
3. 可视化梯度问题
4. 解决方案
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("第3节: 梯度消失/爆炸问题")
print("=" * 60)

# =============================================================================
# 1. 梯度消失的原因
# =============================================================================
print("""
📚 梯度消失的数学原理

链式法则导致的梯度衰减:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ∂hₜ/∂h₁ = ∂hₜ/∂hₜ₋₁ · ∂hₜ₋₁/∂hₜ₋₂ · ... · ∂h₂/∂h₁         │
│                                                              │
│  其中每一项:                                                  │
│    ∂hₖ/∂hₖ₋₁ = Wₕₕᵀ · diag(tanh'(zₖ))                       │
│                                                              │
│  tanh 的导数范围: (0, 1]                                      │
│  最大值 = 1 (当 z = 0 时)                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

关键问题:
  • 如果 ||Wₕₕ|| < 1 且 |tanh'| < 1: 梯度指数级衰减 → 消失
  • 如果 ||Wₕₕ|| > 1: 梯度指数级增长 → 爆炸
""")


# =============================================================================
# 2. 可视化 tanh 及其导数
# =============================================================================
print("\n" + "=" * 60)
print("📌 2. 可视化 tanh 及其导数")
print("-" * 60)

x = np.linspace(-5, 5, 200)
y_tanh = np.tanh(x)
y_tanh_deriv = 1 - np.tanh(x) ** 2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# tanh 函数
axes[0].plot(x, y_tanh, "b-", linewidth=2, label="tanh(x)")
axes[0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
axes[0].axhline(y=1, color="r", linestyle="--", alpha=0.5)
axes[0].axhline(y=-1, color="r", linestyle="--", alpha=0.5)
axes[0].set_xlabel("x")
axes[0].set_ylabel("tanh(x)")
axes[0].set_title("tanh 函数")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# tanh 导数
axes[1].plot(x, y_tanh_deriv, "orange", linewidth=2, label="tanh'(x)")
axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.5, label="最大值=1")
axes[1].axhline(y=0.25, color="g", linestyle="--", alpha=0.5)
axes[1].fill_between(x, y_tanh_deriv, alpha=0.3)
axes[1].set_xlabel("x")
axes[1].set_ylabel("tanh'(x)")
axes[1].set_title("tanh 导数 (最大值=1, |x|>2 时接近0)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/tanh_derivative.png", dpi=100)
plt.close()
print("tanh 导数图已保存: outputs/tanh_derivative.png")


# =============================================================================
# 3. 梯度消失的数值实验
# =============================================================================
print("\n" + "=" * 60)
print("📌 3. 梯度消失的数值实验")
print("-" * 60)


def compute_gradient_norms(model, seq_len, input_size=10, hidden_size=50):
    """计算不同时间步的梯度范数"""
    x = torch.randn(1, seq_len, input_size, requires_grad=True)
    h0 = torch.zeros(1, 1, hidden_size)

    output, _ = model(x, h0)

    # 只对最后一个时刻的输出求导
    loss = output[:, -1, :].sum()
    loss.backward()

    # 计算输入各时间步的梯度范数
    grad_norms = []
    for t in range(seq_len):
        grad_norm = x.grad[0, t, :].norm().item()
        grad_norms.append(grad_norm)

    return grad_norms


# 测试不同序列长度
seq_lengths = [10, 25, 50, 100]
hidden_size = 50
input_size = 10

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, seq_len in enumerate(seq_lengths):
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    grad_norms = compute_gradient_norms(rnn, seq_len, input_size, hidden_size)

    # 从最后时刻往前看梯度
    time_steps = list(range(seq_len))
    distance_from_end = [seq_len - t for t in time_steps]

    axes[idx].plot(distance_from_end, grad_norms, "b-o", markersize=3)
    axes[idx].set_xlabel("距离最后时刻的步数")
    axes[idx].set_ylabel("梯度范数")
    axes[idx].set_title(f"序列长度 = {seq_len}")
    axes[idx].set_yscale("log")  # 对数刻度更清楚
    axes[idx].grid(True, alpha=0.3)
    axes[idx].invert_xaxis()  # 反转 x 轴

plt.suptitle("RNN 梯度消失现象 (越远离输出，梯度越小)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/gradient_vanishing.png", dpi=100)
plt.close()
print("梯度消失可视化已保存: outputs/gradient_vanishing.png")


# =============================================================================
# 4. 梯度爆炸实验
# =============================================================================
print("\n" + "=" * 60)
print("📌 4. 梯度爆炸实验")
print("-" * 60)


def demonstrate_gradient_explosion():
    """演示梯度爆炸"""
    # 使用较大的权重初始化
    rnn = nn.RNN(10, 50, batch_first=True)

    # 人为放大权重
    with torch.no_grad():
        rnn.weight_hh_l0 *= 2.0

    x = torch.randn(1, 50, 10, requires_grad=True)
    h0 = torch.zeros(1, 1, 50)

    output, _ = rnn(x, h0)
    loss = output[:, -1, :].sum()

    try:
        loss.backward()
        grad_norm = x.grad.norm().item()
        print(f"  梯度范数: {grad_norm:.2f}")
        if grad_norm > 1000:
            print("  ⚠️ 梯度爆炸！")
        elif grad_norm < 0.01:
            print("  ⚠️ 梯度消失！")
        else:
            print("  ✓ 梯度正常")
    except RuntimeError as e:
        print(f"  发生错误: {e}")


print("\n测试梯度爆炸:")
demonstrate_gradient_explosion()


# =============================================================================
# 5. 解决方案
# =============================================================================
print("\n" + "=" * 60)
print("📌 5. 解决方案")
print("-" * 60)

print("""
解决梯度消失/爆炸的方法:

┌─────────────────────────────────────────────────────────────┐
│  方法              │ 解决问题     │ 说明                     │
├─────────────────────────────────────────────────────────────┤
│  LSTM/GRU          │ 梯度消失    │ 门控机制提供梯度高速公路  │
│  梯度裁剪          │ 梯度爆炸    │ 限制梯度范数             │
│  权重初始化        │ 两者        │ 正交初始化保持梯度幅度    │
│  LayerNorm         │ 两者        │ 归一化隐藏状态           │
│  残差连接          │ 梯度消失    │ 提供梯度直通路径         │
└─────────────────────────────────────────────────────────────┘
""")

# 梯度裁剪示例
print("\n梯度裁剪示例:")


def train_with_gradient_clipping():
    """使用梯度裁剪训练"""
    model = nn.RNN(10, 50, batch_first=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(1, 100, 10)
    h0 = torch.zeros(1, 1, 50)

    output, _ = model(x, h0)
    loss = output.sum()

    optimizer.zero_grad()
    loss.backward()

    # 裁剪前
    total_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.data.norm(2).item() ** 2
    total_norm_before = total_norm_before**0.5

    # 梯度裁剪
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # 裁剪后
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.data.norm(2).item() ** 2
    total_norm_after = total_norm_after**0.5

    print(f"  裁剪前梯度范数: {total_norm_before:.4f}")
    print(f"  裁剪后梯度范数: {total_norm_after:.4f}")
    print(f"  最大允许范数: {max_norm}")


train_with_gradient_clipping()


# =============================================================================
# 6. LSTM 如何解决梯度消失
# =============================================================================
print("\n" + "=" * 60)
print("📌 6. LSTM 如何解决梯度消失")
print("-" * 60)

print("""
LSTM 的梯度传播:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  细胞状态更新: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ                     │
│                                                              │
│  梯度: ∂Cₜ/∂Cₜ₋₁ = fₜ  (仅仅是遗忘门的值！)                  │
│                                                              │
│  对比:                                                        │
│    RNN:  ∂hₜ/∂hₜ₋₁ = Wₕₕᵀ · diag(tanh'(z))  (矩阵乘法)       │
│    LSTM: ∂Cₜ/∂Cₜ₋₁ = fₜ                      (标量乘法)       │
│                                                              │
│  当 fₜ ≈ 1 时，梯度几乎无损传播！                            │
│  这就是"信息高速公路"的含义                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")

# 对比 RNN 和 LSTM 的梯度
print("\n对比 RNN 和 LSTM 的梯度传播:")


def compare_rnn_lstm_gradients(seq_len=50):
    """对比 RNN 和 LSTM 的梯度传播"""
    input_size = 10
    hidden_size = 50

    results = {}

    for name, model_class in [("RNN", nn.RNN), ("LSTM", nn.LSTM)]:
        model = model_class(input_size, hidden_size, batch_first=True)
        x = torch.randn(1, seq_len, input_size, requires_grad=True)

        output, _ = model(x)
        loss = output[:, -1, :].sum()
        loss.backward()

        # 计算各时间步梯度范数
        grad_norms = [x.grad[0, t, :].norm().item() for t in range(seq_len)]
        results[name] = grad_norms

    return results


seq_len = 50
results = compare_rnn_lstm_gradients(seq_len)

# 可视化对比
plt.figure(figsize=(10, 5))
time_steps = list(range(seq_len))
distance_from_end = [seq_len - t for t in time_steps]

plt.plot(distance_from_end, results["RNN"], "r-o", label="RNN", markersize=3)
plt.plot(distance_from_end, results["LSTM"], "b-o", label="LSTM", markersize=3)
plt.xlabel("距离最后时刻的步数")
plt.ylabel("梯度范数 (对数刻度)")
plt.title("RNN vs LSTM 梯度传播对比")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig("outputs/rnn_vs_lstm_gradient.png", dpi=100)
plt.close()
print("RNN vs LSTM 梯度对比图已保存: outputs/rnn_vs_lstm_gradient.png")

print(f"\n  RNN 第一时刻梯度范数: {results['RNN'][0]:.6f}")
print(f"  LSTM 第一时刻梯度范数: {results['LSTM'][0]:.6f}")
print(f"  比值 (LSTM/RNN): {results['LSTM'][0] / results['RNN'][0]:.2f}x")


# =============================================================================
# 7. 练习
# =============================================================================
print("\n" + "=" * 60)
print("📝 练习题")
print("-" * 60)

print("""
1. 为什么 sigmoid 和 tanh 容易导致梯度消失？
   答：它们的导数值域在 (0, 1) 或 (0, 0.25)，多次相乘后趋于 0

2. 梯度裁剪是如何工作的？
   答：当梯度范数超过阈值时，等比例缩小所有梯度

3. LSTM 中遗忘门值为 1 时，梯度传播有什么特点？
   答：梯度可以完全无损地传递到前一时刻

4. 为什么正交初始化有助于缓解梯度问题？
   答：正交矩阵的特征值都是 1，梯度不会指数级放大或缩小
""")

print("\n✅ 第3节完成！")
print("下一节：04-lstm.py - LSTM 门控机制")
