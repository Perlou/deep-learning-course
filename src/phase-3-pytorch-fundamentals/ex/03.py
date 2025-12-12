"""
03-tensor-autograd.py
Phase 3: PyTorch 核心技能

自动微分 (Autograd) - 深度学习的核心机制

学习目标：
1. 理解计算图的概念
2. 掌握 requires_grad 和梯度计算
3. 理解 backward() 的工作原理
4. 掌握梯度控制方法
"""

import torch

print("=" * 60)
print("PyTorch 核心技能 - 自动微分")
print("=" * 60)

print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 计算 f(x) = x³ + 2x² - x + 1 在 x=2 处的导数

2. 创建一个 2x2 矩阵 W（需要梯度），计算 L = (W @ x).sum() 的梯度

3. 实现一个简单的二次函数最小化：f(x) = (x - 5)²

4. 解释为什么在更新参数时需要 with torch.no_grad()

5. 实现梯度累积：累积 4 个 mini-batch 的梯度后再更新
""")

print("\n" + "=" * 60)
print("【8. 实用示例：简单线性回归】")

torch.manual_seed(42)
X = torch.randn(100, 1)
y_true = 3 * X + 2 + 0.1 * torch.randn(100, 1)

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
for epoch in range(100):
    y_pred = X @ w + b
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n最终: w={w.item():.4f} (真实=3), b={b.item():.4f} (真实=2)")

# 1
x = torch.tensor(2.0, requires_grad=True)
f = x **3 + 2*x**2 - x + 1
f.backward()
print(f"f'(2) = {x.grad}")

# 2
W = torch.randn(2, 2, requires_grad=True)
x = torch.randn(2)
L = (W @ x).sum()
L.backward()
print(f"dL/dW:\n{W.grad}")

# 3
x = torch.tensor(0.0, requires_grad=True)
for i in range(50):
    f = (x - 5) ** 2
    f.backward()
    with torch.no_grad():
        x -= 0.1 * x.grad
    x.grad.zero_()
print(f"最优 x = {x.item()}")  # 应该接近 5

# 4
# 答案：如果不用 no_grad，参数更新操作会被记录到计算图中，
# 导致：1) 内存泄漏  2) 下次 backward 时计算错误的梯度

# 5
# accumulation_steps = 4
# for i, (x, y) in enumerate(dataloader):
#     loss = model(x, y)/ accumulation_steps
#     loss.backward()
#     if (i + 1) % accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()
