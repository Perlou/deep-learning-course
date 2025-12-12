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

# =============================================================================
# 1. Autograd 简介
# =============================================================================
print("\n【1. Autograd 简介】")

print("""
PyTorch 的自动微分机制：
- 动态计算图：每次前向传播创建新的计算图
- requires_grad=True：告诉 PyTorch 追踪这个张量的操作
- backward()：反向传播，计算梯度
- .grad：存储计算得到的梯度
""")

# =============================================================================
# 2. 基本梯度计算
# =============================================================================
print("\n【2. 基本梯度计算】")

# 创建需要梯度的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")

# 前向传播：y = x^2 + 2x
y = x ** 2 + 2 * x
print(f"y = x² + 2x = {y}")

# 计算标量损失
loss = y.sum()
print(f"loss = y.sum() = {loss}")

# 反向传播
loss.backward()

# 查看梯度: dy/dx = 2x + 2
print(f"\n梯度 x.grad = {x.grad}")
print(f"理论值 dy/dx = 2x + 2 = {2 * x.detach() + 2}")

# =============================================================================
# 3. 计算图
# =============================================================================
print("\n" + "=" * 60)
print("【3. 计算图】")

print("""
计算图关键属性：
- .grad_fn: 创建该张量的函数（反向传播入口）
- .is_leaf: 是否为叶子节点（用户创建的张量）
- .grad: 只有叶子节点才会保留梯度
""")

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y ** 2
loss = z.sum()

print(f"x: requires_grad={x.requires_grad}, is_leaf={x.is_leaf}, grad_fn={x.grad_fn}")
print(f"y: requires_grad={y.requires_grad}, is_leaf={y.is_leaf}, grad_fn={y.grad_fn}")
print(f"z: requires_grad={z.requires_grad}, is_leaf={z.is_leaf}, grad_fn={z.grad_fn}")

loss.backward()
print(f"\n只有叶子节点 x 有梯度: {x.grad}")

# =============================================================================
# 4. 梯度累积
# =============================================================================
print("\n" + "=" * 60)
print("【4. 梯度累积】")

print("""
⚠️ 重要：PyTorch 的梯度是累积的！
每次 backward() 会把梯度加到 .grad 上，而不是替换。
需要手动清零：x.grad.zero_() 或 optimizer.zero_grad()
""")

x = torch.tensor([1.0], requires_grad=True)

# 第一次
y = x ** 2
y.backward()
print(f"第一次 backward: x.grad = {x.grad}")

# 第二次（累积）
y = x ** 2
y.backward()
print(f"第二次 backward (累积): x.grad = {x.grad}")

# 清零后再计算
x.grad.zero_()
y = x ** 2
y.backward()
print(f"清零后 backward: x.grad = {x.grad}")

# =============================================================================
# 5. 禁用梯度追踪
# =============================================================================
print("\n" + "=" * 60)
print("【5. 禁用梯度追踪】")

x = torch.tensor([1.0, 2.0], requires_grad=True)

# 方法 1: torch.no_grad()
print("方法 1: with torch.no_grad():")
with torch.no_grad():
    y = x * 2
    print(f"  y.requires_grad = {y.requires_grad}")

# 方法 2: .detach()
print("\n方法 2: .detach():")
y = x.detach()
print(f"  y.requires_grad = {y.requires_grad}")
print(f"  x 和 y 共享数据: {x.data_ptr() == y.data_ptr()}")

# 方法 3: @torch.no_grad() 装饰器
@torch.no_grad()
def inference(x):
    return x * 2

print("\n方法 3: @torch.no_grad() 装饰器:")
y = inference(x)
print(f"  y.requires_grad = {y.requires_grad}")

# 方法 4: torch.inference_mode() (更高效)
print("\n方法 4: with torch.inference_mode():")
with torch.inference_mode():
    y = x * 2
    print(f"  y.requires_grad = {y.requires_grad}")

# =============================================================================
# 6. 非标量反向传播
# =============================================================================
print("\n" + "=" * 60)
print("【6. 非标量反向传播】")

print("""
backward() 只能对标量调用。
如果 y 不是标量，需要传入 gradient 参数：
    y.backward(gradient=v)  # 计算 v·(dy/dx)
""")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # y 是向量

# 方法 1: 先 sum 变标量
loss = y.sum()
loss.backward()
print(f"方法 1 (sum): x.grad = {x.grad}")

# 清零重来
x.grad.zero_()

# 方法 2: 传入 gradient
gradient = torch.tensor([1.0, 1.0, 1.0])  # 权重向量
y = x ** 2
y.backward(gradient=gradient)
print(f"方法 2 (gradient): x.grad = {x.grad}")

# =============================================================================
# 7. 保留计算图
# =============================================================================
print("\n" + "=" * 60)
print("【7. 保留计算图】")

print("""
默认情况下，backward() 后计算图被释放。
如果需要多次 backward（如 MAML），使用 retain_graph=True
""")

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# 第一次 backward，保留计算图
y.backward(retain_graph=True)
print(f"第一次 backward: x.grad = {x.grad}")

# 第二次 backward（需要 retain_graph）
x.grad.zero_()
y.backward()  # 如果上面没有 retain_graph=True，这里会报错
print(f"第二次 backward: x.grad = {x.grad}")

# =============================================================================
# 8. 实用示例：简单线性回归
# =============================================================================
print("\n" + "=" * 60)
print("【8. 实用示例：简单线性回归】")

# 数据
torch.manual_seed(42)
X = torch.randn(100, 1)
y_true = 3 * X + 2 + 0.1 * torch.randn(100, 1)

# 参数 (需要梯度)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
lr = 0.1
for epoch in range(100):
    # 前向传播
    y_pred = X @ w + b
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 梯度下降（在 no_grad 中更新，避免追踪）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n最终: w={w.item():.4f} (真实=3), b={b.item():.4f} (真实=2)")

# =============================================================================
# 9. 练习题
# =============================================================================
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

# === 练习答案 ===
# 1
# x = torch.tensor(2.0, requires_grad=True)
# f = x**3 + 2*x**2 - x + 1
# f.backward()
# print(f"f'(2) = {x.grad}")  # 3x² + 4x - 1 = 12 + 8 - 1 = 19

# 2
# W = torch.randn(2, 2, requires_grad=True)
# x = torch.randn(2)
# L = (W @ x).sum()
# L.backward()
# print(f"dL/dW:\n{W.grad}")

# 3
# x = torch.tensor(0.0, requires_grad=True)
# for i in range(50):
#     f = (x - 5) ** 2
#     f.backward()
#     with torch.no_grad():
#         x -= 0.1 * x.grad
#     x.grad.zero_()
# print(f"最优 x = {x.item()}")  # 应该接近 5

# 4
# 答案：如果不用 no_grad，参数更新操作会被记录到计算图中，
# 导致：1) 内存泄漏  2) 下次 backward 时计算错误的梯度

# 5
# accumulation_steps = 4
# for i, (x, y) in enumerate(dataloader):
#     loss = model(x, y) / accumulation_steps
#     loss.backward()  # 梯度累积
#     if (i + 1) % accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()

print("\n✅ 自动微分完成！")
print("下一步：04-nn-module.py - nn.Module 深入")
