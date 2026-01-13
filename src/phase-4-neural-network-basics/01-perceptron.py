"""
01-perceptron.py
Phase 4: 神经网络基础

单层感知机 - 神经网络的起点

学习目标：
1. 理解感知机的数学模型
2. 实现感知机学习算法
3. 理解线性可分与 XOR 问题
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("神经网络基础 - 单层感知机")
print("=" * 60)

# =============================================================================
# 1. 感知机简介
# =============================================================================
print("\n【1. 感知机简介】")

print("""
感知机 (Perceptron) 是最简单的神经网络单元：

    输出 = sign(w·x + b)

    其中：
    - x: 输入向量
    - w: 权重向量
    - b: 偏置
    - sign: 符号函数 (>0 返回 1，否则返回 -1)

几何意义：
    w·x + b = 0 定义了一个超平面，将空间分成两部分
    感知机学习的目标是找到能正确分类的超平面
""")

# =============================================================================
# 2. 手动实现感知机
# =============================================================================
print("\n" + "=" * 60)
print("【2. 手动实现感知机】")


class Perceptron:
    """感知机实现"""

    def __init__(self, n_features, learning_rate=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate
        self.history = []

    def predict(self, x):
        """预测"""
        return np.sign(np.dot(x, self.w) + self.b)

    def fit(self, X, y, max_epochs=100):
        """训练"""
        for epoch in range(max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                if pred != yi:
                    # 更新规则：w = w + lr * y * x
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1

            self.history.append(errors)

            if errors == 0:
                print(f"收敛于第 {epoch + 1} 轮")
                break

        return self.history


# 创建线性可分数据
np.random.seed(42)
X_pos = np.random.randn(20, 2) + np.array([2, 2])
X_neg = np.random.randn(20, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])
y = np.array([1] * 20 + [-1] * 20)

# 训练感知机
perceptron = Perceptron(n_features=2, learning_rate=0.1)
history = perceptron.fit(X, y)

print(f"训练完成，权重: w={perceptron.w}, b={perceptron.b:.4f}")


# 可视化决策边界
def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(10, 4))

    # 子图1：数据和决策边界
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="正类", marker="o")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c="red", label="负类", marker="x")

    # 画决策边界
    x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    if model.w[1] != 0:
        y_line = -(model.w[0] * x_line + model.b) / model.w[1]
        plt.plot(x_line, y_line, "g-", linewidth=2, label="决策边界")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：训练曲线
    plt.subplot(1, 2, 2)
    plt.plot(model.history, "b-o")
    plt.xlabel("Epoch")
    plt.ylabel("错误数")
    plt.title("训练过程")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/perceptron_result.png", dpi=100)
    plt.close()
    print("结果已保存: outputs/perceptron_result.png")


plot_decision_boundary(perceptron, X, y, "感知机分类结果")

# =============================================================================
# 3. XOR 问题
# =============================================================================
print("\n" + "=" * 60)
print("【3. XOR 问题 - 感知机的局限】")

print("""
XOR (异或) 问题：
    输入: (0,0) -> 0
    输入: (0,1) -> 1
    输入: (1,0) -> 1
    输入: (1,1) -> 0

XOR 不是线性可分的！
单层感知机无法解决 XOR，这导致了第一次 AI 寒冬。
解决方案：多层感知机 (MLP)
""")

# 尝试用感知机解决 XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([-1, 1, 1, -1])  # -1 代表 0，1 代表 1

perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
history_xor = perceptron_xor.fit(X_xor, y_xor, max_epochs=100)

# 检查结果
print("\nXOR 感知机预测:")
for xi, yi in zip(X_xor, y_xor):
    pred = perceptron_xor.predict(xi)
    status = "✓" if pred == yi else "✗"
    print(f"  {xi} -> 预测: {int(pred)}, 真实: {int(yi)} {status}")

# 可视化 XOR 问题
plt.figure(figsize=(10, 4))

# XOR 数据
plt.subplot(1, 2, 1)
colors = ["blue" if yi == 1 else "red" for yi in y_xor]
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, edgecolors="black")
for i, (xi, yi) in enumerate(zip(X_xor, y_xor)):
    plt.annotate(
        f"XOR={0 if yi == -1 else 1}", xy=xi, xytext=(xi[0] + 0.1, xi[1] + 0.1)
    )
plt.title("XOR 问题 - 线性不可分")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True, alpha=0.3)

# 线性可分 vs 不可分
plt.subplot(1, 2, 2)
# AND 门 (线性可分)
plt.scatter([0, 0, 1], [0, 1, 0], c="red", s=100, label="AND=0", marker="x")
plt.scatter([1], [1], c="blue", s=100, label="AND=1", marker="o")
# 画一条可能的决策边界
x_line = np.linspace(-0.5, 1.5, 100)
plt.plot(x_line, 1.5 - x_line, "g--", label="可能的边界")
plt.title("AND 门 - 线性可分")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/xor_problem.png", dpi=100)
plt.close()
print("XOR 问题图已保存: outputs/xor_problem.png")

# =============================================================================
# 4. 用 PyTorch 实现感知机
# =============================================================================
print("\n" + "=" * 60)
print("【4. PyTorch 实现感知机】")


class PerceptronPyTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


# 准备数据
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# 创建模型
model = PerceptronPyTorch(2)
criterion = nn.MSELoss()  # 使用 MSE 而不是符号函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        # 计算准确率
        preds = torch.sign(outputs)
        acc = (preds == y_tensor).float().mean()
        print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

print("\nPyTorch 权重:", model.linear.weight.data.numpy())
print("PyTorch 偏置:", model.linear.bias.data.numpy())

# =============================================================================
# 5. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 实现 OR 门的感知机分类器

2. 解释为什么感知机无法解决 XOR 问题（从几何角度）

3. 修改学习率，观察收敛速度的变化

4. 如果数据有噪声（类别重叠），感知机会怎样？

5. 证明：感知机收敛定理保证线性可分数据一定能收敛
""")

# === 练习答案 ===
# 1 OR 门
# X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_or = np.array([-1, 1, 1, 1])
# p = Perceptron(2)
# p.fit(X_or, y_or)

# 2 答案: XOR 需要两条直线才能分开四个点
#   (0,0) 和 (1,1) 是一类，(0,1) 和 (1,0) 是另一类
#   无法用单条直线分开

# 3 学习率影响
# 太大：可能震荡不收敛
# 太小：收敛速度慢
# 适中：快速稳定收敛

# 4 噪声数据
# 感知机会不断更新权重，无法收敛
# 需要设置最大迭代次数或使用其他算法

# 5 收敛定理：如果数据线性可分且存在间隔 γ
#   则感知机在有限步内收敛

print("\n✅ 感知机完成！")
print("下一步：02-mlp-basic.py - 多层感知机")
