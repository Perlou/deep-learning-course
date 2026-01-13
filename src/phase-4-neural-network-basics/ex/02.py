import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("神经网络基础 - 多层感知机 (MLP)")
print("=" * 60)


print("\n" + "=" * 60)
print("【2. 用 MLP 解决 XOR】")

X_xor = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = torch.FloatTensor([[0], [1], [1], [0]])


class MLP_XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


# 训练
model = MLP_XOR()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1.0)

print("训练 MLP 解决 XOR:")
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_xor)
    loss = criterion(outputs, y_xor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 200 == 0:
        preds = (outputs > 0.5).float()
        acc = (preds == y_xor).float().mean()
        print(f"  Epoch {epoch + 1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

# 测试
print("\n最终预测:")
with torch.no_grad():
    preds = model(X_xor)
    for xi, yi, pi in zip(X_xor, y_xor, preds):
        xi_list = [int(x) for x in xi.tolist()]
        print(f"  {xi_list} -> 预测: {pi.item():.4f}, 真实: {int(yi.item())}")

# =============================================================================
# 3. 可视化决策边界
# =============================================================================
print("\n" + "=" * 60)
print("【3. 可视化决策边界】")


def plot_decision_boundary_mlp(model, X, y):
    """可视化 MLP 的决策边界"""
    plt.figure(figsize=(12, 4))

    # 子图1：决策边界
    plt.subplot(1, 3, 1)

    # 创建网格
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.8)
    plt.colorbar(label="输出概率")

    # 画数据点
    colors = ["red" if yi == 0 else "blue" for yi in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors="black", zorder=5)
    for xi, yi in zip(X.numpy(), y.numpy()):
        plt.annotate(
            f"{int(yi[0])}", xy=xi, ha="center", va="center", fontsize=12, color="white"
        )

    plt.title("MLP 决策边界 (XOR)")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # 子图2：损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练损失")
    plt.grid(True, alpha=0.3)

    # 子图3：网络结构
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.text(0.5, 0.9, "MLP 结构", ha="center", fontsize=14, fontweight="bold")
    plt.text(0.5, 0.7, "输入层: 2 个神经元", ha="center", fontsize=12)
    plt.text(0.5, 0.5, "隐藏层: 4 个神经元 + Sigmoid", ha="center", fontsize=12)
    plt.text(0.5, 0.3, "输出层: 1 个神经元 + Sigmoid", ha="center", fontsize=12)
    plt.text(
        0.5,
        0.1,
        f"总参数: {sum(p.numel() for p in model.parameters())}",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig("../outputs/mlp_xor.png", dpi=100)
    plt.close()
    print("结果已保存: outputs/mlp_xor.png")


plot_decision_boundary_mlp(model, X_xor, y_xor)

print("\n" + "=" * 60)
print("【5. 更大的 MLP - 分类螺旋数据】")


def generate_spiral_data(n_samples=100, n_classes=2):
    X = []
    y = []
    for i in range(n_classes):
        r = np.linspace(0.1, 1, n_samples)
        theta = (
            np.linspace(i * 4, (i + 1) * 4, n_samples)
            + np.random.randn(n_samples) * 0.2
        )
        X.append(
            np.column_stack(
                [
                    r * np.cos(theta),
                ]
            )
        )
