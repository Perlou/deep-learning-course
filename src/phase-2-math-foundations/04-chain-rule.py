"""
04-chain-rule.py
Phase 2: 深度学习数学基础

链式法则 - 反向传播的数学基础

学习目标：
1. 理解复合函数的求导法则
2. 掌握计算图的概念
3. 理解反向传播算法的本质
4. 手动实现简单的反向传播
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("深度学习数学基础 - 链式法则与反向传播")
print("=" * 60)

# =============================================================================
# 1. 链式法则基础
# =============================================================================
print("\n【1. 链式法则基础】")

print("""
链式法则：复合函数的求导

对于 y = f(g(x))，设 u = g(x)，则：
    dy/dx = (dy/du) × (du/dx)
    
或写成：
    dy/dx = f'(g(x)) × g'(x)

这是神经网络反向传播的数学基础！
""")

# 示例 1：y = (3x + 2)^2
def example1():
    print("示例 1: y = (3x + 2)²")
    print("  设 u = 3x + 2, 则 y = u²")
    print("  dy/du = 2u, du/dx = 3")
    print("  dy/dx = 2u × 3 = 6(3x + 2)")
    
    x = 1.0
    y = (3*x + 2)**2
    # 解析导数
    dy_dx = 6 * (3*x + 2)
    # 数值验证
    h = 1e-7
    dy_dx_numerical = ((3*(x+h) + 2)**2 - (3*(x-h) + 2)**2) / (2*h)
    
    print(f"\n  在 x = {x} 处:")
    print(f"  y = {y}")
    print(f"  dy/dx (解析) = {dy_dx}")
    print(f"  dy/dx (数值) = {dy_dx_numerical:.6f}")

example1()

# 示例 2：多层复合 y = sin(exp(x²))
def example2():
    print("\n示例 2: y = sin(exp(x²))")
    print("  设 u = x², v = exp(u), y = sin(v)")
    print("  dy/dv = cos(v)")
    print("  dv/du = exp(u)")
    print("  du/dx = 2x")
    print("  dy/dx = cos(exp(x²)) × exp(x²) × 2x")
    
    x = 0.5
    y = np.sin(np.exp(x**2))
    # 解析导数
    dy_dx = np.cos(np.exp(x**2)) * np.exp(x**2) * 2*x
    # 数值验证
    h = 1e-7
    dy_dx_numerical = (np.sin(np.exp((x+h)**2)) - np.sin(np.exp((x-h)**2))) / (2*h)
    
    print(f"\n  在 x = {x} 处:")
    print(f"  y = {y:.6f}")
    print(f"  dy/dx (解析) = {dy_dx:.6f}")
    print(f"  dy/dx (数值) = {dy_dx_numerical:.6f}")

example2()

# =============================================================================
# 2. 计算图
# =============================================================================
print("\n" + "=" * 60)
print("【2. 计算图】")

print("""
神经网络可以表示为计算图：
- 节点：变量或操作
- 边：数据流

示例：简单神经元
    z = w*x + b
    a = sigmoid(z)
    L = (a - y)²
    
前向传播：从输入到输出计算每个节点
反向传播：从输出到输入计算梯度
""")

# 可视化简单的计算图
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# 节点
nodes = {
    'x': (1, 3), 'w': (1, 5), 'b': (1, 1),
    '*': (3, 4), '+': (5, 3),
    'z': (6, 3), 'σ': (7, 3), 'a': (8, 3),
    'y': (8, 1), 'L': (9, 2)
}

# 绘制节点
for name, pos in nodes.items():
    if name in ['*', '+', 'σ']:
        circle = plt.Circle(pos, 0.3, fill=True, color='lightblue', ec='black')
    else:
        circle = plt.Circle(pos, 0.3, fill=True, color='lightyellow', ec='black')
    ax.add_patch(circle)
    ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=12, fontweight='bold')

# 绘制边
edges = [
    ('x', '*'), ('w', '*'),
    ('*', '+'), ('b', '+'),
    ('+', 'z'), ('z', 'σ'), ('σ', 'a'),
    ('a', 'L'), ('y', 'L')
]

for start, end in edges:
    start_pos = nodes[start]
    end_pos = nodes[end]
    ax.annotate('', xy=end_pos, xytext=start_pos,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# 添加标签
ax.text(5, 5.5, '前向传播：z = w*x + b, a = σ(z), L = (a-y)²', fontsize=12)
ax.text(5, 0.5, '反向传播：∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w', fontsize=12, color='red')

plt.title('神经元计算图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/04_computation_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: outputs/04_computation_graph.png")

# =============================================================================
# 3. 手动实现反向传播
# =============================================================================
print("\n" + "=" * 60)
print("【3. 手动实现反向传播】")

print("""
实现一个简单神经元的前向和反向传播：
    z = w*x + b
    a = sigmoid(z)
    L = (a - y)²
""")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

class SimpleNeuron:
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()
        
    def forward(self, x, y):
        """前向传播"""
        self.x = x
        self.y = y
        
        # z = w*x + b
        self.z = self.w * x + self.b
        
        # a = sigmoid(z)
        self.a = sigmoid(self.z)
        
        # L = (a - y)^2
        self.L = (self.a - y)**2
        
        return self.L
    
    def backward(self):
        """反向传播"""
        # dL/da = 2(a - y)
        dL_da = 2 * (self.a - self.y)
        
        # da/dz = sigmoid'(z) = a(1-a)
        da_dz = self.a * (1 - self.a)
        
        # dz/dw = x, dz/db = 1
        dz_dw = self.x
        dz_db = 1
        
        # 链式法则
        # dL/dz = dL/da × da/dz
        dL_dz = dL_da * da_dz
        
        # dL/dw = dL/dz × dz/dw
        self.dw = dL_dz * dz_dw
        
        # dL/db = dL/dz × dz/db
        self.db = dL_dz * dz_db
        
        return self.dw, self.db
    
    def update(self, lr=0.1):
        """梯度下降更新"""
        self.w -= lr * self.dw
        self.b -= lr * self.db

# 训练
np.random.seed(42)
neuron = SimpleNeuron()

# 简单数据
x_data = np.array([0.5, 1.0, 1.5, 2.0])
y_data = np.array([0, 0, 1, 1])  # 二分类

print(f"初始参数: w={neuron.w:.4f}, b={neuron.b:.4f}")

losses = []
for epoch in range(100):
    epoch_loss = 0
    for x, y in zip(x_data, y_data):
        loss = neuron.forward(x, y)
        neuron.backward()
        neuron.update(lr=0.5)
        epoch_loss += loss
    losses.append(epoch_loss / len(x_data))
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={epoch_loss/len(x_data):.4f}, w={neuron.w:.4f}, b={neuron.b:.4f}")

print(f"\n最终参数: w={neuron.w:.4f}, b={neuron.b:.4f}")

# 验证预测
print("\n预测结果:")
for x, y in zip(x_data, y_data):
    z = neuron.w * x + neuron.b
    pred = sigmoid(z)
    print(f"  x={x:.1f}, y_true={y}, y_pred={pred:.4f}, 预测类别={int(pred > 0.5)}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 决策边界
x_line = np.linspace(-0.5, 3, 100)
y_line = sigmoid(neuron.w * x_line + neuron.b)
axes[1].plot(x_line, y_line, 'b-', linewidth=2, label='sigmoid(wx+b)')
axes[1].scatter(x_data, y_data, c='red', s=100, zorder=5, label='数据点')
axes[1].axhline(y=0.5, color='gray', linestyle='--', label='决策边界')
axes[1].set_xlabel('x')
axes[1].set_ylabel('预测值')
axes[1].set_title('单神经元分类', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/04_backprop_neuron.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: outputs/04_backprop_neuron.png")

# =============================================================================
# 4. 多变量链式法则
# =============================================================================
print("\n" + "=" * 60)
print("【4. 多变量链式法则】")

print("""
当函数有多个路径时，需要求和：

如果 z = f(x, y) 且 x = x(t), y = y(t)，则：
    dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)

在神经网络中：
- 一个参数可能影响多个输出
- 反向传播时需要累加所有路径的梯度
""")

# 示例：z = x*y, x = 2t, y = t^2
def multivar_chain_rule():
    print("\n示例: z = x*y, 其中 x = 2t, y = t²")
    print("  ∂z/∂x = y")
    print("  ∂z/∂y = x")
    print("  dx/dt = 2")
    print("  dy/dt = 2t")
    print("  dz/dt = y × 2 + x × 2t = 2t² × 2 + 2t × 2t = 4t² + 4t² = 8t²")
    
    t = 2.0
    x = 2 * t
    y = t**2
    z = x * y
    
    # 解析导数
    dz_dt = 8 * t**2
    
    # 数值验证
    h = 1e-7
    z1 = 2*(t+h) * (t+h)**2
    z2 = 2*(t-h) * (t-h)**2
    dz_dt_numerical = (z1 - z2) / (2*h)
    
    print(f"\n  在 t = {t} 处:")
    print(f"  z = {z}")
    print(f"  dz/dt (解析) = {dz_dt}")
    print(f"  dz/dt (数值) = {dz_dt_numerical:.6f}")

multivar_chain_rule()

# =============================================================================
# 5. 全连接层的反向传播
# =============================================================================
print("\n" + "=" * 60)
print("【5. 全连接层的反向传播】")

print("""
全连接层: y = Wx + b

前向传播:
    y = W @ x + b

反向传播（给定 dL/dy）:
    dL/dW = dL/dy @ x^T  (外积)
    dL/db = dL/dy
    dL/dx = W^T @ dL/dy  (传递给上一层)
""")

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros(out_features)
        
    def forward(self, x):
        """x: (in_features,)"""
        self.x = x
        return self.W @ x + self.b
    
    def backward(self, grad_output):
        """grad_output: dL/dy"""
        # dL/dW = dL/dy @ x^T
        self.grad_W = np.outer(grad_output, self.x)
        # dL/db = dL/dy
        self.grad_b = grad_output
        # dL/dx = W^T @ dL/dy
        grad_input = self.W.T @ grad_output
        return grad_input

# 示例
layer = LinearLayer(3, 2)
x = np.array([1.0, 2.0, 3.0])
y = layer.forward(x)

print(f"W 形状: {layer.W.shape}")
print(f"输入 x: {x}")
print(f"输出 y: {y}")

# 假设上游梯度
grad_output = np.array([0.1, -0.2])
grad_input = layer.backward(grad_output)

print(f"\n上游梯度 dL/dy: {grad_output}")
print(f"dL/dW 形状: {layer.grad_W.shape}")
print(f"dL/db: {layer.grad_b}")
print(f"传递给下层的梯度 dL/dx: {grad_input}")

# =============================================================================
# 6. 完整的两层网络反向传播
# =============================================================================
print("\n" + "=" * 60)
print("【6. 两层网络反向传播】")

class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 层 1: input -> hidden
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        # 层 2: hidden -> output
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(output_dim)
        
    def forward(self, x):
        # 层 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # 层 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.z2  # 线性输出
        
        return self.a2
    
    def backward(self, x, y):
        m = 1  # 单样本
        
        # 输出层梯度（MSE 损失）
        dz2 = (self.a2 - y)
        
        self.dW2 = np.outer(dz2, self.a1)
        self.db2 = dz2
        
        # 隐藏层梯度
        da1 = self.W2.T @ dz2
        dz1 = da1 * (self.z1 > 0)  # ReLU 导数
        
        self.dW1 = np.outer(dz1, x)
        self.db1 = dz1
        
    def update(self, lr=0.01):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

# 训练 XOR 问题
np.random.seed(42)
net = TwoLayerNet(2, 4, 1)

# XOR 数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

print("训练 XOR 问题（需要非线性）")
losses = []
for epoch in range(1000):
    epoch_loss = 0
    for x, y in zip(X, Y):
        pred = net.forward(x)
        loss = 0.5 * (pred - y)**2
        epoch_loss += loss[0]
        net.backward(x, y)
        net.update(lr=0.5)
    losses.append(epoch_loss / 4)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss={epoch_loss/4:.6f}")

print("\n预测结果:")
for x, y in zip(X, Y):
    pred = net.forward(x)
    print(f"  输入: {x}, 真实: {y[0]}, 预测: {pred[0]:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('两层网络训练 XOR', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/04_xor_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: outputs/04_xor_training.png")

# =============================================================================
# 7. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 用链式法则求 y = exp(sin(x²)) 的导数

2. 对于神经元 z = w₁x₁ + w₂x₂ + b, a = tanh(z)：
   - 写出 ∂a/∂w₁, ∂a/∂w₂, ∂a/∂b 的表达式

3. 实现 softmax + 交叉熵的反向传播：
   - softmax: p_i = exp(z_i) / Σexp(z_j)
   - 交叉熵: L = -Σy_i × log(p_i)
   - 证明: ∂L/∂z = p - y

4. 扩展两层网络，添加 L2 正则化项
""")

# === 练习代码 ===
# 练习 1: y = exp(sin(x²))
# print("y = exp(sin(x²))")
# print("设 u = x², v = sin(u), y = exp(v)")
# print("dy/dv = exp(v) = exp(sin(x²))")
# print("dv/du = cos(u) = cos(x²)")
# print("du/dx = 2x")
# print("dy/dx = exp(sin(x²)) × cos(x²) × 2x")
# # 数值验证
# x = 1.0
# dy_dx = np.exp(np.sin(x**2)) * np.cos(x**2) * 2*x
# print(f"在 x=1: dy/dx = {dy_dx:.6f}")

# 练习 2: 神经元梯度
# print("z = w₁x₁ + w₂x₂ + b, a = tanh(z)")
# print("∂a/∂w₁ = sech²(z) × x₁ = (1 - tanh²(z)) × x₁")
# print("∂a/∂w₂ = sech²(z) × x₂")
# print("∂a/∂b = sech²(z)")

# 练习 3: Softmax + 交叉熵
# print("Softmax: p_i = exp(z_i) / Σexp(z_j)")
# print("交叉熵: L = -Σy_i × log(p_i)")
# print("证明: ∂L/∂z_i = p_i - y_i")
# print("推导:")
# print("  对于真实类别 i: ∂L/∂z_i = p_i - 1")
# print("  对于其他类别 j: ∂L/∂z_j = p_j")
# print("  合并 (y 为 one-hot): ∂L/∂z = p - y")
# 
# z = np.array([2.0, 1.0, 0.1])
# y = np.array([1, 0, 0])  # one-hot
# p = np.exp(z) / np.sum(np.exp(z))
# grad = p - y
# print(f"logits: {z}")
# print(f"softmax: {np.round(p, 4)}")
# print(f"∂L/∂z = p - y = {np.round(grad, 4)}")

# 练习 4: L2 正则化
# print("L_total = L_data + (λ/2)||W||²")
# print("∂L_total/∂W = ∂L_data/∂W + λW")
# print("实现: grad_W_total = grad_W + lambda * W")

print("\n✅ 链式法则完成！")
print("下一步：05-optimization-basics.py - 梯度下降优化")
