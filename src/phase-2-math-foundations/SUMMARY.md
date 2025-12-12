# Phase 2 学习总结：深度学习数学基础

## 📚 模块概览

| 模块 | 主题           | 核心概念                 |
| ---- | -------------- | ------------------------ |
| 01   | 向量与矩阵     | 点积、矩阵乘法、线性变换 |
| 02   | 特征分解与 SVD | 特征值、SVD、PCA、LoRA   |
| 03   | 偏导数与梯度   | 偏导数、梯度、梯度下降   |
| 04   | 链式法则       | 复合函数求导、反向传播   |
| 05   | 优化基础       | SGD、Momentum、Adam      |
| 06   | 概率论基础     | 概率分布、贝叶斯定理     |
| 07   | 熵与 KL 散度   | 熵、交叉熵、KL 散度      |

---

## 1️⃣ 向量与矩阵

### 核心公式

```
点积: a · b = Σ aᵢbᵢ = ||a|| ||b|| cos(θ)
矩阵乘法: C[i,j] = Σ A[i,k] × B[k,j]
形状规则: (m×n) @ (n×p) = (m×p)
```

### 深度学习应用

- **全连接层**: `y = Wx + b`
- **注意力机制**: `Attention = softmax(QK^T/√d)V`
- **线性变换**: 旋转、缩放、剪切

### 关键代码

```python
# 点积
dot = np.dot(a, b)  # 或 a @ b

# 矩阵乘法
Y = X @ W + b  # 全连接层

# 批量矩阵乘法 (注意力)
scores = Q @ K.transpose(0, 2, 1)
```

---

## 2️⃣ 特征分解与 SVD

### 核心公式

```
特征分解: A = Q Λ Q^T  (对称矩阵)
SVD分解: A = U Σ V^T
低秩近似: A_k = U[:,:k] @ Σ[:k,:k] @ V[:k,:]^T
```

### PCA 步骤

1. 中心化: `X̃ = X - mean(X)`
2. SVD: `U, s, Vt = svd(X̃)`
3. 投影: `Z = X̃ @ V[:, :k]`

### LoRA 参数高效微调

```
ΔW = A @ B
A: (d × r), B: (r × k), r << min(d, k)
压缩比: dk / (dr + rk)
```

### 关键代码

```python
# 特征分解
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# PCA 降维
X_2d = X_centered @ V[:, :2]
var_explained = np.sum(s[:k]**2) / np.sum(s**2)
```

---

## 3️⃣ 偏导数与梯度

### 核心公式

```
偏导数: ∂f/∂x = lim[h→0] (f(x+h, y) - f(x, y)) / h
梯度: ∇f = [∂f/∂x, ∂f/∂y, ...]
梯度下降: θ_new = θ_old - lr × ∇f(θ)
```

### 常用导数

| 函数   | 导数        |
| ------ | ----------- |
| x^n    | n × x^(n-1) |
| e^x    | e^x         |
| ln(x)  | 1/x         |
| sin(x) | cos(x)      |
| cos(x) | -sin(x)     |

### 关键代码

```python
# 梯度下降
for _ in range(n_epochs):
    grad = compute_gradient(theta)
    theta = theta - lr * grad
```

---

## 4️⃣ 链式法则与反向传播

### 核心公式

```
链式法则: dy/dx = (dy/du) × (du/dx)
多路径: dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
```

### 反向传播（神经元）

```
前向: z = wx + b, a = σ(z), L = (a - y)²
反向: dL/dw = dL/da × da/dz × dz/dw
```

### 全连接层梯度

```
前向: Y = W @ X + b
反向: dL/dW = dL/dY @ X^T
      dL/db = dL/dY
      dL/dX = W^T @ dL/dY
```

### Softmax + 交叉熵

```
∂L/∂z = p - y  (重要结论!)
```

---

## 5️⃣ 优化基础

### 优化器公式

| 优化器   | 更新规则                                               |
| -------- | ------------------------------------------------------ |
| SGD      | θ = θ - η∇f                                            |
| Momentum | v = βv + ∇f, θ = θ - ηv                                |
| RMSprop  | s = βs + (1-β)g², θ = θ - ηg/√s                        |
| Adam     | m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², θ = θ - η(m̂/√v̂) |

### Adam 默认超参数

```python
lr = 0.001
beta1 = 0.9   # 一阶矩
beta2 = 0.999 # 二阶矩
eps = 1e-8
```

### 学习率调度

- **Step Decay**: 每 N epoch 减半
- **Cosine Annealing**: `lr = lr_min + (lr_max - lr_min)(1 + cos(πt/T))/2`
- **Warmup**: 前几个 epoch 线性增加

---

## 6️⃣ 概率论基础

### 核心公式

```
条件概率: P(A|B) = P(A∩B) / P(B)
贝叶斯定理: P(A|B) = P(B|A)P(A) / P(B)
期望: E[X] = ∫ x p(x) dx
方差: Var(X) = E[(X - μ)²]
```

### 常见分布

| 分布        | 用途            |
| ----------- | --------------- |
| Bernoulli   | 二分类          |
| Categorical | 多分类          |
| Gaussian    | 权重初始化、VAE |
| Uniform     | 随机采样        |

### 重参数化技巧 (VAE)

```python
# 可微采样
z = mu + sigma * epsilon  # epsilon ~ N(0, 1)
```

---

## 7️⃣ 熵与 KL 散度

### 核心公式

```
信息量: I(x) = -log₂(p(x))
熵: H(X) = -Σ p(x) log₂(p(x))
交叉熵: H(p, q) = -Σ p(x) log(q(x))
KL散度: D_KL(p||q) = Σ p(x) log(p(x)/q(x)) = H(p,q) - H(p)
```

### 关系

```
交叉熵 = 熵 + KL散度
H(p, q) = H(p) + D_KL(p || q)
```

### 损失函数

```python
# 二分类交叉熵
BCE = -[y*log(p) + (1-y)*log(1-p)]

# 多分类交叉熵
CE = -Σ y_i * log(p_i)

# VAE KL 损失
KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

---

## 🔗 概念连接图

```
向量/矩阵 ──→ 线性变换 ──→ 全连接层
    │                        │
    ↓                        ↓
特征分解 ──→ PCA/LoRA    前向传播
    │                        │
    ↓                        ↓
梯度/偏导 ──→ 链式法则 ──→ 反向传播
    │                        │
    ↓                        ↓
优化器 ───→ 损失下降 ←── 交叉熵损失
    ↑                        ↑
    └── 概率分布 ←── 熵/KL散度 ──┘
```

---

## ✅ 自检清单

- [ ] 能计算矩阵乘法并理解形状变化
- [ ] 能解释特征值/SVD 的几何意义
- [ ] 能手推简单函数的梯度
- [ ] 理解反向传播的链式法则
- [ ] 能解释 SGD、Momentum、Adam 的区别
- [ ] 理解交叉熵损失的来源
- [ ] 能计算 KL 散度并解释其意义

---

## 📖 推荐资源

1. [3Blue1Brown 线性代数](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
2. [深度学习花书 第 2-4 章](https://www.deeplearningbook.org/)
3. [李宏毅机器学习](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)
