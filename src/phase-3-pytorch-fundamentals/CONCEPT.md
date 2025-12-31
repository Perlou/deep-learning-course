# PyTorch 深度解析：从零开始的完整指南

---

## 📑 目录

1. [PyTorch 简介](#1-pytorch-简介)
2. [安装与环境配置](#2-安装与环境配置)
3. [张量(Tensor)详解](#3-张量tensor详解)
4. [自动求导(Autograd)机制](#4-自动求导autograd机制)
5. [神经网络模块(nn.Module)](#5-神经网络模块nnmodule)
6. [损失函数与优化器](#6-损失函数与优化器)
7. [数据加载与处理](#7-数据加载与处理)
8. [完整训练流程](#8-完整训练流程)
9. [模型保存与加载](#9-模型保存与加载)
10. [GPU 加速](#10-gpu-加速)
11. [高级主题](#11-高级主题)
12. [实战案例](#12-实战案例)
13. [最佳实践与调试技巧](#13-最佳实践与调试技巧)

---

## 1. PyTorch 简介

### 1.1 什么是 PyTorch

PyTorch 是一个开源的深度学习框架，由 Facebook（现 Meta）的 AI 研究团队开发。

```
┌─────────────────────────────────────────────────────────────┐
│                      PyTorch 架构                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ torchvision │  │  torchaudio │  │     torchtext       │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         └────────────────┼────────────────────┘              │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    torch.nn                            │  │
│  │    (神经网络层、损失函数、容器)                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │                  torch.autograd                        │  │
│  │              (自动微分引擎)                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │                   torch.Tensor                         │  │
│  │              (多维数组/张量)                            │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              CUDA / CPU Backend                        │  │
│  │            (底层计算引擎)                               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 PyTorch vs TensorFlow 对比

| 特性     | PyTorch                  | TensorFlow                   |
| -------- | ------------------------ | ---------------------------- |
| 计算图   | 动态图（默认）           | 静态图（1.x）/ 动态图（2.x） |
| 调试     | 简单，可用 Python 调试器 | 相对复杂                     |
| 学习曲线 | 较平缓                   | 较陡峭                       |
| 部署     | TorchScript / ONNX       | TensorFlow Serving / TFLite  |
| 社区     | 学术界主流               | 工业界广泛                   |

### 1.3 核心特性

```python
# PyTorch 的核心特性演示
import torch

# 1. 动态计算图 - 每次前向传播构建新图
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # 计算图在此动态构建
y.sum().backward()

# 2. Pythonic - 与 Python 生态无缝集成
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor = torch.from_numpy(numpy_array)

# 3. GPU 加速 - 简单的设备切换
if torch.cuda.is_available():
    tensor = tensor.cuda()
```

---

## 2. 安装与环境配置

### 2.1 安装方法

```bash
# CPU 版本
pip install torch torchvision torchaudio

# CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Conda 安装
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2.2 验证安装

```python
import torch

# 基本信息
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"GPU 数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
```

---

## 3. 张量(Tensor)详解

### 3.1 张量基础

张量是 PyTorch 中最核心的数据结构，是一个多维数组。

```
┌─────────────────────────────────────────────────────────────┐
│                      张量维度示意                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   标量 (0-D)      向量 (1-D)      矩阵 (2-D)      3-D张量    │
│                                                             │
│      5            [1,2,3]        [[1,2],         [[[1,2],   │
│                                   [3,4]]           [3,4]],  │
│                                                   [[5,6],   │
│     ( )           ─────          ┌───┐            [7,8]]]   │
│                                  │   │                      │
│                                  └───┘           ┌─────┐    │
│                                                  │┌───┐│    │
│                                                  ││   ││    │
│                                                  │└───┘│    │
│                                                  └─────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 创建张量

```python
import torch

# ==================== 1. 从数据创建 ====================
# 从列表创建
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([[1, 2], [3, 4]])

# 从 NumPy 创建（共享内存）
import numpy as np
np_array = np.array([1, 2, 3])
t3 = torch.from_numpy(np_array)

# 从另一个张量创建
t4 = torch.tensor(t1)           # 复制数据
t5 = t1.clone()                 # 克隆
t6 = t1.detach()                # 分离计算图

# ==================== 2. 特殊张量 ====================
# 全零/全一
zeros = torch.zeros(3, 4)       # 3x4 全零矩阵
ones = torch.ones(3, 4)         # 3x4 全一矩阵
full = torch.full((3, 4), 7)    # 3x4 全7矩阵

# 单位矩阵
eye = torch.eye(4)              # 4x4 单位矩阵

# 未初始化（随机值，速度快）
empty = torch.empty(3, 4)

# ==================== 3. 随机张量 ====================
# 均匀分布 [0, 1)
rand = torch.rand(3, 4)

# 标准正态分布 N(0, 1)
randn = torch.randn(3, 4)

# 指定范围的随机整数
randint = torch.randint(0, 10, (3, 4))  # [0, 10) 的整数

# 指定正态分布
normal = torch.normal(mean=0, std=1, size=(3, 4))

# ==================== 4. 序列张量 ====================
# 等差序列
arange = torch.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1]
logspace = torch.logspace(0, 2, 3)      # [1, 10, 100]

# ==================== 5. 类似形状的张量 ====================
x = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(x)        # 相同形状的全零
ones_like = torch.ones_like(x)          # 相同形状的全一
rand_like = torch.rand_like(x.float())  # 相同形状的随机
```

### 3.3 张量属性

```python
import torch

t = torch.randn(3, 4, 5)

# 基本属性
print(f"形状: {t.shape}")           # torch.Size([3, 4, 5])
print(f"维度: {t.dim()}")           # 3
print(f"元素总数: {t.numel()}")     # 60
print(f"数据类型: {t.dtype}")       # torch.float32
print(f"设备: {t.device}")          # cpu
print(f"是否需要梯度: {t.requires_grad}")  # False

# 步长（内存布局）
print(f"步长: {t.stride()}")        # (20, 5, 1)
print(f"是否连续: {t.is_contiguous()}")  # True
```

### 3.4 数据类型

```python
# PyTorch 数据类型
"""
┌──────────────────────────────────────────────────────────────┐
│                      数据类型对照表                           │
├──────────────────┬─────────────────┬────────────────────────┤
│    PyTorch       │    Python/NumPy  │        说明            │
├──────────────────┼─────────────────┼────────────────────────┤
│ torch.float16    │ np.float16      │ 半精度浮点             │
│ torch.float32    │ np.float32      │ 单精度浮点（默认）      │
│ torch.float64    │ np.float64      │ 双精度浮点             │
│ torch.int8       │ np.int8         │ 8位有符号整数           │
│ torch.int16      │ np.int16        │ 16位有符号整数          │
│ torch.int32      │ np.int32        │ 32位有符号整数          │
│ torch.int64      │ np.int64        │ 64位有符号整数（默认）   │
│ torch.bool       │ np.bool_        │ 布尔类型               │
│ torch.complex64  │ np.complex64    │ 复数（实虚部各32位）    │
│ torch.complex128 │ np.complex128   │ 复数（实虚部各64位）    │
└──────────────────┴─────────────────┴────────────────────────┘
"""

# 类型转换
t = torch.tensor([1, 2, 3])
t_float = t.float()               # 转换为 float32
t_double = t.double()             # 转换为 float64
t_int = t.int()                   # 转换为 int32
t_long = t.long()                 # 转换为 int64
t_bool = t.bool()                 # 转换为 bool

# 使用 to() 方法
t_float16 = t.to(torch.float16)
t_cuda = t.to('cuda')             # 移动到 GPU
t_cpu = t.to('cpu')               # 移动到 CPU
```

### 3.5 张量操作

#### 3.5.1 索引与切片

```python
import torch

t = torch.arange(12).reshape(3, 4)
print(t)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 基本索引
print(t[0])           # 第一行: tensor([0, 1, 2, 3])
print(t[0, 1])        # 第一行第二列: tensor(1)
print(t[:, 0])        # 第一列: tensor([0, 4, 8])
print(t[1:, :2])      # 第2行起，前2列

# 高级索引
indices = torch.tensor([0, 2])
print(t[indices])     # 第1、3行

# 布尔索引
mask = t > 5
print(t[mask])        # tensor([ 6,  7,  8,  9, 10, 11])

# 修改值
t[0, 0] = 100
t[:, -1] = 0          # 最后一列置0
```

#### 3.5.2 形状操作

```python
import torch

t = torch.arange(12)

# reshape：改变形状（可能返回视图或副本）
t1 = t.reshape(3, 4)
t2 = t.reshape(2, -1)     # -1 自动计算

# view：改变形状（必须连续，返回视图）
t3 = t.view(3, 4)

# flatten：展平
t4 = t1.flatten()         # 完全展平
t5 = t1.flatten(0, 1)     # 展平指定维度

# squeeze/unsqueeze：压缩/扩展维度
t6 = torch.zeros(1, 3, 1, 4)
print(t6.squeeze().shape)        # torch.Size([3, 4])
print(t6.squeeze(0).shape)       # torch.Size([3, 1, 4])

t7 = torch.zeros(3, 4)
print(t7.unsqueeze(0).shape)     # torch.Size([1, 3, 4])
print(t7.unsqueeze(-1).shape)    # torch.Size([3, 4, 1])

# transpose/permute：转置
t8 = torch.randn(2, 3, 4)
print(t8.transpose(0, 2).shape)  # torch.Size([4, 3, 2])
print(t8.permute(2, 0, 1).shape) # torch.Size([4, 2, 3])
```

#### 3.5.3 数学运算

```python
import torch

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

# ============ 基本运算 ============
print(a + b)          # 加法
print(a - b)          # 减法
print(a * b)          # 逐元素乘法
print(a / b)          # 除法
print(a ** 2)         # 幂运算
print(a % 2)          # 取模

# ============ 矩阵运算 ============
m1 = torch.randn(2, 3)
m2 = torch.randn(3, 4)

# 矩阵乘法
result = m1 @ m2                    # 推荐
result = torch.mm(m1, m2)           # 2D矩阵
result = torch.matmul(m1, m2)       # 通用

# 批量矩阵乘法
batch_m1 = torch.randn(10, 2, 3)
batch_m2 = torch.randn(10, 3, 4)
result = torch.bmm(batch_m1, batch_m2)  # (10, 2, 4)

# 点积
dot = torch.dot(a, b)

# ============ 聚合运算 ============
t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

print(t.sum())            # 所有元素求和: 21
print(t.sum(dim=0))       # 按列求和: [5, 7, 9]
print(t.sum(dim=1))       # 按行求和: [6, 15]
print(t.mean())           # 均值
print(t.std())            # 标准差
print(t.var())            # 方差
print(t.max())            # 最大值
print(t.min())            # 最小值
print(t.argmax())         # 最大值索引
print(t.argmin())         # 最小值索引

# ============ 数学函数 ============
x = torch.tensor([1., 2., 3.])
print(torch.exp(x))       # 指数
print(torch.log(x))       # 对数
print(torch.sqrt(x))      # 平方根
print(torch.abs(x))       # 绝对值
print(torch.sin(x))       # 正弦
print(torch.cos(x))       # 余弦
print(torch.tanh(x))      # 双曲正切

# ============ 比较运算 ============
print(a > b)              # 逐元素比较
print(a == b)
print(torch.eq(a, b))     # 相等
print(torch.gt(a, b))     # 大于
print(torch.lt(a, b))     # 小于
print(torch.all(a > 0))   # 全部满足
print(torch.any(a > 2))   # 存在满足
```

#### 3.5.4 广播机制

```python
import torch

"""
广播规则：
1. 如果两个张量维度不同，在维度少的张量前面补1
2. 从右向左比较各维度，维度相同或其中一个为1时可以广播
3. 为1的维度会扩展成与另一个张量相同

示例:
    a: (3, 4)
    b: (   4)  → 补齐为 (1, 4) → 广播为 (3, 4)
    结果: (3, 4)
"""

# 示例1：向量与矩阵
a = torch.ones(3, 4)
b = torch.tensor([1, 2, 3, 4])
print((a + b).shape)  # torch.Size([3, 4])

# 示例2：列向量与行向量
row = torch.tensor([[1, 2, 3]])       # (1, 3)
col = torch.tensor([[1], [2], [3]])   # (3, 1)
print((row + col).shape)  # torch.Size([3, 3])
# 结果:
# [[2, 3, 4],
#  [3, 4, 5],
#  [4, 5, 6]]

# 示例3：3D广播
a = torch.randn(2, 3, 4)
b = torch.randn(   3, 1)
print((a + b).shape)  # torch.Size([2, 3, 4])
```

#### 3.5.5 拼接与分割

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# ============ 拼接 ============
# cat: 沿现有维度拼接
cat_0 = torch.cat([a, b], dim=0)  # (4, 2) 垂直拼接
cat_1 = torch.cat([a, b], dim=1)  # (2, 4) 水平拼接

# stack: 沿新维度堆叠
stack = torch.stack([a, b], dim=0)  # (2, 2, 2)

# ============ 分割 ============
t = torch.arange(12).reshape(4, 3)

# split: 按大小分割
parts = torch.split(t, 2, dim=0)     # 每份2行
parts = torch.split(t, [1, 3], dim=0) # 第1份1行，第2份3行

# chunk: 分成n份
chunks = torch.chunk(t, 2, dim=0)    # 分成2份

# unbind: 沿维度解开（移除该维度）
rows = torch.unbind(t, dim=0)        # 返回4个1D张量
```

---

## 4. 自动求导(Autograd)机制

### 4.1 计算图原理

```
┌─────────────────────────────────────────────────────────────────┐
│                      计算图示意                                  │
│                                                                 │
│    前向传播 (Forward Pass)                                       │
│    ─────────────────────►                                       │
│                                                                 │
│    ┌───┐      ┌───────┐      ┌───────┐      ┌───────┐          │
│    │ x │─────►│  y=   │─────►│  z=   │─────►│  L=   │          │
│    │   │      │ x*w+b │      │ ReLU  │      │ loss  │          │
│    └───┘      └───────┘      └───────┘      └───────┘          │
│      ▲            │              │              │               │
│      │            ▼              ▼              ▼               │
│    ┌───┐      ┌───────┐      ┌───────┐      ┌───────┐          │
│    │∂L/│◄─────│ ∂L/∂y │◄─────│ ∂L/∂z │◄─────│ ∂L/∂L │          │
│    │∂x │      │       │      │       │      │  = 1  │          │
│    └───┘      └───────┘      └───────┘      └───────┘          │
│                                                                 │
│    ◄─────────────────────                                       │
│    反向传播 (Backward Pass)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 自动求导基础

```python
import torch

# ============ 基本用法 ============
# 创建需要梯度的张量
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([4., 5., 6.], requires_grad=True)

# 计算
z = x * y
out = z.sum()

# 反向传播
out.backward()

# 查看梯度
print(f"x.grad: {x.grad}")  # tensor([4., 5., 6.])
print(f"y.grad: {y.grad}")  # tensor([1., 2., 3.])

# ============ 梯度累积 ============
# 默认情况下梯度会累积，需要手动清零
x = torch.tensor([1.], requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第{i+1}次: x.grad = {x.grad}")
    # 第1次: 2, 第2次: 4, 第3次: 6 (累积!)

# 正确做法：清零梯度
x = torch.tensor([1.], requires_grad=True)
for i in range(3):
    if x.grad is not None:
        x.grad.zero_()  # 清零
    y = x ** 2
    y.backward()
    print(f"第{i+1}次: x.grad = {x.grad}")  # 每次都是2
```

### 4.3 控制梯度计算

```python
import torch

# ============ 禁用梯度追踪 ============
x = torch.tensor([1., 2., 3.], requires_grad=True)

# 方法1：with torch.no_grad()
with torch.no_grad():
    y = x * 2
    print(y.requires_grad)  # False

# 方法2：装饰器
@torch.no_grad()
def inference(x):
    return x * 2

# 方法3：detach()
y = x.detach()
print(y.requires_grad)  # False

# ============ 推理模式（更高效）============
with torch.inference_mode():
    y = x * 2  # 比 no_grad 更快

# ============ 启用/禁用全局梯度 ============
torch.set_grad_enabled(False)  # 禁用
torch.set_grad_enabled(True)   # 启用

# ============ 冻结参数 ============
# 常用于迁移学习
model = torch.nn.Linear(10, 2)
for param in model.parameters():
    param.requires_grad = False
```

### 4.4 高级自动求导

```python
import torch

# ============ 计算高阶导数 ============
x = torch.tensor([1.], requires_grad=True)
y = x ** 3

# 一阶导数
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {dy_dx}")  # 3

# 二阶导数
d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
print(f"d²y/dx² = {d2y_dx2}")  # 6

# ============ 雅可比矩阵 ============
x = torch.randn(3, requires_grad=True)
y = x ** 2

# 计算雅可比矩阵
jacobian = torch.autograd.functional.jacobian(lambda x: x**2, x)
print(jacobian)  # 对角矩阵，对角元素为 2*x

# ============ 梯度裁剪（防止梯度爆炸）============
parameters = [torch.randn(10, requires_grad=True) for _ in range(3)]
for p in parameters:
    p.grad = torch.randn_like(p) * 100  # 模拟大梯度

# 按范数裁剪
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(parameters, clip_value=0.5)
```

### 4.5 自定义梯度

```python
import torch

# ============ 自定义 autograd Function ============
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        ctx: 上下文对象，用于保存反向传播所需的信息
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: 上游传来的梯度
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 使用自定义函数
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(x.grad)

# ============ 使用装饰器简化 ============
@torch.no_grad()
def custom_backward_hook(grad):
    """梯度钩子"""
    return grad * 0.1  # 缩放梯度

x = torch.randn(3, requires_grad=True)
x.register_hook(custom_backward_hook)
y = x ** 2
y.sum().backward()
print(x.grad)  # 梯度被缩放了
```

---

## 5. 神经网络模块(nn.Module)

### 5.1 nn.Module 基础

```python
import torch
import torch.nn as nn

# ============ 最简单的网络 ============
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # 必须调用父类初始化
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 使用
model = SimpleNet(784, 256, 10)
x = torch.randn(32, 784)  # batch_size=32
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

### 5.2 常用网络层

```python
import torch.nn as nn

# ============ 线性层 ============
linear = nn.Linear(in_features=100, out_features=50, bias=True)

# ============ 卷积层 ============
# 1D卷积 (序列数据，如文本、音频)
conv1d = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                   stride=1, padding=1)

# 2D卷积 (图像)
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                   stride=1, padding=1, bias=True)

# 3D卷积 (视频)
conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3)

# ============ 转置卷积（上采样）============
deconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

# ============ 池化层 ============
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
adaptivepool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 全局平均池化

# ============ 归一化层 ============
batchnorm1d = nn.BatchNorm1d(num_features=100)
batchnorm2d = nn.BatchNorm2d(num_features=64)
layernorm = nn.LayerNorm(normalized_shape=[64, 32, 32])
instancenorm = nn.InstanceNorm2d(num_features=64)
groupnorm = nn.GroupNorm(num_groups=8, num_channels=64)

# ============ Dropout ============
dropout = nn.Dropout(p=0.5)
dropout2d = nn.Dropout2d(p=0.5)  # 空间Dropout

# ============ 循环层 ============
rnn = nn.RNN(input_size=100, hidden_size=256, num_layers=2,
             batch_first=True, bidirectional=True)
lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2,
               batch_first=True, bidirectional=True)
gru = nn.GRU(input_size=100, hidden_size=256, num_layers=2,
             batch_first=True)

# ============ Transformer层 ============
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
                             num_decoder_layers=6)
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# ============ Embedding ============
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)
```

### 5.3 激活函数

```python
import torch.nn as nn
import torch.nn.functional as F

# ============ 常用激活函数 ============
"""
┌────────────────────────────────────────────────────────────────┐
│                      激活函数对比                               │
├───────────────┬─────────────────────┬─────────────────────────┤
│    函数       │       公式           │         特点            │
├───────────────┼─────────────────────┼─────────────────────────┤
│  ReLU         │  max(0, x)          │ 简单高效，可能死亡神经元 │
│  LeakyReLU    │  max(αx, x)         │ 解决死亡神经元问题       │
│  PReLU        │  max(αx, x), α可学习│ 自适应负斜率            │
│  ELU          │  x if x>0 else α(eˣ-1)│ 平滑，均值接近0        │
│  GELU         │  x·Φ(x)             │ Transformer常用         │
│  Sigmoid      │  1/(1+e⁻ˣ)          │ 输出(0,1)，梯度消失     │
│  Tanh         │  (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)  │ 输出(-1,1)，零中心      │
│  Softmax      │  eˣⁱ/Σeˣʲ           │ 多分类输出层            │
│  SiLU/Swish   │  x·σ(x)             │ 现代网络常用            │
└───────────────┴─────────────────────┴─────────────────────────┘
"""

# 作为模块使用
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
gelu = nn.GELU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=-1)
silu = nn.SiLU()

# 作为函数使用
x = torch.randn(10)
y = F.relu(x)
y = F.leaky_relu(x, negative_slope=0.01)
y = F.gelu(x)
y = F.sigmoid(x)
y = F.softmax(x, dim=-1)
```

### 5.4 容器模块

```python
import torch.nn as nn

# ============ Sequential：顺序容器 ============
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 带名称的Sequential
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
]))

# ============ ModuleList：模块列表 ============
class MyModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(100, 100) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

# ============ ModuleDict：模块字典 ============
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(100, 64)
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(64, 10),
            'regression': nn.Linear(64, 1)
        })

    def forward(self, x, task):
        x = F.relu(self.shared(x))
        return self.heads[task](x)
```

### 5.5 参数管理

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# ============ 查看参数 ============
# 所有参数
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ============ 参数初始化 ============
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)

# 常用初始化方法
"""
nn.init.xavier_uniform_(tensor)      # Xavier均匀分布
nn.init.xavier_normal_(tensor)       # Xavier正态分布
nn.init.kaiming_uniform_(tensor)     # Kaiming均匀分布
nn.init.kaiming_normal_(tensor)      # Kaiming正态分布
nn.init.zeros_(tensor)               # 全零
nn.init.ones_(tensor)                # 全一
nn.init.constant_(tensor, val)       # 常数
nn.init.normal_(tensor, mean, std)   # 正态分布
nn.init.uniform_(tensor, a, b)       # 均匀分布
nn.init.orthogonal_(tensor)          # 正交初始化
"""

# ============ 冻结/解冻参数 ============
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只解冻最后一层
for param in model[-1].parameters():
    param.requires_grad = True
```

---

## 6. 损失函数与优化器

### 6.1 常用损失函数

```python
import torch
import torch.nn as nn

# ============ 分类损失 ============
# 交叉熵损失（多分类，内置Softmax）
criterion = nn.CrossEntropyLoss()
logits = torch.randn(3, 5)  # 3个样本，5个类别
targets = torch.tensor([1, 0, 4])  # 真实标签
loss = criterion(logits, targets)

# 二元交叉熵（二分类，需要先Sigmoid）
criterion = nn.BCELoss()
probs = torch.sigmoid(torch.randn(3))
targets = torch.tensor([1., 0., 1.])
loss = criterion(probs, targets)

# BCEWithLogitsLoss（内置Sigmoid，更稳定）
criterion = nn.BCEWithLogitsLoss()
logits = torch.randn(3)
loss = criterion(logits, targets)

# ============ 回归损失 ============
# MSE损失
criterion = nn.MSELoss()
predictions = torch.randn(3)
targets = torch.randn(3)
loss = criterion(predictions, targets)

# L1损失（MAE）
criterion = nn.L1Loss()
loss = criterion(predictions, targets)

# Smooth L1损失（Huber Loss）
criterion = nn.SmoothL1Loss()
loss = criterion(predictions, targets)

# ============ 其他损失 ============
# 负对数似然损失（NLL）
criterion = nn.NLLLoss()
log_probs = F.log_softmax(logits, dim=1)
loss = criterion(log_probs, targets)

# KL散度
criterion = nn.KLDivLoss(reduction='batchmean')
log_probs = F.log_softmax(logits, dim=1)
target_probs = F.softmax(torch.randn(3, 5), dim=1)
loss = criterion(log_probs, target_probs)

# 余弦嵌入损失
criterion = nn.CosineEmbeddingLoss()
x1 = torch.randn(3, 10)
x2 = torch.randn(3, 10)
y = torch.tensor([1, -1, 1])  # 1表示相似，-1表示不相似
loss = criterion(x1, x2, y)

# Triplet损失
criterion = nn.TripletMarginLoss(margin=1.0)
anchor = torch.randn(3, 10)
positive = torch.randn(3, 10)
negative = torch.randn(3, 10)
loss = criterion(anchor, positive, negative)
```

### 6.2 自定义损失函数

```python
import torch
import torch.nn as nn

# 方法1：继承 nn.Module
class FocalLoss(nn.Module):
    """Focal Loss - 解决类别不平衡问题"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 方法2：直接定义函数
def dice_loss(pred, target, smooth=1e-6):
    """Dice Loss - 常用于分割任务"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# 方法3：组合损失
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice = dice_loss(pred, target)
        return self.alpha * ce_loss + (1 - self.alpha) * dice
```

### 6.3 优化器

```python
import torch.optim as optim

model = nn.Linear(10, 2)

# ============ 常用优化器 ============
# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=1e-4, nesterov=True)

# Adam（最常用）
optimizer = optim.Adam(model.parameters(), lr=0.001,
                       betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

# AdamW（解耦权重衰减）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adagrad
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# ============ 分组参数（不同层不同学习率）============
optimizer = optim.Adam([
    {'params': model.base.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-4)  # 默认学习率

# ============ 优化器基本操作 ============
optimizer.zero_grad()  # 清空梯度
loss.backward()        # 计算梯度
optimizer.step()       # 更新参数

# 手动调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr
```

### 6.4 学习率调度器

```python
import torch.optim.lr_scheduler as lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============ 常用调度器 ============
# 阶梯衰减
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 多阶段衰减
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# 指数衰减
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# 余弦退火
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 带热重启的余弦退火
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# 按验证指标调整（plateau时降低）
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                           factor=0.1, patience=10)

# 线性预热
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

# ============ 使用调度器 ============
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)

    # 常规调度器
    scheduler.step()

    # ReduceLROnPlateau需要传入指标
    # scheduler.step(val_loss)

    print(f"当前学习率: {scheduler.get_last_lr()}")
```

---

## 7. 数据加载与处理

### 7.1 Dataset 类

```python
from torch.utils.data import Dataset, DataLoader
import torch

# ============ 自定义Dataset ============
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """返回单个样本"""
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# 使用示例
data = torch.randn(1000, 28, 28)
labels = torch.randint(0, 10, (1000,))
dataset = CustomDataset(data, labels)

# ============ 常用内置Dataset ============
from torchvision import datasets
from torchvision.transforms import ToTensor

# MNIST
mnist_train = datasets.MNIST(root='./data', train=True,
                             download=True, transform=ToTensor())

# CIFAR-10
cifar_train = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=ToTensor())

# ImageFolder（自定义图片数据集）
# 目录结构: root/class1/xxx.png, root/class2/xxx.png, ...
from torchvision.datasets import ImageFolder
dataset = ImageFolder(root='./data/train', transform=ToTensor())
```

### 7.2 DataLoader

```python
from torch.utils.data import DataLoader

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,           # 批大小
    shuffle=True,            # 是否打乱
    num_workers=4,           # 多进程加载
    pin_memory=True,         # 加速GPU传输
    drop_last=False,         # 丢弃最后不完整的batch
    collate_fn=None,         # 自定义batch整理函数
    sampler=None,            # 自定义采样器
    persistent_workers=True  # 保持worker进程
)

# 遍历数据
for batch_idx, (data, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape = {data.shape}")
    break

# ============ 自定义 collate_fn ============
def custom_collate_fn(batch):
    """处理变长序列"""
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 填充到相同长度
    data_padded = nn.utils.rnn.pad_sequence(data, batch_first=True)
    labels = torch.stack(labels)

    return data_padded, labels

# ============ 自定义采样器 ============
from torch.utils.data import WeightedRandomSampler

# 处理类别不平衡
class_weights = [1.0, 2.0, 0.5]  # 各类别权重
sample_weights = [class_weights[label] for label in dataset.labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset))

dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### 7.3 数据变换

```python
from torchvision import transforms

# ============ 常用变换 ============
transform = transforms.Compose([
    # 几何变换
    transforms.Resize((256, 256)),           # 调整大小
    transforms.CenterCrop(224),              # 中心裁剪
    transforms.RandomCrop(224),              # 随机裁剪
    transforms.RandomResizedCrop(224),       # 随机裁剪+缩放
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.RandomVerticalFlip(p=0.5),    # 垂直翻转
    transforms.RandomRotation(degrees=15),   # 随机旋转

    # 颜色变换
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),       # 随机灰度化

    # 转换
    transforms.ToTensor(),                   # PIL/numpy -> Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                        std=[0.229, 0.224, 0.225]),

    # 数据增强
    transforms.RandomErasing(p=0.5),         # 随机擦除
])

# ============ 训练/验证不同变换 ============
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(),  # 自动数据增强
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============ 自定义变换 ============
class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
```

---

## 8. 完整训练流程

### 8.1 训练框架

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion,
                 optimizer, scheduler=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data, targets in self.val_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(self.val_loader), correct / total

    def fit(self, epochs, save_path='best_model.pth'):
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved best model with val_loss: {val_loss:.4f}')

# ============ 使用示例 ============
# 模型
model = SimpleNet(784, 256, 10)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)
trainer.fit(epochs=100)
```

### 8.2 早停机制

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        return False

# 使用
early_stopping = EarlyStopping(patience=10)
for epoch in range(max_epochs):
    train(...)
    val_loss = validate(...)
    if early_stopping(val_loss, model):
        print("Early stopping triggered")
        break
```

---

## 9. 模型保存与加载

### 9.1 保存与加载方法

```python
import torch

# ============ 方法1：只保存参数（推荐）============
# 保存
torch.save(model.state_dict(), 'model_weights.pth')

# 加载
model = MyModel()  # 需要先创建模型结构
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 推理模式

# ============ 方法2：保存整个模型 ============
# 保存
torch.save(model, 'model_complete.pth')

# 加载
model = torch.load('model_complete.pth')

# ============ 方法3：保存检查点（训练恢复）============
# 保存
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# ============ 跨设备加载 ============
# GPU模型加载到CPU
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# CPU模型加载到GPU
model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))

# ============ 部分加载 ============
pretrained_dict = torch.load('pretrained.pth')
model_dict = model.state_dict()

# 过滤不匹配的键
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

### 9.2 导出为 ONNX

```python
import torch

# 导出
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 验证
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# 推理
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {'input': input_data.numpy()})
```

---

## 10. GPU 加速

### 10.1 基本 GPU 操作

```python
import torch

# ============ 检查CUDA ============
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")

# ============ 设备设置 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 或指定具体GPU
device = torch.device('cuda:0')

# ============ 数据移动 ============
# 张量
tensor = torch.randn(3, 4)
tensor = tensor.to(device)
tensor = tensor.cuda()
tensor = tensor.cpu()

# 模型
model = model.to(device)
model = model.cuda()

# ============ 内存管理 ============
# 查看内存使用
print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# 清空缓存
torch.cuda.empty_cache()

# 同步
torch.cuda.synchronize()
```

### 10.2 多 GPU 训练

```python
import torch
import torch.nn as nn

# ============ DataParallel（简单但效率较低）============
model = MyModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# ============ DistributedDataParallel（推荐）============
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # 创建模型
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 创建数据加载器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # 训练
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 确保每个epoch打乱不同
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            # ... 训练代码

    cleanup()

# 启动
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

### 10.3 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

# 创建GradScaler
scaler = GradScaler()

model = model.to(device)
optimizer = optim.Adam(model.parameters())

for data, target in dataloader:
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    # 混合精度前向传播
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # 缩放后反向传播
    scaler.scale(loss).backward()

    # 更新参数
    scaler.step(optimizer)
    scaler.update()
```

---

## 11. 高级主题

### 11.1 自定义层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # 线性投影
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)

        return self.out_linear(output)
```

### 11.2 钩子(Hooks)

```python
import torch
import torch.nn as nn

# ============ 前向钩子 ============
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model = models.resnet18(pretrained=True)
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))

# 运行模型
x = torch.randn(1, 3, 224, 224)
output = model(x)

# 获取中间特征
print(activations['layer1'].shape)
print(activations['layer2'].shape)

# ============ 反向钩子 ============
gradients = {}

def get_gradient(name):
    def hook(model, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

model.layer4.register_full_backward_hook(get_gradient('layer4'))

output = model(x)
output.sum().backward()
print(gradients['layer4'].shape)

# ============ 使用钩子进行特征可视化（CAM）============
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, target_class):
        output = self.model(x)

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam
```

### 11.3 TorchScript（模型编译）

```python
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
model.eval()

# ============ Tracing（追踪）============
# 适合没有控制流的模型
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("traced_model.pt")

# ============ Scripting（脚本化）============
# 适合有控制流的模型
class ModelWithControl(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        else:
            return x * 3

scripted_model = torch.jit.script(ModelWithControl())
scripted_model.save("scripted_model.pt")

# ============ 加载和使用 ============
loaded_model = torch.jit.load("traced_model.pt")
output = loaded_model(example_input)
```

---

## 12. 实战案例

### 12.1 图像分类（ResNet）

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ============ 数据准备 ============
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ============ 模型（迁移学习）============
model = torchvision.models.resnet18(pretrained=True)

# 冻结特征提取层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
model.fc = nn.Linear(model.fc.in_features, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ============ 训练设置 ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ============ 训练循环 ============
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()

    # 验证
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {running_loss/len(train_loader):.4f}, '
          f'Train Acc: {100*correct/total:.2f}%, '
          f'Test Acc: {100*test_correct/test_total:.2f}%')
```

### 12.2 文本分类（Transformer）

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        if mask is not None:
            # 创建 attention mask (True 表示屏蔽)
            padding_mask = (mask == 0)
        else:
            padding_mask = None

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 使用 [CLS] token 或平均池化
        x = x.mean(dim=1)  # 平均池化

        return self.classifier(x)

# 使用
model = TextClassifier(
    vocab_size=30000,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    num_classes=10
)

# 示例输入
batch_size = 16
seq_len = 128
input_ids = torch.randint(0, 30000, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

output = model(input_ids, attention_mask)
print(output.shape)  # torch.Size([16, 10])
```

### 12.3 生成对抗网络（GAN）

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, feature_dim=64):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (batch, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_dim*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim*8),
            nn.ReLU(True),
            # (batch, feature_dim*8, 4, 4)

            nn.ConvTranspose2d(feature_dim*8, feature_dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim*4),
            nn.ReLU(True),
            # (batch, feature_dim*4, 8, 8)

            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim*2),
            nn.ReLU(True),
            # (batch, feature_dim*2, 16, 16)

            nn.ConvTranspose2d(feature_dim*2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            # (batch, feature_dim, 32, 32)

            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # (batch, img_channels, 64, 64)
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_dim=64):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (batch, img_channels, 64, 64)
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim, feature_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim*2, feature_dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim*4, feature_dim*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.main(img).view(-1, 1)


# ============ 训练 GAN ============
def train_gan(generator, discriminator, dataloader, num_epochs, device):
    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    latent_dim = 100

    for epoch in range(num_epochs):
        for real_imgs, _ in dataloader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # 真实和假标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ============ 训练判别器 ============
            optimizer_D.zero_grad()

            # 真实图片
            outputs = discriminator(real_imgs)
            d_loss_real = criterion(outputs, real_labels)

            # 假图片
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_imgs = generator(z)
            outputs = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ============ 训练生成器 ============
            optimizer_G.zero_grad()

            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
```

---

## 13. 最佳实践与调试技巧

### 13.1 代码规范

```python
# ============ 项目结构 ============
"""
project/
├── config/
│   └── config.yaml         # 配置文件
├── data/
│   ├── dataset.py          # 数据集类
│   └── transforms.py       # 数据变换
├── models/
│   ├── __init__.py
│   ├── backbone.py         # 主干网络
│   └── head.py             # 任务头
├── utils/
│   ├── logger.py           # 日志
│   ├── metrics.py          # 评估指标
│   └── visualization.py    # 可视化
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
└── requirements.txt        # 依赖
"""

# ============ 设置随机种子（可复现性）============
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ 设备无关代码 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自动选择最佳设备
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device('mps')
    return torch.device('cpu')
```

### 13.2 调试技巧

```python
# ============ 形状检查 ============
def debug_shapes(model, input_shape):
    """打印每一层的输出形状"""
    x = torch.randn(*input_shape)
    for name, layer in model.named_children():
        x = layer(x)
        print(f"{name}: {x.shape}")

# ============ 梯度检查 ============
def check_gradients(model):
    """检查梯度是否正常"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"Warning: {name} 梯度为零")
            elif grad_norm > 1000:
                print(f"Warning: {name} 梯度过大: {grad_norm}")
            elif torch.isnan(param.grad).any():
                print(f"Error: {name} 梯度为NaN")

# ============ 异常检测 ============
torch.autograd.set_detect_anomaly(True)  # 开启异常检测

# ============ 内存分析 ============
def memory_stats():
    if torch.cuda.is_available():
        print(f"已分配: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"最大分配: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        print(f"已缓存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

# ============ 性能分析 ============
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 13.3 常见问题与解决方案

```python
"""
┌────────────────────────────────────────────────────────────────────┐
│                      常见问题与解决方案                              │
├────────────────────────┬───────────────────────────────────────────┤
│        问题            │              解决方案                       │
├────────────────────────┼───────────────────────────────────────────┤
│ CUDA out of memory     │ - 减小 batch_size                         │
│                        │ - 使用梯度累积                             │
│                        │ - 使用混合精度训练                         │
│                        │ - 使用梯度检查点                           │
├────────────────────────┼───────────────────────────────────────────┤
│ Loss 为 NaN            │ - 降低学习率                               │
│                        │ - 检查数据是否有 NaN                       │
│                        │ - 添加梯度裁剪                             │
│                        │ - 检查除零操作                             │
├────────────────────────┼───────────────────────────────────────────┤
│ 模型不收敛             │ - 检查数据加载是否正确                      │
│                        │ - 调整学习率                               │
│                        │ - 检查标签是否正确                         │
│                        │ - 简化模型验证基本功能                      │
├────────────────────────┼───────────────────────────────────────────┤
│ 过拟合                 │ - 增加数据增强                             │
│                        │ - 添加 Dropout                            │
│                        │ - 添加正则化 (weight_decay)               │
│                        │ - 早停                                    │
├────────────────────────┼───────────────────────────────────────────┤
│ 训练速度慢             │ - 增加 num_workers                         │
│                        │ - 使用 pin_memory=True                    │
│                        │ - 使用混合精度训练                         │
│                        │ - 检查是否有不必要的计算                   │
└────────────────────────┴───────────────────────────────────────────┘
"""

# ============ 梯度累积（内存不足时）============
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ============ 梯度检查点（节省内存）============
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)

    def forward(self, x):
        x = checkpoint(self.layer1, x)  # 不保存中间激活值
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

---

## 📚 附录

### 常用资源

| 资源             | 链接                               |
| ---------------- | ---------------------------------- |
| PyTorch 官方文档 | https://pytorch.org/docs/          |
| PyTorch 教程     | https://pytorch.org/tutorials/     |
| PyTorch Hub      | https://pytorch.org/hub/           |
| PyTorch 论坛     | https://discuss.pytorch.org/       |
| PyTorch GitHub   | https://github.com/pytorch/pytorch |

### 版本信息

```python
# 检查完整环境信息
import torch
print(torch.__config__.show())
```

---

> 📝 **文档信息**
>
> - 版本：1.0
> - 最后更新：2024 年
> - 适用 PyTorch 版本：2.0+
