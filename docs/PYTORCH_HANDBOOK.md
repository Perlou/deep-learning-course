# PyTorch 常用函数与方法速查手册

---

## 目录

- [1. 张量基础操作](#1-张量基础操作)
- [2. 张量数学运算](#2-张量数学运算)
- [3. 张量形状操作](#3-张量形状操作)
- [4. 索引与切片](#4-索引与切片)
- [5. 神经网络层 (nn.Module)](#5-神经网络层-nnmodule)
- [6. 激活函数](#6-激活函数)
- [7. 损失函数](#7-损失函数)
- [8. 优化器](#8-优化器)
- [9. 学习率调度器](#9-学习率调度器)
- [10. 数据加载](#10-数据加载)
- [11. 自动求导](#11-自动求导)
- [12. 模型保存与加载](#12-模型保存与加载)
- [13. GPU 操作](#13-gpu-操作)
- [14. 常用工具函数](#14-常用工具函数)
- [15. 模型构建模板](#15-模型构建模板)

---

## 1. 张量基础操作

### 1.1 张量创建

| 函数               | 说明                 | 示例                                |
| :----------------- | :------------------- | :---------------------------------- |
| `torch.tensor()`   | 从数据创建张量       | `torch.tensor([1, 2, 3])`           |
| `torch.zeros()`    | 创建全零张量         | `torch.zeros(3, 4)`                 |
| `torch.ones()`     | 创建全一张量         | `torch.ones(3, 4)`                  |
| `torch.empty()`    | 创建未初始化张量     | `torch.empty(3, 4)`                 |
| `torch.full()`     | 创建填充指定值的张量 | `torch.full((3, 4), 5.0)`           |
| `torch.arange()`   | 创建等差序列         | `torch.arange(0, 10, 2)`            |
| `torch.linspace()` | 创建等分序列         | `torch.linspace(0, 1, 5)`           |
| `torch.logspace()` | 创建对数等分序列     | `torch.logspace(0, 2, 5)`           |
| `torch.eye()`      | 创建单位矩阵         | `torch.eye(3)`                      |
| `torch.diag()`     | 创建对角矩阵         | `torch.diag(torch.tensor([1,2,3]))` |

```python
import torch

# 基本创建
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
empty = torch.empty(3, 4)  # 未初始化，值随机

# 序列创建
seq = torch.arange(0, 10, 2)        # tensor([0, 2, 4, 6, 8])
lin = torch.linspace(0, 1, 5)       # tensor([0.00, 0.25, 0.50, 0.75, 1.00])

# 特殊矩阵
eye = torch.eye(3)                  # 3x3 单位矩阵
diag = torch.diag(torch.tensor([1, 2, 3]))  # 对角矩阵
```

### 1.2 随机张量

| 函数                | 说明             | 示例                                     |
| :------------------ | :--------------- | :--------------------------------------- |
| `torch.rand()`      | 均匀分布 [0, 1)  | `torch.rand(3, 4)`                       |
| `torch.randn()`     | 标准正态分布     | `torch.randn(3, 4)`                      |
| `torch.randint()`   | 随机整数         | `torch.randint(0, 10, (3, 4))`           |
| `torch.randperm()`  | 随机排列         | `torch.randperm(10)`                     |
| `torch.normal()`    | 指定均值和标准差 | `torch.normal(0, 1, (3, 4))`             |
| `torch.bernoulli()` | 伯努利分布       | `torch.bernoulli(torch.full((3,), 0.5))` |

```python
# 设置随机种子
torch.manual_seed(42)

# 随机张量
uniform = torch.rand(3, 4)           # 均匀分布 [0, 1)
normal = torch.randn(3, 4)           # 标准正态分布
integers = torch.randint(0, 10, (3, 4))  # 随机整数 [0, 10)
perm = torch.randperm(10)            # 0-9 的随机排列

# 指定分布参数
custom_normal = torch.normal(mean=0.0, std=1.0, size=(3, 4))
```

### 1.3 从其他数据创建

| 函数                 | 说明                   | 示例                         |
| :------------------- | :--------------------- | :--------------------------- |
| `torch.from_numpy()` | 从 NumPy 创建          | `torch.from_numpy(np_array)` |
| `torch.as_tensor()`  | 从数据创建（共享内存） | `torch.as_tensor(data)`      |
| `tensor.numpy()`     | 转换为 NumPy           | `x.numpy()`                  |
| `tensor.tolist()`    | 转换为 Python 列表     | `x.tolist()`                 |
| `tensor.item()`      | 获取标量值             | `x.item()`                   |

```python
import numpy as np

# NumPy 互转
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)  # 共享内存
back_to_numpy = tensor.numpy()       # 共享内存

# 获取值
scalar = torch.tensor(3.14)
value = scalar.item()  # 3.14

# 转列表
x = torch.tensor([[1, 2], [3, 4]])
lst = x.tolist()  # [[1, 2], [3, 4]]
```

### 1.4 张量属性

| 属性/方法              | 说明             | 示例              |
| :--------------------- | :--------------- | :---------------- |
| `tensor.shape`         | 张量形状         | `x.shape`         |
| `tensor.size()`        | 张量形状（方法） | `x.size()`        |
| `tensor.dtype`         | 数据类型         | `x.dtype`         |
| `tensor.device`        | 所在设备         | `x.device`        |
| `tensor.ndim`          | 维度数           | `x.ndim`          |
| `tensor.numel()`       | 元素总数         | `x.numel()`       |
| `tensor.requires_grad` | 是否需要梯度     | `x.requires_grad` |

```python
x = torch.randn(3, 4, 5)

print(x.shape)          # torch.Size([3, 4, 5])
print(x.size())         # torch.Size([3, 4, 5])
print(x.size(0))        # 3 (第0维大小)
print(x.dtype)          # torch.float32
print(x.device)         # cpu
print(x.ndim)           # 3
print(x.numel())        # 60
print(x.requires_grad)  # False
```

### 1.5 类型转换

| 方法              | 说明         | 示例                        |
| :---------------- | :----------- | :-------------------------- |
| `tensor.float()`  | 转为 float32 | `x.float()`                 |
| `tensor.double()` | 转为 float64 | `x.double()`                |
| `tensor.half()`   | 转为 float16 | `x.half()`                  |
| `tensor.int()`    | 转为 int32   | `x.int()`                   |
| `tensor.long()`   | 转为 int64   | `x.long()`                  |
| `tensor.bool()`   | 转为 bool    | `x.bool()`                  |
| `tensor.to()`     | 通用转换     | `x.to(torch.float64)`       |
| `tensor.type()`   | 指定类型     | `x.type(torch.FloatTensor)` |

```python
x = torch.tensor([1, 2, 3])

# 类型转换
x_float = x.float()     # torch.float32
x_double = x.double()   # torch.float64
x_half = x.half()       # torch.float16
x_long = x.long()       # torch.int64
x_bool = x.bool()       # torch.bool

# 通用方法
x_converted = x.to(torch.float32)
x_typed = x.type(torch.FloatTensor)
```

---

## 2. 张量数学运算

### 2.1 基本运算

| 运算   | 函数形式                   | 运算符   | 原地操作    |
| :----- | :------------------------- | :------- | :---------- |
| 加法   | `torch.add(a, b)`          | `a + b`  | `a.add_(b)` |
| 减法   | `torch.sub(a, b)`          | `a - b`  | `a.sub_(b)` |
| 乘法   | `torch.mul(a, b)`          | `a * b`  | `a.mul_(b)` |
| 除法   | `torch.div(a, b)`          | `a / b`  | `a.div_(b)` |
| 整除   | `torch.floor_divide(a, b)` | `a // b` | -           |
| 取余   | `torch.remainder(a, b)`    | `a % b`  | -           |
| 幂运算 | `torch.pow(a, n)`          | `a ** n` | `a.pow_(n)` |
| 负数   | `torch.neg(a)`             | `-a`     | `a.neg_()`  |
| 绝对值 | `torch.abs(a)`             | -        | `a.abs_()`  |

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 基本运算
add_result = a + b           # tensor([5., 7., 9.])
sub_result = a - b           # tensor([-3., -3., -3.])
mul_result = a * b           # tensor([4., 10., 18.])
div_result = a / b           # tensor([0.25, 0.4, 0.5])
pow_result = a ** 2          # tensor([1., 4., 9.])

# 原地操作（节省内存）
a.add_(1)  # a = a + 1，修改 a 本身
```

### 2.2 矩阵运算

| 函数                | 说明              | 示例                            |
| :------------------ | :---------------- | :------------------------------ |
| `torch.mm()`        | 矩阵乘法 (2D)     | `torch.mm(A, B)`                |
| `torch.bmm()`       | 批量矩阵乘法 (3D) | `torch.bmm(A, B)`               |
| `torch.matmul()`    | 通用矩阵乘法      | `torch.matmul(A, B)` 或 `A @ B` |
| `torch.mv()`        | 矩阵-向量乘法     | `torch.mv(A, v)`                |
| `torch.dot()`       | 向量点积          | `torch.dot(a, b)`               |
| `torch.outer()`     | 向量外积          | `torch.outer(a, b)`             |
| `torch.transpose()` | 转置              | `torch.transpose(A, 0, 1)`      |
| `tensor.T`          | 转置 (2D)         | `A.T`                           |
| `tensor.mT`         | 批量转置          | `A.mT`                          |
| `torch.inverse()`   | 矩阵求逆          | `torch.inverse(A)`              |
| `torch.det()`       | 行列式            | `torch.det(A)`                  |
| `torch.trace()`     | 迹                | `torch.trace(A)`                |

```python
# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.mm(A, B)           # [3, 5]
C = A @ B                    # 等价写法

# 批量矩阵乘法
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
batch_C = torch.bmm(batch_A, batch_B)  # [10, 3, 5]

# 通用矩阵乘法 (自动广播)
result = torch.matmul(batch_A, batch_B)

# 向量运算
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
dot_product = torch.dot(v1, v2)    # 32.0
outer_product = torch.outer(v1, v2)  # [3, 3]

# 转置
A_T = A.T                    # 2D 转置
A_T = A.transpose(0, 1)      # 指定维度转置
A_T = A.permute(1, 0)        # 维度重排
```

### 2.3 统计运算

| 函数              | 说明       | 示例                             |
| :---------------- | :--------- | :------------------------------- |
| `torch.sum()`     | 求和       | `x.sum()` / `x.sum(dim=0)`       |
| `torch.mean()`    | 均值       | `x.mean()` / `x.mean(dim=0)`     |
| `torch.std()`     | 标准差     | `x.std()`                        |
| `torch.var()`     | 方差       | `x.var()`                        |
| `torch.max()`     | 最大值     | `x.max()` / `x.max(dim=0)`       |
| `torch.min()`     | 最小值     | `x.min()` / `x.min(dim=0)`       |
| `torch.argmax()`  | 最大值索引 | `x.argmax()` / `x.argmax(dim=0)` |
| `torch.argmin()`  | 最小值索引 | `x.argmin()`                     |
| `torch.median()`  | 中位数     | `x.median()`                     |
| `torch.mode()`    | 众数       | `x.mode()`                       |
| `torch.prod()`    | 累积乘积   | `x.prod()`                       |
| `torch.cumsum()`  | 累积和     | `x.cumsum(dim=0)`                |
| `torch.cumprod()` | 累积积     | `x.cumprod(dim=0)`               |

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# 全局统计
print(x.sum())       # 21.0
print(x.mean())      # 3.5
print(x.std())       # 1.8708
print(x.max())       # 6.0
print(x.min())       # 1.0

# 按维度统计
print(x.sum(dim=0))     # tensor([5., 7., 9.])  沿行求和
print(x.sum(dim=1))     # tensor([6., 15.])     沿列求和
print(x.mean(dim=0))    # tensor([2.5, 3.5, 4.5])

# 保持维度
print(x.sum(dim=1, keepdim=True))  # [[6.], [15.]]

# 最值及索引
max_val, max_idx = x.max(dim=1)
print(max_val)  # tensor([3., 6.])
print(max_idx)  # tensor([2, 2])

# 累积运算
print(x.cumsum(dim=1))  # [[1, 3, 6], [4, 9, 15]]
```

### 2.4 比较运算

| 函数               | 运算符 | 说明         |
| :----------------- | :----- | :----------- |
| `torch.eq()`       | `==`   | 等于         |
| `torch.ne()`       | `!=`   | 不等于       |
| `torch.gt()`       | `>`    | 大于         |
| `torch.ge()`       | `>=`   | 大于等于     |
| `torch.lt()`       | `<`    | 小于         |
| `torch.le()`       | `<=`   | 小于等于     |
| `torch.equal()`    | -      | 张量完全相等 |
| `torch.allclose()` | -      | 近似相等     |
| `torch.isnan()`    | -      | 检测 NaN     |
| `torch.isinf()`    | -      | 检测 Inf     |
| `torch.isfinite()` | -      | 检测有限值   |

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])

# 元素比较
print(a > b)         # tensor([False, False, True])
print(a == b)        # tensor([False, True, False])
print(torch.eq(a, b))

# 张量比较
print(torch.equal(a, b))  # False

# 近似比较
x = torch.tensor([1.0, 2.0])
y = torch.tensor([1.0001, 2.0001])
print(torch.allclose(x, y, atol=1e-3))  # True

# 特殊值检测
z = torch.tensor([1.0, float('nan'), float('inf')])
print(torch.isnan(z))     # tensor([False, True, False])
print(torch.isinf(z))     # tensor([False, False, True])
print(torch.isfinite(z))  # tensor([True, False, False])
```

### 2.5 数学函数

| 函数                                 | 说明              |
| :----------------------------------- | :---------------- |
| `torch.exp()`                        | 指数 e^x          |
| `torch.log()`                        | 自然对数 ln(x)    |
| `torch.log2()`                       | 以 2 为底对数     |
| `torch.log10()`                      | 以 10 为底对数    |
| `torch.log1p()`                      | ln(1+x)，数值稳定 |
| `torch.sqrt()`                       | 平方根            |
| `torch.rsqrt()`                      | 平方根倒数 1/√x   |
| `torch.square()`                     | 平方 x²           |
| `torch.sin()` / `cos()` / `tan()`    | 三角函数          |
| `torch.sinh()` / `cosh()` / `tanh()` | 双曲函数          |
| `torch.sigmoid()`                    | Sigmoid 函数      |
| `torch.sign()`                       | 符号函数          |
| `torch.floor()`                      | 向下取整          |
| `torch.ceil()`                       | 向上取整          |
| `torch.round()`                      | 四舍五入          |
| `torch.trunc()`                      | 截断取整          |
| `torch.frac()`                       | 小数部分          |
| `torch.clamp()`                      | 裁剪到范围        |

```python
x = torch.tensor([0.5, 1.0, 2.0])

# 指数和对数
print(torch.exp(x))      # e^x
print(torch.log(x))      # ln(x)
print(torch.log2(x))     # log2(x)
print(torch.log1p(x))    # ln(1+x)

# 幂和根
print(torch.sqrt(x))     # √x
print(torch.square(x))   # x²

# 三角函数
print(torch.sin(x))
print(torch.cos(x))

# 取整
y = torch.tensor([1.2, 2.5, 3.8])
print(torch.floor(y))    # tensor([1., 2., 3.])
print(torch.ceil(y))     # tensor([2., 3., 4.])
print(torch.round(y))    # tensor([1., 2., 4.])

# 裁剪
print(torch.clamp(y, min=2.0, max=3.0))  # tensor([2., 2.5, 3.])
```

---

## 3. 张量形状操作

### 3.1 形状变换

| 函数                  | 说明                 | 示例              |
| :-------------------- | :------------------- | :---------------- |
| `tensor.view()`       | 改变形状（共享内存） | `x.view(3, 4)`    |
| `tensor.reshape()`    | 改变形状（可能拷贝） | `x.reshape(3, 4)` |
| `tensor.contiguous()` | 确保内存连续         | `x.contiguous()`  |
| `tensor.flatten()`    | 展平                 | `x.flatten()`     |
| `tensor.ravel()`      | 展平（一维视图）     | `x.ravel()`       |
| `torch.squeeze()`     | 移除维度为 1 的维    | `x.squeeze()`     |
| `torch.unsqueeze()`   | 增加维度             | `x.unsqueeze(0)`  |
| `tensor.expand()`     | 扩展维度             | `x.expand(3, 4)`  |
| `tensor.repeat()`     | 重复张量             | `x.repeat(2, 3)`  |
| `tensor.tile()`       | 平铺张量             | `x.tile((2, 3))`  |

```python
x = torch.arange(12)

# 改变形状
a = x.view(3, 4)         # 必须内存连续
b = x.reshape(3, 4)      # 更灵活
c = x.view(3, -1)        # -1 自动推断

# 展平
d = a.flatten()          # 一维
e = a.flatten(0, 1)      # 指定维度范围

# 增删维度
x = torch.randn(3, 4)
y = x.unsqueeze(0)       # [1, 3, 4]
y = x.unsqueeze(-1)      # [3, 4, 1]
z = y.squeeze()          # 移除所有维度为1的维

# 扩展和重复
x = torch.tensor([[1, 2], [3, 4]])
y = x.unsqueeze(0).expand(3, 2, 2)  # [3, 2, 2] 广播扩展
z = x.repeat(2, 3)                   # [4, 6] 复制数据
```

### 3.2 维度操作

| 函数                 | 说明         | 示例                 |
| :------------------- | :----------- | :------------------- |
| `tensor.transpose()` | 交换两个维度 | `x.transpose(0, 1)`  |
| `tensor.permute()`   | 重排所有维度 | `x.permute(2, 0, 1)` |
| `tensor.movedim()`   | 移动维度     | `x.movedim(0, -1)`   |
| `tensor.swapaxes()`  | 交换轴       | `x.swapaxes(0, 1)`   |
| `tensor.T`           | 2D 转置      | `x.T`                |
| `tensor.mT`          | 批量矩阵转置 | `x.mT`               |

```python
x = torch.randn(2, 3, 4)

# 交换维度
y = x.transpose(0, 2)    # [4, 3, 2]
y = x.permute(2, 0, 1)   # [4, 2, 3]
y = x.movedim(0, -1)     # [3, 4, 2]

# 矩阵转置
A = torch.randn(3, 4)
A_T = A.T                # [4, 3]

# 批量转置
batch = torch.randn(10, 3, 4)
batch_T = batch.mT       # [10, 4, 3] 最后两维转置
```

### 3.3 拼接与分割

| 函数             | 说明           | 示例                         |
| :--------------- | :------------- | :--------------------------- |
| `torch.cat()`    | 沿现有维度拼接 | `torch.cat([a, b], dim=0)`   |
| `torch.stack()`  | 沿新维度堆叠   | `torch.stack([a, b], dim=0)` |
| `torch.vstack()` | 垂直堆叠       | `torch.vstack([a, b])`       |
| `torch.hstack()` | 水平堆叠       | `torch.hstack([a, b])`       |
| `torch.dstack()` | 深度堆叠       | `torch.dstack([a, b])`       |
| `torch.split()`  | 按大小分割     | `torch.split(x, 2, dim=0)`   |
| `torch.chunk()`  | 按数量分割     | `torch.chunk(x, 3, dim=0)`   |
| `torch.unbind()` | 移除维度并分解 | `torch.unbind(x, dim=0)`     |

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# 拼接
c = torch.cat([a, b], dim=0)     # [4, 3] 沿行拼接
d = torch.cat([a, b], dim=1)     # [2, 6] 沿列拼接

# 堆叠（新增维度）
e = torch.stack([a, b], dim=0)   # [2, 2, 3]
f = torch.stack([a, b], dim=1)   # [2, 2, 3]

# 分割
x = torch.randn(6, 4)
chunks = torch.chunk(x, 3, dim=0)  # 分成3份
splits = torch.split(x, 2, dim=0)  # 每份大小为2
splits = torch.split(x, [1, 2, 3], dim=0)  # 指定每份大小

# 解绑
x = torch.randn(3, 4)
a, b, c = torch.unbind(x, dim=0)  # 返回3个 [4] 张量
```

---

## 4. 索引与切片

### 4.1 基本索引

```python
x = torch.randn(4, 5, 6)

# 基本索引
a = x[0]           # 第一个元素 [5, 6]
b = x[0, 1]        # [6]
c = x[0, 1, 2]     # 标量

# 切片
d = x[1:3]         # [2, 5, 6]
e = x[:, 1:4]      # [4, 3, 6]
f = x[::2]         # 步长为2

# 负索引
g = x[-1]          # 最后一个
h = x[:, -1]       # 最后一列
```

### 4.2 高级索引

| 函数                    | 说明         | 示例                        |
| :---------------------- | :----------- | :-------------------------- |
| `torch.index_select()`  | 按索引选择   | `x.index_select(0, idx)`    |
| `torch.gather()`        | 按索引收集   | `torch.gather(x, dim, idx)` |
| `torch.scatter()`       | 按索引分散   | `x.scatter(dim, idx, src)`  |
| `torch.take()`          | 展平后取值   | `torch.take(x, idx)`        |
| `torch.masked_select()` | 按掩码选择   | `x[mask]`                   |
| `torch.where()`         | 条件选择     | `torch.where(cond, x, y)`   |
| `torch.nonzero()`       | 非零元素索引 | `torch.nonzero(x)`          |

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 索引选择
idx = torch.tensor([0, 2])
selected = x.index_select(0, idx)  # 选择第0和第2行

# gather - 按索引收集
idx = torch.tensor([[0, 1, 2],
                    [2, 1, 0]])
gathered = torch.gather(x, 1, idx)  # 按行索引收集

# scatter - 按索引分散
dst = torch.zeros(3, 3)
idx = torch.tensor([[0], [1], [2]])
src = torch.tensor([[1.0], [2.0], [3.0]])
dst.scatter_(1, idx, src)  # 原地操作

# 布尔索引
mask = x > 5
selected = x[mask]         # tensor([6, 7, 8, 9])

# where 条件选择
result = torch.where(x > 5, x, torch.zeros_like(x))

# 非零索引
indices = torch.nonzero(x > 5)  # 返回坐标
```

### 4.3 索引操作对比

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 形状: [2, 3]

# index_select: 选择完整的行/列
idx = torch.tensor([0, 2])
result = x.index_select(1, idx)  # [[1, 3], [4, 6]]

# gather: 每行/列选择不同位置
idx = torch.tensor([[0, 1], [2, 0]])  # 形状需匹配输出
result = torch.gather(x, 1, idx)       # [[1, 2], [6, 4]]

"""
gather 理解:
  dim=1时，按列索引
  idx[i,j] 表示取 x[i, idx[i,j]]

  idx = [[0, 1],    result = [[x[0,0], x[0,1]],
         [2, 0]]              [x[1,2], x[1,0]]]
                           = [[1, 2],
                              [6, 4]]
"""
```

---

## 5. 神经网络层 (nn.Module)

### 5.1 线性层

| 层              | 说明       | 参数                                     |
| :-------------- | :--------- | :--------------------------------------- |
| `nn.Linear`     | 全连接层   | `(in_features, out_features, bias=True)` |
| `nn.Bilinear`   | 双线性层   | `(in1, in2, out, bias=True)`             |
| `nn.LazyLinear` | 延迟初始化 | `(out_features, bias=True)`              |
| `nn.Identity`   | 恒等映射   | -                                        |

```python
import torch.nn as nn

# 全连接层
linear = nn.Linear(in_features=128, out_features=64, bias=True)
x = torch.randn(32, 128)
y = linear(x)  # [32, 64]

# 查看参数
print(linear.weight.shape)  # [64, 128]
print(linear.bias.shape)    # [64]

# 延迟初始化 (自动推断输入维度)
lazy_linear = nn.LazyLinear(64)
y = lazy_linear(x)  # 首次调用时初始化
```

### 5.2 卷积层

| 层                   | 说明     | 关键参数                                   |
| :------------------- | :------- | :----------------------------------------- |
| `nn.Conv1d`          | 1D 卷积  | `(in_ch, out_ch, kernel, stride, padding)` |
| `nn.Conv2d`          | 2D 卷积  | `(in_ch, out_ch, kernel, stride, padding)` |
| `nn.Conv3d`          | 3D 卷积  | `(in_ch, out_ch, kernel, stride, padding)` |
| `nn.ConvTranspose2d` | 转置卷积 | 同上 + `output_padding`                    |
| `nn.Unfold`          | 滑窗展开 | `(kernel_size)`                            |
| `nn.Fold`            | 滑窗折叠 | `(output_size, kernel_size)`               |

```python
# 2D 卷积
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros'
)
x = torch.randn(16, 3, 224, 224)  # [B, C, H, W]
y = conv(x)  # [16, 64, 224, 224]

# 计算输出尺寸
# H_out = (H_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1

# 转置卷积 (上采样)
deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
z = deconv(y)  # [16, 32, 448, 448]

# 深度可分离卷积
depthwise = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
pointwise = nn.Conv2d(64, 128, kernel_size=1)
```

### 5.3 池化层

| 层                     | 说明           | 关键参数                    |
| :--------------------- | :------------- | :-------------------------- |
| `nn.MaxPool2d`         | 最大池化       | `(kernel, stride, padding)` |
| `nn.AvgPool2d`         | 平均池化       | `(kernel, stride, padding)` |
| `nn.AdaptiveMaxPool2d` | 自适应最大池化 | `(output_size)`             |
| `nn.AdaptiveAvgPool2d` | 自适应平均池化 | `(output_size)`             |
| `nn.MaxUnpool2d`       | 反池化         | `(kernel, stride, padding)` |

```python
# 最大池化
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(16, 64, 224, 224)
y = maxpool(x)  # [16, 64, 112, 112]

# 带索引的最大池化 (用于反池化)
maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
y, indices = maxpool(x)

# 自适应池化 (输出固定尺寸)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
z = adaptive_pool(x)  # [16, 64, 1, 1]
z = z.flatten(1)      # [16, 64]

# 全局平均池化的常用写法
gap = nn.AdaptiveAvgPool2d(1)
```

### 5.4 归一化层

| 层                  | 说明        | 输入形状                | 归一化维度   |
| :------------------ | :---------- | :---------------------- | :----------- |
| `nn.BatchNorm1d`    | 批归一化 1D | `[N, C]` 或 `[N, C, L]` | 沿 N 维      |
| `nn.BatchNorm2d`    | 批归一化 2D | `[N, C, H, W]`          | 沿 N 维      |
| `nn.LayerNorm`      | 层归一化    | 任意                    | 沿特征维     |
| `nn.GroupNorm`      | 组归一化    | `[N, C, *]`             | 沿组内通道   |
| `nn.InstanceNorm2d` | 实例归一化  | `[N, C, H, W]`          | 单样本单通道 |

```python
# 批归一化
bn = nn.BatchNorm2d(num_features=64)
x = torch.randn(16, 64, 32, 32)
y = bn(x)  # 训练时使用 batch 统计量，推理时使用移动平均

# 层归一化 (常用于 NLP/Transformer)
ln = nn.LayerNorm(normalized_shape=768)
x = torch.randn(32, 10, 768)  # [batch, seq_len, hidden]
y = ln(x)

# 也可以归一化最后几个维度
ln = nn.LayerNorm([10, 768])

# 组归一化 (对 batch size 不敏感)
gn = nn.GroupNorm(num_groups=8, num_channels=64)
x = torch.randn(16, 64, 32, 32)
y = gn(x)

# 实例归一化 (风格迁移常用)
ins_norm = nn.InstanceNorm2d(64)
```

### 5.5 Dropout 层

| 层                | 说明                  | 参数      |
| :---------------- | :-------------------- | :-------- |
| `nn.Dropout`      | 标准 Dropout          | `(p=0.5)` |
| `nn.Dropout2d`    | 2D Dropout (整个通道) | `(p=0.5)` |
| `nn.Dropout3d`    | 3D Dropout            | `(p=0.5)` |
| `nn.AlphaDropout` | SELU 激活配套         | `(p=0.5)` |

```python
# 标准 Dropout
dropout = nn.Dropout(p=0.5)
x = torch.randn(32, 128)
y = dropout(x)  # 训练时随机置零，推理时不变

# 2D Dropout (丢弃整个通道)
dropout2d = nn.Dropout2d(p=0.5)
x = torch.randn(16, 64, 32, 32)
y = dropout2d(x)  # 随机将某些通道全部置零

# 注意：推理时需设置 eval 模式
model.eval()  # Dropout 不起作用
model.train()  # Dropout 起作用
```

### 5.6 循环层

| 层            | 说明      | 关键参数                                |
| :------------ | :-------- | :-------------------------------------- |
| `nn.RNN`      | 基础 RNN  | `(input_size, hidden_size, num_layers)` |
| `nn.LSTM`     | LSTM      | `(input_size, hidden_size, num_layers)` |
| `nn.GRU`      | GRU       | `(input_size, hidden_size, num_layers)` |
| `nn.RNNCell`  | RNN 单元  | `(input_size, hidden_size)`             |
| `nn.LSTMCell` | LSTM 单元 | `(input_size, hidden_size)`             |
| `nn.GRUCell`  | GRU 单元  | `(input_size, hidden_size)`             |

```python
# LSTM
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,   # 输入形状 [batch, seq, feature]
    bidirectional=True,
    dropout=0.5
)

x = torch.randn(32, 50, 128)  # [batch, seq_len, input_size]
output, (h_n, c_n) = lstm(x)
# output: [32, 50, 512] (双向所以 256*2)
# h_n: [4, 32, 256] (2层*双向, batch, hidden)
# c_n: [4, 32, 256]

# GRU
gru = nn.GRU(128, 256, num_layers=2, batch_first=True)
output, h_n = gru(x)

# 手动循环 (使用 Cell)
lstm_cell = nn.LSTMCell(128, 256)
h = torch.zeros(32, 256)
c = torch.zeros(32, 256)
outputs = []
for t in range(50):
    h, c = lstm_cell(x[:, t, :], (h, c))
    outputs.append(h)
output = torch.stack(outputs, dim=1)
```

### 5.7 Transformer 层

| 层                           | 说明               |
| :--------------------------- | :----------------- |
| `nn.Transformer`             | 完整 Transformer   |
| `nn.TransformerEncoder`      | Transformer 编码器 |
| `nn.TransformerDecoder`      | Transformer 解码器 |
| `nn.TransformerEncoderLayer` | 编码器单层         |
| `nn.TransformerDecoderLayer` | 解码器单层         |
| `nn.MultiheadAttention`      | 多头注意力         |

```python
# 多头注意力
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(32, 50, 512)  # [batch, seq, embed]
attn_output, attn_weights = mha(x, x, x)  # self-attention

# Transformer 编码器层
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

x = torch.randn(32, 50, 512)
output = encoder(x)  # [32, 50, 512]

# 完整 Transformer
transformer = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    batch_first=True
)
src = torch.randn(32, 10, 512)
tgt = torch.randn(32, 20, 512)
output = transformer(src, tgt)  # [32, 20, 512]
```

### 5.8 Embedding 层

| 层                | 说明     | 参数                              |
| :---------------- | :------- | :-------------------------------- |
| `nn.Embedding`    | 词嵌入   | `(num_embeddings, embedding_dim)` |
| `nn.EmbeddingBag` | 嵌入池化 | `(num, dim, mode='mean')`         |

```python
# 词嵌入
embedding = nn.Embedding(
    num_embeddings=10000,  # 词表大小
    embedding_dim=256,      # 嵌入维度
    padding_idx=0           # 填充索引 (梯度为0)
)

input_ids = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 0]])
embedded = embedding(input_ids)  # [2, 4, 256]

# 预训练词向量
pretrained = torch.randn(10000, 256)  # 假设的预训练向量
embedding = nn.Embedding.from_pretrained(pretrained, freeze=False)

# EmbeddingBag (适合不定长输入)
embag = nn.EmbeddingBag(1000, 128, mode='mean')
input_ids = torch.LongTensor([1, 2, 4, 5, 4, 3, 2])
offsets = torch.LongTensor([0, 4])  # 每个样本的起始位置
output = embag(input_ids, offsets)  # [2, 128]
```

### 5.9 容器模块

| 容器               | 说明     |
| :----------------- | :------- |
| `nn.Sequential`    | 顺序容器 |
| `nn.ModuleList`    | 模块列表 |
| `nn.ModuleDict`    | 模块字典 |
| `nn.ParameterList` | 参数列表 |
| `nn.ParameterDict` | 参数字典 |

```python
# Sequential - 顺序执行
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 带名称的 Sequential
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
]))
print(model.fc1)  # 可以按名称访问

# ModuleList - 动态模块列表
class MyModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ModuleDict - 模块字典
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(128, 64)
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(64, 10),
            'regression': nn.Linear(64, 1)
        })

    def forward(self, x, task):
        x = self.backbone(x)
        return self.heads[task](x)
```

---

## 6. 激活函数

### 6.1 常用激活函数

| 函数            | 公式                      | 用途             |
| :-------------- | :------------------------ | :--------------- |
| `nn.ReLU`       | max(0, x)                 | 隐藏层默认       |
| `nn.LeakyReLU`  | max(0.01x, x)             | 避免死神经元     |
| `nn.PReLU`      | max(αx, x), α 可学习      | 参数化 ReLU      |
| `nn.ELU`        | x if x>0 else α(e^x-1)    | 平滑 ReLU        |
| `nn.SELU`       | λ·ELU                     | 自归一化         |
| `nn.GELU`       | x·Φ(x)                    | Transformer 常用 |
| `nn.SiLU/Swish` | x·σ(x)                    | 现代网络常用     |
| `nn.Mish`       | x·tanh(softplus(x))       | 平滑激活         |
| `nn.Sigmoid`    | 1/(1+e^(-x))              | 二分类输出       |
| `nn.Tanh`       | (e^x-e^(-x))/(e^x+e^(-x)) | RNN              |
| `nn.Softmax`    | e^xi/Σe^xj                | 多分类输出       |
| `nn.LogSoftmax` | log(Softmax)              | 配合 NLLLoss     |
| `nn.Softplus`   | log(1+e^x)                | 平滑 ReLU        |
| `nn.Hardtanh`   | 裁剪版 tanh               | 轻量化           |
| `nn.Hardswish`  | 近似 Swish                | 移动端           |

```python
import torch.nn.functional as F

# 模块形式
relu = nn.ReLU(inplace=True)
gelu = nn.GELU()
silu = nn.SiLU()  # Swish

# 函数形式
x = torch.randn(32, 128)
y = F.relu(x)
y = F.gelu(x)
y = F.silu(x)
y = F.leaky_relu(x, negative_slope=0.01)

# Softmax (注意 dim 参数)
logits = torch.randn(32, 10)
probs = F.softmax(logits, dim=1)  # 沿类别维度
log_probs = F.log_softmax(logits, dim=1)
```

### 6.2 激活函数对比图

```
ReLU:           LeakyReLU:       GELU:            SiLU/Swish:
    │  ╱            │  ╱            │   _--          │   __--
    │ ╱             │ ╱            _│--              │_--
────┼╱────      ──_─┼╱────      ──╱─┼────         ─_╱┼────
   ╱│             ╱ │            ╱  │              ╱  │
  ╱ │            ╱  │
```

---

## 7. 损失函数

### 7.1 回归损失

| 损失函数          | 公式              | 适用场景     |
| :---------------- | :---------------- | :----------- |
| `nn.MSELoss`      | $(y-\hat{y})^2$   | 标准回归     |
| `nn.L1Loss`       | $\|y-\hat{y}\|$   | 鲁棒回归     |
| `nn.SmoothL1Loss` | Huber Loss        | 目标检测     |
| `nn.HuberLoss`    | 可配置 δ 的 Huber | 通用鲁棒回归 |

```python
# MSE Loss
mse = nn.MSELoss(reduction='mean')  # 'none', 'mean', 'sum'
y_pred = torch.randn(32, 1)
y_true = torch.randn(32, 1)
loss = mse(y_pred, y_true)

# Huber Loss
huber = nn.HuberLoss(delta=1.0)
loss = huber(y_pred, y_true)

# Smooth L1 (等价于 delta=1 的 Huber)
smooth_l1 = nn.SmoothL1Loss()
loss = smooth_l1(y_pred, y_true)
```

### 7.2 分类损失

| 损失函数                      | 输入     | 适用场景            |
| :---------------------------- | :------- | :------------------ |
| `nn.CrossEntropyLoss`         | logits   | 多分类              |
| `nn.NLLLoss`                  | log 概率 | 配合 LogSoftmax     |
| `nn.BCELoss`                  | 概率     | 二分类/多标签       |
| `nn.BCEWithLogitsLoss`        | logits   | 二分类/多标签(推荐) |
| `nn.MultiLabelSoftMarginLoss` | logits   | 多标签              |
| `nn.MultiMarginLoss`          | logits   | 多分类间隔          |

```python
# 多分类 (推荐)
ce_loss = nn.CrossEntropyLoss(
    weight=None,           # 类别权重
    ignore_index=-100,     # 忽略的标签
    label_smoothing=0.0    # 标签平滑
)
logits = torch.randn(32, 10)   # [batch, num_classes]
targets = torch.randint(0, 10, (32,))  # 类别索引
loss = ce_loss(logits, targets)

# 二分类 (推荐)
bce_logits = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()
loss = bce_logits(logits, targets)

# 多标签分类
logits = torch.randn(32, 5)  # 5个标签
targets = torch.randint(0, 2, (32, 5)).float()
loss = bce_logits(logits, targets)
```

### 7.3 其他损失

| 损失函数                 | 说明                |
| :----------------------- | :------------------ |
| `nn.KLDivLoss`           | KL 散度             |
| `nn.CosineEmbeddingLoss` | 余弦相似度损失      |
| `nn.TripletMarginLoss`   | 三元组损失          |
| `nn.MarginRankingLoss`   | 排序损失            |
| `nn.CTCLoss`             | CTC 损失 (语音识别) |

```python
# KL散度
kl_loss = nn.KLDivLoss(reduction='batchmean')
log_probs = F.log_softmax(torch.randn(32, 10), dim=1)
targets = F.softmax(torch.randn(32, 10), dim=1)
loss = kl_loss(log_probs, targets)

# 三元组损失
triplet_loss = nn.TripletMarginLoss(margin=1.0)
anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negative = torch.randn(32, 128)
loss = triplet_loss(anchor, positive, negative)

# 余弦嵌入损失
cos_loss = nn.CosineEmbeddingLoss()
x1 = torch.randn(32, 128)
x2 = torch.randn(32, 128)
labels = torch.randint(-1, 2, (32,)).float()  # 1: 相似, -1: 不相似
loss = cos_loss(x1, x2, labels)
```

---

## 8. 优化器

### 8.1 常用优化器

| 优化器           | 说明                | 推荐场景    |
| :--------------- | :------------------ | :---------- |
| `optim.SGD`      | 随机梯度下降        | 通用、CV    |
| `optim.Adam`     | 自适应矩估计        | 通用默认    |
| `optim.AdamW`    | Adam + 解耦权重衰减 | Transformer |
| `optim.RMSprop`  | 自适应学习率        | RNN         |
| `optim.Adagrad`  | 自适应梯度          | 稀疏数据    |
| `optim.Adadelta` | Adagrad 改进        | -           |
| `optim.NAdam`    | Adam + Nesterov     | -           |
| `optim.RAdam`    | 带整流的 Adam       | -           |
| `optim.LBFGS`    | 拟牛顿法            | 小数据/二阶 |

```python
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,  # L2正则
    nesterov=True
)

# Adam
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)

# AdamW (推荐用于 Transformer)
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# 不同参数组使用不同学习率
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

### 8.2 优化器使用

```python
# 标准训练循环
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()  # 清除梯度
        loss.backward()        # 计算梯度

        # 梯度裁剪 (可选)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()       # 更新参数

# 获取/设置学习率
for param_group in optimizer.param_groups:
    print(param_group['lr'])
    param_group['lr'] = 0.0001  # 手动调整
```

---

## 9. 学习率调度器

### 9.1 常用调度器

| 调度器                        | 说明               |
| :---------------------------- | :----------------- |
| `StepLR`                      | 固定间隔衰减       |
| `MultiStepLR`                 | 多个里程碑衰减     |
| `ExponentialLR`               | 指数衰减           |
| `CosineAnnealingLR`           | 余弦退火           |
| `CosineAnnealingWarmRestarts` | 带重启的余弦退火   |
| `ReduceLROnPlateau`           | 验证集无改善时衰减 |
| `CyclicLR`                    | 循环学习率         |
| `OneCycleLR`                  | 单周期策略         |
| `LinearLR`                    | 线性调整           |
| `PolynomialLR`                | 多项式衰减         |
| `LambdaLR`                    | 自定义函数         |
| `SequentialLR`                | 组合多个调度器     |

```python
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, LambdaLR
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 每 30 个 epoch 学习率乘以 0.1
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 在指定 epoch 衰减
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 验证集无改善时衰减
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控指标是否下降
    factor=0.1,      # 衰减因子
    patience=10,     # 容忍多少个epoch无改善
    verbose=True
)

# OneCycle (适合快速训练)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=len(dataloader) * num_epochs,
    pct_start=0.3    # 30% 时间用于 warmup
)

# Warmup + 余弦退火
def warmup_cosine(epoch):
    if epoch < 5:
        return epoch / 5
    return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / 95))
scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)
```

### 9.2 调度器使用

```python
# 每个 epoch 调用
for epoch in range(num_epochs):
    train_one_epoch()
    validate()
    scheduler.step()  # epoch 级别调度器

# 每个 batch 调用 (OneCycleLR)
for epoch in range(num_epochs):
    for batch in dataloader:
        train_step()
        scheduler.step()  # step 级别调度器

# ReduceLROnPlateau 需要传入指标
val_loss = validate()
scheduler.step(val_loss)

# 获取当前学习率
print(scheduler.get_last_lr())
```

---

## 10. 数据加载

### 10.1 Dataset

```python
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# 使用
dataset = CustomDataset(data, labels)
print(len(dataset))
sample = dataset[0]
```

### 10.2 DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # 训练时打乱
    num_workers=4,          # 多进程加载
    pin_memory=True,        # 加速 GPU 传输
    drop_last=True,         # 丢弃不完整的 batch
    collate_fn=None,        # 自定义批处理函数
    prefetch_factor=2,      # 预取数量
    persistent_workers=True # 保持 worker 进程
)

# 迭代
for batch_x, batch_y in dataloader:
    # batch_x: [batch_size, ...]
    pass

# 获取单个 batch
batch = next(iter(dataloader))
```

### 10.3 常用数据工具

| 类/函数                 | 说明             |
| :---------------------- | :--------------- |
| `TensorDataset`         | 从张量创建数据集 |
| `ConcatDataset`         | 合并数据集       |
| `Subset`                | 数据集子集       |
| `random_split`          | 随机划分         |
| `WeightedRandomSampler` | 加权采样         |
| `SequentialSampler`     | 顺序采样         |
| `RandomSampler`         | 随机采样         |
| `BatchSampler`          | 批次采样         |

```python
from torch.utils.data import (
    TensorDataset, random_split,
    WeightedRandomSampler, Subset
)

# 从张量创建数据集
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)

# 随机划分
train_set, val_set = random_split(dataset, [800, 200])

# 子集
indices = [0, 1, 2, 3, 4]
subset = Subset(dataset, indices)

# 加权采样 (处理类别不平衡)
class_counts = [900, 100]  # 类别数量
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[y]  # 每个样本的权重

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### 10.4 自定义 collate_fn

```python
def custom_collate_fn(batch):
    """处理变长序列"""
    # batch: [(x1, y1), (x2, y2), ...]
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    # 填充到相同长度
    sequences_padded = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )

    # 计算实际长度
    lengths = torch.tensor([len(seq) for seq in sequences])

    return sequences_padded, labels, lengths

dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
```

---

## 11. 自动求导

### 11.1 基本操作

```python
# 创建需要梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 或后续设置
x = torch.randn(3)
x.requires_grad_(True)

# 计算
y = x ** 2
z = y.sum()

# 反向传播
z.backward()
print(x.grad)  # dz/dx = 2x = [2, 4, 6]

# 清除梯度
x.grad.zero_()

# 梯度累加（默认行为）
z.backward()  # 梯度会累加
```

### 11.2 控制梯度

```python
# 停止梯度追踪
with torch.no_grad():
    y = x * 2  # 不会记录操作

# 分离张量
y = x.detach()  # 返回不需要梯度的新张量

# 临时启用/禁用梯度
with torch.enable_grad():
    pass

# 设置全局梯度计算
torch.set_grad_enabled(False)
torch.set_grad_enabled(True)

# 推理模式 (比 no_grad 更高效)
with torch.inference_mode():
    y = model(x)
```

### 11.3 高级自动求导

```python
# 高阶导数
x = torch.tensor([1.0], requires_grad=True)
y = x ** 3
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]  # dy/dx = 3x²
grad2 = torch.autograd.grad(grad1, x)[0]  # d²y/dx² = 6x

# 向量-雅可比积
x = torch.randn(3, requires_grad=True)
y = x ** 2
v = torch.tensor([1.0, 0.1, 0.01])
y.backward(v)  # 计算 v^T @ (dy/dx)

# 获取梯度函数
print(y.grad_fn)  # <PowBackward0>

# 保留计算图
y.backward(retain_graph=True)  # 可以再次 backward

# 钩子函数
def print_grad(grad):
    print(f"Gradient: {grad}")
x.register_hook(print_grad)
```

---

## 12. 模型保存与加载

### 12.1 保存与加载方式

```python
# 方式1: 只保存参数 (推荐)
torch.save(model.state_dict(), 'model_weights.pth')

# 加载参数
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))

# 方式2: 保存整个模型 (包含结构)
torch.save(model, 'model_complete.pth')

# 加载完整模型
model = torch.load('model_complete.pth')
```

### 12.2 保存检查点

```python
# 保存完整检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'best_acc': best_acc,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
best_acc = checkpoint['best_acc']
```

### 12.3 跨设备加载

```python
# GPU 保存，CPU 加载
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# CPU 保存，GPU 加载
model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))

# 多 GPU 保存，单 GPU 加载
state_dict = torch.load('model.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')  # 移除 'module.' 前缀
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
```

### 12.4 部分加载

```python
# 加载部分参数 (迁移学习)
pretrained_dict = torch.load('pretrained.pth')
model_dict = model.state_dict()

# 过滤不匹配的键
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}

# 更新并加载
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 或使用 strict=False
model.load_state_dict(torch.load('model.pth'), strict=False)
```

---

## 13. GPU 操作

### 13.1 设备管理

```python
# 检查 CUDA 可用性
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')  # 指定 GPU

# 张量移动到 GPU
x = torch.randn(3, 4)
x = x.to(device)
x = x.cuda()
x = x.cuda(0)  # 指定 GPU

# 模型移动到 GPU
model = model.to(device)
model = model.cuda()
```

### 13.2 内存管理

```python
# 显存查看
print(torch.cuda.memory_allocated())  # 已分配
print(torch.cuda.memory_reserved())   # 已预留
print(torch.cuda.max_memory_allocated())

# 清理缓存
torch.cuda.empty_cache()

# 重置统计
torch.cuda.reset_peak_memory_stats()

# 内存快照
torch.cuda.memory_summary()
```

### 13.3 多 GPU 训练

```python
# DataParallel (简单)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# DistributedDataParallel (推荐)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# 数据加载器需要使用 DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

### 13.4 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 自动混合精度
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # 缩放梯度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 14. 常用工具函数

### 14.1 初始化方法

```python
import torch.nn.init as init

# Xavier 初始化
init.xavier_uniform_(layer.weight)
init.xavier_normal_(layer.weight)

# Kaiming 初始化 (ReLU推荐)
init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

# 其他初始化
init.constant_(layer.bias, 0)
init.zeros_(layer.weight)
init.ones_(layer.weight)
init.normal_(layer.weight, mean=0, std=0.01)
init.uniform_(layer.weight, a=-1, b=1)
init.orthogonal_(layer.weight)

# 自定义模型初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')

model.apply(init_weights)
```

### 14.2 梯度操作

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")

# 冻结参数
for param in model.backbone.parameters():
    param.requires_grad = False

# 只训练部分参数
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params)
```

### 14.3 模型信息

```python
# 查看模型结构
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

# 查看所有参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# 查看所有模块
for name, module in model.named_modules():
    print(f"{name}: {module.__class__.__name__}")

# 使用 torchinfo (需安装)
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))
```

### 14.4 序列处理工具

```python
from torch.nn.utils.rnn import (
    pad_sequence, pack_padded_sequence,
    pad_packed_sequence, pack_sequence
)

# 填充序列
sequences = [torch.tensor([1, 2, 3]),
             torch.tensor([4, 5]),
             torch.tensor([6, 7, 8, 9])]
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
# tensor([[1, 2, 3, 0],
#         [4, 5, 0, 0],
#         [6, 7, 8, 9]])

# 打包填充序列 (RNN使用)
lengths = torch.tensor([3, 2, 4])
packed = pack_padded_sequence(padded, lengths, batch_first=True,
                               enforce_sorted=False)

# 解包
output, lengths = pad_packed_sequence(packed, batch_first=True)
```

### 14.5 其他实用函数

```python
# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 计时器
import time
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")

with Timer():
    output = model(input)

# CUDA 同步计时
torch.cuda.synchronize()
start = time.time()
output = model(input)
torch.cuda.synchronize()
elapsed = time.time() - start
```

---

## 15. 模型构建模板

### 15.1 基础分类模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # 返回 logits
        return x
```

### 15.2 CNN 模型

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
```

### 15.3 完整训练脚本

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in tqdm(dataloader, desc="Training"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(dataloader), correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(dataloader), correct / total

def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001

    # 数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # 模型
    model = Classifier(input_dim=784, hidden_dim=256, num_classes=10)
    model = model.to(device)

    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')

if __name__ == '__main__':
    main()
```

---

## 快速参考卡片

```
╔═══════════════════════════════════════════════════════════════╗
║                    PyTorch 速查表                              ║
╠═══════════════════════════════════════════════════════════════╣
║  张量创建:                                                     ║
║    torch.tensor(), zeros(), ones(), randn(), arange()         ║
║                                                               ║
║  形状操作:                                                     ║
║    view(), reshape(), squeeze(), unsqueeze(), permute()       ║
║    cat(), stack(), split(), chunk()                           ║
║                                                               ║
║  数学运算:                                                     ║
║    +, -, *, /, @, mm(), bmm(), matmul()                       ║
║    sum(), mean(), max(), min(), argmax()                      ║
║                                                               ║
║  常用层:                                                       ║
║    Linear, Conv2d, BatchNorm2d, Dropout, LSTM, Embedding      ║
║                                                               ║
║  损失函数:                                                     ║
║    CrossEntropyLoss (多分类), BCEWithLogitsLoss (二分类)       ║
║    MSELoss (回归), L1Loss, SmoothL1Loss                       ║
║                                                               ║
║  优化器:                                                       ║
║    Adam, AdamW, SGD (momentum=0.9)                            ║
║                                                               ║
║  训练循环:                                                     ║
║    optimizer.zero_grad() → loss.backward() → optimizer.step() ║
║                                                               ║
║  保存加载:                                                     ║
║    torch.save(model.state_dict(), 'model.pth')                ║
║    model.load_state_dict(torch.load('model.pth'))             ║
║                                                               ║
║  GPU:                                                          ║
║    device = torch.device('cuda' if available else 'cpu')      ║
║    model.to(device), tensor.to(device)                        ║
╚═══════════════════════════════════════════════════════════════╝
```

---

> 📌 **版本**: PyTorch 2.0+  
> 📅 **更新**: 2024  
> 💡 **提示**: 使用 `Ctrl+F` 快速查找所需函数
