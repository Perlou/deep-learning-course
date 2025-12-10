# NumPy 常用方法和函数速查手册

## 目录

1. [数组创建](#1-数组创建)
2. [数组属性](#2-数组属性)
3. [索引与切片](#3-索引与切片)
4. [数组变形](#4-数组变形)
5. [数学运算](#5-数学运算)
6. [统计函数](#6-统计函数)
7. [线性代数](#7-线性代数-nplinalg)
8. [排序与搜索](#8-排序与搜索)
9. [广播机制](#9-广播机制-broadcasting)
10. [实用技巧](#10-实用技巧)
11. [速查表](#11-速查表)

---

## 1. 数组创建

```python
import numpy as np

# ===== 基础创建 =====
a = np.array([1, 2, 3])              # 从列表创建
b = np.array([[1,2], [3,4]])         # 2D数组

# ===== 特殊数组 =====
np.zeros((3, 4))                     # 全0数组 (3行4列)
np.ones((2, 3))                      # 全1数组
np.full((2, 2), 7)                   # 全填充7
np.empty((2, 3))                     # 未初始化（快但值随机）
np.eye(3)                            # 3×3 单位矩阵
np.diag([1, 2, 3])                   # 对角矩阵

# ===== 序列数组 =====
np.arange(0, 10, 2)                  # [0, 2, 4, 6, 8] 步长2
np.linspace(0, 1, 5)                 # [0, 0.25, 0.5, 0.75, 1] 等分5份
np.logspace(0, 2, 3)                 # [1, 10, 100] 对数等分

# ===== 随机数组 =====
np.random.rand(3, 3)                 # 0~1 均匀分布
np.random.randn(3, 3)                # 标准正态分布
np.random.randint(0, 10, (3, 3))     # 0~9 随机整数
np.random.choice([1,2,3], size=5)    # 从列表中随机选择
np.random.seed(42)                   # 设置随机种子
np.random.shuffle(arr)               # 随机打乱数组
```

---

## 2. 数组属性

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape      # (2, 3) 形状
arr.ndim       # 2      维度数
arr.size       # 6      元素总数
arr.dtype      # int64  数据类型
arr.itemsize   # 8      每个元素字节数
arr.nbytes     # 48     总字节数
len(arr)       # 2      第一维长度
```

---

## 3. 索引与切片

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# ===== 基本索引 =====
arr[0, 1]           # 2 (第0行第1列)
arr[1]              # [4, 5, 6] (第1行)
arr[:, 0]           # [1, 4, 7] (第0列)
arr[-1]             # [7, 8, 9] (最后一行)

# ===== 切片 =====
arr[0:2, 1:3]       # [[2,3], [5,6]] (前2行，1-2列)
arr[::2]            # [[1,2,3], [7,8,9]] (隔行取)
arr[::-1]           # 反转数组

# ===== 高级索引 =====
arr[[0, 2]]              # 取第0和第2行
arr[arr > 5]             # [6, 7, 8, 9] 布尔索引
np.where(arr > 5)        # 返回满足条件的索引
np.where(arr > 5, 1, 0)  # 满足条件填1，否则填0
```

---

## 4. 数组变形

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# ===== 变形 =====
arr.reshape(3, 4)        # 变成 3×4
arr.reshape(3, -1)       # -1 自动计算 → 3×4
arr.flatten()            # 展平成1D（返回副本）
arr.ravel()              # 展平成1D（返回视图）
arr.resize(3, 4)         # 原地修改形状

# ===== 转置 =====
arr2d = np.array([[1,2], [3,4]])
arr2d.T                  # 转置
np.transpose(arr2d)      # 同上
np.swapaxes(arr, 0, 1)   # 交换轴

# ===== 维度操作 =====
np.expand_dims(arr, axis=0)  # 增加维度
np.squeeze(arr)              # 删除长度为1的维度
arr[np.newaxis, :]           # 增加新轴

# ===== 合并 =====
a = np.array([1, 2])
b = np.array([3, 4])
np.concatenate([a, b])       # [1, 2, 3, 4]
np.vstack([a, b])            # 垂直堆叠 [[1,2], [3,4]]
np.hstack([a, b])            # 水平堆叠 [1, 2, 3, 4]
np.stack([a, b], axis=0)     # 沿新轴堆叠

# ===== 分割 =====
np.split(arr, 3)             # 等分成3份
np.array_split(arr, 3)       # 不等分（允许不均匀）
np.vsplit(arr2d, 2)          # 垂直分割
np.hsplit(arr2d, 2)          # 水平分割
```

---

## 5. 数学运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# ===== 基本运算（逐元素）=====
a + b       # [5, 7, 9]
a - b       # [-3, -3, -3]
a * b       # [4, 10, 18]  逐元素乘
a / b       # [0.25, 0.4, 0.5]
a // b      # 整除
a ** 2      # [1, 4, 9]
a % 2       # [1, 0, 1] 取模

# ===== 比较运算 =====
a > 2       # [False, False, True]
a == b      # [False, False, False]
np.equal(a, b)
np.greater(a, 2)

# ===== 数学函数 =====
np.sqrt(a)       # 平方根
np.exp(a)        # e^x
np.log(a)        # 自然对数
np.log2(a)       # 以2为底
np.log10(a)      # 以10为底
np.abs(a)        # 绝对值
np.sign(a)       # 符号函数
np.floor(a)      # 向下取整
np.ceil(a)       # 向上取整
np.round(a, 2)   # 四舍五入保留2位
np.clip(a, 2, 5) # 裁剪到[2,5]范围

# ===== 三角函数 =====
np.sin(a)        # 正弦
np.cos(a)        # 余弦
np.tan(a)        # 正切
np.arcsin(a)     # 反正弦
np.degrees(a)    # 弧度转角度
np.radians(a)    # 角度转弧度

# ===== 矩阵运算 =====
np.dot(a, b)     # 点积：32
a @ b            # 同上（Python 3.5+）
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
A @ B            # 矩阵乘法
np.multiply(A, B)    # 逐元素乘
np.outer(a, b)       # 外积
np.inner(a, b)       # 内积
np.cross(a, b)       # 叉积（3D向量）
```

---

## 6. 统计函数

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# ===== 基本统计 =====
np.sum(arr)           # 21 总和
np.sum(arr, axis=0)   # [5, 7, 9] 按列求和
np.sum(arr, axis=1)   # [6, 15] 按行求和

np.mean(arr)          # 3.5 平均值
np.median(arr)        # 3.5 中位数
np.std(arr)           # 标准差
np.var(arr)           # 方差

np.min(arr)           # 1 最小值
np.max(arr)           # 6 最大值
np.ptp(arr)           # 5 极差 (max - min)
np.argmin(arr)        # 0 最小值索引（展平后）
np.argmax(arr)        # 5 最大值索引（展平后）

np.cumsum(arr)        # 累积和 [1, 3, 6, 10, 15, 21]
np.cumprod(arr)       # 累积积
np.diff(arr)          # 差分

# ===== 高级统计 =====
np.percentile(arr, 50)        # 50%分位数
np.quantile(arr, 0.5)         # 同上
np.corrcoef(arr)              # 相关系数矩阵
np.cov(arr)                   # 协方差矩阵
np.histogram(arr, bins=3)     # 直方图
np.bincount(arr.flatten())    # 统计非负整数出现次数

# ===== 逻辑统计 =====
np.any(arr > 3)       # True  是否存在满足条件
np.all(arr > 0)       # True  是否全部满足
np.count_nonzero(arr) # 6     非零元素个数
```

---

## 7. 线性代数 (np.linalg)

```python
A = np.array([[1, 2],
              [3, 4]])

# ===== 基本操作 =====
np.linalg.inv(A)          # 逆矩阵
np.linalg.pinv(A)         # 伪逆矩阵
np.linalg.det(A)          # 行列式 = -2
np.linalg.matrix_rank(A)  # 秩 = 2
np.trace(A)               # 迹 = 5 (对角线之和)

# ===== 范数 =====
np.linalg.norm(A)         # Frobenius范数
np.linalg.norm(A, ord=1)  # 1范数
np.linalg.norm(A, ord=2)  # 2范数（最大奇异值）
np.linalg.norm(A, ord=np.inf)  # 无穷范数

# ===== 特征值分解 =====
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues = np.linalg.eigvals(A)  # 只要特征值

# ===== 奇异值分解 (SVD) =====
U, S, Vt = np.linalg.svd(A)
S = np.linalg.svd(A, compute_uv=False)  # 只要奇异值

# ===== 解线性方程组 Ax = b =====
b = np.array([5, 6])
x = np.linalg.solve(A, b)

# ===== 最小二乘解 =====
result = np.linalg.lstsq(A, b, rcond=None)
x = result[0]  # 解

# ===== 矩阵分解 =====
Q, R = np.linalg.qr(A)        # QR分解
L = np.linalg.cholesky(A)     # Cholesky分解（需正定矩阵）
```

---

## 8. 排序与搜索

```python
arr = np.array([3, 1, 4, 1, 5, 9])

# ===== 排序 =====
np.sort(arr)              # [1, 1, 3, 4, 5, 9] 返回副本
arr.sort()                # 原地排序
np.argsort(arr)           # 返回排序后的索引
np.sort(arr)[::-1]        # 降序排序

# 按列/行排序
arr2d = np.array([[3,1], [2,4]])
np.sort(arr2d, axis=0)    # 按列排序
np.sort(arr2d, axis=1)    # 按行排序

# ===== 搜索 =====
np.where(arr > 3)         # 返回满足条件的索引
np.argwhere(arr > 3)      # 同上，不同格式
np.searchsorted(np.sort(arr), 3)  # 二分查找插入位置
np.extract(arr > 3, arr)  # 提取满足条件的元素

# ===== 最值索引 =====
np.argmax(arr)            # 最大值索引
np.argmin(arr)            # 最小值索引
np.unravel_index(np.argmax(arr2d), arr2d.shape)  # 多维最大值索引

# ===== 唯一值 =====
np.unique(arr)                           # [1, 3, 4, 5, 9]
np.unique(arr, return_counts=True)       # 返回计数
np.unique(arr, return_index=True)        # 返回首次出现索引
np.unique(arr, return_inverse=True)      # 返回重建索引
```

---

## 9. 广播机制 (Broadcasting)

```python
# 不同形状的数组可以运算
a = np.array([[1, 2, 3],
              [4, 5, 6]])    # 2×3

b = np.array([10, 20, 30])   # 1×3

a + b   # b自动广播成2×3
# [[11, 22, 33],
#  [14, 25, 36]]
```

**广播规则：**

1. 从后往前对齐维度
2. 维度为 1 的可以广播
3. 维度不存在的视为 1

```
示例：
     2 × 3       2 × 3
   +     3   →  +     3 (广播为 2×3)
   ───────      ───────
     2 × 3        2 × 3

更多示例：
(3, 4) + (4,)     → (3, 4)  ✓
(3, 4) + (3, 1)   → (3, 4)  ✓
(3, 4) + (3,)     → Error   ✗
(1, 5) + (4, 1)   → (4, 5)  ✓
```

---

## 10. 实用技巧

```python
# ===== 复制 =====
b = a.copy()              # 深拷贝
b = a.view()              # 浅拷贝（共享数据）
b = a[:]                  # 浅拷贝

# ===== 类型转换 =====
arr.astype(np.float32)    # 转换数据类型
arr.astype(int)           # 转为整数

# ===== NaN/Inf 处理 =====
np.isnan(arr)             # 检测 NaN
np.isinf(arr)             # 检测 Inf
np.isfinite(arr)          # 检测有限数
np.nan_to_num(arr)        # NaN→0, Inf→大数
np.nansum(arr)            # 忽略NaN求和
np.nanmean(arr)           # 忽略NaN求平均

# ===== 保存与加载 =====
np.save('data.npy', arr)               # 保存单个数组
arr = np.load('data.npy')              # 加载
np.savez('data.npz', a=arr1, b=arr2)   # 保存多个数组
data = np.load('data.npz')             # 加载
arr1 = data['a']

# 文本文件
np.savetxt('data.csv', arr, delimiter=',')         # 保存CSV
arr = np.loadtxt('data.csv', delimiter=',')        # 加载CSV
arr = np.genfromtxt('data.csv', delimiter=',')     # 更灵活的加载

# ===== 设置打印选项 =====
np.set_printoptions(precision=3, suppress=True)    # 精度3位，不用科学计数法
np.set_printoptions(threshold=np.inf)              # 打印完整数组

# ===== 性能优化 =====
np.vectorize(func)        # 向量化自定义函数
np.frompyfunc(func, 1, 1) # 更快的向量化
```

---

## 11. 速查表

### 创建数组

| 函数               | 说明       |
| ------------------ | ---------- |
| `np.array()`       | 从列表创建 |
| `np.zeros()`       | 全 0 数组  |
| `np.ones()`        | 全 1 数组  |
| `np.full()`        | 填充指定值 |
| `np.eye()`         | 单位矩阵   |
| `np.arange()`      | 等差序列   |
| `np.linspace()`    | 等分序列   |
| `np.random.rand()` | 随机数组   |

### 数组操作

| 函数                    | 说明          |
| ----------------------- | ------------- |
| `reshape()`             | 改变形状      |
| `flatten()`             | 展平（副本）  |
| `ravel()`               | 展平（视图）  |
| `T` / `transpose()`     | 转置          |
| `concatenate()`         | 拼接          |
| `vstack()` / `hstack()` | 垂直/水平堆叠 |
| `split()`               | 分割          |

### 数学运算

| 函数               | 说明     |
| ------------------ | -------- |
| `+`, `-`, `*`, `/` | 基本运算 |
| `@` / `np.dot()`   | 矩阵乘法 |
| `np.sqrt()`        | 平方根   |
| `np.exp()`         | 指数     |
| `np.log()`         | 对数     |
| `np.sin()`         | 三角函数 |

### 统计函数

| 函数                          | 说明     |
| ----------------------------- | -------- |
| `np.sum()`                    | 求和     |
| `np.mean()`                   | 平均值   |
| `np.std()`                    | 标准差   |
| `np.var()`                    | 方差     |
| `np.min()` / `np.max()`       | 最值     |
| `np.argmin()` / `np.argmax()` | 最值索引 |
| `np.cumsum()`                 | 累积和   |

### 线性代数

| 函数                | 说明       |
| ------------------- | ---------- |
| `np.linalg.inv()`   | 逆矩阵     |
| `np.linalg.det()`   | 行列式     |
| `np.linalg.eig()`   | 特征分解   |
| `np.linalg.svd()`   | 奇异值分解 |
| `np.linalg.solve()` | 解方程组   |
| `np.linalg.norm()`  | 范数       |

### 排序搜索

| 函数                | 说明     |
| ------------------- | -------- |
| `np.sort()`         | 排序     |
| `np.argsort()`      | 排序索引 |
| `np.where()`        | 条件索引 |
| `np.unique()`       | 唯一值   |
| `np.searchsorted()` | 二分查找 |

---

## 附录：常用数据类型

| dtype                                             | 说明        |
| ------------------------------------------------- | ----------- |
| `np.int8`, `np.int16`, `np.int32`, `np.int64`     | 整数        |
| `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64` | 无符号整数  |
| `np.float16`, `np.float32`, `np.float64`          | 浮点数      |
| `np.complex64`, `np.complex128`                   | 复数        |
| `np.bool_`                                        | 布尔值      |
| `np.object_`                                      | Python 对象 |
| `np.string_`                                      | 字节字符串  |

---

> 📝 **作者提示**：本文档涵盖了 NumPy 最常用的函数和方法，适合日常开发查阅使用。

```

---

以上是完整的 Markdown 文档，您可以直接复制保存为 `.md` 文件使用！
```
