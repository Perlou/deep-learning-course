# Phase 1: Python 数据科学基础 - 测试题

> **题目数量**：30 道  
> **题型分布**：选择题 15 道 + 填空题 8 道 + 编程题 7 道

---

## 一、选择题（每题 2 分，共 30 分）

### NumPy 部分

**1. 以下哪个函数用于创建一个从 0 到 9 的等差数组？**

- A. `np.linspace(0, 9, 10)`
- B. `np.arange(10)`
- C. `np.zeros(10)`
- D. `np.ones(10)`

**2. 对于数组 `arr = np.array([[1,2,3], [4,5,6]])`，`arr.shape` 的值是？**

- A. `(6,)`
- B. `(3, 2)`
- C. `(2, 3)`
- D. `6`

**3. 以下哪个操作表示矩阵乘法？**

- A. `A * B`
- B. `A @ B`
- C. `A + B`
- D. `A / B`

**4. 广播机制中，以下哪种形状组合是兼容的？**

- A. `(4, 3)` 和 `(3, 4)`
- B. `(2, 3)` 和 `(3,)`
- C. `(4, 5)` 和 `(5, 4)`
- D. `(2, 3)` 和 `(4, 3)`

**5. `np.linalg.inv(A)` 计算的是矩阵 A 的？**

- A. 转置
- B. 行列式
- C. 逆矩阵
- D. 特征值

**6. 对于 NumPy 数组进行向量化操作相比 Python 循环，速度通常快多少？**

- A. 约 2-5 倍
- B. 约 10-50 倍
- C. 约 100-1000 倍
- D. 没有明显差异

### Pandas 部分

**7. 以下哪个方法用于查看 DataFrame 的前 5 行？**

- A. `df.tail()`
- B. `df.head()`
- C. `df.describe()`
- D. `df.info()`

**8. `df.loc[0:2]` 和 `df.iloc[0:2]` 的区别是？**

- A. 没有区别
- B. `loc` 包含结束位置，`iloc` 不包含
- C. `iloc` 包含结束位置，`loc` 不包含
- D. `loc` 用于列索引，`iloc` 用于行索引

**9. 以下哪种方法不能用于填充缺失值？**

- A. `df.fillna(0)`
- B. `df.fillna(method='ffill')`
- C. `df.dropna()`
- D. `df.fillna(df.mean())`

**10. `pd.merge(df1, df2, on='id', how='left')` 执行的是？**

- A. 内连接
- B. 左连接
- C. 右连接
- D. 外连接

**11. 以下哪个方法用于检测重复行？**

- A. `df.isnull()`
- B. `df.duplicated()`
- C. `df.unique()`
- D. `df.isna()`

**12. 使用 IQR 方法检测异常值时，下边界的计算公式是？**

- A. `Q1 - 1.5 * IQR`
- B. `Q1 - 2 * IQR`
- C. `Q3 - 1.5 * IQR`
- D. `mean - 2 * std`

### Matplotlib 部分

**13. 以下哪个函数用于创建多个子图？**

- A. `plt.plot()`
- B. `plt.subplot()`
- C. `plt.subplots()`
- D. B 和 C 都可以

**14. 要在同一图表中显示两个不同刻度的 Y 轴，应使用？**

- A. `plt.subplot()`
- B. `ax.twinx()`
- C. `plt.subplots()`
- D. `GridSpec`

**15. 以下哪种图表最适合展示数据分布？**

- A. 折线图
- B. 散点图
- C. 直方图
- D. 饼图

---

## 二、填空题（每题 3 分，共 24 分）

**16.** NumPy 中，使用 `________` 函数可以创建一个 5x5 的单位矩阵（对角线为 1，其他为 0）。

**17.** 对于数组 `arr = np.array([1, 2, 3, 4, 5])`，`arr[arr > 3]` 的结果是 `________`。

**18.** Pandas 中，使用 `df.________()` 方法可以查看数据的统计摘要，包括均值、标准差等。

**19.** 要将字符串列 "100,000" 转换为数值，首先需要使用 `str.________(',', '')` 去除逗号。

**20.** 在 Matplotlib 中，设置中文字体的代码是 `plt.rcParams['________'] = ['Arial Unicode MS']`。

**21.** `np.sum(arr, axis=0)` 表示对数组按 **\_\_\_\_** 方向求和（行/列）。

**22.** SVD 分解的公式是 `A = U @ ________ @ V^T`。

**23.** 计算向量 `v = [3, 4]` 的 L2 范数，结果是 **\_\_\_\_**。

---

## 三、编程题（每题 6 分，共 46 分）

**24. NumPy 数组操作**

创建一个 4x4 的随机整数矩阵（范围 1-10），然后：

- (a) 提取对角线元素
- (b) 计算每行的和
- (c) 找出所有大于 5 的元素

```python
# 请在此编写代码
import numpy as np

# 你的代码...
```

---

**25. 实现 Softmax 函数**

实现一个数值稳定的 Softmax 函数，并用 `x = [1, 2, 3, 4, 5]` 测试。

```python
# 请在此编写代码
import numpy as np

def softmax(x):
    # 你的代码...
    pass

# 测试
x = np.array([1, 2, 3, 4, 5])
print(softmax(x))
print(f"验证和为1: {np.sum(softmax(x))}")
```

---

**26. Pandas 数据处理**

给定以下数据，完成处理：

```python
import pandas as pd
import numpy as np

data = {
    'name': ['Alice', 'Bob', None, 'David', 'Eva'],
    'age': [25, np.nan, 35, 28, np.nan],
    'salary': [10000, 15000, np.nan, 12000, 18000],
    'department': ['技术', '销售', '技术', None, '销售']
}
df = pd.DataFrame(data)

# (a) 用均值填充 age 列的缺失值
# (b) 用众数填充 department 列的缺失值
# (c) 删除 name 为空的行
# 你的代码...
```

---

**27. 解线性方程组**

使用 NumPy 解以下线性方程组：

```
x + 2y + 3z = 14
2x + y + z = 7
3x + 2y + z = 10
```

```python
# 请在此编写代码
import numpy as np

# 你的代码...
```

---

**28. 数据可视化**

使用 Matplotlib 绑制一个包含以下元素的图表：

- 正弦函数和余弦函数（使用不同颜色和线型）
- 图例、标题、网格线
- 保存为 PNG 文件

```python
# 请在此编写代码
import matplotlib.pyplot as plt
import numpy as np

# 你的代码...
```

---

**29. PCA 降维（简化版）**

给定以下 2D 数据，计算其协方差矩阵并进行特征分解：

```python
import numpy as np

# 生成数据
np.random.seed(42)
data = np.random.randn(100, 2) @ np.array([[2, 1], [1, 1.5]])

# (a) 中心化数据
# (b) 计算协方差矩阵
# (c) 进行特征分解
# (d) 输出主成分方向（最大特征值对应的特征向量）

# 你的代码...
```

---

**30. 综合数据分析**

给定一个销售数据，完成以下分析：

```python
import pandas as pd
import numpy as np

# 创建数据
np.random.seed(42)
sales = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['北区', '南区', '东区'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'quantity': np.random.randint(1, 50, 100)
})

# (a) 按产品分组，计算平均销售额
# (b) 按地区和产品创建透视表，统计总销售额
# (c) 找出销售额排名前5的记录

# 你的代码...
```

---

# 参考答案

## 一、选择题答案

| 题号 | 答案 | 解析                                              |
| ---- | ---- | ------------------------------------------------- |
| 1    | B    | `np.arange(10)` 生成 0-9 共 10 个数的等差数组     |
| 2    | C    | 2 行 3 列的数组，shape 为 (2, 3)                  |
| 3    | B    | `@` 是矩阵乘法运算符，`*` 是元素级乘法            |
| 4    | B    | (2,3) 和 (3,) 可广播为 (2,3) + (1,3) → (2,3)      |
| 5    | C    | `np.linalg.inv()` 计算逆矩阵                      |
| 6    | C    | 向量化操作通常快 100-1000 倍                      |
| 7    | B    | `head()` 默认显示前 5 行                          |
| 8    | B    | `loc` 基于标签包含结束位置，`iloc` 基于位置不包含 |
| 9    | C    | `dropna()` 是删除缺失值，不是填充                 |
| 10   | B    | `how='left'` 表示左连接                           |
| 11   | B    | `duplicated()` 用于检测重复行                     |
| 12   | A    | IQR 方法下边界 = Q1 - 1.5 × IQR                   |
| 13   | D    | `subplot()` 和 `subplots()` 都可以创建多子图      |
| 14   | B    | `twinx()` 创建共享 X 轴的第二个 Y 轴              |
| 15   | C    | 直方图最适合展示数据分布                          |

---

## 二、填空题答案

| 题号 | 答案                            |
| ---- | ------------------------------- |
| 16   | `np.eye(5)` 或 `np.identity(5)` |
| 17   | `[4, 5]`                        |
| 18   | `describe`                      |
| 19   | `replace`                       |
| 20   | `font.sans-serif`               |
| 21   | 列 (沿行方向，结果为每列的和)   |
| 22   | `Σ` (对角矩阵/Sigma)            |
| 23   | `5` (√(3² + 4²) = 5)            |

---

## 三、编程题答案

### 24. NumPy 数组操作

```python
import numpy as np

np.random.seed(42)
matrix = np.random.randint(1, 11, (4, 4))
print(f"原矩阵:\n{matrix}")

# (a) 提取对角线元素
diagonal = np.diag(matrix)
print(f"对角线元素: {diagonal}")

# (b) 计算每行的和
row_sums = np.sum(matrix, axis=1)
print(f"每行的和: {row_sums}")

# (c) 找出所有大于5的元素
greater_than_5 = matrix[matrix > 5]
print(f"大于5的元素: {greater_than_5}")
```

---

### 25. 实现 Softmax 函数

```python
import numpy as np

def softmax(x):
    # 数值稳定性：减去最大值防止 exp 溢出
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

# 测试
x = np.array([1, 2, 3, 4, 5])
result = softmax(x)
print(f"softmax([1,2,3,4,5]) = {result}")
print(f"验证和为1: {np.sum(result):.6f}")
# 输出: [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]
```

---

### 26. Pandas 数据处理

```python
import pandas as pd
import numpy as np

data = {
    'name': ['Alice', 'Bob', None, 'David', 'Eva'],
    'age': [25, np.nan, 35, 28, np.nan],
    'salary': [10000, 15000, np.nan, 12000, 18000],
    'department': ['技术', '销售', '技术', None, '销售']
}
df = pd.DataFrame(data)

# (a) 用均值填充 age 列的缺失值
df['age'] = df['age'].fillna(df['age'].mean())

# (b) 用众数填充 department 列的缺失值
df['department'] = df['department'].fillna(df['department'].mode()[0])

# (c) 删除 name 为空的行
df = df.dropna(subset=['name'])

print(df)
```

---

### 27. 解线性方程组

```python
import numpy as np

# x + 2y + 3z = 14
# 2x + y + z = 7
# 3x + 2y + z = 10

A = np.array([[1, 2, 3],
              [2, 1, 1],
              [3, 2, 1]])
b = np.array([14, 7, 10])

x = np.linalg.solve(A, b)
print(f"解: x={x[0]:.2f}, y={x[1]:.2f}, z={x[2]:.2f}")
print(f"验证 A @ x = {A @ x}")
# 解: x=1.00, y=2.00, z=3.00
```

---

### 28. 数据可视化

```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(0, 2 * np.pi, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
plt.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦和余弦函数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sin_cos_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 sin_cos_plot.png")
```

---

### 29. PCA 降维（简化版）

```python
import numpy as np

np.random.seed(42)
data = np.random.randn(100, 2) @ np.array([[2, 1], [1, 1.5]])

# (a) 中心化数据
data_centered = data - np.mean(data, axis=0)

# (b) 计算协方差矩阵
cov_matrix = np.cov(data_centered.T)
print(f"协方差矩阵:\n{cov_matrix}")

# (c) 进行特征分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# (d) 主成分方向
idx = np.argmax(eigenvalues)
principal_component = eigenvectors[:, idx]
print(f"主成分方向: {principal_component}")
print(f"方差解释比例: {eigenvalues[idx] / np.sum(eigenvalues):.2%}")
```

---

### 30. 综合数据分析

```python
import pandas as pd
import numpy as np

np.random.seed(42)
sales = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['北区', '南区', '东区'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'quantity': np.random.randint(1, 50, 100)
})

# (a) 按产品分组，计算平均销售额
avg_by_product = sales.groupby('product')['sales'].mean()
print("按产品平均销售额:")
print(avg_by_product)

# (b) 按地区和产品创建透视表
pivot = pd.pivot_table(
    sales,
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum'
)
print("\n透视表（地区 × 产品 总销售额）:")
print(pivot)

# (c) 找出销售额排名前5的记录
top5 = sales.nlargest(5, 'sales')
print("\n销售额前5:")
print(top5)
```

---

## 📊 评分标准

| 部分     | 题数      | 满分       |
| -------- | --------- | ---------- |
| 选择题   | 15 题     | 30 分      |
| 填空题   | 8 题      | 24 分      |
| 编程题   | 7 题      | 46 分      |
| **总分** | **30 题** | **100 分** |

### 成绩等级

- **90-100 分**：优秀，可以进入 Phase 2
- **75-89 分**：良好，建议复习薄弱点后进入 Phase 2
- **60-74 分**：及格，需要重点复习后再进入 Phase 2
- **60 分以下**：需要重新学习 Phase 1 相关内容
