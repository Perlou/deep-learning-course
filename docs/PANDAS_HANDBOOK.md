# Pandas 常用方法和函数速查手册

> 📌 版本：适用于 Pandas 1.x / 2.x  
> 💡 使用前请先导入：`import pandas as pd`

---

## 目录

1. [数据结构创建](#一数据结构创建)
2. [数据读取与写入](#二数据读取与写入)
3. [数据查看与基本信息](#三数据查看与基本信息)
4. [数据选择与索引](#四数据选择与索引)
5. [数据清洗](#五数据清洗)
6. [数据操作与转换](#六数据操作与转换)
7. [统计函数](#七统计函数)
8. [分组与聚合](#八分组与聚合)
9. [数据合并与连接](#九数据合并与连接)
10. [时间序列](#十时间序列)
11. [字符串处理](#十一字符串处理)
12. [数据透视表](#十二数据透视表)
13. [窗口函数](#十三窗口函数)
14. [绘图](#十四绑图)
15. [其他实用方法](#十五其他实用方法)

---

## 一、数据结构创建

### 1.1 Series 创建

| 方法                     | 说明       | 示例                                          |
| ------------------------ | ---------- | --------------------------------------------- |
| `pd.Series(data)`        | 从列表创建 | `pd.Series([1, 2, 3])`                        |
| `pd.Series(data, index)` | 指定索引   | `pd.Series([1, 2, 3], index=['a', 'b', 'c'])` |
| `pd.Series(dict)`        | 从字典创建 | `pd.Series({'a': 1, 'b': 2})`                 |

### 1.2 DataFrame 创建

| 方法                    | 说明              | 示例                                                 |
| ----------------------- | ----------------- | ---------------------------------------------------- |
| `pd.DataFrame(dict)`    | 从字典创建        | `pd.DataFrame({'A': [1, 2], 'B': [3, 4]})`           |
| `pd.DataFrame(list)`    | 从列表创建        | `pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])` |
| `pd.DataFrame(ndarray)` | 从 NumPy 数组创建 | `pd.DataFrame(np.array([[1, 2], [3, 4]]))`           |

```python
# 创建示例
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Beijing', 'Shanghai', 'Guangzhou']
})
```

---

## 二、数据读取与写入

### 2.1 读取数据

| 方法                  | 说明              | 常用参数                                                                                 |
| --------------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `pd.read_csv()`       | 读取 CSV 文件     | `filepath, sep, header, index_col, usecols, dtype, nrows, skiprows, encoding, na_values` |
| `pd.read_excel()`     | 读取 Excel 文件   | `filepath, sheet_name, header, index_col, usecols`                                       |
| `pd.read_json()`      | 读取 JSON 文件    | `filepath, orient, lines`                                                                |
| `pd.read_sql()`       | 读取 SQL 查询     | `sql, con, index_col, columns`                                                           |
| `pd.read_html()`      | 读取 HTML 表格    | `url, header, index_col`                                                                 |
| `pd.read_parquet()`   | 读取 Parquet 文件 | `filepath, columns`                                                                      |
| `pd.read_pickle()`    | 读取 Pickle 文件  | `filepath`                                                                               |
| `pd.read_clipboard()` | 读取剪贴板        | `sep`                                                                                    |

```python
# 常用读取示例
df = pd.read_csv('data.csv',
                 encoding='utf-8',
                 sep=',',
                 header=0,
                 index_col=None,
                 usecols=['col1', 'col2'],
                 dtype={'col1': str, 'col2': int},
                 nrows=1000,
                 skiprows=[1, 2],
                 na_values=['NA', 'NULL', ''])

df = pd.read_excel('data.xlsx',
                   sheet_name='Sheet1',  # 或 sheet_name=0
                   header=0,
                   usecols='A:C')
```

### 2.2 写入数据

| 方法                | 说明               | 常用参数                                          |
| ------------------- | ------------------ | ------------------------------------------------- |
| `df.to_csv()`       | 写入 CSV           | `filepath, sep, index, header, columns, encoding` |
| `df.to_excel()`     | 写入 Excel         | `filepath, sheet_name, index, header`             |
| `df.to_json()`      | 写入 JSON          | `filepath, orient, lines`                         |
| `df.to_sql()`       | 写入数据库         | `name, con, if_exists, index`                     |
| `df.to_parquet()`   | 写入 Parquet       | `filepath, compression`                           |
| `df.to_pickle()`    | 写入 Pickle        | `filepath`                                        |
| `df.to_clipboard()` | 写入剪贴板         | `sep`                                             |
| `df.to_markdown()`  | 输出 Markdown 表格 | `index`                                           |
| `df.to_dict()`      | 转为字典           | `orient`                                          |
| `df.to_numpy()`     | 转为 NumPy 数组    | -                                                 |

```python
# 常用写入示例
df.to_csv('output.csv', index=False, encoding='utf-8-sig')  # utf-8-sig 解决Excel中文乱码
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
df.to_json('output.json', orient='records', lines=True)
```

---

## 三、数据查看与基本信息

### 3.1 数据预览

| 方法              | 说明                  | 示例                  |
| ----------------- | --------------------- | --------------------- |
| `df.head(n)`      | 查看前 n 行（默认 5） | `df.head(10)`         |
| `df.tail(n)`      | 查看后 n 行（默认 5） | `df.tail(10)`         |
| `df.sample(n)`    | 随机抽取 n 行         | `df.sample(5)`        |
| `df.sample(frac)` | 随机抽取比例          | `df.sample(frac=0.1)` |

### 3.2 基本信息

| 属性/方法                    | 说明              | 返回类型    |
| ---------------------------- | ----------------- | ----------- |
| `df.shape`                   | 形状 (行数, 列数) | tuple       |
| `df.columns`                 | 列名              | Index       |
| `df.index`                   | 索引              | Index       |
| `df.dtypes`                  | 每列数据类型      | Series      |
| `df.info()`                  | 整体信息概览      | None (打印) |
| `df.describe()`              | 数值列统计摘要    | DataFrame   |
| `df.describe(include='all')` | 所有列统计摘要    | DataFrame   |
| `df.memory_usage()`          | 内存使用          | Series      |
| `df.ndim`                    | 维度数            | int         |
| `df.size`                    | 元素总数          | int         |
| `len(df)`                    | 行数              | int         |

```python
# describe() 输出说明
>>> df.describe()
              age       salary
count    100.000    100.000     # 非空数量
mean      35.500   50000.000    # 均值
std       10.200   15000.000    # 标准差
min       22.000   25000.000    # 最小值
25%       28.000   40000.000    # 25%分位数
50%       35.000   50000.000    # 中位数
75%       42.000   60000.000    # 75%分位数
max       55.000   90000.000    # 最大值
```

### 3.3 数据类型

| Pandas 类型   | Python 类型 | 说明     |
| ------------- | ----------- | -------- |
| `object`      | str         | 字符串   |
| `int64`       | int         | 整数     |
| `float64`     | float       | 浮点数   |
| `bool`        | bool        | 布尔值   |
| `datetime64`  | datetime    | 日期时间 |
| `timedelta64` | timedelta   | 时间差   |
| `category`    | -           | 分类类型 |

---

## 四、数据选择与索引

### 4.1 列选择

| 方法                                | 说明                 | 返回类型  |
| ----------------------------------- | -------------------- | --------- |
| `df['col']`                         | 选择单列             | Series    |
| `df.col`                            | 选择单列（属性方式） | Series    |
| `df[['col1', 'col2']]`              | 选择多列             | DataFrame |
| `df.filter(items=['col1', 'col2'])` | 按名称筛选列         | DataFrame |
| `df.filter(regex='pattern')`        | 按正则筛选列         | DataFrame |
| `df.filter(like='str')`             | 按包含字符筛选列     | DataFrame |

```python
# 示例
df['name']                    # 单列 → Series
df[['name', 'age']]          # 多列 → DataFrame
df.filter(regex='^col_')     # 选择以 col_ 开头的列
df.filter(like='date')       # 选择包含 date 的列
```

### 4.2 行选择

| 方法            | 说明       | 示例                                         |
| --------------- | ---------- | -------------------------------------------- |
| `df[start:end]` | 切片选择   | `df[0:5]`                                    |
| `df[condition]` | 条件筛选   | `df[df['age'] > 30]`                         |
| `df.query()`    | 查询表达式 | `df.query('age > 30 and city == "Beijing"')` |

### 4.3 loc 和 iloc

| 方法        | 索引类型               | 语法                           |
| ----------- | ---------------------- | ------------------------------ |
| `df.loc[]`  | 标签索引               | `df.loc[row_label, col_label]` |
| `df.iloc[]` | 位置索引               | `df.iloc[row_pos, col_pos]`    |
| `df.at[]`   | 标签索引（单值，更快） | `df.at[row_label, col_label]`  |
| `df.iat[]`  | 位置索引（单值，更快） | `df.iat[row_pos, col_pos]`     |

```python
# loc 用法（基于标签）
df.loc[0]                         # 第0行（标签为0）
df.loc[0, 'name']                 # 单个值
df.loc[0:2, 'name']               # 行0-2的name列（包含2）
df.loc[0:2, ['name', 'age']]      # 行0-2，多列
df.loc[:, 'name':'age']           # 所有行，name到age列
df.loc[df['age'] > 30]            # 条件筛选
df.loc[df['age'] > 30, 'name']    # 条件筛选 + 列选择

# iloc 用法（基于位置）
df.iloc[0]                        # 第0行
df.iloc[0, 1]                     # 第0行第1列
df.iloc[0:3, 0:2]                 # 切片（不包含3和2）
df.iloc[[0, 2, 4], [1, 3]]        # 指定行列位置
df.iloc[:, 1:]                    # 所有行，第1列之后

# at/iat 用法（单值访问，更快）
df.at[0, 'name']                  # 标签方式获取单值
df.iat[0, 1]                      # 位置方式获取单值
```

### 4.4 条件筛选

| 方法           | 说明             | 示例                                           |
| -------------- | ---------------- | ---------------------------------------------- | ----------------------------------------- |
| `==, !=`       | 等于/不等于      | `df[df['city'] == 'Beijing']`                  |
| `>, <, >=, <=` | 比较运算         | `df[df['age'] >= 30]`                          |
| `&,            | , ~`             | 与/或/非                                       | `df[(df['age'] > 20) & (df['age'] < 40)]` |
| `df.isin()`    | 包含在列表中     | `df[df['city'].isin(['Beijing', 'Shanghai'])]` |
| `df.between()` | 范围内           | `df[df['age'].between(20, 40)]`                |
| `df.query()`   | 查询表达式       | `df.query('age > 30 & city == "Beijing"')`     |
| `df.where()`   | 条件替换         | `df.where(df > 0, 0)`                          |
| `df.mask()`    | 条件替换（反向） | `df.mask(df < 0, 0)`                           |

```python
# 多条件筛选示例
df[(df['age'] > 25) & (df['city'] == 'Beijing')]
df[(df['age'] < 20) | (df['age'] > 50)]
df[~df['city'].isin(['Beijing', 'Shanghai'])]

# query 方法（更简洁）
df.query('age > 25 and city == "Beijing"')
df.query('age > @min_age')  # 使用外部变量用 @
```

---

## 五、数据清洗

### 5.1 缺失值处理

| 方法                | 说明                   | 示例                |
| ------------------- | ---------------------- | ------------------- |
| `df.isnull()`       | 检测缺失值（返回布尔） | `df.isnull()`       |
| `df.isna()`         | 同 isnull()            | `df.isna()`         |
| `df.notnull()`      | 检测非缺失值           | `df.notnull()`      |
| `df.notna()`        | 同 notnull()           | `df.notna()`        |
| `df.isnull().sum()` | 每列缺失值计数         | `df.isnull().sum()` |
| `df.isnull().any()` | 是否有缺失值           | `df.isnull().any()` |
| `df.dropna()`       | 删除缺失值             | 见下表              |
| `df.fillna()`       | 填充缺失值             | 见下表              |

#### dropna 参数

| 参数      | 说明                           | 示例                                 |
| --------- | ------------------------------ | ------------------------------------ |
| `axis`    | 0=行, 1=列                     | `df.dropna(axis=0)`                  |
| `how`     | 'any'=任一缺失, 'all'=全部缺失 | `df.dropna(how='all')`               |
| `subset`  | 指定检查的列                   | `df.dropna(subset=['col1', 'col2'])` |
| `thresh`  | 非空值最少数量                 | `df.dropna(thresh=3)`                |
| `inplace` | 是否原地修改                   | `df.dropna(inplace=True)`            |

#### fillna 参数

| 参数      | 说明                     | 示例                                 |
| --------- | ------------------------ | ------------------------------------ |
| `value`   | 填充值                   | `df.fillna(0)`                       |
| `method`  | 填充方式 'ffill'/'bfill' | `df.fillna(method='ffill')`          |
| `axis`    | 填充方向                 | `df.fillna(method='ffill', axis=1)`  |
| `limit`   | 最大填充数量             | `df.fillna(method='ffill', limit=2)` |
| `inplace` | 是否原地修改             | `df.fillna(0, inplace=True)`         |

```python
# 缺失值处理示例
df.dropna()                              # 删除有缺失的行
df.dropna(axis=1)                        # 删除有缺失的列
df.dropna(subset=['col1', 'col2'])       # 指定列有缺失才删除

df.fillna(0)                             # 用0填充
df.fillna({'col1': 0, 'col2': 'Unknown'}) # 不同列用不同值
df.fillna(df.mean())                     # 用均值填充
df.fillna(df.median())                   # 用中位数填充
df.fillna(df.mode().iloc[0])             # 用众数填充
df.fillna(method='ffill')                # 前向填充
df.fillna(method='bfill')                # 后向填充
df.interpolate()                         # 插值填充
df.interpolate(method='linear')          # 线性插值
```

### 5.2 重复值处理

| 方法                         | 说明         | 示例                                  |
| ---------------------------- | ------------ | ------------------------------------- |
| `df.duplicated()`            | 检测重复行   | `df.duplicated()`                     |
| `df.duplicated(subset)`      | 按指定列检测 | `df.duplicated(subset=['col1'])`      |
| `df.duplicated(keep)`        | 保留方式     | `df.duplicated(keep='first')`         |
| `df.drop_duplicates()`       | 删除重复行   | `df.drop_duplicates()`                |
| `df.drop_duplicates(subset)` | 按指定列去重 | `df.drop_duplicates(subset=['col1'])` |

```python
# keep 参数：'first'(保留第一个), 'last'(保留最后一个), False(全不保留)
df.duplicated(keep='first')              # 第一个标记为False
df.drop_duplicates(keep='last')          # 保留最后一个
df.drop_duplicates(subset=['name', 'age'], keep='first')
```

### 5.3 数据类型转换

| 方法                | 说明     | 示例                                        |
| ------------------- | -------- | ------------------------------------------- |
| `df.astype()`       | 类型转换 | `df['col'].astype(int)`                     |
| `pd.to_numeric()`   | 转数值   | `pd.to_numeric(df['col'], errors='coerce')` |
| `pd.to_datetime()`  | 转日期   | `pd.to_datetime(df['col'])`                 |
| `pd.to_timedelta()` | 转时间差 | `pd.to_timedelta(df['col'])`                |

```python
# 类型转换示例
df['age'] = df['age'].astype(int)
df['price'] = df['price'].astype(float)
df['name'] = df['name'].astype(str)
df['category'] = df['category'].astype('category')

# errors参数：'raise'(报错), 'coerce'(无效值设为NaN), 'ignore'(忽略)
df['num'] = pd.to_numeric(df['num'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
```

### 5.4 数据替换

| 方法                | 说明     | 示例                                 |
| ------------------- | -------- | ------------------------------------ |
| `df.replace()`      | 替换值   | `df.replace(old, new)`               |
| `df.replace(dict)`  | 多值替换 | `df.replace({0: 'zero', 1: 'one'})`  |
| `df.replace(regex)` | 正则替换 | `df.replace(regex=r'^A', value='B')` |

```python
# 替换示例
df.replace(0, np.nan)                    # 单值替换
df.replace([0, 1, 2], [10, 20, 30])       # 列表替换
df.replace({'col1': {0: 100}})           # 指定列替换
df.replace(regex=r'\s+', value='')       # 正则替换空白
```

---

## 六、数据操作与转换

### 6.1 列操作

| 操作   | 说明        | 示例                                |
| ------ | ----------- | ----------------------------------- |
| 新增列 | 直接赋值    | `df['new_col'] = values`            |
| 删除列 | drop 方法   | `df.drop('col', axis=1)`            |
| 删除列 | del 语句    | `del df['col']`                     |
| 删除列 | pop 方法    | `df.pop('col')`                     |
| 重命名 | rename 方法 | `df.rename(columns={'old': 'new'})` |
| 选择列 | filter 方法 | `df.filter(items=['col1', 'col2'])` |

```python
# 新增列
df['total'] = df['price'] * df['quantity']
df['category'] = 'A'                     # 常量列
df['rank'] = df['score'].rank()          # 排名列

# 使用 assign（链式操作）
df = df.assign(
    total = df['price'] * df['quantity'],
    discount = lambda x: x['total'] * 0.1
)

# 删除列
df.drop('col', axis=1, inplace=True)
df.drop(['col1', 'col2'], axis=1, inplace=True)

# 重命名
df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']    # 直接修改所有列名
```

### 6.2 行操作

| 操作     | 说明        | 示例                         |
| -------- | ----------- | ---------------------------- |
| 新增行   | loc 方法    | `df.loc[new_index] = values` |
| 新增行   | concat 方法 | `pd.concat([df, new_row])`   |
| 删除行   | drop 方法   | `df.drop(index, axis=0)`     |
| 重置索引 | reset_index | `df.reset_index()`           |
| 设置索引 | set_index   | `df.set_index('col')`        |

```python
# 新增行
df.loc[len(df)] = [val1, val2, val3]
new_row = pd.DataFrame({'col1': [v1], 'col2': [v2]})
df = pd.concat([df, new_row], ignore_index=True)

# 删除行
df.drop(0, axis=0, inplace=True)         # 删除索引为0的行
df.drop([0, 1, 2], axis=0, inplace=True) # 删除多行

# 索引操作
df.reset_index(drop=True, inplace=True)  # 重置索引，丢弃原索引
df.set_index('id', inplace=True)         # 设置某列为索引
```

### 6.3 排序

| 方法                              | 说明        | 示例                                     |
| --------------------------------- | ----------- | ---------------------------------------- |
| `df.sort_values()`                | 按值排序    | `df.sort_values('col')`                  |
| `df.sort_values(ascending=False)` | 降序        | `df.sort_values('col', ascending=False)` |
| `df.sort_values(by=[])`           | 多列排序    | `df.sort_values(by=['col1', 'col2'])`    |
| `df.sort_index()`                 | 按索引排序  | `df.sort_index()`                        |
| `df.nlargest()`                   | 最大的 n 行 | `df.nlargest(10, 'col')`                 |
| `df.nsmallest()`                  | 最小的 n 行 | `df.nsmallest(10, 'col')`                |

```python
# 排序示例
df.sort_values('age')                    # 按age升序
df.sort_values('age', ascending=False)   # 按age降序
df.sort_values(['city', 'age'], ascending=[True, False])  # 多列排序
df.sort_index(ascending=False)           # 按索引降序
df.nlargest(5, 'salary')                 # 薪资最高的5行
```

### 6.4 Apply 函数

| 方法                | 说明               | 示例                     |
| ------------------- | ------------------ | ------------------------ |
| `df.apply()`        | 应用函数到行/列    | `df.apply(func, axis=0)` |
| `df['col'].apply()` | 应用函数到 Series  | `df['col'].apply(func)`  |
| `df.applymap()`     | 应用函数到每个元素 | `df.applymap(func)`      |
| `df['col'].map()`   | 映射替换           | `df['col'].map(dict)`    |
| `df.transform()`    | 变换（保持形状）   | `df.transform(func)`     |

```python
# apply 示例
df['age'].apply(lambda x: x * 2)         # Series上应用
df['age'].apply(lambda x: 'adult' if x >= 18 else 'child')

df.apply(lambda row: row['a'] + row['b'], axis=1)  # 按行应用
df.apply(np.sum, axis=0)                  # 按列求和

# map 示例（用于映射）
mapping = {'Beijing': 'BJ', 'Shanghai': 'SH'}
df['city_code'] = df['city'].map(mapping)

# applymap 示例（应用到每个元素）
df.applymap(lambda x: len(str(x)))       # 每个元素的字符串长度

# transform 示例
df.groupby('city')['salary'].transform('mean')  # 组内均值填充
```

### 6.5 数据变形

| 方法               | 说明       | 示例                               |
| ------------------ | ---------- | ---------------------------------- |
| `df.T`             | 转置       | `df.T`                             |
| `df.melt()`        | 宽表转长表 | `df.melt(id_vars, value_vars)`     |
| `df.pivot()`       | 长表转宽表 | `df.pivot(index, columns, values)` |
| `df.pivot_table()` | 透视表     | 见透视表章节                       |
| `df.stack()`       | 列转行     | `df.stack()`                       |
| `df.unstack()`     | 行转列     | `df.unstack()`                     |
| `df.explode()`     | 展开列表列 | `df.explode('col')`                |

```python
# melt 示例（宽转长）
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'math': [90, 85],
    'english': [88, 92]
})
df.melt(id_vars=['name'], value_vars=['math', 'english'],
        var_name='subject', value_name='score')
# 结果：
#     name  subject  score
# 0  Alice     math     90
# 1    Bob     math     85
# 2  Alice  english     88
# 3    Bob  english     92

# pivot 示例（长转宽）
df.pivot(index='name', columns='subject', values='score')

# explode 示例（展开列表）
df = pd.DataFrame({'A': [[1, 2], [3, 4]]})
df.explode('A')
# 结果：
#    A
# 0  1
# 0  2
# 1  3
# 1  4
```

---

## 七、统计函数

### 7.1 描述性统计

| 方法          | 说明       | 示例                 |
| ------------- | ---------- | -------------------- |
| `df.count()`  | 非空值计数 | `df.count()`         |
| `df.sum()`    | 求和       | `df['col'].sum()`    |
| `df.mean()`   | 均值       | `df['col'].mean()`   |
| `df.median()` | 中位数     | `df['col'].median()` |
| `df.mode()`   | 众数       | `df['col'].mode()`   |
| `df.std()`    | 标准差     | `df['col'].std()`    |
| `df.var()`    | 方差       | `df['col'].var()`    |
| `df.min()`    | 最小值     | `df['col'].min()`    |
| `df.max()`    | 最大值     | `df['col'].max()`    |
| `df.abs()`    | 绝对值     | `df['col'].abs()`    |
| `df.prod()`   | 乘积       | `df['col'].prod()`   |
| `df.sem()`    | 标准误差   | `df['col'].sem()`    |
| `df.skew()`   | 偏度       | `df['col'].skew()`   |
| `df.kurt()`   | 峰度       | `df['col'].kurt()`   |

### 7.2 位置统计

| 方法                             | 说明        | 示例                                    |
| -------------------------------- | ----------- | --------------------------------------- |
| `df.quantile()`                  | 分位数      | `df['col'].quantile(0.75)`              |
| `df.quantile([0.25, 0.5, 0.75])` | 多分位数    | `df['col'].quantile([0.25, 0.5, 0.75])` |
| `df.idxmax()`                    | 最大值索引  | `df['col'].idxmax()`                    |
| `df.idxmin()`                    | 最小值索引  | `df['col'].idxmin()`                    |
| `df.nlargest()`                  | 最大 n 个值 | `df['col'].nlargest(5)`                 |
| `df.nsmallest()`                 | 最小 n 个值 | `df['col'].nsmallest(5)`                |

### 7.3 累计统计

| 方法            | 说明       | 示例                           |
| --------------- | ---------- | ------------------------------ |
| `df.cumsum()`   | 累计和     | `df['col'].cumsum()`           |
| `df.cumprod()`  | 累计积     | `df['col'].cumprod()`          |
| `df.cummax()`   | 累计最大值 | `df['col'].cummax()`           |
| `df.cummin()`   | 累计最小值 | `df['col'].cummin()`           |
| `df.cumcount()` | 累计计数   | `df.groupby('col').cumcount()` |

### 7.4 差分与变化

| 方法              | 说明       | 示例                     |
| ----------------- | ---------- | ------------------------ |
| `df.diff()`       | 差分       | `df['col'].diff()`       |
| `df.pct_change()` | 百分比变化 | `df['col'].pct_change()` |
| `df.shift()`      | 移动数据   | `df['col'].shift(1)`     |

### 7.5 相关性

| 方法                    | 说明               | 示例                    |
| ----------------------- | ------------------ | ----------------------- |
| `df.corr()`             | 相关系数矩阵       | `df.corr()`             |
| `df['A'].corr(df['B'])` | 两列相关系数       | `df['A'].corr(df['B'])` |
| `df.cov()`              | 协方差矩阵         | `df.cov()`              |
| `df.corrwith()`         | 与另一对象的相关性 | `df.corrwith(other)`    |

### 7.6 唯一值与频率

| 方法                                     | 说明       | 示例                                      |
| ---------------------------------------- | ---------- | ----------------------------------------- |
| `df['col'].unique()`                     | 唯一值数组 | `df['city'].unique()`                     |
| `df['col'].nunique()`                    | 唯一值数量 | `df['city'].nunique()`                    |
| `df['col'].value_counts()`               | 值频率统计 | `df['city'].value_counts()`               |
| `df['col'].value_counts(normalize=True)` | 频率占比   | `df['city'].value_counts(normalize=True)` |

```python
# value_counts 示例
>>> df['city'].value_counts()
Beijing     50
Shanghai    40
Guangzhou   30
Name: city, dtype: int64

>>> df['city'].value_counts(normalize=True)
Beijing     0.416667
Shanghai    0.333333
Guangzhou   0.250000
Name: city, dtype: float64
```

---

## 八、分组与聚合

### 8.1 GroupBy 基础

| 方法                               | 说明         | 示例                                      |
| ---------------------------------- | ------------ | ----------------------------------------- |
| `df.groupby('col')`                | 按单列分组   | `df.groupby('city')`                      |
| `df.groupby(['col1', 'col2'])`     | 按多列分组   | `df.groupby(['city', 'gender'])`          |
| `df.groupby('col').groups`         | 查看分组情况 | `df.groupby('city').groups`               |
| `df.groupby('col').get_group('x')` | 获取某分组   | `df.groupby('city').get_group('Beijing')` |

### 8.2 聚合函数

| 方法        | 说明       | 示例                                    |
| ----------- | ---------- | --------------------------------------- |
| `.count()`  | 计数       | `df.groupby('city')['age'].count()`     |
| `.sum()`    | 求和       | `df.groupby('city')['salary'].sum()`    |
| `.mean()`   | 均值       | `df.groupby('city')['salary'].mean()`   |
| `.median()` | 中位数     | `df.groupby('city')['salary'].median()` |
| `.std()`    | 标准差     | `df.groupby('city')['salary'].std()`    |
| `.var()`    | 方差       | `df.groupby('city')['salary'].var()`    |
| `.min()`    | 最小值     | `df.groupby('city')['salary'].min()`    |
| `.max()`    | 最大值     | `df.groupby('city')['salary'].max()`    |
| `.first()`  | 第一个值   | `df.groupby('city')['name'].first()`    |
| `.last()`   | 最后一个值 | `df.groupby('city')['name'].last()`     |
| `.size()`   | 组大小     | `df.groupby('city').size()`             |

### 8.3 agg 方法

```python
# 单列多聚合
df.groupby('city')['salary'].agg(['mean', 'max', 'min', 'count'])

# 多列不同聚合
df.groupby('city').agg({
    'salary': ['mean', 'sum'],
    'age': ['min', 'max'],
    'name': 'count'
})

# 自定义聚合函数
df.groupby('city')['salary'].agg(lambda x: x.max() - x.min())

# 命名聚合（Pandas 0.25+）
df.groupby('city').agg(
    avg_salary=('salary', 'mean'),
    max_age=('age', 'max'),
    count=('name', 'count')
)
```

### 8.4 Transform 与 Filter

```python
# transform：返回与原DataFrame同样大小的结果
df['salary_mean'] = df.groupby('city')['salary'].transform('mean')
df['salary_rank'] = df.groupby('city')['salary'].transform('rank')

# filter：过滤分组
df.groupby('city').filter(lambda x: x['salary'].mean() > 50000)  # 保留均薪>50000的城市
df.groupby('city').filter(lambda x: len(x) >= 10)  # 保留人数>=10的城市
```

### 8.5 Apply

```python
# apply：灵活应用函数
def top_n(group, n=3):
    return group.nlargest(n, 'salary')

df.groupby('city').apply(top_n)  # 每个城市薪资前3的人

# 多列操作
df.groupby('city').apply(lambda x: pd.Series({
    'avg_salary': x['salary'].mean(),
    'age_range': x['age'].max() - x['age'].min()
}))
```

---

## 九、数据合并与连接

### 9.1 concat 拼接

| 参数           | 说明                      | 示例                                       |
| -------------- | ------------------------- | ------------------------------------------ |
| `objs`         | 要拼接的对象列表          | `pd.concat([df1, df2])`                    |
| `axis`         | 拼接方向：0=纵向, 1=横向  | `pd.concat([df1, df2], axis=1)`            |
| `join`         | 连接方式：'outer'/'inner' | `pd.concat([df1, df2], join='inner')`      |
| `ignore_index` | 是否忽略原索引            | `pd.concat([df1, df2], ignore_index=True)` |
| `keys`         | 添加层级索引              | `pd.concat([df1, df2], keys=['a', 'b'])`   |

```python
# 纵向拼接（行增加）
pd.concat([df1, df2], axis=0, ignore_index=True)

# 横向拼接（列增加）
pd.concat([df1, df2], axis=1)

# 添加层级索引
pd.concat([df1, df2], keys=['2023', '2024'])
```

### 9.2 merge 合并

| 参数                | 说明           | 示例                                               |
| ------------------- | -------------- | -------------------------------------------------- |
| `left, right`       | 左右 DataFrame | `pd.merge(left, right)`                            |
| `on`                | 连接键         | `pd.merge(left, right, on='key')`                  |
| `left_on, right_on` | 不同名的连接键 | `pd.merge(left, right, left_on='a', right_on='b')` |
| `how`               | 连接方式       | `pd.merge(left, right, how='left')`                |
| `suffixes`          | 重名列后缀     | `pd.merge(left, right, suffixes=('_l', '_r'))`     |
| `indicator`         | 显示来源       | `pd.merge(left, right, indicator=True)`            |

```python
# 连接方式
pd.merge(left, right, on='key', how='inner')   # 内连接（交集）
pd.merge(left, right, on='key', how='left')    # 左连接
pd.merge(left, right, on='key', how='right')   # 右连接
pd.merge(left, right, on='key', how='outer')   # 外连接（并集）
pd.merge(left, right, on='key', how='cross')   # 笛卡尔积

# 多键连接
pd.merge(left, right, on=['key1', 'key2'])

# 不同列名连接
pd.merge(left, right, left_on='lkey', right_on='rkey')

# 索引连接
pd.merge(left, right, left_index=True, right_index=True)
```

### 9.3 join 方法

```python
# 基于索引的连接
left.join(right, how='left')
left.join(right, on='key')  # left的'key'列与right的索引连接
```

### 9.4 append（已弃用，用 concat 代替）

```python
# 旧写法
df.append(new_row, ignore_index=True)

# 新写法
pd.concat([df, new_row], ignore_index=True)
```

---

## 十、时间序列

### 10.1 日期时间创建

| 方法                   | 说明           | 示例                                               |
| ---------------------- | -------------- | -------------------------------------------------- |
| `pd.to_datetime()`     | 转换为日期时间 | `pd.to_datetime('2024-01-01')`                     |
| `pd.Timestamp()`       | 创建时间戳     | `pd.Timestamp('2024-01-01')`                       |
| `pd.date_range()`      | 创建日期范围   | `pd.date_range('2024-01-01', periods=10)`          |
| `pd.period_range()`    | 创建周期范围   | `pd.period_range('2024-01', periods=12, freq='M')` |
| `pd.timedelta_range()` | 创建时间差范围 | `pd.timedelta_range('1 days', periods=5)`          |

```python
# to_datetime
pd.to_datetime('2024-01-01')
pd.to_datetime('01/01/2024', format='%m/%d/%Y')
pd.to_datetime(df['date_col'])
pd.to_datetime(df[['year', 'month', 'day']])

# date_range 频率参数
pd.date_range('2024-01-01', periods=10, freq='D')   # 日
pd.date_range('2024-01-01', periods=10, freq='W')   # 周
pd.date_range('2024-01-01', periods=10, freq='M')   # 月末
pd.date_range('2024-01-01', periods=10, freq='MS')  # 月初
pd.date_range('2024-01-01', periods=10, freq='Q')   # 季度末
pd.date_range('2024-01-01', periods=10, freq='Y')   # 年末
pd.date_range('2024-01-01', periods=10, freq='H')   # 小时
pd.date_range('2024-01-01', periods=10, freq='T')   # 分钟
pd.date_range('2024-01-01', periods=10, freq='B')   # 工作日
```

### 10.2 日期时间属性（dt accessor）

| 属性                 | 说明             | 示例                               |
| -------------------- | ---------------- | ---------------------------------- |
| `.dt.year`           | 年               | `df['date'].dt.year`               |
| `.dt.month`          | 月               | `df['date'].dt.month`              |
| `.dt.day`            | 日               | `df['date'].dt.day`                |
| `.dt.hour`           | 时               | `df['date'].dt.hour`               |
| `.dt.minute`         | 分               | `df['date'].dt.minute`             |
| `.dt.second`         | 秒               | `df['date'].dt.second`             |
| `.dt.dayofweek`      | 星期几（0=周一） | `df['date'].dt.dayofweek`          |
| `.dt.dayofyear`      | 年中第几天       | `df['date'].dt.dayofyear`          |
| `.dt.weekofyear`     | 年中第几周       | `df['date'].dt.isocalendar().week` |
| `.dt.quarter`        | 季度             | `df['date'].dt.quarter`            |
| `.dt.is_month_start` | 是否月初         | `df['date'].dt.is_month_start`     |
| `.dt.is_month_end`   | 是否月末         | `df['date'].dt.is_month_end`       |
| `.dt.date`           | 日期部分         | `df['date'].dt.date`               |
| `.dt.time`           | 时间部分         | `df['date'].dt.time`               |
| `.dt.days_in_month`  | 当月天数         | `df['date'].dt.days_in_month`      |

```python
# 提取日期组件
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()  # Monday, Tuesday...
df['is_weekend'] = df['date'].dt.dayofweek >= 5
```

### 10.3 日期时间方法

| 方法                | 说明         | 示例                                        |
| ------------------- | ------------ | ------------------------------------------- |
| `.dt.strftime()`    | 格式化日期   | `df['date'].dt.strftime('%Y-%m-%d')`        |
| `.dt.floor()`       | 向下取整     | `df['date'].dt.floor('D')`                  |
| `.dt.ceil()`        | 向上取整     | `df['date'].dt.ceil('H')`                   |
| `.dt.round()`       | 四舍五入     | `df['date'].dt.round('H')`                  |
| `.dt.normalize()`   | 归一化到午夜 | `df['date'].dt.normalize()`                 |
| `.dt.tz_localize()` | 设置时区     | `df['date'].dt.tz_localize('UTC')`          |
| `.dt.tz_convert()`  | 转换时区     | `df['date'].dt.tz_convert('Asia/Shanghai')` |

### 10.4 重采样（Resample）

```python
# 设置日期为索引
df = df.set_index('date')

# 重采样
df.resample('D').sum()      # 按日求和
df.resample('W').mean()     # 按周求均值
df.resample('M').agg({'col1': 'sum', 'col2': 'mean'})  # 多列不同聚合
df.resample('Q').last()     # 按季度取最后值

# 升采样（插值）
df.resample('H').ffill()    # 小时级别，前向填充
df.resample('H').interpolate()  # 插值填充
```

---

## 十一、字符串处理

### 11.1 str accessor 方法

| 方法                 | 说明          | 示例                                      |
| -------------------- | ------------- | ----------------------------------------- |
| `.str.lower()`       | 转小写        | `df['name'].str.lower()`                  |
| `.str.upper()`       | 转大写        | `df['name'].str.upper()`                  |
| `.str.title()`       | 首字母大写    | `df['name'].str.title()`                  |
| `.str.capitalize()`  | 首字母大写    | `df['name'].str.capitalize()`             |
| `.str.strip()`       | 去除两端空白  | `df['name'].str.strip()`                  |
| `.str.lstrip()`      | 去除左侧空白  | `df['name'].str.lstrip()`                 |
| `.str.rstrip()`      | 去除右侧空白  | `df['name'].str.rstrip()`                 |
| `.str.len()`         | 字符串长度    | `df['name'].str.len()`                    |
| `.str.replace()`     | 替换          | `df['name'].str.replace('old', 'new')`    |
| `.str.contains()`    | 是否包含      | `df['name'].str.contains('pattern')`      |
| `.str.startswith()`  | 是否以...开头 | `df['name'].str.startswith('A')`          |
| `.str.endswith()`    | 是否以...结尾 | `df['name'].str.endswith('ing')`          |
| `.str.find()`        | 查找位置      | `df['name'].str.find('a')`                |
| `.str.count()`       | 计数匹配      | `df['name'].str.count('a')`               |
| `.str.split()`       | 分割          | `df['name'].str.split(',')`               |
| `.str.join()`        | 连接          | `df['list_col'].str.join('-')`            |
| `.str.cat()`         | 拼接字符串    | `df['col1'].str.cat(df['col2'], sep='-')` |
| `.str.slice()`       | 切片          | `df['name'].str.slice(0, 3)`              |
| `.str.extract()`     | 正则提取      | `df['name'].str.extract(r'(\d+)')`        |
| `.str.extractall()`  | 正则提取所有  | `df['name'].str.extractall(r'(\d+)')`     |
| `.str.match()`       | 正则匹配      | `df['name'].str.match(r'^A')`             |
| `.str.pad()`         | 填充          | `df['id'].str.pad(5, fillchar='0')`       |
| `.str.zfill()`       | 零填充        | `df['id'].str.zfill(5)`                   |
| `.str.get()`         | 获取位置元素  | `df['name'].str.get(0)`                   |
| `.str.get_dummies()` | 哑变量编码    | `df['category'].str.get_dummies(',')`     |

```python
# 常用示例
df['name'].str.lower()                           # 转小写
df['name'].str.strip()                           # 去空白
df['name'].str.replace(' ', '_')                 # 替换
df['name'].str.contains('pattern', case=False)   # 不区分大小写
df['name'].str.split(',', expand=True)           # 分割成多列

# 正则表达式
df['phone'].str.extract(r'(\d{3})-(\d{4})-(\d{4})')  # 提取电话号码
df['email'].str.contains(r'@\w+\.com', regex=True)   # 匹配邮箱
```

---

## 十二、数据透视表

### 12.1 pivot_table

| 参数           | 说明       | 示例                   |
| -------------- | ---------- | ---------------------- |
| `values`       | 值列       | `values='sales'`       |
| `index`        | 行索引     | `index='city'`         |
| `columns`      | 列索引     | `columns='product'`    |
| `aggfunc`      | 聚合函数   | `aggfunc='mean'`       |
| `fill_value`   | 填充缺失值 | `fill_value=0`         |
| `margins`      | 添加总计   | `margins=True`         |
| `margins_name` | 总计名称   | `margins_name='Total'` |

```python
# 创建透视表
pd.pivot_table(df,
               values='sales',
               index='city',
               columns='product',
               aggfunc='sum',
               fill_value=0,
               margins=True)

# 多值多函数
pd.pivot_table(df,
               values=['sales', 'quantity'],
               index=['city', 'store'],
               columns='product',
               aggfunc={'sales': 'sum', 'quantity': 'mean'})
```

### 12.2 crosstab 交叉表

```python
# 频率交叉表
pd.crosstab(df['city'], df['product'])

# 带聚合
pd.crosstab(df['city'], df['product'], values=df['sales'], aggfunc='sum')

# 带总计和比例
pd.crosstab(df['city'], df['product'], margins=True, normalize='all')
```

---

## 十三、窗口函数

### 13.1 Rolling 滑动窗口

| 方法                 | 说明         | 示例                               |
| -------------------- | ------------ | ---------------------------------- |
| `.rolling(window)`   | 创建滑动窗口 | `df['col'].rolling(3)`             |
| `.rolling().mean()`  | 滑动均值     | `df['col'].rolling(7).mean()`      |
| `.rolling().sum()`   | 滑动求和     | `df['col'].rolling(7).sum()`       |
| `.rolling().std()`   | 滑动标准差   | `df['col'].rolling(7).std()`       |
| `.rolling().min()`   | 滑动最小值   | `df['col'].rolling(7).min()`       |
| `.rolling().max()`   | 滑动最大值   | `df['col'].rolling(7).max()`       |
| `.rolling().apply()` | 自定义函数   | `df['col'].rolling(7).apply(func)` |

```python
# 常用参数
df['col'].rolling(window=7,           # 窗口大小
                  min_periods=1,       # 最小观测数
                  center=False).mean() # 是否居中

# 7日移动平均
df['ma7'] = df['sales'].rolling(7).mean()

# 自定义函数
df['rolling_range'] = df['price'].rolling(5).apply(lambda x: x.max() - x.min())
```

### 13.2 Expanding 扩展窗口

```python
# 累计统计（从开始到当前）
df['cumsum'] = df['col'].expanding().sum()
df['cummean'] = df['col'].expanding().mean()
df['cummax'] = df['col'].expanding().max()
```

### 13.3 EWM 指数加权

```python
# 指数加权移动平均
df['ewm'] = df['col'].ewm(span=7).mean()       # 按span
df['ewm'] = df['col'].ewm(alpha=0.5).mean()    # 按alpha
df['ewm'] = df['col'].ewm(halflife=3).mean()   # 按半衰期
```

### 13.4 Rank 排名

| 方法                      | 说明             | 示例                               |
| ------------------------- | ---------------- | ---------------------------------- |
| `.rank()`                 | 排名             | `df['col'].rank()`                 |
| `.rank(method='average')` | 平均排名（默认） | `df['col'].rank(method='average')` |
| `.rank(method='min')`     | 最小排名         | `df['col'].rank(method='min')`     |
| `.rank(method='max')`     | 最大排名         | `df['col'].rank(method='max')`     |
| `.rank(method='first')`   | 按出现顺序       | `df['col'].rank(method='first')`   |
| `.rank(method='dense')`   | 密集排名         | `df['col'].rank(method='dense')`   |
| `.rank(ascending=False)`  | 降序排名         | `df['col'].rank(ascending=False)`  |
| `.rank(pct=True)`         | 百分比排名       | `df['col'].rank(pct=True)`         |

```python
# 分组排名
df['rank'] = df.groupby('city')['sales'].rank(ascending=False)
```

---

## 十四、绑图

### 14.1 基础绑图

| 方法                | 说明         | 示例                            |
| ------------------- | ------------ | ------------------------------- |
| `df.plot()`         | 线图（默认） | `df.plot()`                     |
| `df.plot.line()`    | 线图         | `df.plot.line()`                |
| `df.plot.bar()`     | 柱状图       | `df['col'].plot.bar()`          |
| `df.plot.barh()`    | 水平柱状图   | `df['col'].plot.barh()`         |
| `df.plot.hist()`    | 直方图       | `df['col'].plot.hist(bins=20)`  |
| `df.plot.box()`     | 箱线图       | `df.plot.box()`                 |
| `df.plot.scatter()` | 散点图       | `df.plot.scatter(x='a', y='b')` |
| `df.plot.pie()`     | 饼图         | `df['col'].plot.pie()`          |
| `df.plot.area()`    | 面积图       | `df.plot.area()`                |
| `df.plot.kde()`     | 密度图       | `df['col'].plot.kde()`          |
| `df.plot.hexbin()`  | 六边形图     | `df.plot.hexbin(x='a', y='b')`  |

```python
import matplotlib.pyplot as plt

# 基础绑图
df['sales'].plot(kind='line', figsize=(10, 6), title='Sales Trend')
plt.show()

# 多子图
df.plot(subplots=True, layout=(2, 2), figsize=(12, 8))
plt.tight_layout()
plt.show()

# 常用参数
df.plot(
    kind='bar',           # 图表类型
    figsize=(10, 6),      # 图表大小
    title='Title',        # 标题
    xlabel='X Label',     # X轴标签
    ylabel='Y Label',     # Y轴标签
    legend=True,          # 显示图例
    grid=True,            # 显示网格
    color='blue',         # 颜色
    alpha=0.7             # 透明度
)
```

---

## 十五、其他实用方法

### 15.1 类型判断

| 方法                                  | 说明           | 示例                                           |
| ------------------------------------- | -------------- | ---------------------------------------------- |
| `pd.api.types.is_numeric_dtype()`     | 是否数值类型   | `pd.api.types.is_numeric_dtype(df['col'])`     |
| `pd.api.types.is_string_dtype()`      | 是否字符串类型 | `pd.api.types.is_string_dtype(df['col'])`      |
| `pd.api.types.is_datetime64_dtype()`  | 是否日期类型   | `pd.api.types.is_datetime64_dtype(df['col'])`  |
| `pd.api.types.is_categorical_dtype()` | 是否分类类型   | `pd.api.types.is_categorical_dtype(df['col'])` |

### 15.2 内存优化

```python
# 查看内存使用
df.memory_usage(deep=True)

# 类型优化
df['int_col'] = df['int_col'].astype('int32')    # 降低精度
df['float_col'] = df['float_col'].astype('float32')
df['cat_col'] = df['cat_col'].astype('category')  # 类别型

# 自动优化函数
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df
```

### 15.3 复制

| 方法                  | 说明   | 示例                            |
| --------------------- | ------ | ------------------------------- |
| `df.copy()`           | 深拷贝 | `df_copy = df.copy()`           |
| `df.copy(deep=False)` | 浅拷贝 | `df_copy = df.copy(deep=False)` |

### 15.4 迭代

| 方法              | 说明             | 示例                                    |
| ----------------- | ---------------- | --------------------------------------- |
| `df.iterrows()`   | 按行迭代         | `for index, row in df.iterrows():`      |
| `df.itertuples()` | 按行迭代（更快） | `for row in df.itertuples():`           |
| `df.items()`      | 按列迭代         | `for col_name, col_data in df.items():` |

```python
# iterrows（较慢）
for index, row in df.iterrows():
    print(row['name'], row['age'])

# itertuples（更快）
for row in df.itertuples():
    print(row.name, row.age)

# 尽量用向量化操作代替迭代
df['new_col'] = df['col1'] + df['col2']  # 推荐
```

### 15.5 条件表达式

```python
# np.where
df['status'] = np.where(df['score'] >= 60, 'Pass', 'Fail')

# np.select（多条件）
conditions = [
    df['score'] >= 90,
    df['score'] >= 60,
    df['score'] < 60
]
choices = ['A', 'B', 'C']
df['grade'] = np.select(conditions, choices)

# cut（分箱）
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                         labels=['少年', '青年', '中年', '老年'])

# qcut（等频分箱）
df['score_group'] = pd.qcut(df['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### 15.6 链式调用

```python
# 使用 pipe 进行链式调用
def add_features(df):
    df['total'] = df['price'] * df['quantity']
    return df

def filter_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] > mean - n_std * std) &
              (df[column] < mean + n_std * std)]

result = (df
    .pipe(add_features)
    .pipe(filter_outliers, column='total')
    .groupby('category')
    .agg({'total': 'sum'})
    .reset_index()
    .sort_values('total', ascending=False)
)
```

### 15.7 选项设置

```python
# 显示设置
pd.set_option('display.max_rows', 100)        # 最大显示行数
pd.set_option('display.max_columns', 50)      # 最大显示列数
pd.set_option('display.width', 1000)          # 显示宽度
pd.set_option('display.max_colwidth', 100)    # 列宽
pd.set_option('display.precision', 2)         # 小数位数
pd.set_option('display.float_format', '{:.2f}'.format)  # 浮点数格式

# 查看当前设置
pd.get_option('display.max_rows')

# 重置设置
pd.reset_option('display.max_rows')
pd.reset_option('all')

# 临时设置
with pd.option_context('display.max_rows', 10):
    print(df)
```

---

## 附录：常用速查

### A. 常用导入

```python
import pandas as pd
import numpy as np
```

### B. 快速对照表

| 需求        | 代码                                  |
| ----------- | ------------------------------------- |
| 读取 CSV    | `pd.read_csv('file.csv')`             |
| 查看前 5 行 | `df.head()`                           |
| 查看形状    | `df.shape`                            |
| 查看信息    | `df.info()`                           |
| 统计摘要    | `df.describe()`                       |
| 选择列      | `df['col']` 或 `df[['col1', 'col2']]` |
| 条件筛选    | `df[df['col'] > value]`               |
| 缺失值处理  | `df.fillna(value)` 或 `df.dropna()`   |
| 去重        | `df.drop_duplicates()`                |
| 排序        | `df.sort_values('col')`               |
| 分组聚合    | `df.groupby('col').agg(func)`         |
| 合并        | `pd.merge(df1, df2, on='key')`        |
| 拼接        | `pd.concat([df1, df2])`               |
| 新增列      | `df['new'] = values`                  |
| 删除列      | `df.drop('col', axis=1)`              |
| 重命名      | `df.rename(columns={'old': 'new'})`   |
| 保存 CSV    | `df.to_csv('file.csv', index=False)`  |

---

> 📚 **参考文档**: [Pandas 官方文档](https://pandas.pydata.org/docs/)  
> 💡 **提示**: 遇到问题时，使用 `help(pd.function_name)` 或 `df.method_name?` 查看帮助

---
