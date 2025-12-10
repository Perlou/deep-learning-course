"""
05-pandas-preprocessing.py
Phase 1: Python 数据科学基础

Pandas 数据预处理：数据清洗、转换、合并

学习目标：
1. 掌握缺失值处理的多种方法
2. 学习数据类型转换
3. 掌握字符串处理
4. 理解数据合并操作
"""

import pandas as pd
import numpy as np

print("=" * 50)
print("Pandas 数据预处理")
print("=" * 50)

# =============================================================================
# 1. 缺失值处理
# =============================================================================
print("\n【1. 缺失值处理】")

# 创建含缺失值的数据
data = {
    'name': ['Alice', 'Bob', None, 'David', 'Eva'],
    'age': [25, np.nan, 35, 28, np.nan],
    'salary': [10000, 15000, np.nan, np.nan, 18000],
    'department': ['技术', '销售', '技术', None, '销售']
}
df = pd.DataFrame(data)
print("原始数据（含缺失值）:")
print(df)

# 检查缺失值
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n缺失值占比:\n{df.isnull().mean() * 100}%")

# 填充方法 1：用固定值填充
df_filled_0 = df.copy()
df_filled_0['age'] = df_filled_0['age'].fillna(0)
print("\n用 0 填充 age:")
print(df_filled_0)

# 填充方法 2：用均值填充
df_filled_mean = df.copy()
df_filled_mean['age'] = df_filled_mean['age'].fillna(df['age'].mean())
df_filled_mean['salary'] = df_filled_mean['salary'].fillna(df['salary'].median())
print("\n用均值/中位数填充:")
print(df_filled_mean)

# 填充方法 3：用众数填充（分类数据）
df_filled_mode = df.copy()
df_filled_mode['department'] = df_filled_mode['department'].fillna(
    df['department'].mode()[0]
)
print("\n用众数填充 department:")
print(df_filled_mode)

# 填充方法 4：前向/后向填充
df_ffill = df.copy()
df_ffill['age'] = df_ffill['age'].fillna(method='ffill')  # 前向填充
print("\n前向填充 (ffill):")
print(df_ffill)

# 删除缺失值
print("\n删除含缺失值的行:")
print(df.dropna())

print("\n删除缺失值超过 50% 的列:")
print(df.dropna(axis=1, thresh=len(df) * 0.5))

# =============================================================================
# 2. 数据类型转换
# =============================================================================
print("\n" + "=" * 50)
print("【2. 数据类型转换】")

data = {
    'id': ['001', '002', '003'],
    'price': ['100.5', '200.0', '150.3'],
    'date': ['2024-01-15', '2024-02-20', '2024-03-25'],
    'is_sale': ['True', 'False', 'True']
}
df = pd.DataFrame(data)
print("原始数据:")
print(df)
print(f"数据类型:\n{df.dtypes}")

# 转换数值类型
df['price'] = df['price'].astype(float)
df['id'] = df['id'].astype(int)

# 转换日期类型
df['date'] = pd.to_datetime(df['date'])

# 转换布尔类型
df['is_sale'] = df['is_sale'].map({'True': True, 'False': False})

print("\n转换后:")
print(df)
print(f"数据类型:\n{df.dtypes}")

# 从日期提取信息
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.day_name()
print("\n提取日期信息:")
print(df)

# =============================================================================
# 3. 字符串处理
# =============================================================================
print("\n" + "=" * 50)
print("【3. 字符串处理】")

data = {
    'name': ['  Alice  ', 'BOB', 'charlie', 'David Lee'],
    'email': ['alice@gmail.com', 'bob@yahoo.com', 'charlie@163.com', 'david@qq.com']
}
df = pd.DataFrame(data)
print("原始数据:")
print(df)

# 去除空白
df['name'] = df['name'].str.strip()

# 大小写转换
df['name_lower'] = df['name'].str.lower()
df['name_upper'] = df['name'].str.upper()
df['name_title'] = df['name'].str.title()

print("\n字符串处理:")
print(df[['name', 'name_lower', 'name_upper', 'name_title']])

# 字符串分割
df['first_name'] = df['name'].str.split().str[0]
df['email_domain'] = df['email'].str.split('@').str[1]
print("\n字符串分割:")
print(df[['name', 'first_name', 'email', 'email_domain']])

# 字符串包含
df['is_gmail'] = df['email'].str.contains('gmail')
print("\n是否为 Gmail:")
print(df[['email', 'is_gmail']])

# 字符串替换
df['email_masked'] = df['email'].str.replace(r'@.*', '@***.com', regex=True)
print("\n邮箱脱敏:")
print(df[['email', 'email_masked']])

# =============================================================================
# 4. 重复值处理
# =============================================================================
print("\n" + "=" * 50)
print("【4. 重复值处理】")

data = {
    'id': [1, 2, 2, 3, 4, 4, 4],
    'name': ['A', 'B', 'B', 'C', 'D', 'D', 'D'],
    'value': [10, 20, 20, 30, 40, 40, 40]
}
df = pd.DataFrame(data)
print("含重复值的数据:")
print(df)

# 检查重复
print(f"\n重复行数量: {df.duplicated().sum()}")
print(f"重复的行:\n{df[df.duplicated(keep=False)]}")

# 删除重复
df_unique = df.drop_duplicates()
print(f"\n删除重复后:\n{df_unique}")

# 按特定列删除重复
df_unique_by_id = df.drop_duplicates(subset=['id'], keep='first')
print(f"\n按 id 删除重复 (保留第一个):\n{df_unique_by_id}")

# =============================================================================
# 5. 异常值处理
# =============================================================================
print("\n" + "=" * 50)
print("【5. 异常值处理】")

np.random.seed(42)
data = {
    'value': np.concatenate([
        np.random.normal(100, 10, 98),  # 正常值
        [200, -50]  # 异常值
    ])
}
df = pd.DataFrame(data)

print(f"原始统计:\n{df.describe()}")

# 方法 1：IQR (四分位距) 方法
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"\nIQR 方法：下限 {lower:.2f}，上限 {upper:.2f}")

# 标记异常值
df['is_outlier'] = (df['value'] < lower) | (df['value'] > upper)
print(f"异常值数量: {df['is_outlier'].sum()}")

# 方法 2：Z-score 方法
from scipy import stats
df['zscore'] = np.abs(stats.zscore(df['value']))
df['is_outlier_zscore'] = df['zscore'] > 3
print(f"\nZ-score 方法 (|z| > 3) 异常值数量: {df['is_outlier_zscore'].sum()}")

# 处理异常值：用边界值替换
df['value_clipped'] = df['value'].clip(lower, upper)
print(f"\n裁剪后统计:\n{df['value_clipped'].describe()}")

# =============================================================================
# 6. 数据合并
# =============================================================================
print("\n" + "=" * 50)
print("【6. 数据合并】")

# 创建两个表
df_orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4],
    'user_id': [101, 102, 101, 103],
    'amount': [100, 200, 150, 300]
})

df_users = pd.DataFrame({
    'user_id': [101, 102, 104],
    'name': ['Alice', 'Bob', 'David']
})

print("订单表:")
print(df_orders)
print("\n用户表:")
print(df_users)

# 内连接 (默认)
print("\n内连接 (inner join):")
print(pd.merge(df_orders, df_users, on='user_id'))

# 左连接
print("\n左连接 (left join):")
print(pd.merge(df_orders, df_users, on='user_id', how='left'))

# 右连接
print("\n右连接 (right join):")
print(pd.merge(df_orders, df_users, on='user_id', how='right'))

# 外连接
print("\n外连接 (outer join):")
print(pd.merge(df_orders, df_users, on='user_id', how='outer'))

# =============================================================================
# 7. 数据透视表
# =============================================================================
print("\n" + "=" * 50)
print("【7. 数据透视表】")

data = {
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'] * 2,
    'category': ['电子', '服装', '电子', '服装'] * 2,
    'region': ['北', '北', '北', '北', '南', '南', '南', '南'],
    'sales': [100, 80, 120, 90, 150, 70, 130, 85]
}
df = pd.DataFrame(data)
print("销售数据:")
print(df)

# 创建透视表
pivot = pd.pivot_table(
    df,
    values='sales',
    index='category',
    columns='region',
    aggfunc='sum'
)
print("\n透视表（按品类和地区汇总）:")
print(pivot)

# =============================================================================
# 8. 练习题
# =============================================================================
print("\n" + "=" * 50)
print("【练习题】")
print("=" * 50)

print("""
1. 创建一个包含缺失值的数据集，使用三种不同方法处理缺失值

2. 将字符串列 "1,000.50", "2,500.00", "800.75" 转换为数值类型
   提示：先去掉逗号

3. 给定两个表（学生表和成绩表），使用 merge 合并并计算每个学生的平均成绩

4. 创建一个销售数据透视表，分析不同月份和产品的销售趋势

请在下方编写代码完成练习...
""")

# === 在这里编写你的练习代码 ===
# 练习 1
# df_missing = pd.DataFrame({
#     'A': [1, np.nan, 3, np.nan, 5],
#     'B': [10, 20, np.nan, 40, 50],
#     'C': ['x', 'y', None, 'z', 'w']
# })
# print("原数据:\n", df_missing)
# 
# # 方法1: 用固定值填充
# df1 = df_missing.copy()
# df1['A'] = df1['A'].fillna(0)
# print("\n方法1 - 固定值填充:\n", df1)
# 
# # 方法2: 用均值填充
# df2 = df_missing.copy()
# df2['A'] = df2['A'].fillna(df2['A'].mean())
# df2['B'] = df2['B'].fillna(df2['B'].mean())
# print("\n方法2 - 均值填充:\n", df2)
# 
# # 方法3: 前向填充
# df3 = df_missing.copy()
# df3 = df3.fillna(method='ffill')
# print("\n方法3 - 前向填充:\n", df3)

# 练习 2
# prices = pd.DataFrame({'price': ["1,000.50", "2,500.00", "800.75"]})
# prices['price'] = prices['price'].str.replace(',', '').astype(float)
# print(prices)
# print(prices.dtypes)

# 练习 3
# students = pd.DataFrame({
#     'student_id': [1, 2, 3, 4],
#     'name': ['Alice', 'Bob', 'Charlie', 'David']
# })
# scores = pd.DataFrame({
#     'student_id': [1, 1, 2, 2, 3, 3],
#     'subject': ['math', 'english', 'math', 'english', 'math', 'english'],
#     'score': [90, 85, 78, 92, 88, 76]
# })
# merged = pd.merge(students, scores, on='student_id')
# avg_scores = merged.groupby('name')['score'].mean()
# print("每个学生的平均成绩:\n", avg_scores)

# 练习 4
# sales_data = pd.DataFrame({
#     'month': ['2024-01', '2024-01', '2024-02', '2024-02', '2024-03', '2024-03'],
#     'product': ['A', 'B', 'A', 'B', 'A', 'B'],
#     'sales': [100, 150, 120, 180, 90, 200]
# })
# pivot = pd.pivot_table(sales_data, values='sales', index='product', columns='month', aggfunc='sum')
# print("销售数据透视表:\n", pivot)

print("\n✅ Pandas 数据预处理完成！")
print("下一步：06-matplotlib-basics.py - 数据可视化基础")
