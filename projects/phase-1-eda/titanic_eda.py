"""
Phase 1 实战项目：Titanic 数据集探索性数据分析 (EDA)

项目目标：
1. 综合运用 NumPy、Pandas、Matplotlib 进行真实数据分析
2. 掌握完整的 EDA 流程
3. 为后续机器学习建模做数据准备

数据集说明：
- 泰坦尼克号乘客数据
- 目标：分析影响生存率的因素

EDA 流程：
1. 数据加载与初步探索
2. 缺失值分析与处理
3. 单变量分析
4. 多变量分析
5. 特征工程初步
6. 结论与可视化报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 创建输出目录
OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Phase 1 实战项目：Titanic 生存分析 EDA")
print("=" * 60)

# =============================================================================
# 1. 数据加载与初步探索
# =============================================================================
print("\n" + "=" * 60)
print("【1. 数据加载与初步探索】")
print("=" * 60)

# 从网络加载 Titanic 数据集
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

try:
    df = pd.read_csv(url)
    print("✅ 数据加载成功！")
except Exception as e:
    print(f"❌ 网络加载失败: {e}")
    print("正在使用内置数据...")
    # 使用 seaborn 内置数据集作为备选
    df = sns.load_dataset('titanic')
    # 调整列名以匹配标准 Titanic 数据集
    df = df.rename(columns={
        'survived': 'Survived',
        'pclass': 'Pclass',
        'sex': 'Sex',
        'age': 'Age',
        'sibsp': 'SibSp',
        'parch': 'Parch',
        'fare': 'Fare',
        'embarked': 'Embarked',
        'class': 'Class',
        'who': 'Who',
        'adult_male': 'Adult_Male',
        'deck': 'Deck',
        'embark_town': 'Embark_Town',
        'alive': 'Alive',
        'alone': 'Alone'
    })

# 查看数据基本信息
print(f"\n数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n列名: {df.columns.tolist()}")

print("\n前 5 行数据：")
print(df.head())

print("\n数据类型：")
print(df.dtypes)

print("\n统计摘要：")
print(df.describe())

print("\n非数值列统计：")
print(df.describe(include=['object', 'category']))

# =============================================================================
# 2. 缺失值分析与处理
# =============================================================================
print("\n" + "=" * 60)
print("【2. 缺失值分析与处理】")
print("=" * 60)

# 缺失值统计
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失比例(%)': missing_pct
}).sort_values('缺失比例(%)', ascending=False)

print("\n缺失值统计：")
print(missing_df[missing_df['缺失数量'] > 0])

# 可视化缺失值
fig, ax = plt.subplots(figsize=(10, 6))
missing_cols = missing_df[missing_df['缺失数量'] > 0]['缺失比例(%)']
if len(missing_cols) > 0:
    missing_cols.plot(kind='bar', color='coral', edgecolor='black', ax=ax)
    ax.set_title('各列缺失值比例', fontsize=14, fontweight='bold')
    ax.set_xlabel('列名')
    ax.set_ylabel('缺失比例 (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    for i, v in enumerate(missing_cols):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '01_missing_values.png'}")

# 处理缺失值
df_clean = df.copy()

# Age: 用中位数填充
if 'Age' in df_clean.columns:
    median_age = df_clean['Age'].median()
    df_clean['Age'] = df_clean['Age'].fillna(median_age)
    print(f"\n✅ Age 缺失值用中位数 {median_age:.1f} 填充")

# Embarked: 用众数填充
if 'Embarked' in df_clean.columns:
    mode_embarked = df_clean['Embarked'].mode()[0]
    df_clean['Embarked'] = df_clean['Embarked'].fillna(mode_embarked)
    print(f"✅ Embarked 缺失值用众数 '{mode_embarked}' 填充")

# Cabin: 缺失太多，创建是否有舱位的二值特征
if 'Cabin' in df_clean.columns:
    df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)
    print("✅ Cabin 转换为二值特征 Has_Cabin")

print(f"\n处理后缺失值: {df_clean.isnull().sum().sum()}")

# =============================================================================
# 3. 目标变量分析（生存率）
# =============================================================================
print("\n" + "=" * 60)
print("【3. 目标变量分析】")
print("=" * 60)

survival_counts = df_clean['Survived'].value_counts()
survival_rate = df_clean['Survived'].mean() * 100

print(f"\n生存统计：")
print(f"  - 遇难: {survival_counts[0]} 人")
print(f"  - 生存: {survival_counts[1]} 人")
print(f"  - 总体生存率: {survival_rate:.2f}%")

# 生存率饼图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 饼图
colors = ['#ff6b6b', '#4ecdc4']
axes[0].pie(survival_counts, labels=['遇难', '生存'], autopct='%1.1f%%',
            colors=colors, explode=(0, 0.05), shadow=True, startangle=90)
axes[0].set_title('生存情况分布', fontsize=14, fontweight='bold')

# 柱状图
bars = axes[1].bar(['遇难', '生存'], survival_counts, color=colors, edgecolor='black')
axes[1].set_title('生存人数统计', fontsize=14, fontweight='bold')
axes[1].set_ylabel('人数')
for bar, count in zip(bars, survival_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(count), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_survival_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '02_survival_distribution.png'}")

# =============================================================================
# 4. 单变量分析
# =============================================================================
print("\n" + "=" * 60)
print("【4. 单变量分析】")
print("=" * 60)

# 4.1 性别与生存率
print("\n4.1 性别与生存率：")
sex_survival = df_clean.groupby('Sex')['Survived'].agg(['sum', 'count', 'mean'])
sex_survival.columns = ['生存人数', '总人数', '生存率']
sex_survival['生存率'] = (sex_survival['生存率'] * 100).round(2)
print(sex_survival)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 性别分布
sex_counts = df_clean['Sex'].value_counts()
axes[0].bar(sex_counts.index, sex_counts.values, color=['#3498db', '#e74c3c'], edgecolor='black')
axes[0].set_title('性别分布', fontsize=14, fontweight='bold')
axes[0].set_ylabel('人数')

# 性别生存率
colors = ['#e74c3c', '#3498db']
survival_by_sex = df_clean.groupby('Sex')['Survived'].mean() * 100
bars = axes[1].bar(survival_by_sex.index, survival_by_sex.values, color=colors, edgecolor='black')
axes[1].set_title('不同性别的生存率', fontsize=14, fontweight='bold')
axes[1].set_ylabel('生存率 (%)')
axes[1].set_ylim(0, 100)
for bar, rate in zip(bars, survival_by_sex.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_sex_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '03_sex_analysis.png'}")

# 4.2 船舱等级与生存率
print("\n4.2 船舱等级与生存率：")
pclass_survival = df_clean.groupby('Pclass')['Survived'].agg(['sum', 'count', 'mean'])
pclass_survival.columns = ['生存人数', '总人数', '生存率']
pclass_survival['生存率'] = (pclass_survival['生存率'] * 100).round(2)
print(pclass_survival)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 船舱等级分布
pclass_counts = df_clean['Pclass'].value_counts().sort_index()
colors_pclass = ['#2ecc71', '#f39c12', '#e74c3c']
axes[0].bar([f'{i}等舱' for i in pclass_counts.index], pclass_counts.values, 
            color=colors_pclass, edgecolor='black')
axes[0].set_title('船舱等级分布', fontsize=14, fontweight='bold')
axes[0].set_ylabel('人数')

# 船舱等级生存率
survival_by_pclass = df_clean.groupby('Pclass')['Survived'].mean() * 100
bars = axes[1].bar([f'{i}等舱' for i in survival_by_pclass.index], survival_by_pclass.values,
                   color=colors_pclass, edgecolor='black')
axes[1].set_title('不同船舱等级的生存率', fontsize=14, fontweight='bold')
axes[1].set_ylabel('生存率 (%)')
axes[1].set_ylim(0, 100)
for bar, rate in zip(bars, survival_by_pclass.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_pclass_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '04_pclass_analysis.png'}")

# 4.3 年龄分布与生存
print("\n4.3 年龄分布分析：")
print(f"  年龄范围: {df_clean['Age'].min():.0f} - {df_clean['Age'].max():.0f} 岁")
print(f"  平均年龄: {df_clean['Age'].mean():.1f} 岁")
print(f"  年龄中位数: {df_clean['Age'].median():.1f} 岁")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 年龄分布直方图
axes[0].hist(df_clean['Age'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
axes[0].axvline(df_clean['Age'].mean(), color='red', linestyle='--', 
                label=f'均值: {df_clean["Age"].mean():.1f}')
axes[0].axvline(df_clean['Age'].median(), color='green', linestyle='--',
                label=f'中位数: {df_clean["Age"].median():.1f}')
axes[0].set_title('年龄分布', fontsize=14, fontweight='bold')
axes[0].set_xlabel('年龄')
axes[0].set_ylabel('人数')
axes[0].legend()

# 生存者与遇难者年龄分布对比
axes[1].hist(df_clean[df_clean['Survived'] == 0]['Age'], bins=30, alpha=0.6,
             label='遇难', color='#ff6b6b', edgecolor='white')
axes[1].hist(df_clean[df_clean['Survived'] == 1]['Age'], bins=30, alpha=0.6,
             label='生存', color='#4ecdc4', edgecolor='white')
axes[1].set_title('生存者与遇难者年龄分布对比', fontsize=14, fontweight='bold')
axes[1].set_xlabel('年龄')
axes[1].set_ylabel('人数')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_age_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '05_age_analysis.png'}")

# 4.4 票价分布与生存
print("\n4.4 票价分布分析：")
print(f"  票价范围: ${df_clean['Fare'].min():.2f} - ${df_clean['Fare'].max():.2f}")
print(f"  平均票价: ${df_clean['Fare'].mean():.2f}")
print(f"  票价中位数: ${df_clean['Fare'].median():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 票价分布（对数刻度更清晰）
axes[0].hist(df_clean['Fare'], bins=50, color='coral', edgecolor='white', alpha=0.7)
axes[0].set_title('票价分布', fontsize=14, fontweight='bold')
axes[0].set_xlabel('票价 ($)')
axes[0].set_ylabel('人数')

# 生存者与遇难者票价箱线图
data_to_plot = [df_clean[df_clean['Survived'] == 0]['Fare'],
                df_clean[df_clean['Survived'] == 1]['Fare']]
bp = axes[1].boxplot(data_to_plot, labels=['遇难', '生存'], patch_artist=True)
colors = ['#ff6b6b', '#4ecdc4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_title('生存者与遇难者票价对比', fontsize=14, fontweight='bold')
axes[1].set_ylabel('票价 ($)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_fare_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '06_fare_analysis.png'}")

# =============================================================================
# 5. 多变量分析
# =============================================================================
print("\n" + "=" * 60)
print("【5. 多变量分析】")
print("=" * 60)

# 5.1 性别 × 船舱等级 × 生存率
print("\n5.1 性别与船舱等级交叉分析：")
cross_tab = pd.crosstab([df_clean['Pclass'], df_clean['Sex']], 
                         df_clean['Survived'], margins=True)
print(cross_tab)

# 热力图
survival_pivot = df_clean.pivot_table(values='Survived', index='Sex', 
                                       columns='Pclass', aggfunc='mean') * 100

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(survival_pivot, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('生存率 (%)')

# 添加标签
ax.set_xticks(range(len(survival_pivot.columns)))
ax.set_yticks(range(len(survival_pivot.index)))
ax.set_xticklabels([f'{i}等舱' for i in survival_pivot.columns])
ax.set_yticklabels(survival_pivot.index)

# 在每个格子中显示数值
for i in range(len(survival_pivot.index)):
    for j in range(len(survival_pivot.columns)):
        val = survival_pivot.iloc[i, j]
        text_color = 'white' if val < 50 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                color=text_color, fontsize=14, fontweight='bold')

ax.set_title('性别与船舱等级的生存率热力图', fontsize=14, fontweight='bold')
ax.set_xlabel('船舱等级')
ax.set_ylabel('性别')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_sex_pclass_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '07_sex_pclass_heatmap.png'}")

# 5.2 年龄段与生存率
print("\n5.2 年龄段分析：")
df_clean['Age_Group'] = pd.cut(df_clean['Age'], 
                                bins=[0, 12, 18, 35, 60, 100],
                                labels=['儿童', '青少年', '青年', '中年', '老年'])

age_group_survival = df_clean.groupby('Age_Group')['Survived'].agg(['sum', 'count', 'mean'])
age_group_survival.columns = ['生存人数', '总人数', '生存率']
age_group_survival['生存率'] = (age_group_survival['生存率'] * 100).round(2)
print(age_group_survival)

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(age_group_survival))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], age_group_survival['生存人数'], 
               width, label='生存', color='#4ecdc4', edgecolor='black')
bars2 = ax.bar([i + width/2 for i in x], 
               age_group_survival['总人数'] - age_group_survival['生存人数'],
               width, label='遇难', color='#ff6b6b', edgecolor='black')

ax.set_xlabel('年龄组')
ax.set_ylabel('人数')
ax.set_title('各年龄段生存与遇难人数', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(age_group_survival.index)
ax.legend()

# 添加生存率标注
for i, (idx, row) in enumerate(age_group_survival.iterrows()):
    ax.text(i, row['总人数'] + 5, f'{row["生存率"]:.0f}%', 
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_age_group_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '08_age_group_analysis.png'}")

# 5.3 家庭规模与生存
print("\n5.3 家庭规模分析：")
df_clean['Family_Size'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean['Is_Alone'] = (df_clean['Family_Size'] == 1).astype(int)

family_survival = df_clean.groupby('Family_Size')['Survived'].agg(['sum', 'count', 'mean'])
family_survival.columns = ['生存人数', '总人数', '生存率']
family_survival['生存率'] = (family_survival['生存率'] * 100).round(2)
print(family_survival)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 家庭规模分布
family_counts = df_clean['Family_Size'].value_counts().sort_index()
axes[0].bar(family_counts.index, family_counts.values, color='steelblue', edgecolor='black')
axes[0].set_title('家庭规模分布', fontsize=14, fontweight='bold')
axes[0].set_xlabel('家庭规模')
axes[0].set_ylabel('人数')

# 家庭规模与生存率
survival_by_family = df_clean.groupby('Family_Size')['Survived'].mean() * 100
axes[1].bar(survival_by_family.index, survival_by_family.values, 
            color='coral', edgecolor='black')
axes[1].set_title('不同家庭规模的生存率', fontsize=14, fontweight='bold')
axes[1].set_xlabel('家庭规模')
axes[1].set_ylabel('生存率 (%)')
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_family_size_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '09_family_size_analysis.png'}")

# =============================================================================
# 6. 相关性分析
# =============================================================================
print("\n" + "=" * 60)
print("【6. 相关性分析】")
print("=" * 60)

# 选择数值列
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_Size']
if 'Has_Cabin' in df_clean.columns:
    numeric_cols.append('Has_Cabin')

# 计算相关性矩阵
corr_matrix = df_clean[numeric_cols].corr()

print("\n与生存率的相关性：")
print(corr_matrix['Survived'].sort_values(ascending=False))

# 可视化相关性矩阵
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('相关系数')

# 添加标签
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

# 在每个格子中显示数值
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        text_color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color=text_color, fontsize=10)

ax.set_title('特征相关性矩阵', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '10_correlation_matrix.png'}")

# =============================================================================
# 7. 特征工程预览
# =============================================================================
print("\n" + "=" * 60)
print("【7. 特征工程预览】")
print("=" * 60)

# 提取称呼 (Mr., Mrs., Miss., etc.)
if 'Name' in df_clean.columns:
    df_clean['Title'] = df_clean['Name'].str.extract(r' ([A-Za-z]+)\.')
    
    # 简化称呼
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer', 'Capt': 'Officer',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Countess': 'Royalty', 'Lady': 'Royalty', 'Sir': 'Royalty', 'Don': 'Royalty', 'Jonkheer': 'Royalty', 'Dona': 'Royalty'
    }
    df_clean['Title'] = df_clean['Title'].map(title_mapping).fillna('Other')
    
    print("\n称呼与生存率：")
    title_survival = df_clean.groupby('Title')['Survived'].agg(['sum', 'count', 'mean'])
    title_survival.columns = ['生存人数', '总人数', '生存率']
    title_survival['生存率'] = (title_survival['生存率'] * 100).round(2)
    print(title_survival.sort_values('生存率', ascending=False))

# 票价分组
df_clean['Fare_Group'] = pd.cut(df_clean['Fare'],
                                 bins=[0, 10, 30, 100, 600],
                                 labels=['低', '中', '高', '豪华'])

print("\n票价等级与生存率：")
fare_group_survival = df_clean.groupby('Fare_Group')['Survived'].agg(['sum', 'count', 'mean'])
fare_group_survival.columns = ['生存人数', '总人数', '生存率']
fare_group_survival['生存率'] = (fare_group_survival['生存率'] * 100).round(2)
print(fare_group_survival)

# =============================================================================
# 8. 综合报告与结论
# =============================================================================
print("\n" + "=" * 60)
print("【8. 综合报告与结论】")
print("=" * 60)

# 创建综合仪表板
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 生存率总览
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#ff6b6b', '#4ecdc4']
ax1.pie(df_clean['Survived'].value_counts(), labels=['遇难', '生存'],
        autopct='%1.1f%%', colors=colors, explode=(0, 0.05))
ax1.set_title('总体生存率', fontsize=12, fontweight='bold')

# 2. 性别生存率
ax2 = fig.add_subplot(gs[0, 1])
survival_by_sex = df_clean.groupby('Sex')['Survived'].mean() * 100
bars = ax2.bar(survival_by_sex.index, survival_by_sex.values, 
               color=['#e74c3c', '#3498db'], edgecolor='black')
ax2.set_title('性别生存率', fontsize=12, fontweight='bold')
ax2.set_ylabel('生存率 (%)')
ax2.set_ylim(0, 100)

# 3. 船舱等级生存率
ax3 = fig.add_subplot(gs[0, 2])
survival_by_pclass = df_clean.groupby('Pclass')['Survived'].mean() * 100
bars = ax3.bar([f'{i}等舱' for i in survival_by_pclass.index], survival_by_pclass.values,
               color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
ax3.set_title('船舱等级生存率', fontsize=12, fontweight='bold')
ax3.set_ylabel('生存率 (%)')
ax3.set_ylim(0, 100)

# 4. 年龄分布
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df_clean[df_clean['Survived'] == 0]['Age'], bins=20, alpha=0.6,
         label='遇难', color='#ff6b6b')
ax4.hist(df_clean[df_clean['Survived'] == 1]['Age'], bins=20, alpha=0.6,
         label='生存', color='#4ecdc4')
ax4.set_title('年龄与生存', fontsize=12, fontweight='bold')
ax4.set_xlabel('年龄')
ax4.legend()

# 5. 家庭规模与生存
ax5 = fig.add_subplot(gs[1, 1])
survival_by_family = df_clean.groupby('Family_Size')['Survived'].mean() * 100
ax5.bar(survival_by_family.index, survival_by_family.values, color='coral', edgecolor='black')
ax5.set_title('家庭规模生存率', fontsize=12, fontweight='bold')
ax5.set_xlabel('家庭规模')
ax5.set_ylabel('生存率 (%)')
ax5.set_ylim(0, 100)

# 6. 票价与生存
ax6 = fig.add_subplot(gs[1, 2])
data_to_plot = [df_clean[df_clean['Survived'] == 0]['Fare'],
                df_clean[df_clean['Survived'] == 1]['Fare']]
bp = ax6.boxplot(data_to_plot, labels=['遇难', '生存'], patch_artist=True)
colors_box = ['#ff6b6b', '#4ecdc4']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax6.set_title('票价与生存', fontsize=12, fontweight='bold')
ax6.set_ylabel('票价 ($)')
ax6.set_ylim(0, 150)

# 7. 性别×船舱等级热力图
ax7 = fig.add_subplot(gs[2, :2])
survival_pivot = df_clean.pivot_table(values='Survived', index='Sex',
                                       columns='Pclass', aggfunc='mean') * 100
im = ax7.imshow(survival_pivot, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax7.set_xticks(range(3))
ax7.set_yticks(range(2))
ax7.set_xticklabels([f'{i}等舱' for i in [1, 2, 3]])
ax7.set_yticklabels(survival_pivot.index)
for i in range(2):
    for j in range(3):
        val = survival_pivot.iloc[i, j]
        text_color = 'white' if val < 50 else 'black'
        ax7.text(j, i, f'{val:.0f}%', ha='center', va='center',
                 color=text_color, fontsize=12, fontweight='bold')
ax7.set_title('性别×船舱等级生存率', fontsize=12, fontweight='bold')

# 8. 关键发现总结
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
findings = """
🔍 关键发现

1. 女性生存率显著高于男性
   (74.2% vs 18.9%)

2. 一等舱生存率最高 (62.9%)
   三等舱最低 (24.2%)

3. 儿童生存率较高
   
4. 2-4人家庭生存率最优

5. 高票价乘客生存率更高
"""
ax8.text(0.1, 0.9, findings, transform=ax8.transAxes,
         fontsize=11, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Titanic 数据集探索性分析报告', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(OUTPUT_DIR / '11_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {OUTPUT_DIR / '11_dashboard.png'}")

# 打印最终结论
print("""
╔══════════════════════════════════════════════════════════════╗
║                        EDA 分析结论                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 总体统计:                                                 ║
║     - 总样本数: 891 人                                        ║
║     - 生存率: 38.4%                                          ║
║                                                              ║
║  🔑 关键发现:                                                 ║
║                                                              ║
║  1. 【性别】是最强的生存预测因子                               ║
║     - 女性生存率 74.2%，男性仅 18.9%                          ║
║     - 体现了"妇女和儿童优先"的逃生原则                         ║
║                                                              ║
║  2. 【社会地位】显著影响生存率                                 ║
║     - 一等舱 62.9% > 二等舱 47.3% > 三等舱 24.2%              ║
║     - 高等舱位靠近甲板，更容易获得救生艇                        ║
║                                                              ║
║  3. 【年龄】呈现非线性影响                                     ║
║     - 儿童(0-12岁)生存率较高                                  ║
║     - 老年人生存率较低                                        ║
║                                                              ║
║  4. 【家庭规模】存在最优区间                                   ║
║     - 2-4人家庭生存率最高                                     ║
║     - 独自旅行或大家庭生存率较低                               ║
║                                                              ║
║  5. 【票价】间接反映社会地位                                   ║
║     - 高票价乘客生存率显著更高                                 ║
║                                                              ║
║  💡 特征工程建议:                                             ║
║     - 创建 Title 特征（从姓名中提取）                          ║
║     - 创建 Family_Size 特征                                  ║
║     - 创建 Is_Alone 特征                                     ║
║     - 对 Age 和 Fare 进行分箱                                 ║
║     - 考虑性别×船舱等级的交互特征                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# 保存清洗后的数据
df_clean.to_csv(OUTPUT_DIR / 'titanic_cleaned.csv', index=False)
print(f"\n✅ 清洗后的数据已保存: {OUTPUT_DIR / 'titanic_cleaned.csv'}")

print("\n" + "=" * 60)
print("✅ Phase 1 实战项目完成！")
print("=" * 60)
print(f"\n所有图表已保存至: {OUTPUT_DIR}")
print("\n📂 生成的文件:")
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"   - {f.name}")
print(f"   - titanic_cleaned.csv")

print("""
🎯 项目总结：
   本项目综合运用了 Phase 1 学习的所有技能：
   ✓ NumPy: 数值计算
   ✓ Pandas: 数据处理、清洗、分组聚合  
   ✓ Matplotlib: 多种图表可视化
   ✓ 完整 EDA 流程: 探索 → 清洗 → 分析 → 可视化 → 结论

下一步：进入 Phase 2 深度学习数学基础学习！
""")
