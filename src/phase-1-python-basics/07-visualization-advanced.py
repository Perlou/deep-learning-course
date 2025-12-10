"""
07-visualization-advanced.py
Phase 1: Python 数据科学基础

高级可视化：多子图、自定义样式、统计图表

学习目标：
1. 掌握多子图布局技巧
2. 学习自定义图表样式
3. 掌握统计可视化图表
4. 了解 Seaborn 高级绑图
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 尝试导入 seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Seaborn 未安装，部分示例将跳过")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 50)
print("Matplotlib 高级可视化")
print("=" * 50)

# =============================================================================
# 1. 多子图布局
# =============================================================================
print("\n【1. 多子图布局】")

# 方法1: plt.subplot() 传统方式
fig = plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)  # 1行3列的第1个
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('子图 1')

plt.subplot(1, 3, 2)  # 1行3列的第2个
plt.bar(['A', 'B', 'C'], [3, 7, 5])
plt.title('子图 2')

plt.subplot(1, 3, 3)  # 1行3列的第3个
plt.scatter([1, 2, 3, 4], [4, 1, 3, 2])
plt.title('子图 3')

plt.tight_layout()
plt.savefig('outputs/07_subplot_basic.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_subplot_basic.png")

# 方法2: plt.subplots() 推荐方式
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 使用索引访问每个子图
axes[0, 0].plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
axes[0, 0].set_title('正弦函数')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('sin(x)')

axes[0, 1].plot(np.linspace(0, 10, 100), np.cos(np.linspace(0, 10, 100)), 'r--')
axes[0, 1].set_title('余弦函数')

np.random.seed(42)
axes[1, 0].hist(np.random.randn(1000), bins=30, color='green', alpha=0.7)
axes[1, 0].set_title('正态分布直方图')

axes[1, 1].pie([30, 25, 20, 15, 10], labels=['A', 'B', 'C', 'D', 'E'], autopct='%1.1f%%')
axes[1, 1].set_title('饼图')

plt.tight_layout()
plt.savefig('outputs/07_subplots_2x2.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_subplots_2x2.png")

# 方法3: GridSpec 灵活布局
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)

# 跨行跨列的子图
ax1 = fig.add_subplot(gs[0, :])  # 第一行，占据所有列
ax1.plot(np.random.randn(100).cumsum())
ax1.set_title('累积随机游走 (跨3列)')

ax2 = fig.add_subplot(gs[1, :2])  # 第二行，占据前两列
ax2.bar(['Q1', 'Q2', 'Q3', 'Q4'], [100, 120, 90, 150])
ax2.set_title('季度销售 (跨2列)')

ax3 = fig.add_subplot(gs[1, 2])   # 第二行第三列
ax3.scatter(np.random.randn(50), np.random.randn(50))
ax3.set_title('散点图')

ax4 = fig.add_subplot(gs[2, 0])   # 第三行第一列
ax4.hist(np.random.randn(500), bins=20)
ax4.set_title('直方图')

ax5 = fig.add_subplot(gs[2, 1:])  # 第三行，占据后两列
x = np.linspace(0, 5, 100)
ax5.fill_between(x, np.sin(x), alpha=0.5)
ax5.set_title('填充图 (跨2列)')

plt.tight_layout()
plt.savefig('outputs/07_gridspec.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_gridspec.png")

# =============================================================================
# 2. 自定义样式
# =============================================================================
print("\n" + "=" * 50)
print("【2. 自定义样式】")

# 查看可用样式
print(f"可用样式: {plt.style.available[:5]}...")

# 自定义颜色和线条
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
styles = [
    {'color': '#2E86AB', 'linestyle': '-', 'linewidth': 2, 'label': '实线'},
    {'color': '#A23B72', 'linestyle': '--', 'linewidth': 2, 'label': '虚线'},
    {'color': '#F18F01', 'linestyle': '-.', 'linewidth': 2, 'label': '点划线'},
    {'color': '#C73E1D', 'linestyle': ':', 'linewidth': 3, 'label': '点线'},
]

for i, style in enumerate(styles):
    ax.plot(x, np.sin(x + i * 0.5), **style)

ax.set_xlabel('X 轴', fontsize=12)
ax.set_ylabel('Y 轴', fontsize=12)
ax.set_title('自定义线条样式', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')

plt.savefig('outputs/07_custom_style.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_custom_style.png")

# 使用预定义样式
with plt.style.context('seaborn-v0_8-whitegrid'):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)')
    ax.legend()
    ax.set_title('使用 seaborn-whitegrid 样式')
    plt.savefig('outputs/07_style_seaborn.png', dpi=150, bbox_inches='tight')
    plt.close()
print("已保存 outputs/07_style_seaborn.png")

# =============================================================================
# 3. 双轴图表
# =============================================================================
print("\n" + "=" * 50)
print("【3. 双轴图表】")

fig, ax1 = plt.subplots(figsize=(10, 6))

# 模拟数据：月份、销售额、利润率
months = ['1月', '2月', '3月', '4月', '5月', '6月']
sales = [120, 150, 130, 180, 200, 190]
profit_rate = [12, 15, 10, 18, 22, 20]

# 左轴：柱状图（销售额）
color1 = '#2E86AB'
ax1.bar(months, sales, color=color1, alpha=0.7, label='销售额')
ax1.set_xlabel('月份')
ax1.set_ylabel('销售额（万元）', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# 右轴：折线图（利润率）
ax2 = ax1.twinx()
color2 = '#C73E1D'
ax2.plot(months, profit_rate, color=color2, marker='o', linewidth=2, label='利润率')
ax2.set_ylabel('利润率（%）', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('销售额与利润率对比')
plt.tight_layout()
plt.savefig('outputs/07_twin_axes.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_twin_axes.png")

# =============================================================================
# 4. 热力图
# =============================================================================
print("\n" + "=" * 50)
print("【4. 热力图】")

# 创建相关性矩阵数据
np.random.seed(42)
data = np.random.randn(5, 5)
# 创建对称矩阵模拟相关性
corr_matrix = (data + data.T) / 2
np.fill_diagonal(corr_matrix, 1)
corr_matrix = np.clip(corr_matrix, -1, 1)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('相关系数')

# 添加标签
labels = ['特征A', '特征B', '特征C', '特征D', '特征E']
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# 在每个格子中显示数值
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center', 
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

ax.set_title('特征相关性热力图')
plt.tight_layout()
plt.savefig('outputs/07_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_heatmap.png")

# =============================================================================
# 5. 箱线图与小提琴图
# =============================================================================
print("\n" + "=" * 50)
print("【5. 箱线图与小提琴图】")

np.random.seed(42)
data_groups = [
    np.random.normal(0, 1, 100),
    np.random.normal(1, 1.5, 100),
    np.random.normal(0.5, 0.8, 100),
    np.random.normal(-0.5, 1.2, 100)
]
labels = ['组A', '组B', '组C', '组D']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 箱线图
bp = axes[0].boxplot(data_groups, labels=labels, patch_artist=True)
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('箱线图')
axes[0].set_ylabel('数值')
axes[0].grid(True, alpha=0.3)

# 小提琴图
vp = axes[1].violinplot(data_groups, positions=range(1, 5), showmeans=True, showmedians=True)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(colors[i])
    body.set_alpha(0.7)
axes[1].set_xticks(range(1, 5))
axes[1].set_xticklabels(labels)
axes[1].set_title('小提琴图')
axes[1].set_ylabel('数值')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/07_boxplot_violin.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_boxplot_violin.png")

print("""
💡 箱线图 vs 小提琴图：
- 箱线图：显示四分位数、中位数、异常值
- 小提琴图：显示整体分布形状（核密度估计）
- 两者结合使用可以更全面了解数据分布
""")

# =============================================================================
# 6. 堆叠图与面积图
# =============================================================================
print("=" * 50)
print("【6. 堆叠图与面积图】")

months = np.arange(1, 13)
product_a = np.array([20, 25, 30, 35, 40, 45, 50, 48, 42, 38, 35, 30])
product_b = np.array([15, 18, 20, 22, 25, 28, 30, 32, 28, 25, 22, 18])
product_c = np.array([10, 12, 15, 18, 20, 22, 25, 24, 22, 20, 18, 15])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 堆叠柱状图
width = 0.6
axes[0].bar(months, product_a, width, label='产品A', color='#2E86AB')
axes[0].bar(months, product_b, width, bottom=product_a, label='产品B', color='#A23B72')
axes[0].bar(months, product_c, width, bottom=product_a+product_b, label='产品C', color='#F18F01')
axes[0].set_xlabel('月份')
axes[0].set_ylabel('销量')
axes[0].set_title('堆叠柱状图')
axes[0].legend()
axes[0].set_xticks(months)

# 堆叠面积图
axes[1].stackplot(months, product_a, product_b, product_c, 
                   labels=['产品A', '产品B', '产品C'],
                   colors=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
axes[1].set_xlabel('月份')
axes[1].set_ylabel('销量')
axes[1].set_title('堆叠面积图')
axes[1].legend(loc='upper left')
axes[1].set_xlim(1, 12)

plt.tight_layout()
plt.savefig('outputs/07_stacked.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_stacked.png")

# =============================================================================
# 7. Seaborn 高级图表（可选）
# =============================================================================
print("\n" + "=" * 50)
print("【7. Seaborn 高级图表】")

if HAS_SEABORN:
    # 设置 seaborn 样式
    sns.set_theme(style="whitegrid")
    
    # 创建示例数据
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'size': np.random.randint(10, 100, n)
    })
    df['y'] = df['x'] * 0.5 + df['y'] * 0.5  # 添加相关性
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 散点图 + 回归线
    sns.regplot(data=df, x='x', y='y', ax=axes[0, 0], scatter_kws={'alpha': 0.5})
    axes[0, 0].set_title('散点图 + 回归线')
    
    # 分类散点图
    sns.stripplot(data=df, x='category', y='y', ax=axes[0, 1], jitter=True, alpha=0.5)
    axes[0, 1].set_title('分类散点图')
    
    # KDE 密度图
    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        sns.kdeplot(data=subset, x='x', ax=axes[1, 0], label=cat, fill=True, alpha=0.3)
    axes[1, 0].set_title('核密度估计图')
    axes[1, 0].legend()
    
    # 联合分布图（简化版）
    sns.kdeplot(data=df, x='x', y='y', ax=axes[1, 1], cmap='Blues', fill=True)
    axes[1, 1].set_title('二维 KDE 图')
    
    plt.tight_layout()
    plt.savefig('outputs/07_seaborn_advanced.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存 outputs/07_seaborn_advanced.png")
    
    # Pairplot（配对图）
    iris_data = pd.DataFrame({
        'sepal_length': np.random.normal(5.8, 0.8, 150),
        'sepal_width': np.random.normal(3.0, 0.4, 150),
        'petal_length': np.random.normal(3.7, 1.7, 150),
        'species': np.repeat(['setosa', 'versicolor', 'virginica'], 50)
    })
    
    g = sns.pairplot(iris_data, hue='species', height=2.5)
    g.fig.suptitle('配对图示例', y=1.02)
    plt.savefig('outputs/07_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存 outputs/07_pairplot.png")
    
else:
    print("跳过 Seaborn 示例（未安装）")

# =============================================================================
# 8. 3D 可视化
# =============================================================================
print("\n" + "=" * 50)
print("【8. 3D 可视化】")

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 5))

# 3D 散点图
ax1 = fig.add_subplot(131, projection='3d')
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
ax1.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D 散点图')

# 3D 曲面图
ax2 = fig.add_subplot(132, projection='3d')
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D 曲面图')

# 3D 线图
ax3 = fig.add_subplot(133, projection='3d')
t = np.linspace(0, 10*np.pi, 500)
x = np.sin(t)
y = np.cos(t)
z = t / (5*np.pi)
ax3.plot(x, y, z, 'b-', linewidth=1)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('3D 螺旋线')

plt.tight_layout()
plt.savefig('outputs/07_3d_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/07_3d_plots.png")

# =============================================================================
# 9. 练习题
# =============================================================================
print("\n" + "=" * 50)
print("【练习题】")
print("=" * 50)

print("""
1. 创建一个 2x3 的子图布局，在每个子图中绑制不同类型的图表
   （折线图、散点图、柱状图、直方图、饼图、箱线图）
   
2. 使用 GridSpec 创建一个不规则布局：
   - 第一行一个大图占据全宽
   - 第二行两个小图并排
   - 第三行三个图并排
   
3. 创建一个双轴图表，展示某股票的价格（折线图）和成交量（柱状图）

4. 生成一个 6x6 的随机相关矩阵，绘制热力图并添加数值标注

5. 使用 3D 可视化绘制函数 z = sin(x) * cos(y) 的曲面图

请在下方编写代码完成练习...
""")

# === 在这里编写你的练习代码 ===
# 练习 1
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# 
# # 折线图
# x = np.linspace(0, 10, 100)
# axes[0, 0].plot(x, np.sin(x))
# axes[0, 0].set_title('折线图')
# 
# # 散点图
# axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
# axes[0, 1].set_title('散点图')
# 
# # 柱状图
# axes[0, 2].bar(['A', 'B', 'C', 'D'], [25, 40, 30, 35])
# axes[0, 2].set_title('柱状图')
# 
# # 直方图
# axes[1, 0].hist(np.random.randn(1000), bins=30)
# axes[1, 0].set_title('直方图')
# 
# # 饼图
# axes[1, 1].pie([30, 25, 20, 15, 10], labels=['A', 'B', 'C', 'D', 'E'], autopct='%1.1f%%')
# axes[1, 1].set_title('饼图')
# 
# # 箱线图
# axes[1, 2].boxplot([np.random.randn(100) for _ in range(4)])
# axes[1, 2].set_title('箱线图')
# 
# plt.tight_layout()
# plt.savefig('outputs/exercise_1.png', dpi=150)
# plt.close()
# print("练习1完成")

# 练习 2
# from matplotlib.gridspec import GridSpec
# fig = plt.figure(figsize=(12, 10))
# gs = GridSpec(3, 6, figure=fig)
# 
# ax1 = fig.add_subplot(gs[0, :])
# ax1.plot(np.random.randn(100).cumsum())
# ax1.set_title('大图（占据全宽）')
# 
# ax2 = fig.add_subplot(gs[1, :3])
# ax2.bar(['A', 'B', 'C'], [10, 15, 12])
# ax2.set_title('左半图')
# 
# ax3 = fig.add_subplot(gs[1, 3:])
# ax3.scatter(np.random.randn(50), np.random.randn(50))
# ax3.set_title('右半图')
# 
# ax4 = fig.add_subplot(gs[2, :2])
# ax4.hist(np.random.randn(500), bins=20)
# ax4.set_title('左图')
# 
# ax5 = fig.add_subplot(gs[2, 2:4])
# ax5.pie([30, 40, 30], labels=['X', 'Y', 'Z'])
# ax5.set_title('中图')
# 
# ax6 = fig.add_subplot(gs[2, 4:])
# ax6.boxplot([np.random.randn(100) for _ in range(3)])
# ax6.set_title('右图')
# 
# plt.tight_layout()
# plt.savefig('outputs/exercise_2.png', dpi=150)
# plt.close()
# print("练习2完成")

# 练习 3
# fig, ax1 = plt.subplots(figsize=(10, 6))
# 
# days = np.arange(1, 31)
# price = 100 + np.cumsum(np.random.randn(30))  # 模拟股票价格
# volume = np.random.randint(1000, 5000, 30)     # 模拟成交量
# 
# ax1.bar(days, volume, color='lightblue', alpha=0.7, label='成交量')
# ax1.set_xlabel('日期')
# ax1.set_ylabel('成交量')
# 
# ax2 = ax1.twinx()
# ax2.plot(days, price, color='red', linewidth=2, label='价格')
# ax2.set_ylabel('价格')
# 
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
# 
# plt.title('股票价格与成交量')
# plt.savefig('outputs/exercise_3.png', dpi=150)
# plt.close()
# print("练习3完成")

# 练习 4
# np.random.seed(42)
# n = 6
# random_data = np.random.randn(n, n)
# corr_matrix = (random_data + random_data.T) / 2
# np.fill_diagonal(corr_matrix, 1)
# corr_matrix = np.clip(corr_matrix, -1, 1)
# 
# fig, ax = plt.subplots(figsize=(8, 6))
# im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
# plt.colorbar(im, ax=ax)
# 
# labels = [f'特征{i+1}' for i in range(n)]
# ax.set_xticks(range(n))
# ax.set_yticks(range(n))
# ax.set_xticklabels(labels)
# ax.set_yticklabels(labels)
# 
# for i in range(n):
#     for j in range(n):
#         ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
#                 color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
# 
# ax.set_title('6x6 相关矩阵热力图')
# plt.tight_layout()
# plt.savefig('outputs/exercise_4.png', dpi=150)
# plt.close()
# print("练习4完成")

# 练习 5
# from mpl_toolkits.mplot3d import Axes3D
# 
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# 
# x = np.linspace(-np.pi, np.pi, 50)
# y = np.linspace(-np.pi, np.pi, 50)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(X) * np.cos(Y)
# 
# surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('z = sin(x) * cos(y)')
# plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
# 
# plt.savefig('outputs/exercise_5.png', dpi=150)
# plt.close()
# print("练习5完成")

# =============================================================================
# 额外练习：验证理解
# =============================================================================
print("\n" + "=" * 50)
print("【额外知识点】")
print("=" * 50)

# 1. 保存高质量图像
print("\n1. 保存高质量图像:")
print("""
   # 保存为 PNG（适合网页）
   plt.savefig('figure.png', dpi=300, bbox_inches='tight')
   
   # 保存为 PDF（适合论文）
   plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')
   
   # 保存为 SVG（矢量图）
   plt.savefig('figure.svg', format='svg', bbox_inches='tight')
""")

# 2. 颜色映射表
print("2. 常用颜色映射表 (cmap):")
print("""
   - 连续性：'viridis', 'plasma', 'magma', 'inferno'
   - 发散性：'RdBu', 'coolwarm', 'seismic'
   - 循环性：'twilight', 'hsv'
   - 分类性：'Set1', 'Set2', 'tab10', 'Paired'
""")

# 3. 中文字体设置
print("3. 不同系统的中文字体设置:")
print("""
   # macOS
   plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
   
   # Windows
   plt.rcParams['font.sans-serif'] = ['SimHei']
   
   # Linux
   plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
   
   # 通用：解决负号显示问题
   plt.rcParams['axes.unicode_minus'] = False
""")

print("\n✅ Matplotlib 高级可视化完成！")
print("恭喜完成 Phase 1 所有内容！下一步请进入 Phase 2: 深度学习基础")
