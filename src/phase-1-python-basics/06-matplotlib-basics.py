"""
06-matplotlib-basics.py
Phase 1: Matplotlib 基础可视化

学习目标：
1. 掌握基本图表类型
2. 学习图表自定义
3. 理解多子图布局
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 50)
print("Matplotlib 基础可视化")
print("=" * 50)

# 1. 折线图
print("\n【1. 折线图】")
x = np.linspace(0, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)', color='blue', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', color='red', linestyle='--')
plt.title('正弦和余弦函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/01_line.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/01_line.png")

# 2. 散点图
print("\n【2. 散点图】")
np.random.seed(42)
x, y = np.random.randn(100), np.random.randn(100)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=np.random.rand(100), s=100, alpha=0.6, cmap='viridis')
plt.colorbar(label='颜色值')
plt.title('散点图示例')
plt.savefig('outputs/02_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/02_scatter.png")

# 3. 柱状图
print("\n【3. 柱状图】")
categories = ['A', 'B', 'C', 'D']
values = [35, 48, 42, 55]
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='steelblue')
plt.title('柱状图示例')
plt.savefig('outputs/03_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/03_bar.png")

# 4. 直方图
print("\n【4. 直方图】")
data = np.random.randn(1000)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='coral', edgecolor='white')
plt.axvline(data.mean(), color='red', linestyle='--', label=f'均值: {data.mean():.2f}')
plt.title('直方图示例')
plt.legend()
plt.savefig('outputs/04_hist.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/04_hist.png")

# 5. 多子图
print("\n【5. 多子图布局】")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('折线图')
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('散点图')
axes[1, 0].bar(['A', 'B', 'C'], [10, 20, 15])
axes[1, 0].set_title('柱状图')
axes[1, 1].hist(np.random.randn(500), bins=20)
axes[1, 1].set_title('直方图')
plt.tight_layout()
plt.savefig('outputs/05_subplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 outputs/05_subplots.png")

# =============================================================================
# 6. 练习题
# =============================================================================
print("\n" + "=" * 50)
print("【练习题】")
print("=" * 50)

print("""
1. 绑制一个正弦函数和余弦函数的复合图，要求：
   - 使用不同颜色和线型区分两条曲线
   - 添加图例、标题、网格线
   - 标注 x=π 处的点

2. 创建一个包含两组数据的散点图，使用不同颜色和形状区分
   
3. 绘制一个分组柱状图，展示三个产品在四个季度的销量

4. 绘制一个包含正态分布数据的直方图，并叠加概率密度曲线

请在下方编写代码完成练习...
""")

# === 在这里编写你的练习代码 ===
# 练习 1
# x = np.linspace(0, 2*np.pi, 100)
# plt.figure(figsize=(10, 6))
# plt.plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
# plt.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
# plt.scatter([np.pi], [np.sin(np.pi)], color='green', s=100, zorder=5)
# plt.annotate('x=π', xy=(np.pi, 0), xytext=(np.pi+0.5, 0.3),
#              arrowprops=dict(arrowstyle='->', color='green'))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('正弦和余弦函数')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('outputs/exercise_1.png', dpi=150)
# plt.close()
# print("练习1完成")

# 练习 2
# np.random.seed(42)
# x1, y1 = np.random.randn(30), np.random.randn(30)
# x2, y2 = np.random.randn(30) + 2, np.random.randn(30) + 1
# plt.figure(figsize=(10, 6))
# plt.scatter(x1, y1, c='blue', marker='o', label='组A', alpha=0.7)
# plt.scatter(x2, y2, c='red', marker='^', label='组B', alpha=0.7)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('两组数据散点图')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('outputs/exercise_2.png', dpi=150)
# plt.close()
# print("练习2完成")

# 练习 3
# quarters = ['Q1', 'Q2', 'Q3', 'Q4']
# product_a = [20, 35, 30, 35]
# product_b = [25, 32, 34, 20]
# product_c = [30, 25, 28, 32]
# 
# x = np.arange(len(quarters))
# width = 0.25
# 
# plt.figure(figsize=(10, 6))
# plt.bar(x - width, product_a, width, label='产品A')
# plt.bar(x, product_b, width, label='产品B')
# plt.bar(x + width, product_c, width, label='产品C')
# plt.xticks(x, quarters)
# plt.xlabel('季度')
# plt.ylabel('销量')
# plt.title('分组柱状图 - 产品季度销量')
# plt.legend()
# plt.savefig('outputs/exercise_3.png', dpi=150)
# plt.close()
# print("练习3完成")

# 练习 4
# from scipy import stats
# np.random.seed(42)
# data = np.random.randn(1000) * 15 + 100  # 均值100，标准差15
# 
# plt.figure(figsize=(10, 6))
# # 直方图（归一化）
# n, bins, patches = plt.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', label='直方图')
# # 概率密度曲线
# x = np.linspace(data.min(), data.max(), 100)
# pdf = stats.norm.pdf(x, data.mean(), data.std())
# plt.plot(x, pdf, 'r-', linewidth=2, label=f'正态分布 (μ={data.mean():.1f}, σ={data.std():.1f})')
# plt.xlabel('值')
# plt.ylabel('概率密度')
# plt.title('直方图与概率密度曲线')
# plt.legend()
# plt.savefig('outputs/exercise_4.png', dpi=150)
# plt.close()
# print("练习4完成")

print("\n✅ Matplotlib 基础完成！")
print("下一步：07-visualization-advanced.py - 高级可视化")
