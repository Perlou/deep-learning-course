---
name: explain-concept
description: 解释深度学习概念，结合理论、公式、代码和可视化
---

# 解释概念技能

此技能用于详细解释深度学习中的核心概念，帮助用户深入理解。

## 解释结构

### 1. 概念概述

- 一句话定义
- 为什么重要
- 应用场景

### 2. 直观理解

- 类比解释（用日常生活的例子）
- 可视化图示（使用 ASCII art 或建议查看哪些可视化资源）

### 3. 数学原理

- 核心公式
- 公式推导（如果需要）
- 各变量的含义

### 4. 代码实现

```python
# 从零实现
def concept_from_scratch(...):
    """手动实现，帮助理解原理"""
    pass

# PyTorch 实现
def concept_with_pytorch(...):
    """使用 PyTorch 的标准实现"""
    pass
```

### 5. 常见问题

- 常见误区
- FAQ

### 6. 延伸阅读

- 相关论文
- 推荐资源

## 解释示例：Softmax 函数

### 1. 概念概述

**一句话定义**：Softmax 将任意实数向量转换为概率分布。

**为什么重要**：多分类问题中，我们需要将模型输出转换为概率。

**应用场景**：分类任务的输出层、注意力机制中的权重计算。

### 2. 直观理解

想象一个投票系统：

- 原始分数：[3.0, 1.0, 0.2]
- 这些分数代表三个选项的"支持度"
- Softmax 将它们转换为概率：[0.87, 0.12, 0.01]
- 结果：总和为 1，越高的分数对应越高的概率

### 3. 数学原理

**公式**：

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**数值稳定版本**（减去最大值防止溢出）：

```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

### 4. 代码实现

```python
import numpy as np
import torch
import torch.nn.functional as F

# 从零实现
def softmax_from_scratch(x):
    """手动实现 softmax"""
    # 数值稳定：减去最大值
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

# PyTorch 实现
def softmax_pytorch(x):
    """使用 PyTorch"""
    return F.softmax(torch.tensor(x), dim=0)

# 验证
x = [3.0, 1.0, 0.2]
print(f"从零实现: {softmax_from_scratch(x)}")
print(f"PyTorch:  {softmax_pytorch(x)}")
```

### 5. 常见问题

**Q: 为什么要减去最大值？**
A: 防止 exp(x) 溢出。当 x 很大时，exp(x) 可能超出浮点数范围。

**Q: Softmax 和 Sigmoid 的区别？**
A: Sigmoid 用于二分类（输出单个概率），Softmax 用于多分类（输出概率分布）。

### 6. 延伸阅读

- Bishop, Pattern Recognition and Machine Learning, Chapter 4
- 相关项目中的代码：`src/phase-4-neural-network-basics/`

## 使用提示

1. **根据用户水平调整**：初学者多用类比，进阶者多讲数学
2. **代码优先**：总是提供可运行的代码示例
3. **循序渐进**：从简单到复杂
4. **主动链接**：链接到项目中相关的课程文件
