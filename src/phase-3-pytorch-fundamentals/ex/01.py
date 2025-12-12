"""
01-tensor-basics.py
Phase 3: PyTorch 核心技能

Tensor 基础 - PyTorch 的核心数据结构

学习目标：
1. 理解 Tensor 的概念和属性
2. 掌握 Tensor 创建的各种方法
3. 理解设备 (CPU/GPU) 管理
"""

import torch
import numpy as np

print("=" * 60)
print("PyTorch 核心技能 - Tensor 基础")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")

print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 创建一个 3x4 的随机张量（正态分布），设置 dtype=float32

2. 创建一个 5x5 的单位矩阵

3. 创建一个 2x3 的张量，并将其移动到 GPU（如果可用）

4. 创建一个 NumPy 数组 [[1,2],[3,4]]，转换为 Tensor，
   然后检查它们是否共享内存

5. 查看当前 PyTorch 版本和 CUDA 版本
""")

# 1
x = torch.randn(3, 4, dtype=torch.float32)
print(x)

# 2
eye = torch.eye(5)
print(eye)

# 3
x = torch.randn(2, 3)
if torch.cuda.is_available():
    x = x.cuda()
elif torch.backends.mps.is_available():
    x = x.to('mps')
print(x.device)

# 4
np_arr = np.array([[1, 2], [3, 4]])
t = torch.from_numpy(np_arr)
np_arr[0, 0] = 100
print(f"NumPy: {np_arr}")
print(f"Tensor: {t}")

# 5
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
