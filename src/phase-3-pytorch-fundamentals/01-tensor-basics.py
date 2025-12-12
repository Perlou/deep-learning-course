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

# =============================================================================
# 1. Tensor 简介
# =============================================================================
print("\n【1. Tensor 简介】")

print("""
Tensor（张量）是 PyTorch 的核心数据结构：
- 类似于 NumPy 的 ndarray
- 支持 GPU 加速
- 支持自动微分（autograd）

张量的维度：
- 0D: 标量 (scalar)
- 1D: 向量 (vector)
- 2D: 矩阵 (matrix)
- 3D+: 高维张量 (tensor)
""")

# =============================================================================
# 2. 创建 Tensor
# =============================================================================
print("\n【2. 创建 Tensor】")

# 2.1 从 Python 列表创建
print("\n2.1 从列表创建:")
x = torch.tensor([1, 2, 3, 4])
print(f"  x = {x}")

# 2.2 从 NumPy 创建
print("\n2.2 从 NumPy 创建:")
np_array = np.array([[1, 2], [3, 4]])
x_np = torch.from_numpy(np_array)
print(f"  NumPy array:\n{np_array}")
print(f"  Tensor:\n{x_np}")
# 注意：共享内存！
np_array[0, 0] = 100
print(f"  修改 NumPy 后, Tensor 也变了:\n{x_np}")

# 2.3 特殊张量
print("\n2.3 特殊张量:")
zeros = torch.zeros(2, 3)
print(f"  zeros(2, 3):\n{zeros}")

ones = torch.ones(2, 3)
print(f"  ones(2, 3):\n{ones}")

eye = torch.eye(3)
print(f"  eye(3):\n{eye}")

# 2.4 随机张量
print("\n2.4 随机张量:")
rand_uniform = torch.rand(2, 3)        # 均匀分布 [0, 1)
print(f"  rand(2, 3):\n{rand_uniform}")

rand_normal = torch.randn(2, 3)        # 标准正态分布
print(f"  randn(2, 3):\n{rand_normal}")

rand_int = torch.randint(0, 10, (2, 3))  # 整数
print(f"  randint(0, 10, (2, 3)):\n{rand_int}")

# 2.5 创建相同形状的张量
print("\n2.5 创建相同形状的张量:")
x = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x.float())  # 需要浮点类型
print(f"  原张量 x:\n{x}")
print(f"  zeros_like(x):\n{zeros_like}")

# 2.6 指定数据类型
print("\n2.6 指定数据类型:")
x_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
x_float64 = torch.tensor([1, 2, 3], dtype=torch.float64)
x_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"  float32: {x_float32}, dtype: {x_float32.dtype}")
print(f"  float64: {x_float64}, dtype: {x_float64.dtype}")
print(f"  int64: {x_int64}, dtype: {x_int64.dtype}")

# =============================================================================
# 3. Tensor 属性
# =============================================================================
print("\n" + "=" * 60)
print("【3. Tensor 属性】")

x = torch.randn(3, 4, 5)
print(f"张量 x 形状: {x.shape}")
print(f"张量 x 维度数: {x.dim()} (或 x.ndim)")
print(f"张量 x 元素数: {x.numel()}")
print(f"张量 x 数据类型: {x.dtype}")
print(f"张量 x 设备: {x.device}")
print(f"张量 x 是否需要梯度: {x.requires_grad}")

# =============================================================================
# 4. 设备管理 (CPU/GPU)
# =============================================================================
print("\n" + "=" * 60)
print("【4. 设备管理】")

# 检查 GPU
cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()  # Apple Silicon
print(f"CUDA (NVIDIA GPU) 可用: {cuda_available}")
print(f"MPS (Apple GPU) 可用: {mps_available}")

# 设置设备
if cuda_available:
    device = torch.device("cuda")
elif mps_available:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"当前使用设备: {device}")

# 创建在特定设备上的张量
x_cpu = torch.randn(2, 3)
print(f"\nCPU 张量: {x_cpu.device}")

# 移动到 GPU (如果可用)
if device.type != "cpu":
    x_gpu = x_cpu.to(device)
    print(f"GPU 张量: {x_gpu.device}")
    
    # 也可以用 .cuda() 或 .mps()
    # x_gpu = x_cpu.cuda()  # NVIDIA
    # x_gpu = x_cpu.to('mps')  # Apple

# 直接在 GPU 创建
x_on_device = torch.randn(2, 3, device=device)
print(f"直接在 {device} 创建: {x_on_device.device}")

# =============================================================================
# 5. 与 NumPy 互操作
# =============================================================================
print("\n" + "=" * 60)
print("【5. 与 NumPy 互操作】")

# Tensor → NumPy
t = torch.tensor([1.0, 2.0, 3.0])
n = t.numpy()
print(f"Tensor: {t}")
print(f"NumPy: {n}")

# 注意：共享内存（CPU 上）
t[0] = 100
print(f"修改 Tensor 后, NumPy 也变了: {n}")

# 如果不想共享内存，用 clone()
t2 = torch.tensor([1.0, 2.0, 3.0])
n2 = t2.clone().numpy()
t2[0] = 100
print(f"\n使用 clone(): Tensor={t2}, NumPy={n2}")

# GPU 张量需要先移到 CPU
if device.type != "cpu":
    x_gpu = torch.randn(3, device=device)
    x_np = x_gpu.cpu().numpy()  # 先 .cpu() 再 .numpy()
    print(f"GPU → NumPy: {x_np}")

# =============================================================================
# 6. 常用属性方法
# =============================================================================
print("\n" + "=" * 60)
print("【6. 常用属性方法】")

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"原张量:\n{x}")
print(f"形状: {x.shape} 或 {x.size()}")
print(f"第一维大小: {x.size(0)}")
print(f"元素总数: {x.numel()}")

# 类型转换
print(f"\n类型转换:")
print(f"转 int: {x.int()}")
print(f"转 long: {x.long()}")
print(f"转 float: {x.float()}")
print(f"转 double: {x.double()}")

# =============================================================================
# 7. 练习题
# =============================================================================
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

# === 练习答案 ===
# 1
# x = torch.randn(3, 4, dtype=torch.float32)
# print(x)

# 2
# eye = torch.eye(5)
# print(eye)

# 3
# x = torch.randn(2, 3)
# if torch.cuda.is_available():
#     x = x.cuda()
# elif torch.backends.mps.is_available():
#     x = x.to('mps')
# print(x.device)

# 4
# np_arr = np.array([[1, 2], [3, 4]])
# t = torch.from_numpy(np_arr)
# np_arr[0, 0] = 100
# print(f"NumPy: {np_arr}")
# print(f"Tensor: {t}")  # 也变成 100 了

# 5
# print(f"PyTorch: {torch.__version__}")
# print(f"CUDA: {torch.version.cuda}")

print("\n✅ Tensor 基础完成！")
print("下一步：02-tensor-operations.py - Tensor 运算")
