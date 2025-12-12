"""
04-nn-module.py
Phase 3: PyTorch 核心技能

nn.Module - 神经网络的基础构建块

学习目标：
1. 理解 nn.Module 的结构和作用
2. 掌握自定义模型的方法
3. 了解常用层和参数管理
4. 理解模型的 train/eval 模式
"""

import torch
import torch.nn as nn

print("=" * 60)
print("PyTorch 核心技能 - nn.Module")
print("=" * 60)

# =============================================================================
# 1. nn.Module 简介
# =============================================================================
print("\n【1. nn.Module 简介】")

print("""
nn.Module 是所有神经网络模块的基类：
- 管理参数（自动追踪所有 nn.Parameter）
- 提供 forward() 方法定义前向传播
- 支持 GPU 移动、保存/加载、train/eval 模式
- 可以嵌套（模块包含子模块）
""")

# =============================================================================
# 2. 基本使用
# =============================================================================
print("\n【2. 基本使用】")

# 直接使用内置层
linear = nn.Linear(10, 5)  # 10 输入，5 输出
print(f"Linear 层: {linear}")
print(f"权重形状: {linear.weight.shape}")
print(f"偏置形状: {linear.bias.shape}")

# 前向传播
x = torch.randn(3, 10)  # batch_size=3, features=10
y = linear(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {y.shape}")

# =============================================================================
# 3. 自定义模型
# =============================================================================
print("\n" + "=" * 60)
print("【3. 自定义模型】")

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # 必须调用父类初始化
        
        # 定义层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # 定义前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet(input_dim=10, hidden_dim=20, output_dim=5)
print(f"模型结构:\n{model}")

# 前向传播
x = torch.randn(3, 10)
y = model(x)  # 自动调用 forward()
print(f"\n输入: {x.shape} -> 输出: {y.shape}")

# =============================================================================
# 4. 参数管理
# =============================================================================
print("\n" + "=" * 60)
print("【4. 参数管理】")

# 查看所有参数
print("所有参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# 参数总数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数: {total_params}")
print(f"可训练参数: {trainable_params}")

# 冻结参数
print("\n冻结 fc1 层:")
for param in model.fc1.parameters():
    param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"冻结后可训练参数: {trainable_params}")

# 解冻
for param in model.fc1.parameters():
    param.requires_grad = True

# =============================================================================
# 5. 常用层
# =============================================================================
print("\n" + "=" * 60)
print("【5. 常用层】")

print("""
常用层类型:
- nn.Linear(in, out)      全连接层
- nn.Conv2d(in_ch, out_ch, kernel)  2D 卷积
- nn.ReLU(), nn.Sigmoid(), nn.Tanh()  激活函数
- nn.BatchNorm1d/2d       批归一化
- nn.Dropout(p)           Dropout
- nn.Embedding(num, dim)  嵌入层
- nn.LSTM, nn.GRU         循环神经网络
- nn.Transformer          Transformer
""")

# 激活函数
print("\n激活函数示例:")
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"  输入: {x}")
print(f"  ReLU: {nn.ReLU()(x)}")
print(f"  Sigmoid: {nn.Sigmoid()(x)}")
print(f"  Tanh: {nn.Tanh()(x)}")

# Dropout
print("\nDropout 示例:")
dropout = nn.Dropout(p=0.5)
x = torch.ones(10)
print(f"  训练模式: {dropout(x)}")  # 随机置零
dropout.eval()
print(f"  评估模式: {dropout(x)}")  # 不置零
dropout.train()

# =============================================================================
# 6. nn.Sequential
# =============================================================================
print("\n" + "=" * 60)
print("【6. nn.Sequential】")

# 方式 1: 顺序传入
model_seq = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5)
)
print(f"Sequential 模型:\n{model_seq}")

# 方式 2: 使用 OrderedDict 命名
from collections import OrderedDict

model_named = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 20)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(20, 5))
]))
print(f"\n命名 Sequential:\n{model_named}")

# 访问子模块
print(f"\n访问子模块 model_named.fc1: {model_named.fc1}")

# =============================================================================
# 7. train/eval 模式
# =============================================================================
print("\n" + "=" * 60)
print("【7. train/eval 模式】")

print("""
某些层在训练和推理时行为不同:
- Dropout: 训练时随机置零，推理时不变
- BatchNorm: 训练时用 batch 统计，推理时用全局统计

需要正确设置模式：
- model.train()  训练模式
- model.eval()   推理模式
""")

class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(10)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

model = ModelWithDropout()
x = torch.randn(4, 10)

model.train()
print(f"training 属性: {model.training}")
y_train = model(x)

model.eval()
print(f"training 属性: {model.training}")
with torch.no_grad():  # 推理时也要禁用梯度
    y_eval = model(x)

# =============================================================================
# 8. 设备移动
# =============================================================================
print("\n" + "=" * 60)
print("【8. 设备移动】")

model = SimpleNet(10, 20, 5)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
print(f"使用设备: {device}")

# 移动模型到设备
model = model.to(device)
print(f"模型设备: {next(model.parameters()).device}")

# 输入也要在同一设备
x = torch.randn(3, 10).to(device)
y = model(x)
print(f"输入设备: {x.device}, 输出设备: {y.device}")

# =============================================================================
# 9. 模型初始化
# =============================================================================
print("\n" + "=" * 60)
print("【9. 模型初始化】")

def init_weights(m):
    """自定义权重初始化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = SimpleNet(10, 20, 5)
model.apply(init_weights)  # 递归应用到所有子模块
print("已应用 Xavier 初始化")

# 常用初始化方法
print("""
常用初始化方法:
- nn.init.xavier_uniform_()   Xavier 均匀
- nn.init.xavier_normal_()    Xavier 正态
- nn.init.kaiming_uniform_()  He/Kaiming 均匀
- nn.init.kaiming_normal_()   He/Kaiming 正态
- nn.init.zeros_()            全零
- nn.init.ones_()             全一
- nn.init.constant_(x, val)   常数
""")

# =============================================================================
# 10. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 创建一个三层 MLP: 784 -> 256 -> 128 -> 10，使用 ReLU 激活

2. 统计模型的总参数量和可训练参数量

3. 使用 nn.Sequential 创建相同的网络

4. 实现一个带 Dropout 和 BatchNorm 的模型

5. 将模型移动到 GPU（如果可用）并进行一次前向传播
""")

# === 练习答案 ===
# 1
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# 2
# model = MLP()
# total = sum(p.numel() for p in model.parameters())
# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"总参数: {total}, 可训练: {trainable}")

# 3
# model_seq = nn.Sequential(
#     nn.Linear(784, 256), nn.ReLU(),
#     nn.Linear(256, 128), nn.ReLU(),
#     nn.Linear(128, 10)
# )

# 4
# class RegularizedMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(256, 10)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = torch.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# 5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MLP().to(device)
# x = torch.randn(32, 784).to(device)
# y = model(x)

print("\n✅ nn.Module 完成！")
print("下一步：05-loss-functions.py - 损失函数")
