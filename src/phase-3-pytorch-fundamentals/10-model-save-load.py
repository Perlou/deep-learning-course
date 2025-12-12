"""
10-model-save-load.py
Phase 3: PyTorch 核心技能

模型保存与加载 - 持久化训练成果

学习目标：
1. 掌握模型保存和加载的方法
2. 理解 state_dict 的概念
3. 了解 checkpoint 的最佳实践
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os

print("=" * 60)
print("PyTorch 核心技能 - 模型保存与加载")
print("=" * 60)

# =============================================================================
# 1. 保存和加载的两种方式
# =============================================================================
print("\n【1. 保存和加载的两种方式】")

print("""
方式 1: 保存整个模型 (不推荐)
    torch.save(model, 'model.pth')
    model = torch.load('model.pth')

方式 2: 只保存参数 (推荐)
    torch.save(model.state_dict(), 'model_weights.pth')
    model.load_state_dict(torch.load('model_weights.pth'))

推荐方式 2 的原因:
- 更灵活 (可以用于不同的模型定义)
- 更小的文件
- 更好的兼容性
""")

# =============================================================================
# 2. 基本示例
# =============================================================================
print("\n" + "=" * 60)
print("【2. 基本示例】")

# 创建模型
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet(10, 20, 5)

# 查看 state_dict
print("state_dict 内容:")
for name, param in model.state_dict().items():
    print(f"  {name}: {param.shape}")

# 创建保存目录
save_dir = 'outputs/models'
os.makedirs(save_dir, exist_ok=True)

# 保存参数
save_path = os.path.join(save_dir, 'simple_net.pth')
torch.save(model.state_dict(), save_path)
print(f"\n模型已保存到: {save_path}")

# 加载参数
new_model = SimpleNet(10, 20, 5)
new_model.load_state_dict(torch.load(save_path, weights_only=True))
print("模型已加载")

# 验证
x = torch.randn(3, 10)
with torch.no_grad():
    y1 = model(x)
    y2 = new_model(x)
print(f"输出一致: {torch.allclose(y1, y2)}")

# =============================================================================
# 3. 保存完整 Checkpoint
# =============================================================================
print("\n" + "=" * 60)
print("【3. 保存完整 Checkpoint】")

print("""
Checkpoint 应包含:
- 模型参数 (model.state_dict())
- 优化器状态 (optimizer.state_dict())
- 当前 epoch
- 最佳指标
- 学习率调度器状态
- 随机数状态 (可选)
""")

# 示例
model = SimpleNet(10, 20, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

# 模拟训练
epoch = 25
best_val_acc = 0.95
train_loss = 0.123

# 保存 checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
    'train_loss': train_loss,
}

checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint 已保存: {checkpoint_path}")

# 加载 checkpoint
checkpoint = torch.load(checkpoint_path, weights_only=False)

new_model = SimpleNet(10, 20, 5)
new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=10)

new_model.load_state_dict(checkpoint['model_state_dict'])
new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
new_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1

print(f"恢复训练从 Epoch {start_epoch}")
print(f"之前的最佳验证准确率: {checkpoint['best_val_acc']}")

# =============================================================================
# 4. GPU/CPU 之间的转换
# =============================================================================
print("\n" + "=" * 60)
print("【4. GPU/CPU 之间的转换】")

print("""
# GPU 上保存，CPU 上加载
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# CPU 上保存，GPU 上加载
model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
""")

# =============================================================================
# 5. 部分加载
# =============================================================================
print("\n" + "=" * 60)
print("【5. 部分加载 (迁移学习)】")

# 创建一个不完全匹配的模型
class LargerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)  # 匹配
        self.fc2 = nn.Linear(20, 5)   # 匹配
        self.fc3 = nn.Linear(5, 2)    # 新增层
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

larger_model = LargerNet()

# 加载部分权重
pretrained_dict = torch.load(save_path, weights_only=True)
model_dict = larger_model.state_dict()

# 过滤掉不匹配的键
pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                   if k in model_dict and model_dict[k].shape == v.shape}

print(f"可加载的层: {list(pretrained_dict.keys())}")

# 更新当前模型的 state_dict
model_dict.update(pretrained_dict)
larger_model.load_state_dict(model_dict)
print("部分权重已加载")

# =============================================================================
# 6. 保存用于推理的模型
# =============================================================================
print("\n" + "=" * 60)
print("【6. 保存用于推理的模型】")

print("""
方式 1: 只保存参数
    torch.save(model.state_dict(), 'model_inference.pth')

方式 2: TorchScript (推荐用于部署)
    scripted = torch.jit.script(model)
    scripted.save('model_scripted.pt')
    
    # 或使用 trace
    traced = torch.jit.trace(model, example_input)
    traced.save('model_traced.pt')

方式 3: ONNX 格式 (跨平台)
    torch.onnx.export(model, example_input, 'model.onnx')
""")

# TorchScript 示例
model.eval()
scripted_model = torch.jit.script(model)
scripted_path = os.path.join(save_dir, 'model_scripted.pt')
scripted_model.save(scripted_path)
print(f"TorchScript 模型已保存: {scripted_path}")

# 加载 TorchScript 模型
loaded_scripted = torch.jit.load(scripted_path)
x = torch.randn(3, 10)
with torch.no_grad():
    y = loaded_scripted(x)
print(f"TorchScript 推理输出: {y.shape}")

# =============================================================================
# 7. 最佳实践
# =============================================================================
print("\n" + "=" * 60)
print("【7. 最佳实践】")

print("""
╔════════════════════════════════════════════════════════════╗
║                    保存/加载最佳实践                       ║
╠════════════════════════════════════════════════════════════╣
║  1. 使用 .pth 或 .pt 扩展名                               ║
║  2. 保存完整 checkpoint，不只是 state_dict                ║
║  3. 记录模型配置（版本、超参数等）                        ║
║  4. 保存前调用 model.eval()                               ║
║  5. 使用 map_location 处理设备差异                        ║
║  6. 部署时使用 TorchScript 或 ONNX                        ║
║  7. 定期保存 checkpoint (每 N 个 epoch)                   ║
║  8. 保留最佳模型和最新模型                                ║
╚════════════════════════════════════════════════════════════╝
""")

# 实用的保存函数
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    """保存训练 checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'pytorch_version': torch.__version__,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """加载训练 checkpoint"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0)

print("实用函数 save_checkpoint() 和 load_checkpoint() 已定义")

# =============================================================================
# 8. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 保存一个模型并在新脚本中加载

2. 实现定期保存 checkpoint 的功能 (每 5 个 epoch)

3. 将模型转换为 TorchScript 格式

4. 实现加载预训练权重，冻结部分层，只训练最后一层

5. 解释为什么不推荐使用 torch.save(model, ...)
""")

# === 练习答案 ===
# 1
# torch.save(model.state_dict(), 'model.pth')
# # 新脚本
# model = SimpleNet(10, 20, 5)
# model.load_state_dict(torch.load('model.pth'))

# 2
# for epoch in range(100):
#     train(...)
#     if (epoch + 1) % 5 == 0:
#         save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
#                        f'checkpoint_epoch{epoch+1}.pth')

# 3
# model.eval()
# scripted = torch.jit.script(model)
# scripted.save('model.pt')

# 4
# model = PretrainedModel()
# model.load_state_dict(torch.load('pretrained.pth'))
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(512, 10)  # 替换最后一层
# optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 5
# 答案: 
# - 保存整个模型会序列化类定义
# - 依赖于具体的目录结构和类路径
# - 加载时如果类定义改变会失败
# - 文件更大，不够灵活

# 清理
import shutil
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
    print(f"\n已清理临时目录: {save_dir}")

print("\n✅ 模型保存与加载完成！")
print("🎉 Phase 3 全部模块完成！")
print("\n下一步：完成 MNIST 实战项目，然后进入 Phase 4")
