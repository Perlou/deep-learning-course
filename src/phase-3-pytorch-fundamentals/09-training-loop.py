"""
09-training-loop.py
Phase 3: PyTorch 核心技能

完整训练流程 - 从数据到模型

学习目标：
1. 掌握完整的训练循环结构
2. 理解验证和评估流程
3. 了解训练技巧和最佳实践
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time

print("=" * 60)
print("PyTorch 核心技能 - 完整训练流程")
print("=" * 60)

# =============================================================================
# 1. 训练流程概述
# =============================================================================
print("\n【1. 训练流程概述】")

print("""
完整训练流程:

1. 准备数据
   - 加载数据集
   - 划分训练/验证/测试集
   - 创建 DataLoader

2. 创建模型
   - 定义网络结构
   - 初始化权重
   - 移动到设备 (GPU)

3. 定义损失和优化器
   - 选择损失函数
   - 选择优化器和学习率
   - 设置学习率调度器

4. 训练循环
   - 前向传播
   - 计算损失
   - 反向传播
   - 更新参数

5. 验证和评估
   - 在验证集上评估
   - 保存最佳模型
   - 早停策略

6. 测试
   - 加载最佳模型
   - 在测试集上评估
""")

# =============================================================================
# 2. 准备数据
# =============================================================================
print("\n" + "=" * 60)
print("【2. 准备数据】")

# 生成模拟数据
torch.manual_seed(42)
n_samples = 1000
n_features = 20
n_classes = 3

X = torch.randn(n_samples, n_features)
y = torch.randint(0, n_classes, (n_samples,))

# 创建数据集
dataset = TensorDataset(X, y)

# 划分数据集
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")
print(f"测试集: {len(test_dataset)}")

# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 3. 创建模型
# =============================================================================
print("\n" + "=" * 60)
print("【3. 创建模型】")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
print(f"使用设备: {device}")

# 创建模型
model = MLP(n_features, 64, n_classes).to(device)
print(f"模型:\n{model}")

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")

# =============================================================================
# 4. 定义损失和优化器
# =============================================================================
print("\n" + "=" * 60)
print("【4. 定义损失和优化器】")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print(f"损失函数: {criterion}")
print(f"优化器: AdamW, lr=0.001, weight_decay=0.01")
print(f"学习率调度: CosineAnnealing, T_max=50")

# =============================================================================
# 5. 训练和验证函数
# =============================================================================
print("\n" + "=" * 60)
print("【5. 训练和验证函数】")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 (可选)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    return total_loss / total, correct / total

print("训练和验证函数已定义")

# =============================================================================
# 6. 完整训练循环
# =============================================================================
print("\n" + "=" * 60)
print("【6. 完整训练循环】")

# 训练配置
n_epochs = 30
best_val_acc = 0
best_model_state = None
patience = 5
patience_counter = 0

# 记录历史
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'lr': []
}

print("\n开始训练...")
start_time = time.time()

for epoch in range(n_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # 验证
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    # 打印进度
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d}/{n_epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
              f"LR={current_lr:.6f}")
    
    # 早停
    if patience_counter >= patience:
        print(f"\n早停于 Epoch {epoch}")
        break

elapsed_time = time.time() - start_time
print(f"\n训练完成! 用时: {elapsed_time:.2f}s")
print(f"最佳验证准确率: {best_val_acc:.4f}")

# =============================================================================
# 7. 测试评估
# =============================================================================
print("\n" + "=" * 60)
print("【7. 测试评估】")

# 加载最佳模型
model.load_state_dict(best_model_state)

# 测试
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集 Accuracy: {test_acc:.4f}")

# =============================================================================
# 8. 训练技巧
# =============================================================================
print("\n" + "=" * 60)
print("【8. 训练技巧】")

print("""
╔════════════════════════════════════════════════════════════╗
║                    训练技巧                                ║
╠════════════════════════════════════════════════════════════╣
║  1. 使用 model.train() 和 model.eval()                    ║
║  2. 在 torch.no_grad() 中验证                             ║
║  3. 使用梯度裁剪防止梯度爆炸                               ║
║  4. 使用学习率调度器                                       ║
║  5. 实现早停策略                                           ║
║  6. 保存最佳模型                                           ║
║  7. 记录训练历史 (TensorBoard/W&B)                        ║
║  8. 使用混合精度训练 (torch.cuda.amp)                     ║
╚════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 9. 混合精度训练
# =============================================================================
print("\n" + "=" * 60)
print("【9. 混合精度训练 (可选)】")

amp_code = """
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in train_loader:
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        outputs = model(x)
        loss = criterion(outputs, y)
    
    # 缩放梯度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""
print(amp_code)

# =============================================================================
# 10. 练习题
# =============================================================================
print("\n" + "=" * 60)
print("【练习题】")
print("=" * 60)

print("""
1. 修改训练循环，添加 TensorBoard 日志记录

2. 实现学习率预热 (Warmup)

3. 添加混合精度训练

4. 实现梯度累积 (Gradient Accumulation)

5. 使用 tqdm 显示进度条
""")

# === 练习答案 ===
# 1 TensorBoard
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/experiment1')
# writer.add_scalar('Loss/train', train_loss, epoch)
# writer.add_scalar('Accuracy/val', val_acc, epoch)

# 2 Warmup
# def lr_lambda(epoch):
#     warmup_epochs = 5
#     if epoch < warmup_epochs:
#         return epoch / warmup_epochs
#     return 1.0
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 3 混合精度见上文

# 4 梯度累积
# accumulation_steps = 4
# for i, (x, y) in enumerate(train_loader):
#     loss = criterion(model(x), y) / accumulation_steps
#     loss.backward()
#     if (i + 1) % accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()

# 5 tqdm
# from tqdm import tqdm
# for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
#     ...

print("\n✅ 完整训练流程完成！")
print("下一步：10-model-save-load.py - 模型保存与加载")
