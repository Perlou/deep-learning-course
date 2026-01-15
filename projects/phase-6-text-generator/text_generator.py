"""
Phase 6 实战项目 - 字符级文本生成器 (Character-Level Text Generator)

使用 LSTM 实现字符级语言模型，训练后可生成类似风格的中文文本（古诗词）

核心技术：
- 字符级编码与解码
- 多层 LSTM 网络
- 温度采样（Temperature Sampling）
- 梯度裁剪（Gradient Clipping）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ==================== 1. 数据处理 ====================


class CharDataset(Dataset):
    """字符级数据集"""

    def __init__(self, text, seq_length=100):
        """
        Args:
            text: 训练文本
            seq_length: 序列长度
        """
        self.seq_length = seq_length

        # 构建字符到索引的映射
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # 将文本转换为索引
        self.data = [self.char_to_idx[ch] for ch in text]

        print(f"字符集大小: {self.vocab_size}")
        print(f"训练文本长度: {len(self.data)} 字符")
        print(f"示例字符: {self.chars[:20]}")

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        """
        返回输入序列和目标序列
        目标序列是输入序列向后移动一个字符
        """
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ==================== 2. 模型定义 ====================


class CharLSTM(nn.Module):
    """字符级 LSTM 语言模型"""

    def __init__(
        self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3
    ):
        """
        Args:
            vocab_size: 词汇表大小
            embed_size: 嵌入维度
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            dropout: Dropout 比例
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM 层
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        前向传播

        Args:
            x: 输入序列 (batch_size, seq_length)
            hidden: 隐藏状态

        Returns:
            output: 输出 (batch_size, seq_length, vocab_size)
            hidden: 新的隐藏状态
        """
        # 嵌入: (batch_size, seq_length, embed_size)
        embed = self.embedding(x)

        # LSTM: (batch_size, seq_length, hidden_size)
        lstm_out, hidden = self.lstm(embed, hidden)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 全连接: (batch_size, seq_length, vocab_size)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# ==================== 3. 训练函数 ====================


def train_epoch(model, train_loader, criterion, optimizer, clip_grad=5.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        # 初始化隐藏状态
        hidden = model.init_hidden(batch_size)

        # 前向传播
        output, hidden = model(x, hidden)

        # 计算损失
        # output: (batch_size, seq_length, vocab_size)
        # y: (batch_size, seq_length)
        loss = criterion(output.transpose(1, 2), y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


# ==================== 4. 文本生成函数 ====================


def generate_text(model, dataset, start_str="春", length=200, temperature=0.8):
    """
    生成文本

    Args:
        model: 训练好的模型
        dataset: 数据集（用于字符映射）
        start_str: 起始字符串
        length: 生成长度
        temperature: 温度参数（越高越随机，越低越确定）

    Returns:
        生成的文本
    """
    model.eval()

    # 将起始字符串转换为索引
    chars = [dataset.char_to_idx.get(ch, 0) for ch in start_str]
    input_seq = torch.tensor([chars[-1]], dtype=torch.long).unsqueeze(0).to(device)

    # 初始化隐藏状态
    hidden = model.init_hidden(1)

    generated = start_str

    with torch.no_grad():
        for _ in range(length):
            # 前向传播
            output, hidden = model(input_seq, hidden)

            # 获取最后一个时间步的输出
            output = output[:, -1, :] / temperature

            # 应用 softmax 得到概率分布
            probs = torch.softmax(output, dim=-1).cpu().numpy()[0]

            # 按概率采样下一个字符
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = dataset.idx_to_char[next_idx]

            generated += next_char

            # 更新输入
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return generated


# ==================== 5. 主训练流程 ====================


def main():
    # 配置参数
    config = {
        "seq_length": 100,  # 序列长度
        "embed_size": 128,  # 嵌入维度
        "hidden_size": 256,  # LSTM 隐藏层大小
        "num_layers": 2,  # LSTM 层数
        "dropout": 0.3,  # Dropout
        "batch_size": 64,  # 批量大小
        "learning_rate": 0.001,  # 学习率
        "num_epochs": 50,  # 训练轮数
        "clip_grad": 5.0,  # 梯度裁剪阈值
    }

    # 创建输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)

    # 加载数据
    data_path = Path("data/training_text.txt")
    if not data_path.exists():
        print(f"错误: 找不到训练数据文件 {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n{'=' * 60}")
    print("字符级文本生成 - LSTM 语言模型")
    print(f"{'=' * 60}\n")

    # 创建数据集
    dataset = CharDataset(text, seq_length=config["seq_length"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # 创建模型
    model = CharLSTM(
        vocab_size=dataset.vocab_size,
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    print(f"\n模型参数:")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 训练循环
    print(f"\n开始训练...")
    print(f"{'=' * 60}\n")

    train_losses = []
    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}]")

        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config["clip_grad"]
        )
        train_losses.append(train_loss)

        print(f"  平均损失: {train_loss:.4f}")

        # 学习率调度
        scheduler.step(train_loss)

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "config": config,
                    "char_to_idx": dataset.char_to_idx,
                    "idx_to_char": dataset.idx_to_char,
                },
                checkpoint_dir / "best_model.pth",
            )
            print(f"  ✓ 保存最佳模型")

        # 每 10 个 epoch 生成一些文本
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n  生成文本示例:")
            for temp in [0.5, 0.8, 1.0]:
                generated = generate_text(
                    model, dataset, start_str="春", length=100, temperature=temp
                )
                print(f"    温度={temp}: {generated[:50]}...")
            print()

    print(f"\n{'=' * 60}")
    print("训练完成!")
    print(f"{'=' * 60}\n")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title("训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(output_dir / "training_loss.png", dpi=300, bbox_inches="tight")
    print(f"✓ 训练曲线已保存到 {output_dir / 'training_loss.png'}")

    # 生成最终文本
    print(f"\n最终文本生成:")
    print(f"{'=' * 60}\n")

    start_strings = ["春", "月", "山", "水", "风"]
    temperatures = [0.5, 0.8, 1.0, 1.2]

    with open(generated_dir / "generated_samples.txt", "w", encoding="utf-8") as f:
        for start_str in start_strings:
            f.write(f"\n起始字符: {start_str}\n")
            f.write(f"{'-' * 60}\n\n")

            for temp in temperatures:
                generated = generate_text(
                    model, dataset, start_str=start_str, length=200, temperature=temp
                )
                f.write(f"温度 {temp}:\n{generated}\n\n")

                if temp == 0.8:  # 打印一个示例
                    print(f"起始字符 '{start_str}' (温度={temp}):")
                    print(f"{generated}\n")

    print(f"✓ 生成文本已保存到 {generated_dir / 'generated_samples.txt'}")

    print(f"\n{'=' * 60}")
    print("项目完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
