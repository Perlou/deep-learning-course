---
name: create-project
description: 创建深度学习实战项目，包含完整的项目结构和代码
---

# 创建实战项目技能

此技能用于在 `projects/` 目录下创建完整的深度学习实战项目。

## 项目结构

```
projects/{project-name}/
├── README.md           # 项目说明文档
├── main.py             # 主入口文件
├── model.py            # 模型定义
├── dataset.py          # 数据集加载与预处理
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── utils.py            # 工具函数
├── config.py           # 配置参数（可选）
├── requirements.txt    # 项目依赖（如有额外依赖）
└── outputs/            # 输出目录（模型、日志等）
    ├── models/
    └── logs/
```

## README.md 模板

````markdown
# {项目名称}

## 项目简介

简要描述项目目标和背景...

## 技术栈

- Python 3.10+
- PyTorch 2.x
- 其他依赖...

## 快速开始

### 1. 环境准备

```bash
# 在项目根目录
source venv/bin/activate
```
````

### 2. 数据准备

```bash
# 数据下载/准备步骤
```

### 3. 训练模型

```bash
python projects/{project-name}/train.py
```

### 4. 评估模型

```bash
python projects/{project-name}/evaluate.py
```

## 模型架构

描述模型架构，可配合图示...

## 实验结果

| 指标     | 值   |
| -------- | ---- |
| Accuracy | xx%  |
| Loss     | x.xx |

## 学习要点

1. 要点一
2. 要点二
3. 要点三

## 参考资料

- [论文/文档链接](url)

````

## 创建步骤

1. **确定项目主题**：明确项目要解决的问题和使用的技术
2. **创建目录结构**：按上述结构创建目录和文件
3. **实现核心模块**：
   - `model.py`：定义神经网络模型
   - `dataset.py`：数据加载和预处理
   - `train.py`：训练循环
   - `evaluate.py`：模型评估
4. **编写 README**：详细的项目文档
5. **测试运行**：确保代码可以正常运行

## 代码规范

### model.py 示例

```python
"""
模型定义
"""
import torch
import torch.nn as nn


class MyModel(nn.Module):
    """模型类"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)
````

### train.py 示例

```python
"""
训练脚本
"""
import torch
from torch.utils.data import DataLoader
from model import MyModel
from dataset import MyDataset


def train(config):
    """训练函数"""
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_dataset = MyDataset(...)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])

    # 模型初始化
    model = MyModel(...).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(config["epochs"]):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # 保存模型
    torch.save(model.state_dict(), "outputs/models/model.pth")


if __name__ == "__main__":
    config = {
        "batch_size": 32,
        "lr": 0.001,
        "epochs": 10,
    }
    train(config)
```

## 注意事项

- 确保代码可在 CPU 和 GPU 上运行
- 添加适当的日志输出
- 保存训练过程中的指标
- 使用相对路径处理文件
- 提供清晰的错误信息
