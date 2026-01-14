---
description: 训练深度学习模型的标准流程
---

# 训练模型工作流

使用此工作流进行模型训练。

## 步骤

1. 激活虚拟环境

```bash
source venv/bin/activate
```

2. 检查 GPU 可用性（可选）

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

// turbo 3. 运行训练脚本

```bash
python {训练脚本路径}
```

4. 监控训练过程：

   - 观察 loss 是否下降
   - 检查是否有警告或错误
   - 如果训练时间长，考虑使用 `nohup` 或 `tmux`

5. 训练完成后，检查输出：

   - 模型文件是否保存
   - 日志文件是否生成
   - 最终指标是否合理

6. 如果遇到问题，参考 `.agent/skills/debug-training/SKILL.md` 进行诊断
