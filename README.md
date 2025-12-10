# 🧠 深度学习课程 (Python + PyTorch)

> **定制对象**：资深全栈工程师向 AI/算法岗位转型  
> **学习方式**：基于 Python + PyTorch 的理论与实践结合学习  
> **预计时长**：20-24 周（每周投入 10-15 小时）

---

## 🚀 快速开始

### 1. 环境准备

```bash
cd /deep-learning-course

# 激活虚拟环境（每次打开终端都需要执行）
source venv/bin/activate

# 如果还没创建虚拟环境，先执行：
# python3 -m venv venv && source venv/bin/activate
# pip install -r requirements.txt

# 验证环境
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

### 2. 按阶段学习

所有代码已按学习阶段组织，详见下方 [🎓 按阶段学习](#-按阶段学习) 部分。

```bash
# 第一个示例：NumPy 基础
python src/phase-1-python-basics/01-numpy-basics.py

# 或使用 Jupyter Notebook
jupyter lab notebooks/
```

---

## 📂 项目结构

```
deep-learning-course/
├── README.md                    # 课程介绍与快速开始
├── ROADMAP.md                   # 学习路线图（可视化）
├── LEARNING_PLAN.md             # 详细学习计划
├── CONCEPTS.md                  # 核心概念汇总文档
├── requirements.txt             # Python 依赖
├── src/
│   ├── phase-1-python-basics/   # 第1阶段：Python数据科学基础
│   ├── phase-2-math-foundations/ # 第2阶段：数学基础
│   ├── phase-3-pytorch-core/    # 第3阶段：PyTorch核心
│   ├── phase-4-neural-networks/ # 第4阶段：神经网络基础
│   ├── phase-5-cnn/             # 第5阶段：卷积神经网络
│   ├── phase-6-rnn-lstm/        # 第6阶段：循环神经网络
│   ├── phase-7-transformers/    # 第7阶段：Transformer
│   ├── phase-8-generative/      # 第8阶段：生成模型
│   ├── phase-9-optimization/    # 第9阶段：训练优化
│   ├── phase-10-cv-applications/ # 第10阶段：CV应用
│   ├── phase-11-nlp/            # 第11阶段：NLP
│   ├── phase-12-llm-frontier/   # 第12阶段：大模型前沿
│   └── utils/                   # 工具函数
├── notebooks/                   # Jupyter Notebooks
├── docs/                        # 学习笔记与论文阅读
├── data/                        # 数据集目录
└── projects/                    # 实战项目
```

**重要文档**：

- 📖 [ROADMAP.md](./ROADMAP.md) - 学习路线图
- 📝 [CONCEPTS.md](./CONCEPTS.md) - 核心概念文档
- 🗺️ [LEARNING_PLAN.md](./LEARNING_PLAN.md) - 完整学习计划

---

## 🎓 按阶段学习

每个阶段目录都包含独立的 README.md，详细说明该阶段的学习目标、核心概念和运行方式。

### 第 1 阶段：Python 数据科学基础

```bash
python src/phase-1-python-basics/01-numpy-basics.py
python src/phase-1-python-basics/04-pandas-basics.py
python src/phase-1-python-basics/06-matplotlib-basics.py
```

查看详情：[phase-1-python-basics/README.md](./src/phase-1-python-basics/README.md)

### 第 2 阶段：深度学习数学基础

```bash
python src/phase-2-math-foundations/01-vectors-matrices.py
python src/phase-2-math-foundations/03-derivatives-gradients.py
```

查看详情：[phase-2-math-foundations/README.md](./src/phase-2-math-foundations/README.md)

### 第 3 阶段：PyTorch 核心技能

```bash
python src/phase-3-pytorch-core/01-tensor-basics.py
python src/phase-3-pytorch-core/03-tensor-autograd.py
python src/phase-3-pytorch-core/09-training-loop.py
```

查看详情：[phase-3-pytorch-core/README.md](./src/phase-3-pytorch-core/README.md)

### 第 4 阶段：神经网络基础

```bash
python src/phase-4-neural-networks/01-perceptron.py
python src/phase-4-neural-networks/02-mlp-basic.py
```

查看详情：[phase-4-neural-networks/README.md](./src/phase-4-neural-networks/README.md)

### 第 5-12 阶段

查看完整的后续学习计划：[LEARNING_PLAN.md](./LEARNING_PLAN.md)

---

## 🛠️ 技术栈

- **Python 3.10+**
- **PyTorch 2.x**
- **NumPy / Pandas / Matplotlib**
- **Jupyter Lab**
- **HuggingFace Transformers**

---

## 📈 学习进度追踪

| 阶段     | 主题                | 文件数 | 状态      |
| -------- | ------------------- | ------ | --------- |
| Phase 1  | Python 数据科学基础 | 0/7    | ⏳ 待开始 |
| Phase 2  | 数学基础            | 0/7    | ⏳ 待开始 |
| Phase 3  | PyTorch 核心        | 0/10   | ⏳ 待开始 |
| Phase 4  | 神经网络基础        | 0/9    | ⏳ 待开始 |
| Phase 5  | CNN                 | 0/12   | ⏳ 待开始 |
| Phase 6  | RNN/LSTM            | 0/9    | ⏳ 待开始 |
| Phase 7  | Transformer         | 0/10   | ⏳ 待开始 |
| Phase 8  | 生成模型            | 0/8    | ⏳ 待开始 |
| Phase 9  | 训练优化            | 0/10   | ⏳ 待开始 |
| Phase 10 | CV 应用             | 0/8    | ⏳ 待开始 |
| Phase 11 | NLP                 | 0/8    | ⏳ 待开始 |
| Phase 12 | 大模型前沿          | 0/10   | ⏳ 待开始 |

---

**Good luck! 🚀**

有任何问题随时在代码注释或 `docs/` 中记录，养成持续学习和总结的习惯。
