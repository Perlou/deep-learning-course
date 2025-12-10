# 深度学习完整学习计划

> **定制对象**：资深 Web 全栈开发工程师向 AI/算法岗位转型  
> **学习方式**：基于 Python + PyTorch 的理论与实践结合学习  
> **预计时长**：20-24 周（每周投入 10-15 小时）  
> **当前进度**：🔄 准备开始第 1 阶段

---

## 📊 当前学习状态评估

### 🎯 学习目标

1. 掌握深度学习核心原理与数学基础
2. 熟练使用 PyTorch 实现各类神经网络
3. 具备独立完成 CV/NLP 项目的能力
4. 理解大模型原理，具备微调和应用能力
5. 为算法岗位面试做好准备

---

## 🗺️ 详细学习计划

### 阶段 1：Python 数据科学基础 (1 周)

> **目标**：熟练使用 Python 数据科学工具栈

#### 第 1 周：NumPy、Pandas、Matplotlib

- [ ] **NumPy 核心操作**

  - 创建 `01-numpy-basics.py`：数组创建、索引、切片
  - 创建 `02-numpy-operations.py`：矩阵运算、广播机制
  - 创建 `03-numpy-linear-algebra.py`：线性代数操作

- [ ] **Pandas 数据处理**

  - 创建 `04-pandas-basics.py`：DataFrame 操作
  - 创建 `05-pandas-preprocessing.py`：数据清洗、缺失值处理

- [ ] **Matplotlib 可视化**

  - 创建 `06-matplotlib-basics.py`：基础绑图
  - 创建 `07-visualization-advanced.py`：多子图、自定义样式

- [ ] **实战项目**：探索性数据分析（Titanic/Iris 数据集）

---

### 阶段 2：深度学习数学基础 (1-2 周)

> **目标**：掌握深度学习必需的数学知识

#### 第 2 周：线性代数与微积分

- [ ] **线性代数精要**

  - 创建 `01-vectors-matrices.py`：向量空间、矩阵运算
  - 创建 `02-eigenvalue-svd.py`：特征分解、SVD
  - 文档：`docs/LINEAR_ALGEBRA_DL.md`

- [ ] **微积分与优化**
  - 创建 `03-derivatives-gradients.py`：偏导数、梯度
  - 创建 `04-chain-rule.py`：链式法则（反向传播基础）
  - 创建 `05-optimization-basics.py`：梯度下降可视化

#### 第 3 周：概率论基础

- [ ] **概率论与信息论**
  - 创建 `06-probability-basics.py`：概率分布
  - 创建 `07-entropy-kl-divergence.py`：熵、KL 散度、交叉熵
  - 文档：`docs/PROBABILITY_FOR_DL.md`

---

### 阶段 3：PyTorch 核心技能 (1 周)

> **目标**：深入理解 PyTorch 框架

#### 第 4 周：PyTorch 基础

- [ ] **Tensor 核心**

  - 创建 `01-tensor-basics.py`：创建、属性、设备
  - 创建 `02-tensor-operations.py`：运算、索引、变形
  - 创建 `03-tensor-autograd.py`：自动微分机制

- [ ] **核心模块**

  - 创建 `04-nn-module.py`：nn.Module 深入
  - 创建 `05-loss-functions.py`：损失函数详解
  - 创建 `06-optimizers.py`：优化器原理与使用

- [ ] **数据处理**

  - 创建 `07-dataset-dataloader.py`：Dataset 和 DataLoader
  - 创建 `08-data-augmentation.py`：数据增强技术

- [ ] **训练循环**

  - 创建 `09-training-loop.py`：完整训练流程
  - 创建 `10-model-save-load.py`：模型保存与加载

- [ ] **实战项目**：MNIST 手写数字分类

---

### 阶段 4：神经网络基础 (1 周)

> **目标**：理解神经网络核心原理

#### 第 5 周：MLP 与正则化

- [ ] **感知机与多层网络**

  - 创建 `01-perceptron.py`：单层感知机
  - 创建 `02-mlp-basic.py`：多层感知机
  - 创建 `03-forward-backward.py`：前向传播与反向传播

- [ ] **激活函数**

  - 创建 `04-activation-functions.py`：ReLU、Sigmoid、Tanh、GELU
  - 创建 `05-activation-comparison.py`：激活函数对比分析

- [ ] **正则化技术**

  - 创建 `06-dropout.py`：Dropout 原理与实现
  - 创建 `07-batch-normalization.py`：BatchNorm 详解
  - 创建 `08-weight-regularization.py`：L1/L2 正则化

- [ ] **权重初始化**

  - 创建 `09-weight-init.py`：Xavier、He 初始化

- [ ] **实战项目**：房价预测（回归任务）

---

### 阶段 5：卷积神经网络 CNN (2 周)

> **目标**：掌握计算机视觉基础架构

#### 第 6 周：CNN 基础

- [ ] **卷积操作**

  - 创建 `01-convolution-basics.py`：卷积核、步长、填充
  - 创建 `02-pooling-layers.py`：池化操作
  - 创建 `03-receptive-field.py`：感受野计算

- [ ] **经典架构**
  - 创建 `04-lenet.py`：LeNet 实现
  - 创建 `05-alexnet.py`：AlexNet 架构
  - 创建 `06-vggnet.py`：VGG 深度网络

#### 第 7 周：现代 CNN

- [ ] **残差网络**

  - 创建 `07-resnet.py`：ResNet 及残差连接
  - 创建 `08-resnet-variants.py`：ResNet18/34/50/101

- [ ] **高效架构**

  - 创建 `09-mobilenet.py`：MobileNet 轻量化
  - 创建 `10-efficientnet.py`：EfficientNet

- [ ] **特征可视化**

  - 创建 `11-feature-visualization.py`：特征图可视化
  - 创建 `12-grad-cam.py`：Grad-CAM 热力图

- [ ] **实战项目**：CIFAR-10 分类达到 90%+ 准确率

---

### 阶段 6：循环神经网络 RNN/LSTM (2 周)

> **目标**：掌握序列建模技术

#### 第 8 周：RNN 基础

- [ ] **RNN 原理**

  - 创建 `01-rnn-basics.py`：RNN 结构与前向传播
  - 创建 `02-bptt.py`：时间反向传播
  - 创建 `03-vanishing-gradient.py`：梯度消失问题

- [ ] **LSTM 与 GRU**
  - 创建 `04-lstm.py`：LSTM 门控机制
  - 创建 `05-gru.py`：GRU 简化结构
  - 创建 `06-bidirectional.py`：双向 RNN

#### 第 9 周：序列应用

- [ ] **序列建模**

  - 创建 `07-seq2seq.py`：序列到序列模型
  - 创建 `08-attention-basic.py`：基础注意力机制
  - 创建 `09-time-series.py`：时间序列预测

- [ ] **实战项目**
  - 文本生成：字符级语言模型
  - 股票预测：时间序列预测

---

### 阶段 7：注意力机制与 Transformer (2-3 周)

> **目标**：深入理解现代深度学习核心架构

#### 第 10 周：注意力机制

- [ ] **自注意力**
  - 创建 `01-self-attention.py`：自注意力从零实现
  - 创建 `02-multi-head-attention.py`：多头注意力
  - 创建 `03-positional-encoding.py`：位置编码

#### 第 11 周：Transformer 架构

- [ ] **Transformer 核心**

  - 创建 `04-transformer-encoder.py`：编码器实现
  - 创建 `05-transformer-decoder.py`：解码器实现
  - 创建 `06-transformer-full.py`：完整 Transformer

- [ ] **论文阅读**
  - 文档：`docs/ATTENTION_IS_ALL_YOU_NEED.md`

#### 第 12 周：预训练模型

- [ ] **BERT 家族**

  - 创建 `07-bert-architecture.py`：BERT 结构理解
  - 创建 `08-bert-finetuning.py`：微调实践

- [ ] **GPT 家族**

  - 创建 `09-gpt-architecture.py`：GPT 架构
  - 创建 `10-gpt-generation.py`：文本生成

- [ ] **实战项目**
  - 机器翻译：Transformer 翻译模型
  - 文本分类：BERT 微调情感分析

---

### 阶段 8：生成模型 (2 周)

> **目标**：掌握主流生成模型

#### 第 13 周：GAN 与 VAE

- [ ] **GAN 系列**

  - 创建 `01-gan-basics.py`：原始 GAN
  - 创建 `02-dcgan.py`：深度卷积 GAN
  - 创建 `03-wgan.py`：Wasserstein GAN

- [ ] **变分自编码器**
  - 创建 `04-autoencoder.py`：自编码器基础
  - 创建 `05-vae.py`：VAE 原理与实现

#### 第 14 周：Diffusion Models

- [ ] **扩散模型**

  - 创建 `06-diffusion-theory.py`：DDPM 理论
  - 创建 `07-diffusion-implementation.py`：简易实现
  - 创建 `08-stable-diffusion-intro.py`：Stable Diffusion 架构

- [ ] **实战项目**：GAN 生成人脸/动漫图像

---

### 阶段 9：训练技巧与优化 (1-2 周)

> **目标**：掌握工业级训练技巧

#### 第 15 周：高级训练技术

- [ ] **优化器深入**

  - 创建 `01-sgd-momentum.py`：SGD 及动量
  - 创建 `02-adam-variants.py`：Adam、AdamW
  - 创建 `03-lr-schedulers.py`：学习率调度

- [ ] **训练技巧**

  - 创建 `04-gradient-clipping.py`：梯度裁剪
  - 创建 `05-mixed-precision.py`：混合精度训练
  - 创建 `06-gradient-accumulation.py`：梯度累积

- [ ] **分布式训练**

  - 创建 `07-data-parallel.py`：数据并行
  - 创建 `08-distributed-training.py`：分布式训练基础

- [ ] **超参数调优**
  - 创建 `09-hyperparameter-tuning.py`：网格搜索、随机搜索
  - 创建 `10-optuna-integration.py`：Optuna 自动调参

---

### 阶段 10：计算机视觉应用 (2 周)

> **目标**：掌握 CV 核心任务

#### 第 16-17 周：CV 实战

- [ ] **目标检测**

  - 创建 `01-detection-basics.py`：检测任务概述
  - 创建 `02-yolo-v8.py`：YOLO 系列
  - 创建 `03-rcnn-family.py`：R-CNN 系列

- [ ] **图像分割**

  - 创建 `04-semantic-segmentation.py`：语义分割
  - 创建 `05-unet.py`：U-Net 架构
  - 创建 `06-instance-segmentation.py`：实例分割

- [ ] **其他任务**

  - 创建 `07-pose-estimation.py`：姿态估计
  - 创建 `08-face-recognition.py`：人脸识别

- [ ] **实战项目**：目标检测系统（YOLOv8）

---

### 阶段 11：自然语言处理 NLP (2 周)

> **目标**：掌握 NLP 核心技术

#### 第 18-19 周：NLP 实战

- [ ] **文本表示**

  - 创建 `01-word2vec.py`：词向量训练
  - 创建 `02-embeddings-advanced.py`：词嵌入分析

- [ ] **NLP 任务**

  - 创建 `03-text-classification.py`：文本分类
  - 创建 `04-ner.py`：命名实体识别
  - 创建 `05-question-answering.py`：问答系统

- [ ] **预训练模型应用**

  - 创建 `06-huggingface-basics.py`：HuggingFace 入门
  - 创建 `07-transformer-finetuning.py`：Transformer 微调
  - 创建 `08-peft-lora.py`：参数高效微调

- [ ] **实战项目**：情感分析系统

---

### 阶段 12：大模型与前沿技术 (2-3 周)

> **目标**：理解大模型训练与前沿技术

#### 第 20-22 周：LLM 前沿

- [ ] **大模型基础**

  - 创建 `01-llm-architecture.py`：LLM 架构理解
  - 创建 `02-tokenization-advanced.py`：分词器详解
  - 创建 `03-flash-attention.py`：FlashAttention 原理

- [ ] **大模型训练**

  - 创建 `04-pre-training-basics.py`：预训练基础
  - 创建 `05-instruction-tuning.py`：指令微调
  - 创建 `06-rlhf-basics.py`：RLHF 原理

- [ ] **高效推理**

  - 创建 `07-quantization.py`：模型量化
  - 创建 `08-inference-optimization.py`：推理优化

- [ ] **前沿方向**

  - 创建 `09-multimodal.py`：多模态模型
  - 创建 `10-agents-tools.py`：Agent 与工具调用
  - 文档：`docs/FRONTIER_SURVEY.md`

- [ ] **实战项目**：使用 LoRA 微调开源模型

---

## 🎓 配套学习资源

### 必读书籍

- [ ] 《深度学习》(花书) - Goodfellow
- [ ] 《动手学深度学习》 - 李沐
- [ ] 《PyTorch 深度学习实战》

### 推荐课程

- [ ] **Stanford CS231n**：计算机视觉
- [ ] **Stanford CS224n**：自然语言处理
- [ ] **李宏毅机器学习**：深度学习入门
- [ ] **李沐 动手学深度学习**：PyTorch 实践

### 关键论文阅读

| 阶段     | 论文                          | 必读程度   |
| -------- | ----------------------------- | ---------- |
| Phase 5  | AlexNet, ResNet, EfficientNet | ⭐⭐⭐     |
| Phase 6  | LSTM, Seq2Seq with Attention  | ⭐⭐⭐     |
| Phase 7  | Attention Is All You Need     | ⭐⭐⭐⭐⭐ |
| Phase 7  | BERT, GPT 系列                | ⭐⭐⭐⭐⭐ |
| Phase 8  | GAN, VAE, DDPM                | ⭐⭐⭐⭐   |
| Phase 12 | LoRA, Flash Attention         | ⭐⭐⭐⭐   |

---

## 📝 学习方法建议

### 1️⃣ 代码先行，理论跟进

每个概念先运行代码、观察结果，再深入理解原理。

### 2️⃣ 构建个人知识库

- 将所有学习笔记放入 `docs/` 目录
- 定期复盘总结到 `CONCEPTS.md`

### 3️⃣ 项目驱动学习

每个阶段末尾完成对应实战项目，巩固所学。

### 4️⃣ 论文阅读习惯

从 Phase 5 开始，每周阅读 1-2 篇经典论文。

### 5️⃣ 建立评估习惯

- 每个实验记录准确率、损失曲线
- 对比不同方法的效果

---

## 🎯 学习里程碑检查

### 第 4 周后（Phase 1-3 完成）

- [ ] NumPy/Pandas 熟练操作
- [ ] 理解梯度下降、反向传播原理
- [ ] PyTorch 基础模型能独立实现
- [ ] 完成 MNIST 分类项目

### 第 8 周后（Phase 5-6 完成）

- [ ] CNN 经典架构理解并能实现
- [ ] RNN/LSTM 序列建模掌握
- [ ] 完成 CIFAR-10 分类项目
- [ ] 完成文本生成项目

### 第 12 周后（Phase 7-8 完成）

- [ ] Transformer 架构深入理解
- [ ] 能够微调 BERT/GPT 模型
- [ ] GAN/VAE/Diffusion 原理掌握
- [ ] 完成机器翻译项目

### 第 18 周后（Phase 9-11 完成）

- [ ] 掌握工业级训练技巧
- [ ] CV 核心任务（检测、分割）能独立完成
- [ ] NLP 任务能使用预训练模型完成
- [ ] 完成目标检测和情感分析项目

### 第 24 周后（全部完成）

- [ ] 理解大模型训练流程
- [ ] 能够进行参数高效微调
- [ ] 具备独立设计深度学习系统的能力
- [ ] 完成 LLM 微调项目

---

## 💼 职业发展建议

### 作品集建设

将以下项目放入 GitHub：

1. `deep-learning-course`（学习笔记库）
2. 图像分类系统（ResNet）
3. 目标检测系统（YOLO）
4. 情感分析系统（BERT）
5. LLM 微调项目（LoRA）

### 技能关键词（简历优化）

完成本计划后，可以突出：

- PyTorch 深度学习开发
- CNN/RNN/Transformer 架构实现
- 计算机视觉（分类、检测、分割）
- NLP（文本分类、NER、问答）
- 大模型微调（LoRA、PEFT）
- 分布式训练与推理优化

### 目标岗位

- 深度学习工程师
- 算法工程师
- CV/NLP 算法工程师
- AI 研发工程师
- 大模型应用工程师

---

## 📊 进度追踪

| 周次  | 阶段     | 完成文件数 | 实战项目       | 状态      |
| ----- | -------- | ---------- | -------------- | --------- |
| 1     | Phase 1  | 0/7        | 探索性数据分析 | ⏳ 待开始 |
| 2-3   | Phase 2  | 0/7        | -              | ⏳ 待开始 |
| 4     | Phase 3  | 0/10       | MNIST 分类     | ⏳ 待开始 |
| 5     | Phase 4  | 0/9        | 房价预测       | ⏳ 待开始 |
| 6-7   | Phase 5  | 0/12       | CIFAR-10       | ⏳ 待开始 |
| 8-9   | Phase 6  | 0/9        | 文本生成       | ⏳ 待开始 |
| 10-12 | Phase 7  | 0/10       | 机器翻译       | ⏳ 待开始 |
| 13-14 | Phase 8  | 0/8        | 图像生成       | ⏳ 待开始 |
| 15    | Phase 9  | 0/10       | -              | ⏳ 待开始 |
| 16-17 | Phase 10 | 0/8        | 目标检测       | ⏳ 待开始 |
| 18-19 | Phase 11 | 0/8        | 情感分析       | ⏳ 待开始 |
| 20-22 | Phase 12 | 0/10       | LLM 微调       | ⏳ 待开始 |

---

**Good luck! 🚀**

有任何问题随时在代码注释或 `docs/` 中记录，养成持续学习和总结的习惯。
