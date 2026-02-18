"""
01_prepare_data.py — 数据准备脚本
==================================

下载并准备 ClearMind 训练所需的全部数据:
  1. 预训练数据: 中英文混合文本语料
  2. SFT 数据: 指令-回复对话数据
  3. DPO 数据: 偏好对齐数据

使用方法:
  python scripts/01_prepare_data.py
  python scripts/01_prepare_data.py --data_dir data/
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_sample_pretrain_data(output_path: str, n_samples: int = 5000):
    """创建样例预训练数据

    生成中英混合文本，用于快速验证训练流程。
    正式训练时应替换为真实语料。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 中英文示例文本集合
    chinese_texts = [
        "深度学习是机器学习的一个子领域，它利用多层神经网络来学习数据的层次化表示。",
        "Transformer架构于2017年被提出，彻底改变了自然语言处理领域的研究方向。",
        "注意力机制允许模型在处理序列数据时，关注输入中最相关的部分。",
        "预训练语言模型通过在大量文本数据上进行无监督学习，获得了强大的语言理解能力。",
        "反向传播算法通过链式法则计算损失函数对每个参数的梯度，是训练神经网络的核心方法。",
        "卷积神经网络通过局部连接和权重共享，有效地提取图像的空间特征。",
        "递归神经网络能够处理变长序列，但存在梯度消失和梯度爆炸的问题。",
        "LSTM通过门控机制解决了传统RNN的长期依赖问题。",
        "批归一化技术通过对每个小批量数据进行归一化，加速了网络训练过程。",
        "Dropout是一种正则化技术，通过在训练过程中随机丢弃部分神经元来防止过拟合。",
        "生成对抗网络由生成器和判别器两个网络组成，通过对抗训练来学习数据分布。",
        "强化学习是一种通过与环境交互来学习最优策略的机器学习方法。",
        "迁移学习允许将在一个任务上学到的知识应用到另一个相关任务上。",
        "自监督学习通过设计预训练任务，从无标注数据中学习有用的表示。",
        "知识蒸馏是一种模型压缩技术，通过让小模型学习大模型的输出来提升性能。",
        "多头注意力机制让模型能够同时关注不同位置的不同信息子空间。",
        "位置编码为Transformer提供了序列中元素的位置信息。",
        "残差连接帮助解决深度网络中的退化问题，使得训练更深的网络成为可能。",
        "学习率调度策略通过动态调整学习率来改善训练过程。",
        "梯度累积技术允许在有限的GPU内存下模拟大批量训练。",
    ]

    english_texts = [
        "Deep learning is a subset of machine learning that uses multi-layered neural networks to learn hierarchical representations of data.",
        "The Transformer architecture, introduced in 2017, has revolutionized the field of natural language processing.",
        "Attention mechanisms allow models to focus on the most relevant parts of the input when processing sequential data.",
        "Pre-trained language models acquire powerful language understanding capabilities through unsupervised learning on large text corpora.",
        "Backpropagation uses the chain rule to compute gradients of the loss function with respect to each parameter.",
        "Convolutional neural networks effectively extract spatial features from images through local connectivity and weight sharing.",
        "Recurrent neural networks can process variable-length sequences but suffer from vanishing and exploding gradient problems.",
        "LSTM addresses the long-term dependency problem of traditional RNNs through gating mechanisms.",
        "Batch normalization accelerates network training by normalizing each mini-batch of data.",
        "Dropout is a regularization technique that prevents overfitting by randomly dropping neurons during training.",
        "Generative adversarial networks consist of a generator and discriminator that learn data distributions through adversarial training.",
        "Reinforcement learning is a machine learning approach that learns optimal policies through interaction with an environment.",
        "Transfer learning allows knowledge learned from one task to be applied to another related task.",
        "Self-supervised learning designs pretext tasks to learn useful representations from unlabeled data.",
        "Knowledge distillation is a model compression technique that improves a small model by learning from a large model's outputs.",
        "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
        "Positional encoding provides position information of elements in the sequence to the Transformer.",
        "Residual connections help solve the degradation problem in deep networks, enabling the training of much deeper architectures.",
        "Learning rate scheduling strategies improve the training process by dynamically adjusting the learning rate.",
        "Gradient accumulation techniques allow simulating large batch training with limited GPU memory.",
    ]

    # 组合并重复生成数据
    all_texts = chinese_texts + english_texts
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_samples):
            text = all_texts[i % len(all_texts)]
            # 稍微变化文本, 避免完全重复
            if i >= len(all_texts):
                text = (
                    f"第{i + 1}条: {text}" if i % 2 == 0 else f"Sample {i + 1}: {text}"
                )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"✅ 预训练数据: {output_path}")
    print(f"   样本数: {n_samples:,}")


def create_sample_sft_data(output_path: str, n_samples: int = 2000):
    """创建样例 SFT 数据"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sft_samples = [
        {
            "instruction": "什么是深度学习？",
            "output": "深度学习是机器学习的一个分支，它使用由多层神经元组成的深度神经网络来自动学习数据中的特征和模式。与传统机器学习不同，深度学习能够从原始数据中自动提取层次化的特征表示，无需人工设计特征。",
        },
        {
            "instruction": "请解释Transformer架构的核心思想。",
            "output": "Transformer架构的核心思想是自注意力机制(Self-Attention)。它摒弃了传统的循环结构，完全依赖注意力机制来捕捉序列中任意两个位置之间的依赖关系。这使得模型能够并行处理整个序列，大大提高了训练效率。",
        },
        {
            "instruction": "什么是梯度下降？",
            "output": "梯度下降是一种优化算法，用于最小化目标函数。它通过计算损失函数相对于模型参数的梯度，然后沿梯度的反方向更新参数来逐步降低损失。常见的变体包括随机梯度下降(SGD)、小批量梯度下降和Adam优化器等。",
        },
        {
            "instruction": "Explain what is machine learning.",
            "output": "Machine learning is a branch of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed. It works by identifying patterns in training data and applying those patterns to new, unseen data.",
        },
        {
            "instruction": "What is the attention mechanism?",
            "output": "The attention mechanism is a technique that allows neural networks to focus on specific parts of the input when producing each element of the output. It computes a weighted sum of values based on the similarity between queries and keys, enabling the model to selectively attend to relevant information.",
        },
        {
            "instruction": "解释什么是反向传播算法。",
            "output": "反向传播算法是训练神经网络的核心方法。它首先通过前向传播计算网络的输出和损失，然后从输出层开始，利用链式法则逐层反向计算每个参数的梯度，最后用这些梯度来更新网络参数，使损失函数逐步减小。",
        },
        {
            "instruction": "什么是过拟合？如何解决？",
            "output": "过拟合是指模型在训练数据上表现很好，但在新数据上表现较差的现象。解决方法包括：1)增加训练数据量；2)使用正则化技术如L1/L2正则化；3)使用Dropout；4)减小模型复杂度；5)使用数据增强；6)早停法(Early Stopping)。",
        },
        {
            "instruction": "What is a neural network?",
            "output": "A neural network is a computational model inspired by the structure of biological neural networks. It consists of interconnected layers of artificial neurons that process information. Each neuron receives inputs, applies weights and an activation function, and produces an output that feeds into the next layer.",
        },
        {
            "instruction": "请简要介绍GPT模型。",
            "output": "GPT(Generative Pre-trained Transformer)是OpenAI开发的一系列大规模语言模型。它基于Transformer的Decoder部分，使用自回归方式进行训练，即根据前面的文本预测下一个token。GPT通过在大量文本上预训练，然后通过微调适应各种下游任务，展现了强大的语言生成和理解能力。",
        },
        {
            "instruction": "What is transfer learning?",
            "output": "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different but related task. This is particularly useful when the target task has limited labeled data, as the pre-trained model has already learned useful feature representations that can be fine-tuned for the new task.",
        },
        {
            "instruction": "解释什么是词嵌入。",
            "output": "词嵌入是一种将离散的词语映射到连续向量空间的技术。每个词被表示为一个固定维度的稠密向量，语义相近的词在向量空间中距离更近。常见的词嵌入方法包括Word2Vec、GloVe和FastText等。在现代大语言模型中，词嵌入通常作为模型的第一层进行端到端训练。",
        },
        {
            "instruction": "What is reinforcement learning?",
            "output": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns a policy that maximizes cumulative rewards over time. Key concepts include states, actions, rewards, and the balance between exploration and exploitation.",
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_samples):
            sample = sft_samples[i % len(sft_samples)]
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ SFT 数据: {output_path}")
    print(f"   样本数: {n_samples:,}")


def create_sample_dpo_data(output_path: str, n_samples: int = 500):
    """创建样例 DPO 数据"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dpo_samples = [
        {
            "prompt": "什么是人工智能？",
            "chosen": "人工智能(AI)是计算机科学的一个分支，致力于创造能模拟人类智能行为的系统。它包括机器学习、自然语言处理、计算机视觉等多个子领域。现代AI系统通过从大量数据中学习模式来完成各种任务，如语言翻译、图像识别和决策制定等。",
            "rejected": "人工智能就是让电脑变聪明。",
        },
        {
            "prompt": "How does a neural network learn?",
            "chosen": "A neural network learns through an iterative process involving forward propagation and backpropagation. During forward propagation, input data passes through the network layers, producing predictions. The loss function measures the difference between predictions and actual targets. Backpropagation then computes gradients using the chain rule, and an optimizer updates the network weights to minimize the loss. This process repeats over many iterations until the network converges.",
            "rejected": "It just learns from data somehow.",
        },
        {
            "prompt": "请解释什么是卷积神经网络。",
            "chosen": "卷积神经网络(CNN)是一种专门用于处理网格状数据（如图像）的深度学习架构。它的核心组件包括：1)卷积层：使用滑动窗口的卷积核提取局部特征；2)池化层：降低特征图的空间尺寸；3)全连接层：进行最终的分类或回归。CNN通过权重共享和局部连接大大减少了参数量，使其特别适合处理图像数据。",
            "rejected": "CNN就是一种处理图片的神经网络，用来识别图片中的物体。",
        },
        {
            "prompt": "What is the difference between supervised and unsupervised learning?",
            "chosen": "Supervised learning uses labeled training data where each input has a corresponding target output. The model learns to map inputs to outputs by minimizing prediction errors. Common tasks include classification and regression. Unsupervised learning, on the other hand, works with unlabeled data and aims to discover hidden patterns or structure. Common tasks include clustering, dimensionality reduction, and anomaly detection. The key difference is the availability and use of labeled data during training.",
            "rejected": "Supervised learning uses labels and unsupervised doesn't.",
        },
        {
            "prompt": "请解释什么是自注意力机制。",
            "chosen": "自注意力机制是Transformer架构的核心组件，它让模型能够关注输入序列中不同位置之间的关系。具体而言，对于序列中的每个位置，自注意力计算三个向量：Query(查询)、Key(键)和Value(值)。通过计算Query和所有Key的点积并进行softmax归一化，得到注意力权重，再用这些权重对Value进行加权求和。这种机制使得模型可以捕捉序列中任意距离的依赖关系，且计算可以完全并行化。",
            "rejected": "自注意力就是模型自己关注自己，通过计算相似度来决定关注哪些部分。",
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_samples):
            sample = dpo_samples[i % len(dpo_samples)]
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ DPO 数据: {output_path}")
    print(f"   样本数: {n_samples:,}")


def create_tokenizer_corpus(data_dir: str, output_path: str):
    """从预训练数据中提取纯文本，用于训练分词器"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pretrain_path = os.path.join(data_dir, "pretrain", "pretrain_data.jsonl")
    sft_path = os.path.join(data_dir, "sft", "sft_data.jsonl")

    texts = []

    # 从预训练数据提取
    if os.path.exists(pretrain_path):
        with open(pretrain_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                texts.append(obj.get("text", ""))

    # 从 SFT 数据提取
    if os.path.exists(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                texts.append(obj.get("instruction", ""))
                texts.append(obj.get("output", ""))

    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            if text.strip():
                f.write(text.strip() + "\n")

    print(f"✅ 分词器语料: {output_path}")
    print(f"   文本行数: {len(texts):,}")


def main():
    parser = argparse.ArgumentParser(description="ClearMind 数据准备")
    parser.add_argument("--data_dir", type=str, default="data", help="数据存储目录")
    parser.add_argument(
        "--pretrain_samples", type=int, default=5000, help="预训练样本数"
    )
    parser.add_argument("--sft_samples", type=int, default=2000, help="SFT 样本数")
    parser.add_argument("--dpo_samples", type=int, default=500, help="DPO 样本数")
    args = parser.parse_args()

    data_dir = args.data_dir
    print("=" * 60)
    print("ClearMind 数据准备")
    print("=" * 60)

    # 1. 创建预训练数据
    print(f"\n📥 Step 1/4: 准备预训练数据...")
    create_sample_pretrain_data(
        os.path.join(data_dir, "pretrain", "pretrain_data.jsonl"),
        n_samples=args.pretrain_samples,
    )

    # 2. 创建 SFT 数据
    print(f"\n📥 Step 2/4: 准备 SFT 数据...")
    create_sample_sft_data(
        os.path.join(data_dir, "sft", "sft_data.jsonl"),
        n_samples=args.sft_samples,
    )

    # 3. 创建 DPO 数据
    print(f"\n📥 Step 3/4: 准备 DPO 数据...")
    create_sample_dpo_data(
        os.path.join(data_dir, "dpo", "dpo_data.jsonl"),
        n_samples=args.dpo_samples,
    )

    # 4. 创建分词器训练语料
    print(f"\n📥 Step 4/4: 创建分词器训练语料...")
    create_tokenizer_corpus(
        data_dir,
        os.path.join(data_dir, "tokenizer_corpus.txt"),
    )

    print(f"\n{'=' * 60}")
    print("✅ 数据准备完成!")
    print(f"\n📁 数据目录结构:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, "").count(os.sep)
        indent = "  " * level
        print(f"  {indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {subindent}{file} ({size_kb:.1f} KB)")

    print(f"\n💡 下一步: python scripts/02_train_tokenizer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
