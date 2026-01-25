"""
HuggingFace Transformers 入门
============================

学习目标：
    1. 了解 HuggingFace 生态系统
    2. 掌握 Tokenizer 的使用
    3. 学习加载预训练模型
    4. 使用 Pipeline 快速推理

核心概念：
    - HuggingFace Hub：模型和数据集仓库
    - Tokenizer：文本分词和编码
    - AutoModel：自动加载模型
    - Pipeline：高级推理接口

前置知识：
    - Phase 7: Transformer
    - 前面的 NLP 课程
"""

import torch


# ==================== 第一部分：HuggingFace 生态系统 ====================


def introduction():
    """HuggingFace 生态系统介绍"""
    print("=" * 60)
    print("第一部分：HuggingFace 生态系统")
    print("=" * 60)

    print("""
HuggingFace 生态系统：

┌─────────────────────────────────────────────────────────────┐
│                    HuggingFace 组件                          │
├─────────────────────────────────────────────────────────────┤
│  🤗 Transformers    预训练模型库                             │
│     - BERT, GPT, T5, LLaMA...                               │
│     - 支持 PyTorch, TensorFlow, JAX                         │
├─────────────────────────────────────────────────────────────┤
│  🤗 Datasets        数据集库                                 │
│     - 3000+ 数据集                                          │
│     - 高效的数据加载                                         │
├─────────────────────────────────────────────────────────────┤
│  🤗 Hub             模型托管平台                             │
│     - 分享和下载模型                                         │
│     - 模型卡片和文档                                         │
├─────────────────────────────────────────────────────────────┤
│  🤗 Accelerate      分布式训练                               │
│  🤗 PEFT            参数高效微调                             │
│  🤗 Evaluate        评估指标                                 │
└─────────────────────────────────────────────────────────────┘

安装：
    pip install transformers datasets
    """)


# ==================== 第二部分：Tokenizer ====================


def tokenizer_demo():
    """Tokenizer 使用示例"""
    print("\n" + "=" * 60)
    print("第二部分：Tokenizer")
    print("=" * 60)

    print("""
Tokenizer 的作用：
    文本 → tokens → input_ids → 模型

基本使用：

    from transformers import AutoTokenizer
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # 编码
    text = "深度学习很有趣"
    encoded = tokenizer(text)
    # {'input_ids': [101, 3918, 2428, ...], 'attention_mask': [1, 1, ...]}
    
    # 解码
    decoded = tokenizer.decode(encoded['input_ids'])
    # "[CLS] 深度学习很有趣 [SEP]"

常用参数：

    encoded = tokenizer(
        text,
        max_length=128,           # 最大长度
        padding='max_length',     # 填充策略
        truncation=True,          # 是否截断
        return_tensors='pt'       # 返回 PyTorch 张量
    )

批量处理：

    texts = ["第一段文本", "第二段文本"]
    encoded = tokenizer(texts, padding=True, return_tensors='pt')

特殊 tokens：
    - [CLS]: 句子开始，用于分类
    - [SEP]: 句子分隔/结束
    - [PAD]: 填充
    - [MASK]: 掩码（BERT MLM）
    """)


# ==================== 第三部分：加载预训练模型 ====================


def model_loading():
    """加载预训练模型"""
    print("\n" + "=" * 60)
    print("第三部分：加载预训练模型")
    print("=" * 60)

    print("""
使用 AutoModel 加载模型：

    from transformers import AutoModel, AutoTokenizer
    
    # 加载模型和 tokenizer
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 推理
    text = "深度学习很有趣"
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.last_hidden_state: (batch, seq_len, hidden_size)
    print(f"输出形状: {outputs.last_hidden_state.shape}")

针对特定任务的模型：

    # 文本分类
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-chinese", num_labels=2
    )
    
    # 问答
    from transformers import AutoModelForQuestionAnswering
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese")
    
    # Token 分类 (NER)
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-chinese", num_labels=9
    )

模型配置：

    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained("bert-base-chinese")
    print(config.hidden_size)  # 768
    print(config.num_hidden_layers)  # 12
    """)


# ==================== 第四部分：Pipeline ====================


def pipeline_demo():
    """Pipeline 使用示例"""
    print("\n" + "=" * 60)
    print("第四部分：Pipeline")
    print("=" * 60)

    print("""
Pipeline 是最简单的推理方式：

    from transformers import pipeline
    
    # 情感分析
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love deep learning!")
    # [{'label': 'POSITIVE', 'score': 0.9998}]
    
    # 文本生成
    generator = pipeline("text-generation", model="gpt2")
    result = generator("Deep learning is", max_length=50)
    
    # 问答
    qa = pipeline("question-answering")
    result = qa(
        question="What is deep learning?",
        context="Deep learning is a branch of machine learning..."
    )
    
    # 命名实体识别
    ner = pipeline("ner", aggregation_strategy="simple")
    result = ner("Bill Gates founded Microsoft in Seattle")
    
    # 填空
    fill = pipeline("fill-mask", model="bert-base-chinese")
    result = fill("深度[MASK]是人工智能的分支")

常用 Pipeline 任务：

    - "text-classification / sentiment-analysis"
    - "token-classification / ner"
    - "question-answering"
    - "fill-mask"
    - "text-generation"
    - "summarization"
    - "translation"
    - "zero-shot-classification"
    """)


# ==================== 第五部分：实战示例 ====================


def practical_example():
    """实战示例"""
    print("\n" + "=" * 60)
    print("第五部分：实战示例")
    print("=" * 60)

    print("""
完整的文本分类示例：

    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        Trainer, 
        TrainingArguments
    )
    from datasets import load_dataset
    
    # 1. 加载数据集
    dataset = load_dataset("imdb")
    
    # 2. 加载模型和 tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    
    # 3. 数据预处理
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=512
        )
    
    tokenized = dataset.map(tokenize, batched=True)
    
    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
    )
    
    # 5. 创建 Trainer 并训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )
    
    trainer.train()
    
    # 6. 保存模型
    trainer.save_model("./my_model")
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：使用 Pipeline
    任务：用不同的 Pipeline 处理同一段文本

练习 1 答案：
    from transformers import pipeline
    
    text = "Apple CEO Tim Cook announced the new iPhone in California"
    
    # 情感分析
    sentiment = pipeline("sentiment-analysis")
    print(sentiment(text))
    
    # NER
    ner = pipeline("ner", aggregation_strategy="simple")
    print(ner(text))
    
    # 问答
    qa = pipeline("question-answering")
    print(qa(question="Who is the CEO?", context=text))

练习 2：自定义 Tokenizer
    任务：比较不同模型的 Tokenizer 输出差异

练习 2 答案：
    from transformers import AutoTokenizer
    
    text = "深度学习"
    
    for model_name in ["bert-base-chinese", "hfl/chinese-roberta-wwm-ext"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(text)
        print(f"{model_name}: {tokens}")

思考题 1：AutoModel vs 特定任务模型？
答案：
    - AutoModel：只返回编码器输出，需要自己加分类头
    - AutoModelForXXX：包含任务相关的头，可直接用于特定任务
    - 选择依据：是否需要自定义输出层

思考题 2：如何选择预训练模型？
答案：
    考虑因素：
    - 语言：中文用 bert-base-chinese, 英文用 bert-base-uncased
    - 任务：生成用 GPT，理解用 BERT
    - 规模：资源有限用 base，追求效果用 large
    - 领域：特定领域可能有领域专用模型
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    tokenizer_demo()
    model_loading()
    pipeline_demo()
    practical_example()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！下一步：07-transformer-finetuning.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
