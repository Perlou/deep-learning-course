"""
Transformer 微调
================

学习目标：
    1. 理解预训练-微调范式
    2. 掌握 HuggingFace Trainer 的使用
    3. 学习微调的最佳实践
    4. 了解常见的微调策略

核心概念：
    - 预训练-微调：先通用训练，再任务适配
    - Learning Rate Warmup：学习率预热
    - Gradient Accumulation：梯度累积
    - Early Stopping：提前停止

前置知识：
    - Phase 7: Transformer
    - 06-huggingface-basics.py
"""

import torch
import torch.nn as nn


# ==================== 第一部分：预训练-微调范式 ====================


def introduction():
    """预训练-微调范式介绍"""
    print("=" * 60)
    print("第一部分：预训练-微调范式")
    print("=" * 60)

    print("""
预训练-微调 (Pre-training & Fine-tuning) 范式：

┌─────────────────────────────────────────────────────────────┐
│  阶段一：预训练 (Pre-training)                               │
│                                                              │
│  目标：学习通用语言知识                                       │
│  数据：大规模无标注文本（如维基百科、书籍）                    │
│  任务：                                                      │
│    - MLM (BERT): 预测 [MASK] 词                              │
│    - NSP (BERT): 预测下一句                                   │
│    - CLM (GPT): 预测下一词                                    │
│  资源：需要大量计算资源（TPU/GPU 集群）                        │
├─────────────────────────────────────────────────────────────┤
│  阶段二：微调 (Fine-tuning)                                  │
│                                                              │
│  目标：适配特定下游任务                                       │
│  数据：任务相关的标注数据（可以很小）                          │
│  任务：分类、NER、QA 等                                       │
│  资源：单张 GPU 即可                                          │
└─────────────────────────────────────────────────────────────┘

为什么微调有效？
    - 预训练模型已学习语法、语义、常识
    - 微调只需学习任务特定知识
    - 迁移学习：通用知识 → 特定任务
    """)


# ==================== 第二部分：微调流程 ====================


def finetuning_workflow():
    """微调工作流程"""
    print("\n" + "=" * 60)
    print("第二部分：微调流程")
    print("=" * 60)

    print("""
完整微调流程：

    1. 加载预训练模型和 Tokenizer
    2. 准备数据集
    3. 添加任务头（如分类层）
    4. 设置训练参数
    5. 训练和评估
    6. 保存模型

代码示例：

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments
    )
    from datasets import load_dataset
    
    # 1. 加载模型
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # 2. 准备数据
    dataset = load_dataset("your_dataset")
    
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128
        )
    
    tokenized_dataset = dataset.map(preprocess, batched=True)
    
    # 3. 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 4. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    
    # 5. 训练
    trainer.train()
    
    # 6. 保存
    trainer.save_model("./final_model")
    """)


# ==================== 第三部分：微调技巧 ====================


def finetuning_tips():
    """微调技巧和最佳实践"""
    print("\n" + "=" * 60)
    print("第三部分：微调技巧")
    print("=" * 60)

    print("""
1. 学习率设置：
    - 预训练模型层：较小学习率 (1e-5 ~ 2e-5)
    - 新增分类层：较大学习率 (1e-4)
    - 使用 Warmup：前 10% 步骤线性增加学习率

2. 批量大小和梯度累积：
    - 显存不足时使用梯度累积模拟大批量
    
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,  # 等效 batch_size=32
    )

3. 冻结部分层：
    - 数据量少时可冻结预训练层
    
    # 冻结所有 BERT 参数
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # 只训练分类头
    for param in model.classifier.parameters():
        param.requires_grad = True

4. 分层学习率 (Layerwise LR Decay):
    - 底层学习率小，顶层学习率大
    
    from transformers import AdamW
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() 
                    if "bert.embeddings" in n], "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() 
                    if "bert.encoder.layer.0" in n], "lr": 1.5e-5},
        # ... 更多层
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ]

5. 数据增强：
    - 回译 (Back Translation)
    - 同义词替换
    - 随机删除/交换
    """)


# ==================== 第四部分：评估和调试 ====================


def evaluation_and_debugging():
    """评估和调试"""
    print("\n" + "=" * 60)
    print("第四部分：评估和调试")
    print("=" * 60)

    print("""
自定义评估指标：

    from sklearn.metrics import accuracy_score, f1_score
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="macro"),
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

常见问题排查：

    问题：验证集 accuracy 不提升
    方案：
    - 降低学习率
    - 增加 warmup 步数
    - 检查数据标签是否正确
    
    问题：过拟合
    方案：
    - 增加 dropout
    - 增加 weight_decay
    - 数据增强
    - 减少训练轮数
    
    问题：显存不足 (OOM)
    方案：
    - 减小 batch_size
    - 使用梯度累积
    - 使用混合精度训练 (fp16=True)
    - 截断序列长度
    """)


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：情感分析微调
    任务：在 IMDB 数据集上微调 BERT

练习 1 答案：
    from transformers import AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer
    from datasets import load_dataset
    
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)
    
    dataset = dataset.map(tokenize, batched=True)
    
    args = TrainingArguments(
        output_dir="./imdb-bert",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"])
    trainer.train()

练习 2：实现分层学习率
    任务：为不同层设置不同学习率

练习 2 答案：
    from torch.optim import AdamW
    
    def get_optimizer(model, base_lr=2e-5):
        param_groups = []
        
        # Embedding 层
        param_groups.append({
            "params": model.bert.embeddings.parameters(),
            "lr": base_lr * 0.5
        })
        
        # Encoder 层（逐层增加）
        for i, layer in enumerate(model.bert.encoder.layer):
            lr = base_lr * (0.5 + 0.5 * i / 12)
            param_groups.append({"params": layer.parameters(), "lr": lr})
        
        # 分类头
        param_groups.append({"params": model.classifier.parameters(), "lr": base_lr * 5})
        
        return AdamW(param_groups)

思考题 1：为什么微调时学习率要比预训练小很多？
答案：
    - 预训练模型已经学到了好的表示
    - 大学习率会破坏这些知识
    - 微调是"微调"而非"重训练"

思考题 2：小数据集如何防止过拟合？
答案：
    - 冻结预训练层，只训练分类头
    - 增加 Dropout
    - 使用更小的模型
    - 数据增强
    - 缩短训练轮数
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    finetuning_workflow()
    finetuning_tips()
    evaluation_and_debugging()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！下一步：08-peft-lora.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
