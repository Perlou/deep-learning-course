"""
BERT 微调实践 (BERT Fine-tuning)
================================

学习目标：
    1. 学习如何使用HuggingFace Transformers库
    2. 掌握BERT微调的基本流程
    3. 实现文本分类任务
    4. 理解微调的最佳实践

核心概念：
    - 预训练 vs 微调
    - 加载预训练模型
    - 添加任务特定的head
    - 冻结/解冻层

前置知识：
    - 07-bert-architecture.py
"""

print("""
BERT微调基本流程：

1. 加载预训练BERT
2. 添加任务头（如分类层）
3. 准备数据
4. 微调训练
5. 评估和推理

示例代码框架：
```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# 1. 加载预训练模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# 2. 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 3. 准备数据
texts = ["This is great!", "This is bad..."]
labels = [1, 0]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 4. 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
)

# 5. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

微调技巧：
  ✓ 使用较小的学习率（1e-5 ~ 5e-5）
  ✓ 使用warmup
  ✓ 可以先冻结底层，只训练顶层
  ✓ 注意过拟合
  ✓ 使用适当的正则化

HuggingFace资源：
  - 模型库: https://huggingface.co/models
  - 文档: https://huggingface.co/docs/transformers
  - 教程: https://huggingface.co/course
""")
