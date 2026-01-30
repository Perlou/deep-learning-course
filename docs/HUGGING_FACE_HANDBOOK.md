# Hugging Face 速查文档

## 目录

- [1. 安装](#1-安装)
- [2. Pipeline 快速入门](#2-pipeline-快速入门)
- [3. Tokenizer 分词器](#3-tokenizer-分词器)
- [4. 模型加载与使用](#4-模型加载与使用)
- [5. Datasets 数据集](#5-datasets-数据集)
- [6. 模型训练与微调](#6-模型训练与微调)
- [7. 模型保存与上传](#7-模型保存与上传)
- [8. 常用模型速查](#8-常用模型速查)
- [9. 高级配置](#9-高级配置)
- [10. 常见问题解决](#10-常见问题解决)

---

## 1. 安装

### 基础安装

```bash
# 安装 transformers
pip install transformers

# 安装全套工具
pip install transformers datasets evaluate accelerate

# 安装 PyTorch 版本
pip install transformers[torch]

# 安装 TensorFlow 版本
pip install transformers[tf-cpu]

# 从源码安装（最新版）
pip install git+https://github.com/huggingface/transformers
```

### 验证安装

```python
import transformers
print(transformers.__version__)
```

---

## 2. Pipeline 快速入门

Pipeline 是最简单的使用方式，开箱即用。

### 基本用法

```python
from transformers import pipeline

# 自动下载模型并推理
classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 常用 Pipeline 任务

| 任务名称                       | 描述         | 示例                                       |
| ------------------------------ | ------------ | ------------------------------------------ |
| `sentiment-analysis`           | 情感分析     | `pipeline("sentiment-analysis")`           |
| `text-classification`          | 文本分类     | `pipeline("text-classification")`          |
| `ner`                          | 命名实体识别 | `pipeline("ner")`                          |
| `question-answering`           | 问答         | `pipeline("question-answering")`           |
| `summarization`                | 文本摘要     | `pipeline("summarization")`                |
| `translation`                  | 翻译         | `pipeline("translation_en_to_fr")`         |
| `text-generation`              | 文本生成     | `pipeline("text-generation")`              |
| `fill-mask`                    | 填空         | `pipeline("fill-mask")`                    |
| `zero-shot-classification`     | 零样本分类   | `pipeline("zero-shot-classification")`     |
| `image-classification`         | 图像分类     | `pipeline("image-classification")`         |
| `object-detection`             | 目标检测     | `pipeline("object-detection")`             |
| `automatic-speech-recognition` | 语音识别     | `pipeline("automatic-speech-recognition")` |

### Pipeline 详细示例

```python
from transformers import pipeline

# 1. 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello, I'm a language model", max_length=50, num_return_sequences=2)

# 2. 问答系统
qa = pipeline("question-answering")
result = qa(
    question="What is Hugging Face?",
    context="Hugging Face is a company that provides NLP tools."
)

# 3. 命名实体识别
ner = pipeline("ner", aggregation_strategy="simple")
result = ner("My name is John and I live in New York")

# 4. 文本摘要
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer(long_text, max_length=130, min_length=30)

# 5. 翻译
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
result = translator("Hello, how are you?")

# 6. 零样本分类
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about Python programming",
    candidate_labels=["education", "politics", "business"]
)

# 7. 图像分类
image_classifier = pipeline("image-classification")
result = image_classifier("path/to/image.jpg")

# 8. 语音识别
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
result = asr("path/to/audio.mp3")
```

### 指定设备

```python
# 使用 GPU
pipe = pipeline("text-classification", device=0)  # 第一块 GPU
pipe = pipeline("text-classification", device="cuda:0")

# 使用 CPU
pipe = pipeline("text-classification", device=-1)
pipe = pipeline("text-classification", device="cpu")

# 自动选择
pipe = pipeline("text-classification", device_map="auto")
```

---

## 3. Tokenizer 分词器

### 加载 Tokenizer

```python
from transformers import AutoTokenizer

# 从 Hub 加载
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 从本地加载
tokenizer = AutoTokenizer.from_pretrained("./my_model/")
```

### 基本编码/解码

```python
# 编码
text = "Hello, how are you?"
encoded = tokenizer(text)
# {'input_ids': [101, 7592, ...], 'attention_mask': [1, 1, ...], 'token_type_ids': [...]}

# 仅获取 token ids
ids = tokenizer.encode(text)

# 解码
decoded = tokenizer.decode(ids)

# 批量编码
texts = ["Hello world", "How are you"]
batch_encoded = tokenizer(texts)
```

### 高级编码选项

```python
# 填充和截断
encoded = tokenizer(
    text,
    padding="max_length",       # 填充到最大长度
    truncation=True,            # 启用截断
    max_length=512,             # 最大长度
    return_tensors="pt",        # 返回 PyTorch 张量 ("tf" for TensorFlow)
    add_special_tokens=True,    # 添加特殊 token
)

# 批量处理（动态填充）
encoded = tokenizer(
    texts,
    padding=True,               # 填充到批次最长
    truncation=True,
    return_tensors="pt"
)
```

### 填充选项

| 参数                   | 说明               |
| ---------------------- | ------------------ |
| `padding=True`         | 填充到批次最长序列 |
| `padding="max_length"` | 填充到 max_length  |
| `padding="longest"`    | 同 True            |
| `padding=False`        | 不填充             |

### 返回张量类型

| 参数                  | 说明              |
| --------------------- | ----------------- |
| `return_tensors="pt"` | PyTorch tensor    |
| `return_tensors="tf"` | TensorFlow tensor |
| `return_tensors="np"` | NumPy array       |
| 不设置                | Python list       |

### Tokenizer 常用属性和方法

```python
# 词汇表大小
tokenizer.vocab_size

# 特殊 token
tokenizer.pad_token        # [PAD]
tokenizer.unk_token        # [UNK]
tokenizer.cls_token        # [CLS]
tokenizer.sep_token        # [SEP]
tokenizer.mask_token       # [MASK]

# 特殊 token ID
tokenizer.pad_token_id
tokenizer.unk_token_id

# 将 tokens 转换为 ids
tokenizer.convert_tokens_to_ids(["hello", "world"])

# 将 ids 转换为 tokens
tokenizer.convert_ids_to_tokens([101, 7592])

# 分词（不转换为 ids）
tokenizer.tokenize("Hello world")  # ['hello', 'world']
```

---

## 4. 模型加载与使用

### 自动加载模型

```python
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

# 加载基础模型
model = AutoModel.from_pretrained("bert-base-uncased")

# 加载特定任务模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
```

### 常用 AutoModel 类

| 类名                                 | 用途                   |
| ------------------------------------ | ---------------------- |
| `AutoModel`                          | 基础模型（无任务头）   |
| `AutoModelForSequenceClassification` | 文本分类               |
| `AutoModelForTokenClassification`    | Token 分类 (NER)       |
| `AutoModelForQuestionAnswering`      | 问答                   |
| `AutoModelForCausalLM`               | 因果语言模型（GPT类）  |
| `AutoModelForSeq2SeqLM`              | Seq2Seq（T5, BART类）  |
| `AutoModelForMaskedLM`               | 掩码语言模型（BERT类） |
| `AutoModelForImageClassification`    | 图像分类               |
| `AutoModelForObjectDetection`        | 目标检测               |
| `AutoModelForSpeechSeq2Seq`          | 语音转文本             |

### 模型推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 编码输入
inputs = tokenizer("Hello world", return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取 logits
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
```

### 文本生成

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("Hello, I'm a", return_tensors="pt")

# 生成文本
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 生成参数详解

```python
model.generate(
    inputs,
    # 长度控制
    max_length=100,              # 最大生成长度
    max_new_tokens=50,           # 最大新生成 token 数
    min_length=10,               # 最小长度

    # 采样策略
    do_sample=True,              # 是否采样
    temperature=0.7,             # 温度（越高越随机）
    top_k=50,                    # Top-K 采样
    top_p=0.95,                  # Top-P (nucleus) 采样

    # Beam Search
    num_beams=5,                 # Beam 数量
    early_stopping=True,         # 早停
    no_repeat_ngram_size=2,      # 避免重复 n-gram

    # 其他
    num_return_sequences=3,      # 返回序列数
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

---

## 5. Datasets 数据集

### 安装

```bash
pip install datasets
```

### 加载数据集

```python
from datasets import load_dataset

# 从 Hub 加载
dataset = load_dataset("imdb")
dataset = load_dataset("glue", "mrpc")  # 加载子集

# 加载特定分割
train_dataset = load_dataset("imdb", split="train")
test_dataset = load_dataset("imdb", split="test[:100]")  # 前100条

# 从本地文件加载
dataset = load_dataset("csv", data_files="data.csv")
dataset = load_dataset("json", data_files="data.json")
dataset = load_dataset("text", data_files="data.txt")

# 从字典创建
from datasets import Dataset
data = {"text": ["Hello", "World"], "label": [0, 1]}
dataset = Dataset.from_dict(data)

# 从 pandas 创建
dataset = Dataset.from_pandas(df)
```

### 数据集操作

```python
# 查看数据集信息
print(dataset)
print(dataset.features)
print(dataset.column_names)

# 访问数据
print(dataset[0])                    # 第一条
print(dataset["text"])               # 整列
print(dataset[0:5])                  # 切片

# 分割数据集
dataset = dataset.train_test_split(test_size=0.2)

# 打乱
dataset = dataset.shuffle(seed=42)

# 选择/过滤
dataset = dataset.select(range(100))  # 选择前100条
dataset = dataset.filter(lambda x: len(x["text"]) > 10)

# 映射（预处理）
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

dataset = dataset.map(preprocess, batched=True)

# 重命名/删除列
dataset = dataset.rename_column("text", "input_text")
dataset = dataset.remove_columns(["unnecessary_column"])

# 设置格式
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

### 常用公开数据集

| 数据集       | 加载方式                             | 描述             |
| ------------ | ------------------------------------ | ---------------- |
| IMDb         | `load_dataset("imdb")`               | 电影评论情感分析 |
| GLUE         | `load_dataset("glue", "sst2")`       | NLU 基准测试     |
| SQuAD        | `load_dataset("squad")`              | 问答数据集       |
| CoNLL-2003   | `load_dataset("conll2003")`          | NER 数据集       |
| WMT          | `load_dataset("wmt16", "de-en")`     | 翻译数据集       |
| Common Voice | `load_dataset("common_voice", "en")` | 语音数据集       |
| ImageNet     | `load_dataset("imagenet-1k")`        | 图像分类         |

---

## 6. 模型训练与微调

### 使用 Trainer（推荐）

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate
import numpy as np

# 1. 加载数据
dataset = load_dataset("imdb")

# 2. 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 3. 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. 评估指标
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 6. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# 7. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. 训练
trainer.train()

# 9. 评估
trainer.evaluate()

# 10. 保存模型
trainer.save_model("./my_model")
```

### TrainingArguments 常用参数

```python
training_args = TrainingArguments(
    # 输出设置
    output_dir="./results",
    overwrite_output_dir=True,

    # 训练超参数
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,    # 梯度累积
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    warmup_ratio=0.1,

    # 评估和保存
    evaluation_strategy="steps",       # "no", "steps", "epoch"
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,               # 最多保存模型数
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    # 日志
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard",          # "wandb", "tensorboard", "none"

    # 硬件
    fp16=True,                        # 混合精度训练
    bf16=False,
    dataloader_num_workers=4,

    # Hub
    push_to_hub=False,
    hub_model_id="my-model",
)
```

### 使用原生 PyTorch 训练

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler

# 准备数据
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 创建 DataLoader
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)

# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

### 使用 Accelerate 分布式训练

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

---

## 7. 模型保存与上传

### 本地保存和加载

```python
# 保存
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 加载
model = AutoModel.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

### 上传到 Hub

```python
# 方法1: 使用 CLI 登录
# huggingface-cli login

# 方法2: 在代码中登录
from huggingface_hub import login
login(token="your_token")

# 上传模型
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")

# 通过 Trainer 上传
trainer.push_to_hub()
```

### Hub API 使用

```python
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

# 列出模型
models = api.list_models(filter="text-classification")

# 下载文件
hf_hub_download(repo_id="bert-base-uncased", filename="config.json")

# 创建仓库
api.create_repo(repo_id="my-new-model")

# 上传文件
api.upload_file(
    path_or_fileobj="./model.bin",
    path_in_repo="model.bin",
    repo_id="username/my-model"
)
```

---

## 8. 常用模型速查

### NLP 模型

| 模型       | 用途             | 加载方式                  |
| ---------- | ---------------- | ------------------------- |
| BERT       | 编码器，分类/NER | `bert-base-uncased`       |
| RoBERTa    | 优化的 BERT      | `roberta-base`            |
| DistilBERT | 轻量 BERT        | `distilbert-base-uncased` |
| ALBERT     | 轻量 BERT        | `albert-base-v2`          |
| GPT-2      | 文本生成         | `gpt2`, `gpt2-medium`     |
| GPT-Neo    | 开源 GPT         | `EleutherAI/gpt-neo-1.3B` |
| LLaMA      | Meta 大模型      | `meta-llama/Llama-2-7b`   |
| T5         | Seq2Seq          | `t5-base`, `t5-large`     |
| BART       | Seq2Seq，摘要    | `facebook/bart-large-cnn` |
| XLNet      | 自回归+自编码    | `xlnet-base-cased`        |

### 中文模型

| 模型         | 加载方式                      |
| ------------ | ----------------------------- |
| BERT 中文    | `bert-base-chinese`           |
| MacBERT      | `hfl/chinese-macbert-base`    |
| RoBERTa 中文 | `hfl/chinese-roberta-wwm-ext` |
| ERNIE        | `nghuyong/ernie-3.0-base-zh`  |
| ChatGLM      | `THUDM/chatglm3-6b`           |
| Qwen         | `Qwen/Qwen-7B`                |

### 多模态模型

| 模型    | 用途     | 加载方式                                |
| ------- | -------- | --------------------------------------- |
| CLIP    | 图文匹配 | `openai/clip-vit-base-patch32`          |
| ViT     | 图像分类 | `google/vit-base-patch16-224`           |
| BLIP    | 图像描述 | `Salesforce/blip-image-captioning-base` |
| Whisper | 语音识别 | `openai/whisper-base`                   |

---

## 9. 高级配置

### 加载大模型（内存优化）

```python
from transformers import AutoModelForCausalLM

# 8位量化
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    load_in_8bit=True,
    device_map="auto"
)

# 4位量化
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_4bit=True,
    device_map="auto"
)

# 指定设备映射
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto",              # 自动分配
    torch_dtype=torch.float16,       # 半精度
    low_cpu_mem_usage=True
)

# 使用 offload
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom",
    device_map="auto",
    offload_folder="offload",        # CPU/磁盘 offload
)
```

### PEFT (参数高效微调)

```bash
pip install peft
```

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                              # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 保存 LoRA 权重
model.save_pretrained("./lora_model")

# 加载 LoRA 权重
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(base_model, "./lora_model")
```

### 梯度检查点

```python
model.gradient_checkpointing_enable()
```

### Flash Attention

```python
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)
```

---

## 10. 常见问题解决

### 内存不足 (OOM)

```python
# 1. 减小 batch size
# 2. 使用梯度累积
training_args = TrainingArguments(
    gradient_accumulation_steps=4,
    per_device_train_batch_size=4,  # 有效 batch = 4 * 4 = 16
)

# 3. 使用混合精度
training_args = TrainingArguments(fp16=True)

# 4. 使用梯度检查点
model.gradient_checkpointing_enable()

# 5. 量化加载
model = AutoModel.from_pretrained("model", load_in_8bit=True)
```

### 下载问题

```python
# 设置镜像源（中国大陆用户）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置缓存目录
os.environ['HF_HOME'] = '/path/to/cache'
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

# 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
model = AutoModel.from_pretrained("./local_model")
```

### 警告处理

```python
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 设置日志级别
import logging
logging.set_verbosity_error()

from transformers import logging
logging.set_verbosity_warning()
```

### 常用环境变量

| 环境变量               | 说明                    |
| ---------------------- | ----------------------- |
| `HF_HOME`              | Hugging Face 缓存根目录 |
| `HF_TOKEN`             | API Token               |
| `TRANSFORMERS_CACHE`   | 模型缓存目录            |
| `HF_DATASETS_CACHE`    | 数据集缓存目录          |
| `TRANSFORMERS_OFFLINE` | 离线模式                |
| `HF_ENDPOINT`          | Hub 端点 URL            |

---

## 快速参考卡片

```python
# 🚀 最简使用
from transformers import pipeline
pipe = pipeline("task-name")
result = pipe("input text")

# 📦 标准使用
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModel.from_pretrained("model-name")
inputs = tokenizer("text", return_tensors="pt")
outputs = model(**inputs)

# 🎯 训练流程
from transformers import Trainer, TrainingArguments
args = TrainingArguments(output_dir="./output", ...)
trainer = Trainer(model=model, args=args, train_dataset=dataset, ...)
trainer.train()
trainer.save_model()
```

---

> 📚 **官方文档**: https://huggingface.co/docs/transformers
>
> 🤗 **模型中心**: https://huggingface.co/models
>
> 📊 **数据集中心**: https://huggingface.co/datasets
