# 部署指南

> ClearMind-HF 模型部署方式汇总

## 1. Gradio Web Demo

最简单的部署方式，一键启动浏览器对话界面：

```bash
# 本地启动
python deploy/web_demo.py --model outputs/sft --tokenizer outputs/tokenizer

# 创建公网链接 (Gradio Share)
python deploy/web_demo.py --model outputs/dpo --tokenizer outputs/tokenizer --share

# 自定义端口
python deploy/web_demo.py --model outputs/sft --port 8080
```

## 2. CLI 交互对话

终端对话模式，支持多轮对话和参数调节：

```bash
python scripts/chat.py --model outputs/sft --tokenizer outputs/tokenizer

# 自定义生成参数
python scripts/chat.py --model outputs/dpo --temperature 0.5 --max_tokens 500
```

对话中可用命令：
- `quit` / `exit`: 退出
- `clear`: 清空对话历史
- `params`: 查看/修改生成参数

## 3. Python API

### 3.1 model.generate()

```python
import sys; sys.path.insert(0, "src")
from model import ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer

model = ClearMindForCausalLM.from_pretrained("outputs/sft")
tokenizer = ClearMindTokenizer.load("outputs/tokenizer")

inputs = tokenizer("Human: 你好\nAssistant: ", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.2 pipeline

```python
from inference.generate import create_pipeline

pipe = create_pipeline(model, tokenizer, max_new_tokens=200)
result = pipe("Human: 什么是深度学习？\nAssistant: ")
print(result[0]["generated_text"])
```

### 3.3 流式生成

```python
from inference.generate import generate_stream

# 逐 token 输出到终端
generate_stream(model, tokenizer, "Human: 你好\nAssistant: ", max_new_tokens=200)
```

## 4. 模型格式

ClearMind-HF 使用 HuggingFace 标准格式保存：

```
outputs/sft/
  config.json              # 模型配置
  model.safetensors        # 模型权重
  tokenizer.json           # Tokenizer
  tokenizer_config.json    # Tokenizer 配置
  special_tokens_map.json  # 特殊 token 映射
```

可直接用 `from_pretrained()` 加载，兼容 HF 生态所有工具。

## 5. HuggingFace Hub 部署

```python
# 推送到 Hub
model.push_to_hub("your-username/clearmind-sft")
tokenizer.push_to_hub("your-username/clearmind-sft")

# 从 Hub 加载
model = ClearMindForCausalLM.from_pretrained("your-username/clearmind-sft")
```
