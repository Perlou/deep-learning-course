# HuggingFace 镜像配置说明

## 问题描述

当从 HuggingFace 下载模型时，可能会遇到以下错误：

- `SSLEOFError: EOF occurred in violation of protocol`
- `ReadTimeoutError: Read timed out`
- `MaxRetryError: Max retries exceeded`

这通常是由于网络环境限制导致的。

## 解决方案

### 方法1: 使用环境变量（推荐）

代码已经自动配置了HuggingFace镜像，无需额外操作。如果仍有问题，可以在终端中手动设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train.py
```

### 方法2: 手动下载模型

如果镜像仍然无法工作，可以手动下载模型：

#### 1. 创建模型目录

```bash
mkdir -p models/bert-base-chinese
```

#### 2. 从镜像站下载模型文件

访问 [https://hf-mirror.com/bert-base-chinese](https://hf-mirror.com/bert-base-chinese)

下载以下文件到 `models/bert-base-chinese/` 目录:

- `config.json`
- `pytorch_model.bin`
- `tokenizer_config.json`
- `vocab.txt`

#### 3. 修改代码使用本地模型

在 `train.py` 和 `bert_classifier.py` 中，将：

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
```

改为：

```python
tokenizer = BertTokenizer.from_pretrained("./models/bert-base-chinese")
model = BertModel.from_pretrained("./models/bert-base-chinese")
```

### 方法3: 使用其他镜像源

如果 hf-mirror.com 不可用，可以尝试其他镜像：

```bash
# ModelScope镜像
export HF_ENDPOINT=https://modelscope.cn

# 或者在代码中修改
os.environ["HF_ENDPOINT"] = "https://modelscope.cn"
```

## 验证配置

运行以下命令测试配置是否成功：

```bash
python bert_classifier.py
```

如果看到 "✓ Tokenizer加载成功" 和 "✓ 模型创建成功"，说明配置正确。

## 常见问题

**Q: 为什么下载速度很慢？**
A: 镜像站的速度取决于网络环境。如果太慢，建议使用方法2手动下载。

**Q: 出现其他SSL错误怎么办？**
A: 尝试更新 pip 和相关包：

```bash
pip install --upgrade pip
pip install --upgrade transformers torch
```

**Q: 是否需要代理？**
A: 使用镜像站通常不需要代理。如果必须使用代理，设置：

```bash
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
```

## 技术细节

- **HF_ENDPOINT**: HuggingFace 官方支持的环境变量，用于设置模型下载源
- **镜像原理**: 镜像站会同步 HuggingFace 的模型文件，提供国内访问
- **生效时机**: 必须在导入 `transformers` 之前设置环境变量

## 参考链接

- HuggingFace镜像站: [https://hf-mirror.com](https://hf-mirror.com)
- BERT中文模型: [https://hf-mirror.com/bert-base-chinese](https://hf-mirror.com/bert-base-chinese)
