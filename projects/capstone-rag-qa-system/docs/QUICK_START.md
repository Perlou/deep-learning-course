# DocuMind AI - 快速启动指南

> 本指南帮助您快速启动和运行 DocuMind AI 系统

---

## 📋 前置要求

- Python 3.10+
- (可选) NVIDIA GPU + CUDA 11.8+（用于 LLM 推理加速）

---

## 🚀 快速开始

### 1. 进入项目目录

```bash
cd /Users/perlou/Desktop/personal/deep-learning-course/projects/capstone-rag-qa-system
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> ⚠️ 如果 PyTorch 安装较慢，可以使用国内镜像：
>
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 4. 初始化数据库

```bash
python scripts/init_db.py
```

### 5. （可选）下载模型

```bash
python scripts/download_models.py
```

> 💡 模型较大，首次运行时会自动下载，也可以手动提前下载

---

## 🖥️ 启动服务

### 方式一：分别启动后端和前端

**终端 1 - 启动 FastAPI 后端**：

```bash
cd /Users/perlou/Desktop/personal/deep-learning-course/projects/capstone-rag-qa-system
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**终端 2 - 启动 Streamlit 前端**：

```bash
cd /Users/perlou/Desktop/personal/deep-learning-course/projects/capstone-rag-qa-system
streamlit run src/frontend/app.py --server.port 8501
```

### 方式二：使用启动脚本（待创建）

```bash
./scripts/start.sh
```

---

## 🌐 访问服务

| 服务     | 地址                                       | 说明                  |
| -------- | ------------------------------------------ | --------------------- |
| 前端界面 | http://localhost:8501                      | Streamlit 用户界面    |
| API 文档 | http://localhost:8000/docs                 | Swagger UI 交互式文档 |
| ReDoc    | http://localhost:8000/redoc                | ReDoc 风格文档        |
| 健康检查 | http://localhost:8000/api/v1/system/health | 系统健康状态          |

---

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_api.py -v
```

---

## 📁 项目结构

```
capstone-rag-qa-system/
├── configs/
│   └── config.yaml         # 配置文件
├── src/
│   ├── api/                 # FastAPI 后端
│   │   ├── main.py         # 应用入口
│   │   ├── routes/         # 路由模块
│   │   └── schemas/        # 数据模型
│   ├── core/               # 核心业务逻辑（待实现）
│   ├── frontend/           # Streamlit 前端
│   │   └── app.py          # 前端入口
│   ├── models/             # 数据库模型
│   │   ├── database.py     # 数据库配置
│   │   └── entities.py     # 实体定义
│   ├── parsers/            # 文档解析器（待实现）
│   └── utils/              # 工具模块
│       ├── config.py       # 配置管理
│       ├── logger.py       # 日志管理
│       └── helpers.py      # 辅助函数
├── scripts/
│   ├── init_db.py          # 数据库初始化
│   └── download_models.py  # 模型下载
├── tests/
│   └── test_api.py         # API 测试
├── data/                   # 数据目录
├── logs/                   # 日志目录
└── requirements.txt        # 依赖列表
```

---

## ⚙️ 配置说明

编辑 `configs/config.yaml` 可以修改以下配置：

| 配置项                    | 说明     | 默认值                   |
| ------------------------- | -------- | ------------------------ |
| `models.embedding.name`   | 嵌入模型 | BAAI/bge-large-zh-v1.5   |
| `models.llm.name`         | LLM 模型 | Qwen/Qwen2.5-7B-Instruct |
| `models.llm.quantization` | 量化方式 | none (可选: int4, int8)  |
| `chunking.chunk_size`     | 分块大小 | 500                      |
| `retrieval.top_k`         | 检索数量 | 5                        |

---

## 🔧 常见问题

### Q: 显存不足怎么办？

修改 `configs/config.yaml`，启用量化：

```yaml
models:
  llm:
    quantization: int4 # 或 int8
```

### Q: 模型下载太慢？

使用 ModelScope 镜像下载：

```bash
# 安装 modelscope
pip install modelscope

# 下载模型
modelscope download --model qwen/Qwen2.5-7B-Instruct --local_dir ./models/qwen
```

### Q: 如何使用 CPU 推理？

修改 `configs/config.yaml`：

```yaml
models:
  llm:
    device: cpu
```

---

## 📝 当前开发状态

| 模块              | 状态             |
| ----------------- | ---------------- |
| ✅ 项目结构       | 已完成           |
| ✅ 配置系统       | 已完成           |
| ✅ 数据库模型     | 已完成           |
| ✅ FastAPI 框架   | 已完成           |
| ✅ Streamlit 界面 | 已完成（基础版） |
| ⏳ 文档解析器     | Week 2           |
| ⏳ 向量检索       | Week 3           |
| ⏳ LLM 集成       | Week 4           |

---

## 📞 下一步

1. 安装依赖并启动服务，验证基础框架正常运行
2. 继续 Week 2 开发：实现文档解析器
