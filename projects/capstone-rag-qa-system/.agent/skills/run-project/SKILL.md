---
name: run-project
description: 运行和测试 DocuMind AI 项目
---

# 运行项目技能

此技能用于运行和测试 DocuMind AI 项目。

## 快速启动

### 1. 进入项目目录

```bash
cd /Users/perlou/Desktop/personal/deep-learning-course/projects/capstone-rag-qa-system
```

### 2. 激活虚拟环境

```bash
source venv/bin/activate
# 如果没有虚拟环境，先创建：
# python -m venv venv && source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 初始化数据库

```bash
python scripts/init_db.py
```

## 启动服务

### 启动后端 (FastAPI)

```bash
# 开发模式（热重载）
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 启动前端 (Streamlit)

```bash
streamlit run src/frontend/app.py --server.port 8501
```

### 同时启动（使用两个终端）

**终端 1**:

```bash
uvicorn src.api.main:app --reload --port 8000
```

**终端 2**:

```bash
streamlit run src/frontend/app.py
```

## 访问服务

| 服务     | 地址                                       |
| -------- | ------------------------------------------ |
| 前端界面 | http://localhost:8501                      |
| API 文档 | http://localhost:8000/docs                 |
| 健康检查 | http://localhost:8000/api/v1/system/health |

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_api.py -v

# 运行特定测试类
pytest tests/test_api.py::TestKnowledgeBaseAPI -v

# 运行特定测试方法
pytest tests/test_api.py::TestKnowledgeBaseAPI::test_create_kb -v

# 显示覆盖率
pytest tests/ -v --cov=src --cov-report=html
```

## 开发调试

### 使用 Python REPL 测试

```python
# 进入项目目录
import sys
sys.path.insert(0, "/Users/perlou/Desktop/personal/deep-learning-course/projects/capstone-rag-qa-system")

# 测试配置加载
from src.utils import get_settings
settings = get_settings()
print(settings.app.name)

# 测试数据库连接
from src.models import init_db
init_db()
```

### 使用 curl 测试 API

```bash
# 健康检查
curl http://localhost:8000/api/v1/system/health

# 创建知识库
curl -X POST http://localhost:8000/api/v1/kb \
  -H "Content-Type: application/json" \
  -d '{"name": "测试知识库", "description": "测试"}'

# 获取知识库列表
curl http://localhost:8000/api/v1/kb
```

## 常见问题

### 端口被占用

```bash
# 查找占用端口的进程
lsof -i :8000

# 杀死进程
kill -9 <PID>
```

### 模块导入错误

确保在项目根目录运行，或设置 PYTHONPATH：

```bash
export PYTHONPATH="${PYTHONPATH}:/Users/perlou/Desktop/personal/deep-learning-course/projects/capstone-rag-qa-system"
```

### 数据库错误

重新初始化数据库：

```bash
rm -rf data/db/
python scripts/init_db.py
```
