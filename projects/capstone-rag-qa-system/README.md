# DocuMind AI - 智能文档问答系统

> 基于 RAG + LLM 的企业级文档问答系统 | 深度学习课程结课项目

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![React](https://img.shields.io/badge/react-18.x-61dafb)

## ✨ 核心特性

- 🎨 **现代化界面** - 深色主题 + 玻璃拟态设计
- 🚀 **本地 LLM** - Ollama 支持，数据私有化
- 📄 **多格式解析** - PDF、Word、TXT、Markdown
- 💬 **流式对话** - 打字机效果，实时响应
- 📚 **来源追溯** - 自动标注引用来源
- 📊 **Ragas 评估** - 系统化 RAG 质量评估，支持本地/线上 LLM

---

## 🚀 快速开始

### 前置要求

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.ai/)（可选）

### 一键启动

```bash
# 1. 安装依赖
pip install -r requirements.txt
cd src/frontend && npm install && cd ../..

# 2. 初始化数据库
python scripts/init_db.py

# 3. 启动后端（自动停止占用端口的进程）
./scripts/start_backend.sh

# 4. 启动前端（新终端）
./scripts/start_frontend.sh

# 访问: http://localhost:5173
```

### 使用 Ollama（推荐）

```bash
# 1. 安装并启动 Ollama
ollama pull qwen2.5:7b

# 2. 启动后端
USE_OLLAMA=true ./scripts/start_backend.sh
```

### 停止服务

```bash
./scripts/stop_backend.sh
```

---

## 📋 主要功能

| 功能        | 说明                   |
| ----------- | ---------------------- |
| 📄 文档管理 | 上传、解析、向量化文档 |
| 🗃️ 知识库   | 多知识库分类管理       |
| 💬 智能问答 | RAG 精准问答           |
| 🔄 流式输出 | SSE 实时响应           |
| 📚 来源引用 | 可折叠的引用卡片       |
| 📊 RAG 评估 | Ragas 多维度质量评估   |

---

## 🛠️ 技术栈

**后端**: FastAPI + SQLite + FAISS + BGE-Large-ZH + Ollama + Ragas  
**前端**: React 18 + TypeScript + Vite + Tailwind CSS

---

## 📁 项目结构

```
├── src/
│   ├── api/           # FastAPI 后端
│   ├── core/          # 核心业务（LLM、检索、向量）
│   ├── frontend/      # React 前端
│   ├── models/        # 数据库模型
│   └── parsers/       # 文档解析器
├── scripts/           # 启动脚本
├── tests/             # 测试
└── docs/              # 文档
```

---

## ⚙️ 环境变量

```bash
USE_OLLAMA=true        # 使用 Ollama LLM
USE_MOCK_LLM=true      # 使用 Mock 模式（测试）
OLLAMA_MODEL=qwen2.5:7b
```

---

## 🧪 测试

```bash
pytest tests/ -v
```

---

## 📚 文档

- [技术设计](docs/TECHNICAL_DESIGN.md)
- [API 文档](docs/API_DESIGN.md)
- [用户指南](docs/USER_GUIDE.md)
- [开发进度](docs/PROGRESS_TRACKER.md)

---

## 📄 License

MIT License

---

**状态**: ✅ v1.0.0 已完成
