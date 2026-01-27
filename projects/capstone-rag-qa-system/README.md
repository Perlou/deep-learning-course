# DocuMind AI - 智能文档问答系统

> 深度学习课程结课项目 | RAG + LLM 企业级实战

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![React](https://img.shields.io/badge/react-18.x-61dafb)
![License](https://img.shields.io/badge/license-MIT-yellow)

## 📖 项目简介

DocuMind AI 是一个基于检索增强生成（RAG）技术的企业级智能文档问答系统。用户可以上传各类文档（PDF、Word、TXT、Markdown），系统通过向量检索和大语言模型，实现精准的文档问答能力。

### ✨ 特性亮点

- 🎨 **现代化深色主题界面** - 玻璃拟态设计，视觉效果出众
- 🚀 **Ollama 本地 LLM 支持** - 私有部署，数据安全
- 📄 **多格式文档解析** - PDF、Word、TXT、Markdown
- 🔍 **智能语义检索** - BGE-Large 中文嵌入模型
- 💬 **流式对话响应** - 打字机效果，实时反馈
- 📚 **来源引用追溯** - 答案附带原文出处

---

## 🚀 快速开始

### 前置要求

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.ai/) (推荐，用于本地 LLM)

### 方式一：本地开发（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/capstone-rag-qa-system.git
cd capstone-rag-qa-system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装后端依赖
pip install -r requirements.txt

# 4. 初始化数据库
python scripts/init_db.py

# 5. 启动 Ollama 并拉取模型（可选，推荐）
ollama pull qwen2.5:7b

# 6. 启动后端（启用 Ollama）
USE_OLLAMA=true uvicorn src.api.main:app --reload --port 8000

# 7. 安装前端依赖
cd src/frontend && npm install

# 8. 启动前端
npm run dev

# 9. 访问应用
# 前端: http://localhost:5173
# API 文档: http://localhost:8000/docs
```

### 方式二：Docker 部署

```bash
# 1. 进入 docker 目录
cd docker

# 2. 复制环境变量配置
cp .env.example .env

# 3. 编辑 .env 配置 Ollama 地址（如需要）
# OLLAMA_BASE_URL=http://host.docker.internal:11434

# 4. 启动服务
docker-compose up -d

# 5. 访问应用
# 前端: http://localhost
# API: http://localhost:8000/api/v1
```

---

## 🎯 核心功能

| 功能              | 描述                         |
| ----------------- | ---------------------------- |
| 📄 **文档管理**   | 上传、解析、分块、向量化文档 |
| 🗃️ **知识库管理** | 创建多个知识库，分类管理文档 |
| 💬 **智能问答**   | 基于 RAG 的精准文档问答      |
| 🔄 **流式输出**   | SSE 实时流式响应             |
| 📚 **来源引用**   | 每条回答显示相关度和原文出处 |
| 📊 **对话历史**   | 保存和加载历史对话           |

---

## 🛠️ 技术栈

### 后端

| 组件     | 技术                 | 说明                |
| -------- | -------------------- | ------------------- |
| Web 框架 | FastAPI              | 高性能异步 API      |
| 数据库   | SQLite + SQLAlchemy  | 轻量持久化          |
| 向量存储 | FAISS                | Facebook 向量检索库 |
| 嵌入模型 | BGE-Large-ZH-v1.5    | 中文语义嵌入        |
| LLM      | Ollama (Qwen2.5-7B)  | 本地私有部署        |
| 文档解析 | PyMuPDF, python-docx | 多格式支持          |

### 前端

| 组件 | 技术                  | 说明         |
| ---- | --------------------- | ------------ |
| 框架 | React 18 + TypeScript | 类型安全     |
| 构建 | Vite 5                | 极速开发体验 |
| 样式 | Tailwind CSS          | 原子化 CSS   |
| 路由 | React Router 6        | 声明式路由   |
| 图标 | Lucide React          | 现代图标库   |

---

## 📁 项目结构

```
capstone-rag-qa-system/
├── configs/                 # 配置文件
│   └── config.yaml
├── data/                    # 数据目录
│   ├── uploads/            # 上传的文档
│   └── indices/            # FAISS 向量索引
├── designs/                 # UI 设计稿
├── docker/                  # Docker 部署
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── nginx.conf
├── docs/                    # 项目文档
│   ├── PRD.md              # 产品需求文档
│   ├── TECHNICAL_DESIGN.md # 技术架构
│   ├── API_DESIGN.md       # API 设计
│   └── USER_GUIDE.md       # 用户指南
├── src/
│   ├── api/                # FastAPI 后端
│   │   ├── main.py
│   │   └── routes/         # 路由模块
│   ├── core/               # 核心业务逻辑
│   │   ├── chat_service.py # 问答服务
│   │   ├── ollama_engine.py # Ollama LLM
│   │   ├── embedder.py     # 嵌入服务
│   │   ├── retriever.py    # 检索服务
│   │   └── vector_store.py # 向量存储
│   ├── frontend/           # React 前端
│   │   ├── src/
│   │   │   ├── components/ # 组件
│   │   │   ├── pages/      # 页面
│   │   │   └── lib/        # API 封装
│   │   └── package.json
│   ├── models/             # 数据库模型
│   ├── parsers/            # 文档解析器
│   └── utils/              # 工具函数
├── tests/                   # 测试代码
├── requirements.txt
└── README.md
```

---

## ⚙️ 环境变量

| 变量              | 默认值                   | 说明                    |
| ----------------- | ------------------------ | ----------------------- |
| `USE_OLLAMA`      | `false`                  | 启用 Ollama LLM         |
| `USE_MOCK_LLM`    | `true`                   | 使用 Mock LLM（测试用） |
| `OLLAMA_MODEL`    | `qwen2.5:7b`             | Ollama 模型名称         |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址         |

---

## 📋 文档索引

| 文档                                            | 说明         |
| ----------------------------------------------- | ------------ |
| [PRD.md](docs/PRD.md)                           | 产品需求文档 |
| [TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md) | 技术架构设计 |
| [API_DESIGN.md](docs/API_DESIGN.md)             | API 接口设计 |
| [USER_GUIDE.md](docs/USER_GUIDE.md)             | 用户使用指南 |
| [QUICK_START.md](docs/QUICK_START.md)           | 快速开始指南 |
| [PROGRESS_TRACKER.md](docs/PROGRESS_TRACKER.md) | 开发进度追踪 |
| [UI_DESIGN_SPEC.md](designs/UI_DESIGN_SPEC.md)  | UI 设计规范  |

---

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行端到端测试
pytest tests/test_e2e.py -v

# 手动处理待处理文档
python scripts/process_documents.py
```

---

## 📸 界面预览

### 知识库管理

玻璃拟态卡片设计，显示文档数和分块数统计。

### 文档管理

文件类型彩色图标，拖拽上传，状态实时更新。

### 智能问答

流式输出，来源引用进度条，支持多轮对话。

---

## 📌 学习收获

通过本项目，你将掌握：

1. **RAG 系统架构** - 检索增强生成的完整实现
2. **向量数据库** - FAISS 索引构建与查询优化
3. **LLM 推理服务** - Ollama 本地部署与 API 调用
4. **现代前端开发** - React + TypeScript + Tailwind
5. **企业级后端** - FastAPI 异步高性能 API
6. **容器化部署** - Docker + Nginx 生产部署

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE)。

---

**当前状态**：✅ v1.0.0 开发完成

- ✅ 后端 API 完整实现
- ✅ 前端界面重新设计
- ✅ Ollama LLM 集成
- ✅ Docker 部署支持
