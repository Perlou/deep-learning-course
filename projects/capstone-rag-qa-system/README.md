# DocuMind AI - 智能文档问答系统

> 深度学习课程结课项目 | RAG + LLM 企业级实战

## 📖 项目简介

DocuMind AI 是一个基于检索增强生成（RAG）技术的企业级智能文档问答系统。用户可以上传各类文档（PDF、Word、TXT），系统通过向量检索和大语言模型，实现精准的文档问答能力。

## 🎯 项目目标

1. **技术综合运用**：整合课程所学的 Transformer、NLP、LLM 等核心技术
2. **企业级实践**：遵循工业界最佳实践，构建可部署的完整系统
3. **作品集展示**：作为 AI 工程师岗位求职的核心项目

## 📁 项目结构

```
capstone-rag-qa-system/
├── docs/                    # 项目文档
│   ├── PRD.md              # 产品需求文档
│   ├── TECHNICAL_DESIGN.md # 技术架构文档
│   └── API_DESIGN.md       # API 接口设计
├── designs/                 # UI 设计稿
├── src/                     # 源代码
│   ├── api/                # FastAPI 后端接口
│   ├── core/               # 核心业务逻辑
│   ├── frontend/           # 前端界面
│   ├── models/             # 模型相关代码
│   └── utils/              # 工具函数
├── tests/                   # 测试代码
├── data/                    # 数据目录
├── configs/                 # 配置文件
└── README.md               # 项目说明
```

## 🚀 核心功能

- 📄 **多格式文档解析**：支持 PDF、Word、TXT、Markdown
- 🔍 **智能语义检索**：基于向量相似度的精准段落检索
- 💬 **对话式问答**：支持多轮对话，理解上下文
- 📚 **来源引用**：回答附带原文出处，可溯源验证
- 🔧 **知识库管理**：文档上传、删除、更新

## 📋 文档索引

| 文档                                                  | 说明         | 状态      |
| ----------------------------------------------------- | ------------ | --------- |
| [PRD.md](docs/PRD.md)                                 | 产品需求文档 | 📝 编写中 |
| [TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)       | 技术架构设计 | 📝 编写中 |
| [API_DESIGN.md](docs/API_DESIGN.md)                   | API 接口设计 | 📝 编写中 |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | 实施计划     | 📝 编写中 |

## 🛠️ 技术栈

| 层级       | 技术选型                |
| ---------- | ----------------------- |
| 前端       | Streamlit / React       |
| 后端       | FastAPI + Uvicorn       |
| 向量数据库 | FAISS / Milvus          |
| 嵌入模型   | Sentence Transformers   |
| 大语言模型 | Qwen / LLaMA + LoRA     |
| 文档解析   | PyMuPDF, python-docx    |
| 部署       | Docker + Docker Compose |

## 📌 学习收获

完成本项目后，你将掌握：

1. RAG 系统的完整设计与实现
2. 向量数据库的使用与优化
3. LLM 推理服务的部署
4. 企业级后端 API 开发
5. 前后端分离架构实践

---

**当前状态**：📐 项目规划阶段
