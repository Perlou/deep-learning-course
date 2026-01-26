# DocuMind AI - 技术架构设计文档

> 版本: v1.0  
> 更新日期: 2026-01-26  
> 作者: Deep Learning Course Capstone Project

---

## 1. 系统架构总览

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DocuMind AI 系统架构                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        Presentation Layer                         │   │
│  │  ┌──────────────────────────────────────────────────────────────┐│   │
│  │  │                     React Frontend                           ││   │
│  │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   ││   │
│  │  │   │ 文档管理  │  │ 对话交互  │  │ 知识库   │  │   设置   │   ││   │
│  │  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘   ││   │
│  │  └──────────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                          API Layer                                │   │
│  │  ┌──────────────────────────────────────────────────────────────┐│   │
│  │  │                     FastAPI Backend                          ││   │
│  │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   ││   │
│  │  │   │ /docs    │  │ /chat    │  │ /kb      │  │ /health  │   ││   │
│  │  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘   ││   │
│  │  └──────────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        Service Layer                              │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │ DocumentSvc  │  │   ChatSvc    │  │   KBSvc      │           │   │
│  │  │              │  │              │  │              │           │   │
│  │  │ · 解析       │  │ · 检索       │  │ · CRUD       │           │   │
│  │  │ · 分块       │  │ · 生成       │  │ · 索引管理   │           │   │
│  │  │ · 向量化     │  │ · 会话管理   │  │              │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         Core Layer                                │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │ Embedder     │  │ VectorStore  │  │ LLM Engine   │           │   │
│  │  │              │  │              │  │              │           │   │
│  │  │ Sentence     │  │ FAISS /      │  │ Qwen / LLaMA │           │   │
│  │  │ Transformers │  │ Milvus       │  │ + LoRA       │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        Storage Layer                              │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │   SQLite     │  │ FAISS Index  │  │ File System  │           │   │
│  │  │              │  │              │  │              │           │   │
│  │  │ 元数据存储   │  │ 向量索引     │  │ 原始文档     │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 技术选型

| 层级     | 组件             | 技术选型              | 选型理由                       |
| -------- | ---------------- | --------------------- | ------------------------------ |
| 前端     | Web UI           | React + TypeScript    | 现代化、类型安全、组件化       |
| 前端     | UI 框架          | Tailwind + shadcn/ui  | 高效开发、一致性、响应式       |
| 后端     | API Server       | FastAPI               | 高性能、自动文档、异步支持     |
| 嵌入     | Embedding Model  | bge-large-zh-v1.5     | 中文效果优秀、开源免费         |
| 向量库   | Vector Store     | FAISS                 | 轻量级、无需额外服务、性能优秀 |
| LLM      | Language Model   | Qwen2.5-7B-Instruct   | 中文能力强、开源、可本地部署   |
| 推理     | Inference Engine | vLLM / Transformers   | 高效推理、流式输出             |
| 数据库   | Metadata DB      | SQLite                | 轻量级、无需配置、易于部署     |
| 文档解析 | Parser           | PyMuPDF + python-docx | 成熟稳定、格式支持全           |

---

## 2. 核心模块设计

### 2.1 文档处理模块

#### 2.1.1 文档解析器架构

```python
# 解析器基类
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> str:
        """解析文档返回纯文本"""
        pass

# 具体解析器实现
class PDFParser(BaseParser): ...
class DocxParser(BaseParser): ...
class TxtParser(BaseParser): ...
class MarkdownParser(BaseParser): ...

# 工厂模式创建解析器
class ParserFactory:
    @staticmethod
    def create(file_type: str) -> BaseParser:
        parsers = {
            'pdf': PDFParser,
            'docx': DocxParser,
            'txt': TxtParser,
            'md': MarkdownParser,
        }
        return parsers[file_type]()
```

#### 2.1.2 文本分块策略

```
┌─────────────────────────────────────────────────────────────┐
│                    文本分块流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   原始文档                                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 这是第一段内容，介绍了系统的背景...                    │   │
│   │ 这是第二段内容，描述了主要功能...                      │   │
│   │ 这是第三段内容，说明了技术架构...                      │   │
│   │ ...                                                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              RecursiveCharacterTextSplitter          │   │
│   │                                                      │   │
│   │  · chunk_size: 500 字符                              │   │
│   │  · chunk_overlap: 100 字符                           │   │
│   │  · separators: ["\n\n", "\n", "。", "，", " "]        │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   分块结果                                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│   │ Chunk 1  │  │ Chunk 2  │  │ Chunk 3  │  │ Chunk N  │   │
│   │ 500 chars│  │ 500 chars│  │ 500 chars│  │ ≤500     │   │
│   │ +metadata│  │ +metadata│  │ +metadata│  │ +metadata│   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**分块参数配置**:

```yaml
chunking:
  strategy: recursive # 分块策略
  chunk_size: 500 # 每块字符数
  chunk_overlap: 100 # 重叠字符数
  min_chunk_size: 50 # 最小块大小
  separators: # 分隔符优先级
    - "\n\n" # 段落
    - "\n" # 换行
    - "。" # 句号
    - "；" # 分号
    - "，" # 逗号
```

### 2.2 向量检索模块

#### 2.2.1 嵌入模型

```python
class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 1024

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文档"""
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入查询（添加查询前缀以提升效果）"""
        return self.model.encode(
            f"为这个句子生成表示以用于检索相关文章：{query}",
            normalize_embeddings=True
        )
```

#### 2.2.2 向量存储

```python
class VectorStore:
    def __init__(self, embedding_dim: int = 1024):
        # 使用 IndexFlatIP 内积索引（配合归一化向量等价于余弦相似度）
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunk_mapping: Dict[int, ChunkMetadata] = {}

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """添加向量到索引"""
        start_id = self.index.ntotal
        self.index.add(embeddings)
        for i, chunk in enumerate(chunks):
            self.chunk_mapping[start_id + i] = chunk.metadata

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """向量检索"""
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append(SearchResult(
                    chunk=self.chunk_mapping[idx],
                    score=float(score)
                ))
        return results
```

#### 2.2.3 检索流程

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 检索流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户问题: "这个系统支持哪些文件格式？"                       │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   Query Embedding (bge-large-zh-v1.5)               │   │
│   │   [0.12, -0.34, 0.56, ...] (1024维)                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   FAISS Vector Search (Top-K = 5)                   │   │
│   │                                                      │   │
│   │   Score: 0.89  Chunk: "支持PDF、Word、TXT格式..."    │   │
│   │   Score: 0.76  Chunk: "文档上传支持拖拽操作..."       │   │
│   │   Score: 0.71  Chunk: "格式转换模块负责..."          │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   Reranker (可选，使用 bge-reranker)                │   │
│   │   重排序优化检索结果                                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   返回 Top-K 相关段落给 LLM                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 LLM 生成模块

#### 2.3.1 模型选型

| 模型                     | 参数量 | 显存需求     | 特点                   |
| ------------------------ | ------ | ------------ | ---------------------- |
| Qwen2.5-7B-Instruct      | 7B     | ~16GB (FP16) | 中文效果优秀，推荐首选 |
| Qwen2.5-7B-Instruct-GPTQ | 7B     | ~8GB (INT4)  | 量化版本，显存友好     |
| Yi-6B-Chat               | 6B     | ~14GB        | 中文对话效果好         |
| ChatGLM3-6B              | 6B     | ~13GB        | 轻量级选择             |

#### 2.3.2 Prompt 模板

```python
RAG_PROMPT_TEMPLATE = """你是一个专业的文档问答助手。请基于以下参考内容回答用户的问题。

## 参考内容
{context}

## 注意事项
1. 只基于参考内容回答，不要编造信息
2. 如果参考内容无法回答问题，请明确说明
3. 回答时引用来源，格式：[来源: 文档名, 段落N]
4. 保持回答简洁专业

## 用户问题
{question}

## 回答
"""

class ChatService:
    def __init__(self, llm: BaseLLM, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    def chat(self, query: str, kb_id: str) -> Generator[str, None, None]:
        # 1. 检索相关段落
        chunks = self.retriever.retrieve(query, kb_id, top_k=5)

        # 2. 构建上下文
        context = self._format_context(chunks)

        # 3. 构建 Prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

        # 4. 流式生成
        for token in self.llm.stream_generate(prompt):
            yield token
```

#### 2.3.3 推理优化

```python
class LLMEngine:
    def __init__(self, model_path: str, quantization: str = "none"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载配置
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        # 量化配置
        if quantization == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "int8":
            load_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def stream_generate(self, prompt: str, max_tokens: int = 512) -> Generator:
        """流式生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token
```

---

## 3. 数据流设计

### 3.1 文档入库流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        文档入库数据流                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────┐ │
│   │  上传   │───▶│  解析   │───▶│  分块   │───▶│ 向量化  │───▶│ 入库  │ │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └───────┘ │
│        │              │              │              │              │     │
│        ▼              ▼              ▼              ▼              ▼     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────┐ │
│   │保存原始 │    │提取文本 │    │RecursiveCS│   │ BGE     │    │FAISS │ │
│   │文件到   │    │PyMuPDF │    │500/100   │   │ 1024维  │    │+SQLite│ │
│   │文件系统 │    │         │    │          │   │         │    │       │ │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └───────┘ │
│                                                                          │
│   状态更新: pending → parsing → chunking → embedding → completed         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 问答处理流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        问答处理数据流                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         用户输入问题                                     │
│                              │                                           │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Query Processing                              │   │
│   │  · 问题预处理（去除特殊字符）                                     │   │
│   │  · 构建历史上下文（多轮对话）                                     │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Retrieval Stage                               │   │
│   │  · Query Embedding (BGE)                                         │   │
│   │  · FAISS Search (Top-10)                                         │   │
│   │  · Rerank (BGE-Reranker, Top-5)                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Generation Stage                              │   │
│   │  · Prompt Construction                                           │   │
│   │  · LLM Inference (Stream)                                        │   │
│   │  · Source Attribution                                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Response Processing                           │   │
│   │  · SSE 流式输出                                                  │   │
│   │  · 保存对话历史                                                  │   │
│   │  · 返回来源引用                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 目录结构设计

```
capstone-rag-qa-system/
├── configs/                          # 配置文件
│   ├── config.yaml                   # 主配置文件
│   └── logging.yaml                  # 日志配置
│
├── src/                              # 源代码
│   ├── __init__.py
│   │
│   ├── api/                          # FastAPI 接口
│   │   ├── __init__.py
│   │   ├── main.py                   # 应用入口
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py          # 文档管理接口
│   │   │   ├── chat.py               # 问答接口
│   │   │   └── knowledge_base.py     # 知识库接口
│   │   ├── schemas/                  # Pydantic 模型
│   │   │   ├── __init__.py
│   │   │   ├── document.py
│   │   │   ├── chat.py
│   │   │   └── knowledge_base.py
│   │   └── dependencies.py           # 依赖注入
│   │
│   ├── core/                         # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── document_processor.py     # 文档处理
│   │   ├── chunker.py                # 文本分块
│   │   ├── embedder.py               # 嵌入模型
│   │   ├── vector_store.py           # 向量存储
│   │   ├── retriever.py              # 检索器
│   │   ├── llm_engine.py             # LLM 推理
│   │   └── chat_service.py           # 问答服务
│   │
│   ├── models/                       # 数据模型
│   │   ├── __init__.py
│   │   ├── database.py               # 数据库初始化
│   │   └── entities.py               # ORM 实体
│   │
│   ├── parsers/                      # 文档解析器
│   │   ├── __init__.py
│   │   ├── base.py                   # 解析器基类
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   ├── txt_parser.py
│   │   └── md_parser.py
│   │
├── frontend/                         # React 前端 (独立目录)
│   ├── package.json                  # 依赖配置
│   ├── tsconfig.json                 # TypeScript 配置
│   ├── tailwind.config.js            # Tailwind 配置
│   ├── src/
│   │   ├── App.tsx                   # 主应用
│   │   ├── main.tsx                  # 入口文件
│   │   ├── pages/
│   │   │   ├── ChatPage.tsx          # 对话页面
│   │   │   ├── DocumentsPage.tsx     # 文档管理页
│   │   │   └── SettingsPage.tsx      # 设置页
│   │   ├── components/
│   │   │   ├── ui/                   # shadcn 组件
│   │   │   ├── Sidebar.tsx           # 侧边栏组件
│   │   │   ├── ChatMessage.tsx       # 消息组件
│   │   │   └── FileUploader.tsx      # 上传组件
│   │   ├── hooks/                    # 自定义 Hooks
│   │   ├── lib/                      # 工具函数
│   │   └── types/                    # 类型定义
│   │
│   └── utils/                        # 工具函数
│       ├── __init__.py
│       ├── logger.py                 # 日志工具
│       └── helpers.py                # 辅助函数
│
├── tests/                            # 测试代码
│   ├── __init__.py
│   ├── test_parsers.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_api.py
│
├── data/                             # 数据目录
│   ├── uploads/                      # 上传的原始文件
│   ├── indices/                      # FAISS 索引文件
│   └── db/                           # SQLite 数据库
│       └── documind.db
│
├── scripts/                          # 脚本
│   ├── download_models.py            # 下载模型
│   └── init_db.py                    # 初始化数据库
│
├── docker/                           # Docker 配置
│   ├── Dockerfile
│   └── docker-compose.yaml
│
├── docs/                             # 项目文档
│   ├── PRD.md
│   ├── TECHNICAL_DESIGN.md
│   ├── API_DESIGN.md
│   └── IMPLEMENTATION_PLAN.md
│
├── requirements.txt                  # 依赖列表
├── pyproject.toml                    # 项目配置
└── README.md                         # 项目说明
```

---

## 5. 部署架构

### 5.1 开发环境

```
┌─────────────────────────────────────────────────────────────┐
│                    开发环境部署                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   本地机器 (GPU 可选)                                        │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                      │   │
│   │   ┌──────────────┐     ┌──────────────┐            │   │
│   │   │    React     │────▶│   FastAPI    │            │   │
│   │   │  :5173       │     │   :8000      │            │   │
│   │   └──────────────┘     └──────┬───────┘            │   │
│   │                               │                     │   │
│   │         ┌─────────────────────┼─────────────────┐   │   │
│   │         │                     │                 │   │   │
│   │         ▼                     ▼                 ▼   │   │
│   │   ┌──────────┐   ┌────────────────┐   ┌──────────┐ │   │
│   │   │  SQLite  │   │ FAISS (In-Mem) │   │ LLM/Emb  │ │   │
│   │   └──────────┘   └────────────────┘   └──────────┘ │   │
│   │                                                      │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Docker 部署

```yaml
# docker-compose.yaml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models # 模型挂载
    environment:
      - MODEL_PATH=/app/models/qwen2.5-7b-instruct
      - EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

---

## 6. 性能优化策略

### 6.1 嵌入优化

| 策略     | 说明                                 |
| -------- | ------------------------------------ |
| 批量嵌入 | 文档入库时批量处理，减少模型调用次数 |
| 嵌入缓存 | 相同文本使用缓存嵌入结果             |
| 异步处理 | 后台任务队列处理文档入库             |

### 6.2 检索优化

| 策略     | 说明                          |
| -------- | ----------------------------- |
| 索引分片 | 大规模数据使用分片索引        |
| IVF 索引 | 超过 10 万向量时切换 IVF 索引 |
| Reranker | 使用 reranker 模型二次排序    |

### 6.3 LLM 推理优化

| 策略     | 说明                       |
| -------- | -------------------------- |
| 模型量化 | INT4/INT8 量化减少显存占用 |
| KV Cache | 多轮对话复用 KV Cache      |
| 流式输出 | 降低感知延迟               |
| vLLM     | 使用 vLLM 提升吞吐         |

---

## 7. 监控与日志

### 7.1 日志设计

```python
# logging.yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    formatter: standard
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: standard

loggers:
  documind:
    level: INFO
    handlers: [console, file]
```

### 7.2 关键指标

| 指标                      | 说明         | 阈值          |
| ------------------------- | ------------ | ------------- |
| `document_parse_duration` | 文档解析耗时 | < 5s/MB       |
| `embedding_duration`      | 嵌入计算耗时 | < 100ms/chunk |
| `retrieval_duration`      | 检索耗时     | < 500ms       |
| `llm_first_token_latency` | 首字输出延迟 | < 2s          |
| `llm_tokens_per_second`   | 生成速度     | > 20 tokens/s |

---

## 附录

### A. 依赖列表

```txt
# requirements.txt

# Web Framework
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6
streamlit>=1.31.0

# LLM & Embedding
transformers>=4.37.0
sentence-transformers>=2.3.0
torch>=2.1.0
accelerate>=0.26.0
bitsandbytes>=0.42.0

# Vector Store
faiss-cpu>=1.7.4  # CPU 版本
# faiss-gpu>=1.7.4  # GPU 版本

# Document Parsing
PyMuPDF>=1.23.0
python-docx>=1.1.0
markdown>=3.5.0

# Database
sqlalchemy>=2.0.0
aiosqlite>=0.19.0

# Utils
pydantic>=2.5.0
pyyaml>=6.0.0
python-dotenv>=1.0.0
tqdm>=4.66.0
```

### B. 配置文件示例

```yaml
# config.yaml

app:
  name: DocuMind AI
  version: 1.0.0
  debug: true

server:
  host: 0.0.0.0
  port: 8000

models:
  embedding:
    name: BAAI/bge-large-zh-v1.5
    device: cuda
    batch_size: 32

  llm:
    name: Qwen/Qwen2.5-7B-Instruct
    quantization: int4 # none, int4, int8
    max_tokens: 512
    temperature: 0.7
    top_p: 0.9

retrieval:
  top_k: 5
  score_threshold: 0.5
  use_reranker: false

chunking:
  chunk_size: 500
  chunk_overlap: 100

storage:
  upload_dir: ./data/uploads
  index_dir: ./data/indices
  db_path: ./data/db/documind.db
```
