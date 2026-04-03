# Mini-Dify 技术架构文档

> Mini-Dify —— 简化版 LLM 应用开发平台

---

## 目录

1. [技术栈详细说明](#1-技术栈详细说明)
2. [系统架构图](#2-系统架构图)
3. [前端架构](#3-前端架构)
4. [后端架构](#4-后端架构)
5. [数据流](#5-数据流)
6. [部署架构](#6-部署架构)
7. [安全架构](#7-安全架构)
8. [前后端通信](#8-前后端通信)

---

## 1. 技术栈详细说明

### 前端

| 技术 | 版本 | 选型理由 |
|------|------|----------|
| **Next.js** | 14+ (App Router) | Server Components 减少客户端 JS 体积；App Router 提供基于文件系统的嵌套路由与 Layout；内置 Server Actions 简化表单提交；Middleware 层可拦截认证 |
| **TypeScript** | 5.x | 端到端类型安全，配合 Pydantic Schema 可生成前端类型定义 |
| **shadcn/ui** | latest | 基于 Radix UI 的无样式原语 + Tailwind 组合，组件代码直接拷入项目可深度定制，无运行时依赖 |
| **Tailwind CSS** | 3.x | Utility-first 保证一致性，PurgeCSS 生产包极小，与 shadcn/ui 天然配合 |
| **Zustand** | 4.x | 轻量（<1KB），无 Provider 嵌套，支持 middleware（persist、devtools），适合中等复杂度全局状态 |
| **React Flow** | @xyflow/react v12 | 成熟的节点-边画布库，内置拖拽、缩放、minimap，自定义节点类型简单，适合工作流可视化编排 |

### 后端

| 技术 | 版本 | 选型理由 |
|------|------|----------|
| **FastAPI** | 0.110+ | 原生 async/await，自动 OpenAPI 文档，Pydantic 校验，性能接近 Go/Node |
| **SQLAlchemy** | 2.x (async) | Python ORM 事实标准，2.x 原生 async Session，表达力强 |
| **Alembic** | 1.x | SQLAlchemy 官方迁移工具，支持自动生成迁移脚本与降级回滚 |
| **LangChain** | 0.2+ | 统一的 LLM 调用抽象，内置 RAG、Agent、Tool 调用链，社区生态丰富 |

### 基础设施

| 技术 | 版本 | 选型理由 |
|------|------|----------|
| **PostgreSQL** | 16 | ACID 事务、JSONB 存储灵活 Schema、pg_trgm 全文检索、成熟可靠 |
| **Milvus Standalone** | 2.x | 高性能向量检索，支持 IVF_FLAT / HNSW 索引，Standalone 模式部署简单 |
| **Redis** | 7 | 会话缓存、速率限制（滑动窗口）、Celery Broker、SSE 消息中转 |
| **NextAuth.js** | v5 | Next.js 生态首选认证库，OAuth 开箱即用，JWT + Session 双模式 |
| **Docker Compose** | v2 | 单机多容器编排，一条命令启动全部服务，适合开发与小规模部署 |

---

## 2. 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户浏览器                                 │
│                   Next.js CSR + React Flow Canvas                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  HTTPS
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Next.js Server (SSR)                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────┐  │
│  │  App Router   │  │  NextAuth.js  │  │  Server Components / API │  │
│  │  Middleware    │  │  (JWT 签发)    │  │  Route Handlers          │  │
│  └──────────────┘  └───────────────┘  └──────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  HTTP (内网)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (async)                         │
│  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌───────────────┐  │
│  │ REST API   │  │ SSE Stream │  │  Gateway   │  │  Background   │  │
│  │ /api/v1/*  │  │ /stream/*  │  │ /gateway/* │  │  Workers      │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘  └───────┬───────┘  │
│        └────────┬───────┴──────────┬────┘                │          │
│                 ▼                  ▼                      ▼          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐     │
│  │  Service Layer    │  │  Core Engine      │  │  Task Queue    │     │
│  │  (业务编排)        │  │  LLM / RAG /      │  │  (Celery)      │     │
│  │                   │  │  Agent / Workflow  │  │                │     │
│  └────────┬──────────┘  └────────┬─────────┘  └───────┬────────┘     │
│           ▼                      ▼                     ▼             │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                   Repository Layer                        │       │
│  └──────────────────────────┬────────────────────────────────┘       │
└─────────────────────────────┼────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌──────────────┐ ┌─────────────┐ ┌─────────────┐
     │ PostgreSQL   │ │   Milvus    │ │    Redis    │
     │ (关系数据)    │ │  (向量检索)  │ │ (缓存/队列)  │
     └──────────────┘ └─────────────┘ └─────────────┘
```

---

## 3. 前端架构

### 3.1 App Router 路由结构

```
frontend/src/app/
├── (auth)/                        # 路由组：认证相关（公开访问）
│   ├── login/page.tsx             # 登录页
│   ├── register/page.tsx          # 注册页（如需要）
│   └── layout.tsx                 # 居中卡片布局
├── (dashboard)/                   # 路由组：主面板（需登录）
│   ├── layout.tsx                 # 侧边栏 + TopBar 布局
│   ├── models/                    # 模型供应商配置（API Key 管理）
│   │   └── page.tsx
│   ├── prompts/                   # Prompt 工坊（模板 CRUD + 变量调试）
│   │   ├── page.tsx               # 列表
│   │   └── [id]/page.tsx          # 编辑器
│   ├── knowledge/                 # 知识库管理（文档上传 + 分块 + 向量化）
│   │   ├── page.tsx
│   │   └── [id]/page.tsx
│   ├── agents/                    # Agent 构建（工具绑定 + 对话测试）
│   │   ├── page.tsx
│   │   └── [id]/page.tsx
│   ├── workflows/                 # 工作流编排（React Flow 画布）
│   │   ├── page.tsx
│   │   └── [id]/page.tsx          # 画布编辑器
│   ├── apps/                      # 应用发布与管理
│   │   └── page.tsx
│   └── monitoring/                # 运行日志、Token 消耗、延迟监控
│       └── page.tsx
├── api/auth/[...nextauth]/route.ts  # NextAuth.js API Route
└── layout.tsx                     # 根布局（字体、主题、Provider）
```

**路由组说明：**
- `(auth)` 和 `(dashboard)` 使用 Next.js 路由组语法，不影响 URL 路径，仅用于 Layout 隔离
- `(dashboard)/layout.tsx` 中通过 NextAuth `getServerSession()` 检查登录状态，未登录重定向至 `/login`

### 3.2 组件分层

```
components/
├── ui/                  # shadcn/ui 基础组件（Button, Dialog, Table...）
├── layout/              # 布局组件（Sidebar, TopBar, PageContainer）
├── features/            # 业务功能组件
│   ├── workflow/        # 工作流画布、自定义节点、节点配置面板
│   ├── chat/            # 对话界面、消息气泡、流式渲染
│   ├── knowledge/       # 文档上传、分块预览
│   └── prompt/          # Prompt 编辑器、变量插值
└── shared/              # 跨功能共享组件（搜索框、状态标签、确认弹窗）
```

### 3.3 状态管理方案

采用 **Zustand** 进行客户端全局状态管理，按领域拆分 Store：

```
stores/
├── useAuthStore.ts        # 用户信息、Token（配合 NextAuth session）
├── useWorkflowStore.ts    # 画布节点/边状态、选中节点、执行状态
├── useChatStore.ts        # 对话消息列表、流式 buffer、加载状态
├── useKnowledgeStore.ts   # 知识库列表、上传进度
└── useUIStore.ts          # 侧边栏折叠、主题模式、Toast 队列
```

**状态分层原则：**

| 层级 | 工具 | 适用场景 |
|------|------|----------|
| 服务端状态 | Server Components + `fetch` | 页面初始数据、列表查询 |
| URL 状态 | `searchParams` / `useRouter` | 筛选、分页、Tab 切换 |
| 全局客户端状态 | Zustand | 跨组件共享（工作流画布、对话流） |
| 局部状态 | `useState` / `useReducer` | 表单输入、弹窗开关 |

---

## 4. 后端架构

### 4.1 分层架构

```
请求 → API Layer → Service Layer → Repository Layer → Database
         │              │                │
         │              │                └── SQLAlchemy async Session
         │              └── 业务逻辑编排、事务管理、调用 Core 引擎
         └── 路由定义、参数校验（Pydantic）、权限检查、响应序列化
```

**各层职责：**

- **API Layer**（`app/api/`）：定义路由、请求/响应 Schema 校验、依赖注入（当前用户、DB Session）、HTTP 状态码
- **Service Layer**（`app/services/`）：业务逻辑编排，调用一个或多个 Repository，管理事务边界，调用 Core 引擎
- **Repository Layer**（`app/repositories/`）：纯数据访问，封装 SQLAlchemy 查询，不含业务逻辑
- **Core Layer**（`app/core/`）：领域引擎，不依赖 HTTP 层，可独立测试

### 4.2 目录结构

```
backend/
├── app/
│   ├── main.py                    # FastAPI 实例、中间件注册、路由挂载
│   ├── config.py                  # Pydantic Settings（环境变量）
│   ├── dependencies.py            # 依赖注入（get_db, get_current_user）
│   ├── api/
│   │   ├── v1/                    # 内部 API（前端调用）
│   │   │   ├── models.py          # /api/v1/models
│   │   │   ├── prompts.py         # /api/v1/prompts
│   │   │   ├── knowledge.py       # /api/v1/knowledge
│   │   │   ├── agents.py          # /api/v1/agents
│   │   │   ├── workflows.py       # /api/v1/workflows
│   │   │   ├── apps.py            # /api/v1/apps
│   │   │   └── chat.py            # /api/v1/chat（SSE 流式）
│   │   └── gateway/               # 外部 Gateway API（第三方调用）
│   │       └── completion.py      # API Key 认证，兼容 OpenAI 格式
│   ├── core/
│   │   ├── llm/
│   │   │   ├── provider.py        # LLM Provider 抽象（OpenAI, Anthropic...）
│   │   │   └── callback.py        # LangChain Callback（Token 计量、日志）
│   │   ├── rag/
│   │   │   ├── loader.py          # 文档加载（PDF, Markdown, TXT）
│   │   │   ├── splitter.py        # 文本分块策略
│   │   │   ├── embedder.py        # Embedding 调用
│   │   │   └── retriever.py       # 向量检索 + 重排序
│   │   ├── agent/
│   │   │   ├── executor.py        # Agent 执行循环（ReAct）
│   │   │   └── tools/             # 内置工具（搜索、计算、代码执行）
│   │   └── workflow/
│   │       ├── engine.py          # DAG 执行引擎
│   │       ├── nodes.py           # 节点类型定义（LLM、条件、HTTP...）
│   │       └── context.py         # 执行上下文与变量传递
│   ├── models/                    # SQLAlchemy ORM 模型
│   │   ├── user.py
│   │   ├── app.py
│   │   ├── prompt.py
│   │   ├── knowledge.py
│   │   ├── workflow.py
│   │   └── conversation.py
│   ├── schemas/                   # Pydantic Request/Response Schema
│   ├── services/                  # 业务服务层
│   ├── repositories/              # 数据访问层
│   └── utils/
│       ├── security.py            # 加密、哈希
│       └── pagination.py          # 分页工具
├── alembic/                       # 数据库迁移
│   ├── versions/
│   └── env.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── requirements.txt
└── Dockerfile
```

---

## 5. 数据流

### 5.1 认证流程

```
浏览器                    Next.js Server              FastAPI
  │                           │                          │
  │  1. 点击 GitHub 登录       │                          │
  ├──────────────────────────►│                          │
  │                           │  2. 重定向 GitHub OAuth   │
  │◄──────────────────────────┤                          │
  │  3. GitHub 回调 + code     │                          │
  ├──────────────────────────►│                          │
  │                           │  4. 换取 access_token     │
  │                           │  5. 获取用户信息            │
  │                           │  6. 签发 JWT (NextAuth)   │
  │  7. Set-Cookie (session)  │                          │
  │◄──────────────────────────┤                          │
  │                           │                          │
  │  8. 请求 /api/v1/prompts  │                          │
  ├──────────────────────────►│                          │
  │                           │  9. 附加 Authorization    │
  │                           │     Bearer <JWT>          │
  │                           ├─────────────────────────►│
  │                           │                          │ 10. 验证 JWT
  │                           │                          │ 11. 查询数据
  │                           │  12. JSON Response       │
  │                           │◄─────────────────────────┤
  │  13. 渲染页面              │                          │
  │◄──────────────────────────┤                          │
```

### 5.2 标准 CRUD 数据流（以创建 Prompt 为例）

```
1. 用户填写表单 → 点击保存
2. 前端调用 POST /api/v1/prompts（附带 JWT）
3. API Layer：Pydantic 校验 Request Body → 注入 current_user
4. Service Layer：业务校验（名称唯一性）→ 构建 ORM 对象
5. Repository Layer：async session.add() → commit()
6. Service Layer：返回创建后的对象
7. API Layer：序列化为 Response Schema → 201 Created
8. 前端收到响应 → Zustand 更新列表 → UI 刷新
```

### 5.3 SSE 流式对话数据流

```
浏览器                        FastAPI                     LLM Provider
  │                              │                            │
  │  POST /api/v1/chat/stream    │                            │
  │  { message, agent_id }       │                            │
  ├─────────────────────────────►│                            │
  │                              │  构建 LangChain Chain       │
  │                              │  调用 LLM (stream=True)     │
  │                              ├───────────────────────────►│
  │                              │                            │
  │  SSE: data: {"token":"你"}   │◄─ chunk: "你"              │
  │◄─────────────────────────────┤                            │
  │  SSE: data: {"token":"好"}   │◄─ chunk: "好"              │
  │◄─────────────────────────────┤                            │
  │  SSE: data: {"token":"！"}   │◄─ chunk: "！"              │
  │◄─────────────────────────────┤                            │
  │                              │                            │
  │  SSE: data: {"done":true,    │  保存完整回复到 DB           │
  │       "usage":{...}}         │  记录 Token 用量             │
  │◄─────────────────────────────┤                            │
  │  event: [DONE]               │                            │
  │◄─────────────────────────────┤                            │
```

**前端 SSE 消费伪代码：**

```typescript
const response = await fetch("/api/v1/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
  body: JSON.stringify({ message, agent_id }),
});

const reader = response.body!.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  // 解析 SSE 格式，逐 token 追加到消息 buffer
  useChatStore.getState().appendToken(chunk);
}
```

---

## 6. 部署架构

### 6.1 Docker Compose 服务拓扑

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose Network                 │
│                     (mini-dify-net)                      │
│                                                         │
│  ┌─────────────┐     ┌──────────────┐                   │
│  │  frontend    │────►│   backend    │                   │
│  │  Next.js     │     │   FastAPI    │                   │
│  │  :3000       │     │   :8000      │                   │
│  └─────────────┘     └──────┬───────┘                   │
│                             │                           │
│              ┌──────────────┼──────────────┐            │
│              ▼              ▼              ▼            │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────┐      │
│  │  postgres     │  │   milvus    │  │  redis   │      │
│  │  :5432        │  │   :19530    │  │  :6379   │      │
│  │  Volume:      │  │   Volume:   │  │  Volume: │      │
│  │  pg_data      │  │  milvus_data│  │ redis_data│      │
│  └──────────────┘  └──────┬──────┘  └──────────┘      │
│                           │                             │
│                    ┌──────┴──────┐                      │
│                    │    etcd     │                      │
│                    │   :2379     │                      │
│                    │  (Milvus    │                      │
│                    │   元数据)    │                      │
│                    └─────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### 6.2 服务配置概要

| 服务 | 镜像 | 端口映射 | 持久化卷 | 依赖 |
|------|------|---------|----------|------|
| `frontend` | 自建（Node 20 Alpine） | 3000:3000 | - | `backend` |
| `backend` | 自建（Python 3.11 Slim） | 8000:8000 | - | `postgres`, `redis`, `milvus` |
| `postgres` | postgres:16-alpine | 5432:5432 | `pg_data` | - |
| `redis` | redis:7-alpine | 6379:6379 | `redis_data` | - |
| `milvus` | milvusdb/milvus:v2.4.x | 19530:19530 | `milvus_data` | `etcd` |
| `etcd` | quay.io/coreos/etcd:v3.5.x | 2379 (内部) | `etcd_data` | - |

### 6.3 环境变量管理

```bash
# .env（不提交至 Git）
POSTGRES_URL=postgresql+asyncpg://user:pass@postgres:5432/minidify
REDIS_URL=redis://redis:6379/0
MILVUS_HOST=milvus
MILVUS_PORT=19530

NEXTAUTH_SECRET=<random-32-bytes>
NEXTAUTH_URL=http://localhost:3000
GITHUB_CLIENT_ID=xxx
GITHUB_CLIENT_SECRET=xxx
GOOGLE_CLIENT_ID=xxx
GOOGLE_CLIENT_SECRET=xxx

OPENAI_API_KEY=sk-xxx
```

---

## 7. 安全架构

### 7.1 JWT 验证流程

```
┌──────────┐     ┌────────────────┐     ┌────────────────┐
│  浏览器   │────►│ Next.js        │────►│  FastAPI       │
│          │     │ Middleware      │     │  Dependencies  │
└──────────┘     └────────┬───────┘     └───────┬────────┘
                          │                     │
                 1. 从 Cookie 取出               2. 从 Authorization
                    NextAuth Session               Header 取出 JWT
                 3. 验证签名 + 过期时间            4. 解码 payload
                 4. 附加到请求 Header              5. 查询用户是否存在
                                                 6. 注入 current_user
```

**JWT Payload 结构：**

```json
{
  "sub": "user_id_uuid",
  "email": "user@example.com",
  "name": "User Name",
  "iat": 1700000000,
  "exp": 1700086400
}
```

### 7.2 API Key 加密

外部 Gateway API 使用 API Key 认证（面向第三方集成）：

- **生成**：`secrets.token_urlsafe(32)` 生成明文 Key，仅在创建时展示一次
- **存储**：使用 `bcrypt` 对 Key 做单向哈希后存入 PostgreSQL
- **前缀**：保留前 8 位明文（如 `md-xxxx...`）供用户识别
- **验证**：请求到达时，遍历用户 Key 列表逐一 `bcrypt.checkpw()`（数量有限，性能可接受）
- **传输**：仅支持 HTTPS，Key 通过 `X-API-Key` Header 传递

### 7.3 代码沙箱

Agent 工具中如涉及代码执行（如 Python 计算工具）：

- **进程隔离**：使用 `subprocess` 启动独立进程，设置 `timeout` 上限（默认 30 秒）
- **资源限制**：通过 `resource.setrlimit()` 限制内存（256MB）和 CPU 时间
- **文件系统**：使用临时目录 + 只读挂载，禁止网络访问
- **Docker 方案（可选）**：生产环境可将代码执行放入独立的沙箱容器，通过 gVisor 运行时进一步隔离

### 7.4 其他安全措施

| 措施 | 实现方式 |
|------|----------|
| CORS | FastAPI `CORSMiddleware`，限制 Origin 白名单 |
| 速率限制 | Redis 滑动窗口，按 User ID 限制（如 60 req/min） |
| SQL 注入防护 | SQLAlchemy 参数化查询，禁止拼接原始 SQL |
| XSS 防护 | React 默认转义 + CSP Header |
| 敏感数据脱敏 | API Key 返回时仅展示前 8 位，日志中 mask 敏感字段 |

---

## 8. 前后端通信

### 8.1 REST API 约定

**Base URL：** `/api/v1`

**命名规范：**
- 资源名使用复数形式：`/prompts`, `/workflows`, `/agents`
- 嵌套资源：`/knowledge/{id}/documents`
- 动作型接口：`POST /workflows/{id}/run`

**标准 CRUD 映射：**

| 操作 | 方法 | 路径 | 状态码 |
|------|------|------|--------|
| 列表查询 | GET | `/api/v1/prompts?page=1&size=20` | 200 |
| 单条查询 | GET | `/api/v1/prompts/{id}` | 200 |
| 创建 | POST | `/api/v1/prompts` | 201 |
| 更新 | PUT | `/api/v1/prompts/{id}` | 200 |
| 删除 | DELETE | `/api/v1/prompts/{id}` | 204 |

**分页响应格式：**

```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "size": 20,
  "pages": 5
}
```

### 8.2 SSE 流式协议

**请求方式：** `POST`（需要传递 Body，不使用 GET + EventSource）

**Content-Type：** `text/event-stream`

**事件格式：**

```
event: message
data: {"type":"token","content":"你"}

event: message
data: {"type":"token","content":"好"}

event: message
data: {"type":"tool_call","name":"search","input":{"query":"天气"}}

event: message
data: {"type":"tool_result","name":"search","output":"北京今天晴..."}

event: message
data: {"type":"done","usage":{"prompt_tokens":120,"completion_tokens":45}}

event: done
data: [DONE]
```

**事件类型枚举：**

| type | 含义 |
|------|------|
| `token` | LLM 生成的文本片段 |
| `tool_call` | Agent 发起工具调用 |
| `tool_result` | 工具返回结果 |
| `node_start` | 工作流节点开始执行 |
| `node_finish` | 工作流节点执行完毕 |
| `error` | 执行错误 |
| `done` | 流式结束，附带 Token 用量统计 |

### 8.3 错误处理

**统一错误响应格式：**

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Prompt with id 'xxx' not found",
    "details": null
  }
}
```

**错误码与 HTTP 状态码映射：**

| HTTP 状态码 | error.code | 场景 |
|------------|------------|------|
| 400 | `VALIDATION_ERROR` | 请求参数校验失败 |
| 401 | `UNAUTHORIZED` | JWT 缺失或过期 |
| 403 | `FORBIDDEN` | 无权限访问该资源 |
| 404 | `RESOURCE_NOT_FOUND` | 资源不存在 |
| 409 | `CONFLICT` | 资源冲突（名称重复等） |
| 422 | `UNPROCESSABLE_ENTITY` | 业务逻辑校验失败 |
| 429 | `RATE_LIMITED` | 超出速率限制 |
| 500 | `INTERNAL_ERROR` | 服务器内部错误 |
| 502 | `LLM_PROVIDER_ERROR` | LLM 供应商调用失败 |

**前端错误处理策略：**

```typescript
// lib/api.ts — 统一请求封装
async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json();
    throw new ApiError(res.status, body.error.code, body.error.message);
  }

  return res.json();
}
```

---

## 附：关键设计决策记录

| 决策 | 选项 | 结论 | 理由 |
|------|------|------|------|
| 前后端分离 vs BFF | 分离 + Next.js 中转 | Next.js 作为 BFF 层 | Server Components 可直接调后端，减少客户端请求；NextAuth 处理认证后转发 JWT |
| 流式方案 | WebSocket vs SSE | SSE | 对话场景为单向流（服务端→客户端），SSE 更轻量，兼容 HTTP/2，无需维护连接状态 |
| 向量库 | Chroma vs Milvus | Milvus | Milvus 支持更大数据量、分布式扩展，Standalone 模式部署复杂度可接受 |
| 状态管理 | Redux vs Zustand | Zustand | 本项目状态复杂度中等，Zustand 更轻量、样板代码少 |
| 工作流执行 | 同步 vs 异步队列 | 混合 | 短工作流同步执行直接 SSE 返回；长工作流提交至 Celery 后台，通过 Redis Pub/Sub 推送进度 |
