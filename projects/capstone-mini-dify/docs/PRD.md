# Mini-Dify - 产品需求文档 (PRD)

> 版本: v2.0
> 更新日期: 2026-04-02
> 项目类型: 企业级项目

---

## 1. 项目背景

### 1.1 项目定位

Mini-Dify 是一个**简化版的 LLM 应用开发平台**，灵感来源于开源项目 [Dify](https://github.com/langgenius/dify)。它提供了一套可视化的 LLM 应用构建工具，让用户可以通过图形界面完成从模型配置、Prompt 编写、知识库搭建、Agent 创建到工作流编排的全流程，并将这些能力组装成可对外发布的 AI 应用。

### 1.2 项目目标

作为 LLM 课程的毕业项目，可以最终上线，是一个企业级的项目

### 1.3 目标用户

| 用户类型       | 典型场景                      | 核心诉求             |
| -------------- | ----------------------------- | -------------------- |
| LLM 应用开发者 | 快速搭建和测试 LLM 应用原型   | 低代码、快速迭代     |
| 业务人员       | 通过自然语言配置 AI 能力      | 无需编码、可视化操作 |
| AI 团队负责人  | 管理团队的模型资源和应用      | 统一管理、成本可控   |
| 课程学习者     | 理解 LLM 应用平台的设计与实现 | 学习参考、代码清晰   |

### 1.4 产品愿景

> "让每个人都能像搭积木一样构建 AI 应用"

---

## 2. 功能需求

### 2.1 核心功能架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mini-Dify 功能架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  模型管理中心 │  │  Prompt 工坊  │  │  知识库管理   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐          │
│  │ · 多供应商    │  │ · 模板编辑    │  │ · 文档上传    │          │
│  │ · API Key管理 │  │ · 变量注入    │  │ · 切分策略    │          │
│  │ · 参数预设    │  │ · 多模型测试  │  │ · 检索测试    │          │
│  │ · 模型探测    │  │ · 版本管理    │  │ · 多知识库    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Agent 构建器 │  │  工作流引擎   │  │  应用管理     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐          │
│  │ · ReAct/FC   │  │ · 可视化编排  │  │ · Chatbot    │          │
│  │ · 内置工具    │  │ · 5种节点类型 │  │ · Completion │          │
│  │ · 自定义工具  │  │ · 运行调试    │  │ · Workflow   │          │
│  │ · Playground  │  │ · 流式输出    │  │ · API 发布   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  API 网关     │  │  监控分析     │                             │
│  └──────┬───────┘  └──────┬───────┘                             │
│         │                 │                                      │
│  ┌──────▼───────┐  ┌──────▼───────┐                             │
│  │ · API Key 认证│  │ · 对话日志    │                             │
│  │ · 速率限制    │  │ · Token 统计  │                             │
│  │ · SSE 流式    │  │ · 成本分析    │                             │
│  └──────────────┘  └──────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 功能详细说明

#### 2.2.1 模型管理中心 (Model Hub)

**课程对应**: Phase 1 (LLM API 调用)

| 功能项       | 描述                                       | 优先级 |
| ------------ | ------------------------------------------ | ------ |
| 供应商管理   | 添加 / 配置 LLM 供应商 (OpenAI, Claude 等) | P0     |
| API Key 管理 | 安全存储和管理 API Key（加密）             | P0     |
| 模型列表     | 展示每个供应商可用的模型列表               | P0     |
| 参数预设     | 保存常用的模型参数组合                     | P1     |
| 模型探测     | 检测模型连通性和可用性                     | P1     |
| 用量查看     | 查看每个模型的调用次数和 Token 消耗        | P2     |

**支持的供应商**:

| 供应商    | 模型示例            | 接入方式          |
| --------- | ------------------- | ----------------- |
| OpenAI    | GPT-4o, GPT-4o-mini | API Key           |
| Anthropic | Claude 3.5 Sonnet   | API Key           |
| Google    | Gemini 2.0 Flash    | API Key           |
| Ollama    | Qwen2.5, Llama3     | 本地 API (无 Key) |

#### 2.2.2 Prompt 工坊 (Prompt Studio)

**课程对应**: Phase 2 (Prompt Engineering) + Phase 3 (LangChain)

| 功能项     | 描述                                 | 优先级 |
| ---------- | ------------------------------------ | ------ |
| 模板编辑   | 可视化 Prompt 模板编辑器             | P0     |
| 变量系统   | 支持 `{{variable}}` 语法的变量注入   | P0     |
| 多模型测试 | 同一 Prompt 发送给多个模型并对比响应 | P0     |
| 版本管理   | 每次编辑自动保存版本，支持回滚（回滚 = 将目标版本内容复制为新版本，历史完整保留） | P1     |
| 分类标签   | Prompt 模板的分类和标签管理          | P1     |

**Prompt 模板格式**:

```
System Prompt:
  你是一个{{role}}，擅长{{expertise}}。
  请用{{tone}}的风格回答用户的问题。

User Prompt:
  {{user_input}}
```

#### 2.2.3 知识库管理 (Knowledge Base)

**课程对应**: Phase 4 (RAG 基础) + Phase 5 (RAG 进阶)

| 功能项   | 描述                               | 优先级 |
| -------- | ---------------------------------- | ------ |
| 文档上传 | 支持 PDF、Markdown、TXT、DOCX 格式 | P0     |
| 切分配置 | 可配置切分策略（大小、重叠量）     | P0     |
| 向量索引 | 自动 Embedding + 向量存储          | P0     |
| 检索测试 | 输入查询查看检索结果和相似度评分   | P0     |
| 多知识库 | 支持创建多个独立知识库             | P0     |
| 文档管理 | 查看、删除已索引文档               | P1     |
| 检索策略 | 可选相似度检索 / 混合检索 / Rerank | P1     |

**知识库工作流**:

```
文档上传 → 格式解析 → 文本切分 → Embedding → 向量存储 → 可检索
```

#### 2.2.4 Agent 构建器 (Agent Builder)

**课程对应**: Phase 6 (Agent & Tools) + Phase 7 (Multi-Agent)

| 功能项     | 描述                               | 优先级 |
| ---------- | ---------------------------------- | ------ |
| Agent 配置 | 设置名称、描述、系统提示词、模型   | P0     |
| 内置工具   | 提供 Web 搜索、计算器、代码执行等  | P0     |
| 知识库绑定 | 将知识库作为 Agent 的检索工具      | P0     |
| 自定义工具 | 用户自定义工具（名称、参数、代码） | P1     |
| 策略选择   | ReAct / Function Calling           | P1     |
| Playground | Agent 测试游乐场                   | P0     |

**内置工具集**:

| 工具名              | 描述             | 实现方式       |
| ------------------- | ---------------- | -------------- |
| web_search          | 互联网搜索       | SerpAPI/Tavily |
| calculator          | 数学运算         | Python eval    |
| code_runner         | 执行 Python 代码 | 沙箱执行       |
| http_request        | 发送 HTTP 请求   | requests       |
| knowledge_retrieval | 知识库检索       | RAG Pipeline   |

#### 2.2.5 工作流引擎 (Workflow Engine) ⭐

**课程对应**: Phase 7 (Multi-Agent / LangGraph)

> 本模块是 Mini-Dify 的核心亮点，也是与 MediMind 的最大差异点。

| 功能项       | 描述                               | 优先级 |
| ------------ | ---------------------------------- | ------ |
| 可视化编辑器 | React Flow 拖拽式工作流画布        | P0     |
| LLM 节点     | 调用大模型，配置 Prompt 和模型参数 | P0     |
| 知识检索节点 | 从指定知识库检索内容               | P0     |
| 条件分支节点 | 根据条件路由到不同分支             | P0     |
| 代码节点     | 执行 Python 代码进行数据转换       | P1     |
| HTTP 节点    | 调用外部 API                       | P1     |
| 运行调试     | 执行工作流并查看每个节点的输出     | P0     |
| 流式输出     | 工作流执行过程的实时输出           | P1     |

**架构说明**：
- 首期仅支持 DAG（有向无环图），不支持循环
- 预留循环节点接口，后期可扩展（如 Loop 节点 + 最大迭代次数保护）
- 执行引擎：拓扑排序 → 逐节点执行 → SSE 推送进度

**工作流示例 - 智能客服**:

```
开始 → 意图识别(LLM) → 条件分支
                          ├─ FAQ → 知识检索 → 回答生成(LLM) → 结束
                          ├─ 投诉 → 生成工单(HTTP) → 安抚回复(LLM) → 结束
                          └─ 其他 → 通用回复(LLM) → 结束
```

#### 2.2.6 应用管理 (App Manager)

| 功能项   | 描述                                   | 优先级 |
| -------- | -------------------------------------- | ------ |
| 创建应用 | 选择应用类型并配置                     | P0     |
| 应用类型 | Chatbot / Completion / Workflow        | P0     |
| 应用配置 | 绑定模型、Prompt、知识库、Agent/工作流 | P0     |
| 应用发布 | 生成 API Key，启用 API 访问            | P0     |
| 应用预览 | 内嵌 Chat Widget 预览应用效果          | P1     |

**应用类型说明**:

| 类型       | 说明           | 底层引擎    |
| ---------- | -------------- | ----------- |
| Chatbot    | 多轮对话应用   | Agent + RAG |
| Completion | 单次文本生成   | Prompt 模板 |
| Workflow   | 复杂流程自动化 | 工作流引擎  |

#### 2.2.7 API 网关 (API Gateway)

**课程对应**: Phase 9 (部署与生产化)

| 功能项      | 描述                     | 优先级 |
| ----------- | ------------------------ | ------ |
| API Key     | 为已发布应用生成 API Key | P0     |
| RESTful API | 标准 REST 接口           | P0     |
| 流式响应    | SSE 流式输出             | P0     |
| 速率限制    | 按 API Key 限流          | P1     |

#### 2.2.8 监控分析 (Monitoring & Analytics)

**课程对应**: Phase 10 (评估与优化)

| 功能项     | 描述                        | 优先级 |
| ---------- | --------------------------- | ------ |
| 对话日志   | 记录所有对话的输入/输出     | P0     |
| Token 统计 | 按模型、应用统计 Token 用量 | P0     |
| 成本估算   | 基于 Token 数估算 API 成本  | P1     |
| 图表展示   | 用量趋势图和统计面板        | P1     |

---

## 3. 非功能需求

### 3.1 性能需求

| 指标           | 要求                   |
| -------------- | ---------------------- |
| API 响应延迟   | 首字输出 < 2秒（流式） |
| 知识库检索延迟 | < 500ms                |
| 工作流执行     | 单节点 < 3秒           |
| 并发支持       | ≥ 20 用户同时使用      |

### 3.2 安全需求

| 指标         | 要求                       |
| ------------ | -------------------------- |
| API Key 存储 | AES 加密存储，界面脱敏展示 |
| 代码执行沙箱 | 限制执行时间和系统调用     |
| API 认证     | 所有外部 API 调用需要认证  |

---

## 4. 项目范围

### 4.1 本期包含 (In Scope)

- ✅ 多模型供应商管理
- ✅ Prompt 模板与变量系统
- ✅ 知识库管理（RAG 全流程）
- ✅ Agent 构建与测试
- ✅ 可视化工作流编辑器
- ✅ 三种类型应用的创建与发布
- ✅ API Gateway（Key 认证 + SSE）
- ✅ 基础监控面板

### 4.2 本期不包含 (Out of Scope)

- ❌ 多租户 / RBAC 权限系统
- ❌ 计费和充值系统
- ❌ 插件市场
- ❌ 数据标注与微调
- ❌ 多模态输入（图片/音频）
- ❌ SaaS 部署 / 域名管理

---

## 5. 验收标准

### 5.1 功能验收

- [ ] 能添加和管理至少 3 个模型供应商
- [ ] 能创建 Prompt 模板并用变量填充测试
- [ ] 能上传文档并构建知识库，检索测试返回相关结果
- [ ] 能配置 Agent 并在 Playground 中正确调用工具
- [ ] 能拖拽构建工作流并执行成功
- [ ] 能创建 3 种类型的应用并通过 API 访问
- [ ] 监控面板正确展示 Token 用量和对话日志

### 5.2 技术验收

- [ ] 代码结构清晰，分层合理
- [ ] 核心模块有单元测试
- [ ] API 文档自动生成 (Swagger)
- [ ] 前端 UI 美观、响应式
- [ ] Docker 一键部署

---

## 6. 与课程阶段的映射关系

| 课程阶段               | Mini-Dify 对应模块        | 覆盖度 |
| ---------------------- | ------------------------- | ------ |
| Phase 1: LLM API       | Model Hub                 | ⭐⭐⭐ |
| Phase 2: Prompt 工程   | Prompt Studio             | ⭐⭐⭐ |
| Phase 3: LangChain     | 全局（链式调用、LCEL）    | ⭐⭐   |
| Phase 4: RAG 基础      | Knowledge Base            | ⭐⭐⭐ |
| Phase 5: RAG 进阶      | Knowledge Base (检索策略) | ⭐⭐   |
| Phase 6: Agent & Tools | Agent Builder             | ⭐⭐⭐ |
| Phase 7: Multi-Agent   | Workflow Engine           | ⭐⭐⭐ |
| Phase 8: 微调          | —                         | —      |
| Phase 9: 部署          | API Gateway               | ⭐⭐   |
| Phase 10: 评估         | Monitoring                | ⭐⭐   |

---

## 7. 风险评估

| 风险项               | 可能性 | 影响 | 缓解措施                               |
| -------------------- | ------ | ---- | -------------------------------------- |
| 工作流编辑器复杂度高 | 高     | 高   | 限制节点类型为 5-6 种，使用 React Flow |
| 多模型兼容性问题     | 中     | 中   | 基于 LangChain 统一接口封装            |
| 前端开发量大         | 高     | 中   | 使用 TailwindCSS 加速、组件复用        |
| 代码执行安全风险     | 中     | 高   | 限制执行时间、禁用危险模块             |
| API Key 泄露风险     | 低     | 高   | 加密存储、界面脱敏                     |

---

## 8. 技术架构

### 8.1 技术栈

| 层级 | 技术选型 |
| ---- | -------- |
| 前端框架 | Next.js 14+ (App Router, TypeScript) |
| UI 组件库 | shadcn/ui + Tailwind CSS |
| 工作流编辑器 | React Flow (@xyflow/react v12) |
| 状态管理 | Zustand |
| 后端框架 | Python FastAPI (async) |
| ORM | SQLAlchemy (async) + Alembic 迁移 |
| LLM 框架 | LangChain |
| 关系数据库 | PostgreSQL 16 |
| 向量数据库 | Milvus Standalone |
| 缓存/队列 | Redis 7 |
| 认证 | NextAuth.js v5 (GitHub + Google OAuth) |
| 部署 | Docker Compose |

### 8.2 系统架构概览

```
┌────────────────────────────────────────────────────────────┐
│                       客户端 (Browser)                      │
└──────────────────────────┬─────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼─────────────────────────────────┐
│                    Next.js 前端 (:3000)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Dashboard │ │  Prompt  │ │    KB    │ │ Workflow │      │
│  │  Pages   │ │  Studio  │ │  Manager │ │  Canvas  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │            NextAuth.js (OAuth + JWT)              │      │
│  └──────────────────────────────────────────────────┘      │
└──────────────────────────┬─────────────────────────────────┘
                           │ REST + SSE (JWT)
┌──────────────────────────▼─────────────────────────────────┐
│                   FastAPI 后端 (:8000)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Model Hub│ │  Prompt  │ │   RAG    │ │  Agent   │      │
│  │ Service  │ │ Service  │ │ Pipeline │ │ Executor │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ Workflow │ │   App    │ │  Gateway │                   │
│  │  Engine  │ │ Manager  │ │ Endpoint │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
└────┬──────────────┬──────────────┬─────────────────────────┘
     │              │              │
┌────▼────┐  ┌──────▼──────┐  ┌───▼────┐
│PostgreSQL│  │   Milvus    │  │ Redis  │
│  :5432   │  │   :19530    │  │ :6379  │
└──────────┘  └─────────────┘  └────────┘
```

---

## 9. 模块依赖关系

```
Model Hub ──────────────────────────────────────────┐
  │ (被所有模块依赖：选择模型)                          │
  ├──→ Prompt Studio                                  │
  ├──→ Knowledge Base                                 │
  ├──→ Agent Builder                                  │
  └──→ Workflow Engine                                │
                                                      │
Knowledge Base ──→ Agent Builder (绑定 KB 作为工具)    │
               ──→ Workflow Engine (知识检索节点)       │
                                                      │
Prompt Studio ───→ App Manager (Completion 类型绑定)   │
Agent Builder ───→ App Manager (Chatbot 类型绑定)      │
Workflow Engine ─→ App Manager (Workflow 类型绑定)     │
                                                      │
App Manager ─────→ API Gateway (发布后通过 Gateway 访问)│
API Gateway ─────→ Monitoring (产生数据被消费)          │
```

**依赖方向总结**：Model Hub → Prompt/KB/Agent/Workflow → App Manager → API Gateway → Monitoring

---

## 10. 认证方案

### 10.1 OAuth 登录

- 支持 GitHub 和 Google 两种 OAuth 登录方式
- 使用 NextAuth.js v5 处理 OAuth 流程
- 首次登录自动创建用户记录

### 10.2 多用户数据隔离

- 每个用户只能看到和操作自己创建的数据
- 所有业务表包含 `user_id` 外键
- 后端 Service 层在所有查询中自动注入 `user_id` 过滤

### 10.3 内部 API 认证

- 前端 → 后端：NextAuth 签发 JWT，FastAPI 验证
- JWT Payload 包含：`sub`(user_id)、`email`、`name`、`exp`
- FastAPI 依赖注入 `get_current_user()` 自动解析 JWT

### 10.4 外部 API 认证（Gateway）

- 每个已发布应用可生成 App API Key
- API Key 格式：`sk-mini-{random_32_hex}`
- 存储方式：SHA-256 哈希存储，创建时返回明文（仅一次）
- 请求头：`Authorization: Bearer sk-mini-xxxxx`

---

## 11. 流式输出规范

统一使用 **SSE（Server-Sent Events）**，不使用 WebSocket。

### 11.1 场景覆盖

| 场景 | 事件类型 | 数据内容 |
| ---- | -------- | -------- |
| LLM 回答 | `message_delta` | 逐 token 文本块 |
| 工作流进度 | `node_started` / `node_completed` / `node_error` | 节点 ID、状态、输出 |
| Agent 推理 | `thought` / `tool_call` / `tool_result` / `message_delta` | ReAct 步骤详情 |
| 完成 | `done` | 总 token 数、耗时 |
| 错误 | `error` | 错误消息 |

### 11.2 SSE 事件格式

```
event: message_delta
data: {"content": "你好", "message_id": "xxx"}

event: node_completed
data: {"node_id": "node_1", "output": {...}, "duration_ms": 1200}

event: done
data: {"total_tokens": 256, "duration_ms": 3400}
```

---

## 12. 数据模型概要

### 12.1 核心实体（15 张表）

| 表名 | 说明 | 关键字段 |
| ---- | ---- | -------- |
| `users` | 用户信息（OAuth） | id, email, name, avatar_url, provider |
| `providers` | LLM 供应商配置 | id, user_id, name, provider_type, api_key_encrypted, base_url |
| `model_configs` | 模型参数预设 | id, provider_id, model_name, temperature, max_tokens |
| `prompts` | Prompt 模板 | id, user_id, name, description, current_version_id |
| `prompt_versions` | Prompt 版本 | id, prompt_id, version_number, system_prompt, user_prompt, variables |
| `knowledge_bases` | 知识库 | id, user_id, name, description, embedding_model, chunk_size, chunk_overlap |
| `documents` | 知识库文档 | id, kb_id, filename, file_type, status, chunk_count |
| `agents` | Agent 配置 | id, user_id, name, system_prompt, model_config_id, strategy |
| `agent_tools` | Agent 绑定工具 | id, agent_id, tool_type, tool_config |
| `agent_knowledge_bases` | Agent 绑定知识库 | agent_id, kb_id |
| `workflows` | 工作流定义 | id, user_id, name, description, graph_data |
| `workflow_nodes` | 工作流节点 | id, workflow_id, node_type, position, config |
| `workflow_edges` | 工作流连线 | id, workflow_id, source_node_id, target_node_id, condition |
| `apps` | 应用 | id, user_id, name, app_type, config, is_published |
| `app_api_keys` | 应用 API Key | id, app_id, key_hash, prefix, is_active |
| `conversations` | 对话 | id, app_id, user_id, title |
| `messages` | 消息记录 | id, conversation_id, role, content, token_count, model_name, cost |

### 12.2 实体关系

```
users ─1:N─→ providers ─1:N─→ model_configs
users ─1:N─→ prompts ─1:N─→ prompt_versions
users ─1:N─→ knowledge_bases ─1:N─→ documents
users ─1:N─→ agents ─1:N─→ agent_tools
                     ─N:N─→ knowledge_bases (via agent_knowledge_bases)
users ─1:N─→ workflows ─1:N─→ workflow_nodes
                       ─1:N─→ workflow_edges
users ─1:N─→ apps ─1:N─→ app_api_keys
                  ─1:N─→ conversations ─1:N─→ messages
```

---

## 13. 开发里程碑

| Phase | 名称 | 核心交付物 | 依赖 |
| ----- | ---- | ---------- | ---- |
| 1 | 基础架构 + Model Hub | OAuth 登录、供应商 CRUD、模型连通性验证 | 无 |
| 2 | Prompt Studio + App Manager | Prompt 编辑/测试/版本管理、应用 CRUD | Phase 1 |
| 3 | Knowledge Base (RAG) | 文档上传、向量化、检索测试 | Phase 1 |
| 4 | Agent Builder | Agent CRUD、工具系统、Playground 对话 | Phase 1, 3 |
| 5 | Workflow Engine | 可视化编排、DAG 执行、节点调试 | Phase 1, 3 |
| 6 | API Gateway | API Key 认证、限流、三种应用端点 | Phase 2, 4, 5 |
| 7 | Monitoring + 收尾 | 监控面板、Docker 部署、文档 | Phase 6 |

### 验证方式

| Phase | 验证场景 |
| ----- | -------- |
| 1 | OAuth 登录 → 添加 OpenAI 供应商 → 验证连通 → 查看模型列表 |
| 2 | 创建 Prompt → 填写变量 → 多模型测试 → 查看版本历史 → 创建应用 |
| 3 | 创建知识库 → 上传 PDF → 等待处理完成 → 检索测试返回相关段落 |
| 4 | 创建 Agent → 绑定工具和知识库 → Playground 对话 → 验证工具调用 |
| 5 | 创建工作流 → 拖拽节点连线 → 运行 → 查看每个节点输出 |
| 6 | 发布应用 → 生成 API Key → curl 调用 → 收到流式响应 |
| 7 | 查看监控面板 → docker-compose up 一键启动所有服务 |

---

## 附录

### A. 术语表

| 术语       | 说明                                           |
| ---------- | ---------------------------------------------- |
| Dify       | 开源 LLM 应用开发平台                          |
| RAG        | Retrieval-Augmented Generation，检索增强生成   |
| Agent      | 智能代理，具有推理和工具调用能力               |
| Workflow   | 工作流，由多个节点组成的自动化流程             |
| React Flow | 开源 React 流程图/节点编辑器库                 |
| LangGraph  | LangChain 的图状态机库，用于构建 Agent 工作流  |
| SSE        | Server-Sent Events，服务端推送事件（流式输出） |

### B. 参考项目

| 项目      | 链接                                    | 参考点          |
| --------- | --------------------------------------- | --------------- |
| Dify      | https://github.com/langgenius/dify      | 功能设计、UX    |
| LangFlow  | https://github.com/langflow-ai/langflow | 工作流编辑器    |
| FlowiseAI | https://github.com/FlowiseAI/Flowise    | 低代码 LLM 平台 |
