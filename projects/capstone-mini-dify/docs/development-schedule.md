# Mini-Dify 开发进度表

> LLM 应用开发平台 — 共 7 个 Phase，渐进式交付

---

## Phase 间依赖关系图

```
Phase 1 ──────────────────────────────────────────────────┐
  基础架构 + Model Hub                                     │
  │                                                        │
  ├──► Phase 2                                             │
  │     Prompt Studio + App Manager                        │
  │     │                                                  │
  │     ├──► Phase 4                                       │
  │     │     Agent Builder                                │
  │     │     │                                            │
  │     │     ├──► Phase 5                                 │
  │     │     │     Workflow Engine                         │
  │     │     │     │                                      │
  │     │     │     └──► Phase 6 ──► Phase 7               │
  │     │     │           API Gateway   Monitoring + 收尾  │
  │     │     │                                            │
  │     ├──► Phase 3                                       │
  │     │     Knowledge Base (RAG)                         │
  │     │     │                                            │
  │     │     └──► Phase 4 (知识库检索工具)                 │
  │     │                                                  │
  └─────┘                                                  │
                                                           │
  依赖说明:                                                │
  Phase 1 → 所有后续 Phase 的基础                          │
  Phase 2 → Phase 4 (App/Prompt 能力)                      │
  Phase 3 → Phase 4 (knowledge_retrieval 工具)             │
  Phase 4 → Phase 5 (Agent 节点复用)                       │
  Phase 5 → Phase 6 (Workflow Gateway 端点)                │
  Phase 6 → Phase 7 (API 日志 → Monitoring)                │
```

---

## 总体里程碑甘特图

```
Phase       | W1  | W2  | W3  | W4  | W5  | W6  | W7  | W8  | W9  | W10 | W11 | W12 |
------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
Phase 1     |█████|█████|     |     |     |     |     |     |     |     |     |     |
Phase 2     |     |█████|█████|     |     |     |     |     |     |     |     |     |
Phase 3     |     |     |█████|█████|     |     |     |     |     |     |     |     |
Phase 4     |     |     |     |█████|█████|     |     |     |     |     |     |     |
Phase 5     |     |     |     |     |█████|█████|█████|     |     |     |     |     |
Phase 6     |     |     |     |     |     |     |█████|█████|     |     |     |     |
Phase 7     |     |     |     |     |     |     |     |█████|█████|█████|     |     |
缓冲/联调    |     |     |     |     |     |     |     |     |     |█████|█████|█████|

里程碑:
  M1 (W2 末)  — 可登录 + Model Hub 可用
  M2 (W4 末)  — Prompt Studio + Knowledge Base 可用
  M3 (W6 末)  — Agent 可对话
  M4 (W7 末)  — Workflow 可视化构建 + 执行
  M5 (W8 末)  — API Gateway 对外服务
  M6 (W10 末) — Monitoring + Docker 部署
  M7 (W12 末) — 全部联调完成、端到端测试通过
```

---

## Phase 1: 基础架构 + Model Hub

### 任务清单

| # | 任务 | 工作量 | 关键文件 |
|---|------|--------|----------|
| 1.1 | 初始化项目结构（monorepo: `frontend/` + `backend/`） | 简单 | `package.json`, `pyproject.toml` |
| 1.2 | 编写 `docker-compose.yml`（PostgreSQL、Milvus、Redis）+ `docker-compose.dev.yml` | 简单 | `docker-compose.yml`, `docker-compose.dev.yml` |
| 1.3 | 后端搭建：FastAPI 入口、数据库连接、Alembic 迁移框架、Pydantic Settings | 中等 | `backend/app/main.py`, `backend/alembic.ini`, `backend/app/core/config.py` |
| 1.4 | 前端搭建：Next.js + shadcn/ui + Tailwind、根布局、侧边栏 | 中等 | `frontend/src/app/(dashboard)/layout.tsx`, `frontend/src/components/ui/` |
| 1.5 | OAuth 认证：NextAuth.js 配置 + FastAPI JWT 验证 + 用户自动创建 | 复杂 | `frontend/auth.ts`, `backend/app/security.py`, `backend/app/api/auth/` |
| 1.6 | Model Hub 后端：Provider CRUD、API Key AES 加密存储、模型列表、连通性探测 | 复杂 | `backend/app/core/llm/provider_factory.py`, `backend/app/services/provider_service.py` |
| 1.7 | Model Hub 前端：供应商卡片、添加/验证/删除、API Key 脱敏显示、模型列表 | 中等 | `frontend/src/app/(dashboard)/models/page.tsx` |

### 关键文件

- `backend/app/main.py` — FastAPI 应用入口
- `backend/app/security.py` — JWT 验证、AES 加解密
- `backend/app/core/llm/provider_factory.py` — LLM Provider 工厂
- `frontend/src/app/(dashboard)/layout.tsx` — Dashboard 根布局
- `frontend/auth.ts` — NextAuth.js 配置

### 交付物

可登录系统，能添加 LLM 供应商并验证连通性。

### 验证 Checklist

- [ ] `docker-compose up` 启动 PostgreSQL / Milvus / Redis 无报错
- [ ] Alembic 迁移执行成功，数据库表已创建
- [ ] OAuth 登录流程完整（跳转 → 回调 → JWT 签发 → 用户自动创建）
- [ ] 添加 OpenAI 供应商，填写 API Key
- [ ] 点击"验证连通性"返回成功
- [ ] 模型列表正确展示
- [ ] API Key 脱敏显示（仅显示末尾 4 位）
- [ ] 删除供应商后数据库记录清除

---

## Phase 2: Prompt Studio + 基础 App Manager

### 任务清单

| # | 任务 | 工作量 | 关键文件 |
|---|------|--------|----------|
| 2.1 | Prompt CRUD + 版本自动保存 | 中等 | `backend/app/services/prompt_service.py`, `backend/app/models/prompt.py` |
| 2.2 | `{{variable}}` 变量解析与注入 | 简单 | `backend/app/services/prompt_service.py` |
| 2.3 | 多模型对比测试（SSE 流式返回） | 复杂 | `backend/app/core/llm/streaming.py`, `backend/app/api/prompts/endpoints.py` |
| 2.4 | 版本历史列表 + 回滚（创建新版本） | 中等 | `backend/app/services/prompt_service.py` |
| 2.5 | 前端 Prompt 编辑器（分栏：编辑 + 测试） | 复杂 | `frontend/src/components/prompts/template-editor.tsx` |
| 2.6 | App Manager 后端：CRUD、发布、API Key 生成（返回一次 + 哈希存储） | 中等 | `backend/app/services/app_service.py`, `backend/app/models/app.py` |
| 2.7 | App Manager 前端：应用列表、创建、配置、API Key 管理 | 中等 | `frontend/src/app/(dashboard)/apps/page.tsx` |

### 关键文件

- `backend/app/core/llm/streaming.py` — SSE 流式响应封装
- `backend/app/services/prompt_service.py` — Prompt 业务逻辑 + 版本管理
- `frontend/src/components/prompts/template-editor.tsx` — Prompt 模板编辑器组件

### 交付物

能创建 Prompt 模板、用变量测试、多模型对比，能创建应用。

### 验证 Checklist

- [ ] 创建 Prompt 模板，保存成功
- [ ] 输入含 `{{name}}` 的模板，变量面板自动识别并展示输入框
- [ ] 填写变量后点击测试，SSE 流式返回结果
- [ ] 选择多个模型对比测试，结果并排显示
- [ ] 版本历史列表正确展示，点击回滚生成新版本
- [ ] 创建应用，绑定 Prompt 模板
- [ ] 生成 API Key，仅展示一次，关闭后不可再查看
- [ ] 应用列表展示状态（草稿/已发布）

---

## Phase 3: Knowledge Base（RAG）

### 任务清单

| # | 任务 | 工作量 | 关键文件 |
|---|------|--------|----------|
| 3.1 | 文档上传与解析（PDF/MD/TXT/DOCX） | 中等 | `backend/app/core/rag/parser.py` |
| 3.2 | 文本切分（可配置 `chunk_size` / `chunk_overlap`） | 中等 | `backend/app/core/rag/chunker.py` |
| 3.3 | Embedding 生成 + Milvus 存储（每个 KB 一个 Collection） | 复杂 | `backend/app/core/rag/pipeline.py`, `backend/app/core/rag/embedding.py` |
| 3.4 | 检索测试（相似度搜索 + 分数） | 中等 | `backend/app/core/rag/retriever.py` |
| 3.5 | 多知识库管理 | 简单 | `backend/app/services/knowledge_service.py` |
| 3.6 | 文档列表、状态显示、删除（同步删除向量） | 中等 | `backend/app/services/document_service.py` |
| 3.7 | 前端：上传区域、切分配置、检索测试面板 | 中等 | `frontend/src/app/(dashboard)/knowledge/page.tsx` |

### 关键文件

- `backend/app/core/rag/pipeline.py` — RAG 管线编排（解析 → 切分 → 向量化 → 存储）
- `backend/app/core/rag/chunker.py` — 文本切分器
- `backend/app/core/rag/retriever.py` — 向量检索器

### 交付物

能上传文档、自动向量化、检索测试返回相关结果。

### 验证 Checklist

- [ ] 上传 PDF / MD / TXT / DOCX 文件，解析无报错
- [ ] 文档状态流转：上传中 → 解析中 → 向量化中 → 已完成
- [ ] 切分配置修改后重新处理生效
- [ ] 检索测试输入查询，返回相关分片 + 相似度分数
- [ ] 创建多个知识库，文档隔离正确
- [ ] 删除文档后 Milvus 中对应向量同步删除
- [ ] 删除知识库后整个 Collection 被清除

---

## Phase 4: Agent Builder

### 任务清单

| # | 任务 | 工作量 | 关键文件 |
|---|------|--------|----------|
| 4.1 | Agent CRUD + 工具配置 | 中等 | `backend/app/services/agent_service.py`, `backend/app/models/agent.py` |
| 4.2 | 内置工具实现：`web_search`, `calculator`, `code_runner`, `http_request`, `knowledge_retrieval` | 复杂 | `backend/app/core/agent/tools/` |
| 4.3 | 代码执行沙箱：subprocess + resource 限制（10s/128MB）+ 禁用危险模块 | 复杂 | `backend/app/services/sandbox_service.py` |
| 4.4 | Agent 执行器：基于 LangChain `create_react_agent` / `create_tool_calling_agent` | 复杂 | `backend/app/core/agent/executor.py` |
| 4.5 | SSE 流式返回思考过程 + 工具调用 + 最终回答 | 中等 | `backend/app/api/agents/endpoints.py` |
| 4.6 | Playground 前端：聊天界面 + 思考过程展示 | 中等 | `frontend/src/app/(dashboard)/agents/[id]/playground/page.tsx` |

### 关键文件

- `backend/app/core/agent/executor.py` — Agent 执行器核心逻辑
- `backend/app/core/agent/tools/` — 内置工具目录
- `backend/app/services/sandbox_service.py` — 代码沙箱服务

### 交付物

能配置 Agent、绑定工具和知识库、在 Playground 中对话。

### 验证 Checklist

- [ ] 创建 Agent，选择 LLM 模型，配置 System Prompt
- [ ] 绑定 `calculator` 工具，提问数学计算，Agent 正确调用工具并返回
- [ ] 绑定 `web_search` 工具，提问时事问题，返回搜索结果摘要
- [ ] 绑定 `code_runner`，执行 Python 代码片段，输出正确
- [ ] 沙箱限制生效：超时 10s 中断、内存超限中断、`os.system` 等被禁用
- [ ] 绑定 `knowledge_retrieval`，提问知识库相关问题，返回正确引用
- [ ] SSE 流式展示：思考过程 → 工具调用 → 工具结果 → 最终回答
- [ ] `http_request` 工具能发起 GET/POST 请求并返回结果

---

## Phase 5: Workflow Engine（核心亮点）

### 任务清单

| # | 任务 | 工作量 | 关键文件 |
|---|------|--------|----------|
| 5.1 | 后端节点抽象：BaseNode 接口 + 5 种节点实现（LLM / Knowledge / Condition / Code / HTTP） | 复杂 | `backend/app/core/workflow/nodes/base.py`, `backend/app/core/workflow/nodes/` |
| 5.2 | DAG 执行引擎：拓扑排序 → 逐节点执行 → 上下文传递 → SSE 进度推送 | 复杂 | `backend/app/core/workflow/engine.py` |
| 5.3 | 工作流 CRUD（整体保存 React Flow 状态） | 中等 | `backend/app/services/workflow_service.py` |
| 5.4 | 前端 React Flow 画布：自定义节点组件、拖拽、连线 | 复杂 | `frontend/src/components/workflow/canvas.tsx` |
| 5.5 | 节点侧边栏（拖拽面板）+ 节点配置面板 | 中等 | `frontend/src/components/workflow/node-panel.tsx`, `frontend/src/components/workflow/config-panel.tsx` |
| 5.6 | 调试面板：运行按钮、节点状态高亮、输出日志 | 中等 | `frontend/src/components/workflow/debug-panel.tsx`, `frontend/src/stores/workflow-store.ts` |

### 关键文件

- `backend/app/core/workflow/engine.py` — DAG 执行引擎
- `backend/app/core/workflow/nodes/base.py` — 节点基类
- `frontend/src/components/workflow/canvas.tsx` — React Flow 画布
- `frontend/src/stores/workflow-store.ts` — 工作流前端状态管理

### 交付物

可视化拖拽构建工作流，执行并查看每个节点输出。

### 验证 Checklist

- [ ] 从侧边栏拖拽节点到画布，节点正确渲染
- [ ] 节点间连线，数据流向可视化
- [ ] 配置 LLM 节点：选择模型、设置 Prompt 模板
- [ ] 配置 Knowledge 节点：选择知识库、设置检索参数
- [ ] 配置 Condition 节点：设置分支条件，条件路由正确
- [ ] 配置 Code 节点：编写 Python 代码，输入/输出映射正确
- [ ] 配置 HTTP 节点：设置 URL、Method、Headers、Body
- [ ] 点击运行，节点状态依次高亮（待执行 → 执行中 → 已完成/失败）
- [ ] 上下文在节点间正确传递（上游输出 → 下游输入）
- [ ] SSE 推送每个节点的执行状态和输出
- [ ] 保存工作流，刷新页面后画布状态恢复

---

## Phase 6: API Gateway

### 任务清单

| # | 任务 | 工作量 | 关键文件 |
|---|------|--------|----------|
| 6.1 | Gateway 认证中间件：API Key 哈希查找 + 验证 | 中等 | `backend/app/api/gateway/middleware.py` |
| 6.2 | Redis 滑动窗口限流 | 中等 | `backend/app/utils/rate_limiter.py` |
| 6.3 | 三种应用类型的 Gateway 端点（chat / completion / workflow） | 复杂 | `backend/app/api/gateway/endpoints.py` |
| 6.4 | 自动记录 conversation + message（供 Monitoring 使用） | 中等 | `backend/app/services/conversation_service.py` |
| 6.5 | 前端：curl 示例展示、Chat Widget 预览 | 简单 | `frontend/src/components/apps/api-reference.tsx` |

### 关键文件

- `backend/app/api/gateway/endpoints.py` — Gateway 路由端点
- `backend/app/utils/rate_limiter.py` — Redis 滑动窗口限流器

### 交付物

已发布的应用可通过 API Key 认证的 REST API 访问。

### 验证 Checklist

- [ ] 无 API Key 请求返回 `401 Unauthorized`
- [ ] 错误 API Key 请求返回 `401 Unauthorized`
- [ ] 正确 API Key 请求 chat 端点，SSE 流式返回对话结果
- [ ] 正确 API Key 请求 completion 端点，返回补全结果
- [ ] 正确 API Key 请求 workflow 端点，触发工作流执行
- [ ] 超过速率限制返回 `429 Too Many Requests`
- [ ] conversation 和 message 记录自动写入数据库
- [ ] curl 示例能直接复制运行
- [ ] Chat Widget 预览能正常对话

---

## Phase 7: Monitoring + 收尾

### 任务清单

| # | 任务 | 工作量 | 关键文件 |
|---|------|--------|----------|
| 7.1 | 对话日志查询（分页、筛选） | 中等 | `backend/app/api/monitoring/endpoints.py` |
| 7.2 | Token 用量统计（按模型/应用/天聚合） | 中等 | `backend/app/services/analytics_service.py` |
| 7.3 | 成本估算（基于模型定价表） | 简单 | `backend/app/core/llm/pricing.py` |
| 7.4 | 前端图表面板（recharts） | 中等 | `frontend/src/app/(dashboard)/monitoring/page.tsx` |
| 7.5 | Dockerfile（前端 + 后端） | 中等 | `frontend/Dockerfile`, `backend/Dockerfile` |
| 7.6 | 最终 `docker-compose.yml`（含 nginx 反向代理） | 中等 | `docker-compose.yml`, `nginx/nginx.conf` |
| 7.7 | Swagger API 文档验证 | 简单 | `backend/app/main.py` |
| 7.8 | 端到端测试 | 复杂 | `tests/e2e/` |

### 关键文件

- `backend/app/services/analytics_service.py` — 用量统计与成本计算
- `frontend/src/app/(dashboard)/monitoring/page.tsx` — 监控图表页面
- `docker-compose.yml` — 生产部署编排
- `nginx/nginx.conf` — 反向代理配置

### 交付物

监控面板 + Docker 一键部署 + 完整文档。

### 验证 Checklist

- [ ] 对话日志列表分页加载正常，筛选条件生效
- [ ] Token 用量图表按天/模型/应用维度正确聚合
- [ ] 成本估算数值合理（与 OpenAI 官方定价比对）
- [ ] `docker compose up` 一键启动所有服务（前端 + 后端 + DB + Milvus + Redis + Nginx）
- [ ] 通过 Nginx 反向代理访问前端和 API
- [ ] Swagger 文档页面可访问，所有端点描述完整
- [ ] 端到端测试覆盖核心流程：登录 → 添加模型 → 创建 Prompt → 创建知识库 → 创建 Agent → 构建 Workflow → API 调用 → 查看监控

---

## 风险点与缓解措施

| # | 风险 | 影响 | 概率 | 缓解措施 |
|---|------|------|------|----------|
| R1 | Milvus 部署与连接不稳定 | Phase 3 阻塞 | 中 | 提前在 Phase 1 的 docker-compose 中验证 Milvus 可用性；准备 ChromaDB 作为备选 |
| R2 | SSE 流式传输在复杂网络环境下断连 | 影响用户体验 | 中 | 实现客户端自动重连 + 断点续传机制；设置合理的超时时间 |
| R3 | 代码沙箱安全性不足 | 安全漏洞 | 高 | 使用 Docker 容器隔离替代 subprocess；白名单机制限制可用模块；设置严格的资源限制 |
| R4 | LLM API 调用成本超预期 | 开发成本增加 | 中 | 使用低成本模型（如 gpt-3.5-turbo）进行开发测试；实现 Mock LLM Provider 用于单元测试 |
| R5 | React Flow 画布性能问题（大量节点） | Phase 5 体验差 | 低 | 实现虚拟化渲染；限制单个工作流的最大节点数（如 50 个） |
| R6 | OAuth 第三方服务不可用 | 无法登录 | 低 | 实现本地账号密码登录作为降级方案 |
| R7 | Workflow DAG 出现循环依赖 | 引擎死循环 | 中 | 连线时前端校验 DAG 合法性；后端拓扑排序前检测环并报错 |
| R8 | 多模型对比时并发请求过多 | API 限流 / 超时 | 中 | 实现请求队列 + 并发数控制（如最多同时 3 个模型）；增加超时和重试机制 |

---

## 工作量估算汇总

| Phase | 简单 | 中等 | 复杂 | 预估总工时 |
|-------|------|------|------|-----------|
| Phase 1 | 2 | 2 | 2 | ~2 周 |
| Phase 2 | 1 | 4 | 2 | ~2 周 |
| Phase 3 | 1 | 4 | 1 | ~2 周 |
| Phase 4 | 0 | 3 | 3 | ~2 周 |
| Phase 5 | 0 | 3 | 3 | ~3 周 |
| Phase 6 | 1 | 3 | 1 | ~1.5 周 |
| Phase 7 | 2 | 4 | 1 | ~2.5 周 |
| **合计** | **7** | **23** | **13** | **~15 周** |

> 工作量定义：简单 ≈ 0.5 天，中等 ≈ 1~2 天，复杂 ≈ 2~3 天
