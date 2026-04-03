# Mini-Dify 数据库设计文档

> 数据库：PostgreSQL 16
> 最后更新：2026-04-02

---

## 目录

1. [设计原则](#1-设计原则)
2. [ER 关系图](#2-er-关系图)
3. [表结构详细设计](#3-表结构详细设计)
4. [索引设计](#4-索引设计)
5. [特殊设计决策说明](#5-特殊设计决策说明)
6. [完整 DDL](#6-完整-ddl)

---

## 1. 设计原则

| 原则 | 说明 |
|------|------|
| **主键策略** | 全部使用 UUID v4，避免自增 ID 暴露业务量和顺序，便于分布式环境和前端直接生成 |
| **时间字段** | 所有表包含 `created_at`，可变实体额外包含 `updated_at`，统一使用 `TIMESTAMPTZ` 带时区 |
| **软删除** | 当前版本不使用软删除，通过外键 `ON DELETE CASCADE` 保证级联清理；后续可通过添加 `deleted_at` 列实现 |
| **JSON 使用** | 对结构灵活、schema 可变的配置数据使用 `jsonb`，对结构固定的业务数据使用关系列 |
| **加密存储** | API Key 等敏感字段使用 `bytea` 类型存储加密后的密文，应用层负责加解密 |
| **命名规范** | 表名小写复数（snake_case），字段名小写（snake_case），外键命名 `{引用表单数}_id` |

---

## 2. ER 关系图

```
                                    +----------------+
                                    |     users      |
                                    |----------------|
                                    | id (PK)        |
                                    | email          |
                                    | name           |
                                    | avatar_url     |
                                    | oauth_provider |
                                    | oauth_id       |
                                    +-------+--------+
                                            |
                 +-----------+--------------++-----------+-----------+
                 |           |               |           |           |
                 v           v               v           v           v
          +------+---+ +----+----+  +-------+----+ +----+---+ +----+------+
          | providers| | prompts |  |knowledge_  | | agents | | workflows |
          |----------| |---------|  |bases       | |--------| |-----------|
          | id (PK)  | | id (PK) |  |------------| | id(PK)| | id (PK)   |
          | user_id  | | user_id |  | id (PK)    | |user_id| | user_id   |
          +----+-----+ +----+----+  | user_id    | +---+---+ +-----+-----+
               |            |       +------+-----+     |           |
               v            v              |           |           |
        +------+------+ +--+----------+   |      +----+-----+ +---+----------+
        |model_configs| |prompt_      |   v      |agent_    | |workflow_     |
        |-------------| |versions     | +-+--------+tools   | |nodes         |
        | id (PK)     | |-------------| |documents | |----------| |--------------|
        | provider_id | | id (PK)     | |----------| | id (PK)  | | id (PK)      |
        +------+------+ | prompt_id   | | id (PK)  | | agent_id | | workflow_id  |
               |         +-----------+  | kb_id    | +----------+ +---+---+------+
               |                        +----------+       |          |   |
               |                             |             v          |   |
               |              +--------------+    +--------+------+   |   |
               |              |               |   |agent_         |   |   |
               |              |               |   |knowledge_bases|   |   |
               |              |               |   |---------------|   |   |
               |              |               |   | agent_id (PK) |   |   |
               |              |               |   | kb_id (PK)    |   |   |
               |              |               |   +---------------+   |   |
               |              |               |                       |   |
               |         +----+----+          |            +----------+   |
               +-------->+  apps   +<---------+            |              |
                         |---------|                       v              |
                         | id (PK) |              +--------+-------+      |
                         | user_id |              |workflow_edges  |      |
                         +----+----+              |----------------|      |
                              |                   | id (PK)        |      |
                    +---------+---------+         | workflow_id    |      |
                    |                   |         | source_node_id +------+
                    v                   v         | target_node_id |
             +------+------+   +-------+-------+ +----------------+
             |app_api_keys |   |conversations  |
             |-------------|   |---------------|
             | id (PK)     |   | id (PK)       |
             | app_id      |   | app_id        |
             +-------------+   +-------+-------+
                                       |
                                       v
                               +-------+-------+
                               |   messages    |
                               |---------------|
                               | id (PK)       |
                               |conversation_id|
                               +---------------+
```

---

## 3. 表结构详细设计

### 3.1 users - 用户表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 用户唯一标识 |
| email | VARCHAR(255) | NOT NULL, UNIQUE | 邮箱地址 |
| name | VARCHAR(100) | NOT NULL | 用户昵称 |
| avatar_url | VARCHAR(500) | | 头像 URL |
| oauth_provider | VARCHAR(20) | NOT NULL | OAuth 供应商：github / google |
| oauth_id | VARCHAR(100) | NOT NULL | OAuth 供应商用户 ID |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.2 providers - LLM 供应商配置表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 供应商配置唯一标识 |
| user_id | UUID | NOT NULL, FK → users(id) ON DELETE CASCADE | 所属用户 |
| name | VARCHAR(100) | NOT NULL | 显示名称，如 "我的 OpenAI" |
| provider_type | VARCHAR(20) | NOT NULL | 供应商类型：openai / anthropic / google / ollama |
| api_key_encrypted | BYTEA | | 加密后的 API Key |
| base_url | VARCHAR(500) | | 自定义 API 地址（ollama / 代理场景） |
| is_active | BOOLEAN | NOT NULL, DEFAULT true | 是否启用 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.3 model_configs - 模型参数预设表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 预设唯一标识 |
| provider_id | UUID | NOT NULL, FK → providers(id) ON DELETE CASCADE | 所属供应商配置 |
| model_name | VARCHAR(100) | NOT NULL | 模型标识，如 gpt-4o, claude-sonnet-4-20250514 |
| display_name | VARCHAR(100) | | 显示名称 |
| temperature | DECIMAL(3,2) | DEFAULT 0.7, CHECK (0 <= temperature <= 2) | 温度参数 |
| max_tokens | INTEGER | DEFAULT 2048, CHECK (max_tokens > 0) | 最大输出 token 数 |
| top_p | DECIMAL(3,2) | DEFAULT 1.0, CHECK (0 <= top_p <= 1) | Top-P 采样参数 |
| is_default | BOOLEAN | NOT NULL, DEFAULT false | 是否为该供应商的默认模型 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |

### 3.4 prompts - 提示词表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 提示词唯一标识 |
| user_id | UUID | NOT NULL, FK → users(id) ON DELETE CASCADE | 所属用户 |
| name | VARCHAR(200) | NOT NULL | 提示词名称 |
| description | TEXT | | 描述 |
| current_version_id | UUID | FK → prompt_versions(id) ON DELETE SET NULL | 当前生效版本，延迟设置 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.5 prompt_versions - 提示词版本表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 版本唯一标识 |
| prompt_id | UUID | NOT NULL, FK → prompts(id) ON DELETE CASCADE | 所属提示词 |
| version_number | INTEGER | NOT NULL | 版本号，从 1 开始递增 |
| system_prompt | TEXT | NOT NULL | 系统提示词内容 |
| user_prompt | TEXT | | 用户提示词模板 |
| variables | JSONB | DEFAULT '[]'::jsonb | 变量名数组，如 ["name", "topic"] |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |

**唯一约束**：`UNIQUE (prompt_id, version_number)`

### 3.6 knowledge_bases - 知识库表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 知识库唯一标识 |
| user_id | UUID | NOT NULL, FK → users(id) ON DELETE CASCADE | 所属用户 |
| name | VARCHAR(200) | NOT NULL | 知识库名称 |
| description | TEXT | | 描述 |
| embedding_model | VARCHAR(100) | NOT NULL | 向量模型标识，如 text-embedding-3-small |
| chunk_size | INTEGER | NOT NULL, DEFAULT 500 | 分块大小（字符数） |
| chunk_overlap | INTEGER | NOT NULL, DEFAULT 50 | 分块重叠字符数 |
| status | VARCHAR(20) | NOT NULL, DEFAULT 'active' | 状态：active / processing |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.7 documents - 文档表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 文档唯一标识 |
| kb_id | UUID | NOT NULL, FK → knowledge_bases(id) ON DELETE CASCADE | 所属知识库 |
| filename | VARCHAR(500) | NOT NULL | 原始文件名 |
| file_type | VARCHAR(10) | NOT NULL | 文件类型：pdf / md / txt / docx |
| file_size | BIGINT | NOT NULL | 文件大小（字节） |
| status | VARCHAR(20) | NOT NULL, DEFAULT 'pending' | 状态：pending / processing / completed / error |
| chunk_count | INTEGER | DEFAULT 0 | 分块数量 |
| error_message | TEXT | | 处理失败时的错误信息 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.8 agents - 智能体表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 智能体唯一标识 |
| user_id | UUID | NOT NULL, FK → users(id) ON DELETE CASCADE | 所属用户 |
| name | VARCHAR(200) | NOT NULL | 智能体名称 |
| description | TEXT | | 描述 |
| system_prompt | TEXT | | 系统提示词 |
| model_config_id | UUID | FK → model_configs(id) ON DELETE SET NULL | 关联的模型配置 |
| strategy | VARCHAR(30) | NOT NULL, DEFAULT 'function_calling' | 执行策略：react / function_calling |
| max_iterations | INTEGER | NOT NULL, DEFAULT 10, CHECK (max_iterations > 0) | 最大迭代次数 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.9 agent_tools - 智能体工具表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 工具唯一标识 |
| agent_id | UUID | NOT NULL, FK → agents(id) ON DELETE CASCADE | 所属智能体 |
| tool_type | VARCHAR(30) | NOT NULL | 工具类型：web_search / calculator / code_runner / http_request / knowledge_retrieval |
| tool_config | JSONB | DEFAULT '{}'::jsonb | 工具配置参数 |
| is_enabled | BOOLEAN | NOT NULL, DEFAULT true | 是否启用 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |

### 3.10 agent_knowledge_bases - 智能体-知识库关联表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| agent_id | UUID | PK, FK → agents(id) ON DELETE CASCADE | 智能体 ID |
| kb_id | UUID | PK, FK → knowledge_bases(id) ON DELETE CASCADE | 知识库 ID |

### 3.11 workflows - 工作流表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 工作流唯一标识 |
| user_id | UUID | NOT NULL, FK → users(id) ON DELETE CASCADE | 所属用户 |
| name | VARCHAR(200) | NOT NULL | 工作流名称 |
| description | TEXT | | 描述 |
| graph_data | JSONB | | React Flow 完整画布状态（nodes + edges + viewport） |
| is_published | BOOLEAN | NOT NULL, DEFAULT false | 是否已发布 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.12 workflow_nodes - 工作流节点表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 节点唯一标识 |
| workflow_id | UUID | NOT NULL, FK → workflows(id) ON DELETE CASCADE | 所属工作流 |
| node_type | VARCHAR(30) | NOT NULL | 节点类型：start / llm / knowledge_retrieval / condition / code / http / end |
| label | VARCHAR(100) | | 节点显示名称 |
| position_x | DOUBLE PRECISION | NOT NULL | 画布 X 坐标 |
| position_y | DOUBLE PRECISION | NOT NULL | 画布 Y 坐标 |
| config | JSONB | DEFAULT '{}'::jsonb | 节点配置（不同类型 schema 不同） |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |

### 3.13 workflow_edges - 工作流边表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 边唯一标识 |
| workflow_id | UUID | NOT NULL, FK → workflows(id) ON DELETE CASCADE | 所属工作流 |
| source_node_id | UUID | NOT NULL, FK → workflow_nodes(id) ON DELETE CASCADE | 源节点 |
| target_node_id | UUID | NOT NULL, FK → workflow_nodes(id) ON DELETE CASCADE | 目标节点 |
| source_handle | VARCHAR(50) | | 源节点连接点标识 |
| target_handle | VARCHAR(50) | | 目标节点连接点标识 |
| condition | JSONB | | 条件分支表达式（仅 condition 节点出边使用） |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |

### 3.14 apps - 应用表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 应用唯一标识 |
| user_id | UUID | NOT NULL, FK → users(id) ON DELETE CASCADE | 所属用户 |
| name | VARCHAR(200) | NOT NULL | 应用名称 |
| description | TEXT | | 描述 |
| app_type | VARCHAR(20) | NOT NULL | 应用类型：chatbot / completion / workflow |
| config | JSONB | NOT NULL, DEFAULT '{}'::jsonb | 应用配置（结构见下方说明） |
| is_published | BOOLEAN | NOT NULL, DEFAULT false | 是否已发布 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

**config 字段结构说明**：

```jsonc
// app_type = "chatbot"
{ "agent_id": "uuid" }

// app_type = "completion"
{ "prompt_id": "uuid", "model_config_id": "uuid" }

// app_type = "workflow"
{ "workflow_id": "uuid" }
```

### 3.15 app_api_keys - 应用 API Key 表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | Key 唯一标识 |
| app_id | UUID | NOT NULL, FK → apps(id) ON DELETE CASCADE | 所属应用 |
| key_hash | VARCHAR(64) | NOT NULL | SHA-256 哈希值，用于鉴权查找 |
| key_prefix | VARCHAR(12) | NOT NULL | 前缀用于脱敏显示，如 "sk-mini-abcd" |
| name | VARCHAR(100) | | Key 备注名称 |
| is_active | BOOLEAN | NOT NULL, DEFAULT true | 是否启用 |
| rate_limit | INTEGER | NOT NULL, DEFAULT 60 | 每分钟请求限制 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| last_used_at | TIMESTAMPTZ | | 最后使用时间 |

### 3.16 conversations - 会话表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 会话唯一标识 |
| app_id | UUID | NOT NULL, FK → apps(id) ON DELETE CASCADE | 所属应用 |
| external_user_id | VARCHAR(100) | | 外部调用者标识（Gateway 场景） |
| title | VARCHAR(200) | | 会话标题 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 更新时间 |

### 3.17 messages - 消息表

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | UUID | PK, DEFAULT gen_random_uuid() | 消息唯一标识 |
| conversation_id | UUID | NOT NULL, FK → conversations(id) ON DELETE CASCADE | 所属会话 |
| role | VARCHAR(20) | NOT NULL | 角色：system / user / assistant / tool |
| content | TEXT | NOT NULL | 消息内容 |
| token_count | INTEGER | | Token 用量 |
| model_name | VARCHAR(100) | | 使用的模型标识 |
| cost | DECIMAL(10,6) | | 本次调用费用（美元） |
| metadata | JSONB | DEFAULT '{}'::jsonb | 元数据：工具调用详情、引用来源等 |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() | 创建时间 |

---

## 4. 索引设计

### 4.1 主键索引（自动创建）

所有表的 `id` 字段（或复合主键）均自动创建唯一索引，不再赘述。

### 4.2 唯一索引

| 表 | 索引名 | 字段 | 说明 |
|----|--------|------|------|
| users | uk_users_email | email | 邮箱唯一 |
| users | uk_users_oauth | (oauth_provider, oauth_id) | 同一 OAuth 供应商下用户唯一 |
| prompt_versions | uk_prompt_versions_number | (prompt_id, version_number) | 同一提示词下版本号唯一 |

### 4.3 外键 / 查询索引

| 表 | 索引名 | 字段 | 说明 |
|----|--------|------|------|
| providers | idx_providers_user_id | user_id | 按用户查询供应商 |
| model_configs | idx_model_configs_provider_id | provider_id | 按供应商查询模型配置 |
| prompts | idx_prompts_user_id | user_id | 按用户查询提示词 |
| prompt_versions | idx_prompt_versions_prompt_id | prompt_id | 按提示词查询版本列表 |
| knowledge_bases | idx_knowledge_bases_user_id | user_id | 按用户查询知识库 |
| documents | idx_documents_kb_id | kb_id | 按知识库查询文档 |
| documents | idx_documents_status | (kb_id, status) | 按知识库过滤文档状态 |
| agents | idx_agents_user_id | user_id | 按用户查询智能体 |
| agent_tools | idx_agent_tools_agent_id | agent_id | 按智能体查询工具 |
| workflows | idx_workflows_user_id | user_id | 按用户查询工作流 |
| workflow_nodes | idx_workflow_nodes_workflow_id | workflow_id | 按工作流查询节点 |
| workflow_edges | idx_workflow_edges_workflow_id | workflow_id | 按工作流查询边 |
| apps | idx_apps_user_id | user_id | 按用户查询应用 |
| app_api_keys | idx_app_api_keys_key_hash | key_hash | 通过哈希值快速鉴权查找 |
| app_api_keys | idx_app_api_keys_app_id | app_id | 按应用查询 Key 列表 |
| conversations | idx_conversations_app_id | app_id | 按应用查询会话 |
| conversations | idx_conversations_app_user | (app_id, external_user_id) | 按应用 + 外部用户查询会话 |
| messages | idx_messages_conversation_id | conversation_id | 按会话查询消息（按创建时间排序） |
| messages | idx_messages_created_at | (conversation_id, created_at) | 会话内消息按时间排序查询 |

---

## 5. 特殊设计决策说明

### 5.1 UUID vs 自增 ID

选择 UUID v4 作为主键，原因如下：

- **安全性**：自增 ID 暴露记录数量和创建顺序，UUID 不可预测
- **分布式友好**：无需中心化的序列生成器，客户端可直接生成
- **合并友好**：多环境数据合并不会产生 ID 冲突
- **代价**：UUID 占用 16 字节（vs BIGINT 8 字节），索引稍大。对于本项目规模完全可接受
- **实现**：使用 PostgreSQL 内置的 `gen_random_uuid()` 函数

### 5.2 JSONB 字段使用策略

| 字段 | 选择 JSONB 的原因 |
|------|-------------------|
| prompt_versions.variables | 变量列表长度不定，且仅存储变量名 |
| agent_tools.tool_config | 不同工具类型的配置 schema 差异大 |
| workflows.graph_data | React Flow 画布完整状态，结构复杂且前端直出 |
| workflow_nodes.config | 不同节点类型配置差异大（LLM 节点 vs HTTP 节点） |
| workflow_edges.condition | 仅条件分支边使用，且表达式结构灵活 |
| apps.config | 不同应用类型引用不同实体，用 JSONB 统一存储 |
| messages.metadata | 工具调用结果、引用来源等结构可变的附加信息 |

### 5.3 提示词版本管理

采用 `prompts` + `prompt_versions` 两表设计：

- `prompts` 表维护元信息和 `current_version_id` 指针
- `prompt_versions` 表存储每个版本的完整快照（不使用 diff）
- 通过 `current_version_id` 可快速定位当前生效版本
- 版本号 `version_number` 单调递增，配合唯一约束防止并发冲突

### 5.4 API Key 安全存储

- 生成时：创建随机 Key → 返回给用户（仅此一次）→ 存储 SHA-256 哈希 + 前缀
- 鉴权时：收到 Key → 计算 SHA-256 → 通过 `key_hash` 索引查找
- `key_prefix` 仅用于列表展示脱敏（如 `sk-mini-abcd...`）
- 原始 Key 不持久化，丢失后只能重新生成

### 5.5 apps.config 多态设计

不同 `app_type` 对应不同的 config 结构，使用 JSONB 实现「多态配置」而非建立多张子表。优势是查询简单，只需一次 JOIN；代价是外键约束需要在应用层校验。

### 5.6 级联删除策略

所有外键均设置 `ON DELETE CASCADE`，删除父记录时自动清理子记录。例外情况：

- `agents.model_config_id` → `ON DELETE SET NULL`（删除模型配置时不删智能体，只清空引用）
- `prompts.current_version_id` → `ON DELETE SET NULL`（删除版本时不删提示词）

---

## 6. 完整 DDL

```sql
-- ============================================================
-- Mini-Dify Database Schema
-- PostgreSQL 16
-- ============================================================

-- 启用 pgcrypto 扩展（用于 gen_random_uuid）
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- 1. users
-- ============================================================
CREATE TABLE users (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL,
    name            VARCHAR(100) NOT NULL,
    avatar_url      VARCHAR(500),
    oauth_provider  VARCHAR(20)  NOT NULL,
    oauth_id        VARCHAR(100) NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT uk_users_email UNIQUE (email),
    CONSTRAINT uk_users_oauth UNIQUE (oauth_provider, oauth_id)
);

-- ============================================================
-- 2. providers
-- ============================================================
CREATE TABLE providers (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name              VARCHAR(100) NOT NULL,
    provider_type     VARCHAR(20)  NOT NULL,
    api_key_encrypted BYTEA,
    base_url          VARCHAR(500),
    is_active         BOOLEAN      NOT NULL DEFAULT true,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT chk_providers_type CHECK (provider_type IN ('openai', 'anthropic', 'google', 'ollama'))
);

CREATE INDEX idx_providers_user_id ON providers(user_id);

-- ============================================================
-- 3. model_configs
-- ============================================================
CREATE TABLE model_configs (
    id            UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_id   UUID           NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    model_name    VARCHAR(100)   NOT NULL,
    display_name  VARCHAR(100),
    temperature   DECIMAL(3,2)   DEFAULT 0.70,
    max_tokens    INTEGER        DEFAULT 2048,
    top_p         DECIMAL(3,2)   DEFAULT 1.00,
    is_default    BOOLEAN        NOT NULL DEFAULT false,
    created_at    TIMESTAMPTZ    NOT NULL DEFAULT now(),

    CONSTRAINT chk_model_configs_temperature CHECK (temperature >= 0 AND temperature <= 2),
    CONSTRAINT chk_model_configs_max_tokens  CHECK (max_tokens > 0),
    CONSTRAINT chk_model_configs_top_p       CHECK (top_p >= 0 AND top_p <= 1)
);

CREATE INDEX idx_model_configs_provider_id ON model_configs(provider_id);

-- ============================================================
-- 4. prompts
-- ============================================================
CREATE TABLE prompts (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name                VARCHAR(200) NOT NULL,
    description         TEXT,
    current_version_id  UUID,       -- FK 延迟添加（循环引用）
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_prompts_user_id ON prompts(user_id);

-- ============================================================
-- 5. prompt_versions
-- ============================================================
CREATE TABLE prompt_versions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_id       UUID        NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    version_number  INTEGER     NOT NULL,
    system_prompt   TEXT        NOT NULL,
    user_prompt     TEXT,
    variables       JSONB       DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uk_prompt_versions_number UNIQUE (prompt_id, version_number)
);

CREATE INDEX idx_prompt_versions_prompt_id ON prompt_versions(prompt_id);

-- 回填 prompts.current_version_id 外键
ALTER TABLE prompts
    ADD CONSTRAINT fk_prompts_current_version
    FOREIGN KEY (current_version_id) REFERENCES prompt_versions(id) ON DELETE SET NULL;

-- ============================================================
-- 6. knowledge_bases
-- ============================================================
CREATE TABLE knowledge_bases (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            VARCHAR(200) NOT NULL,
    description     TEXT,
    embedding_model VARCHAR(100) NOT NULL,
    chunk_size      INTEGER     NOT NULL DEFAULT 500,
    chunk_overlap   INTEGER     NOT NULL DEFAULT 50,
    status          VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_knowledge_bases_status CHECK (status IN ('active', 'processing'))
);

CREATE INDEX idx_knowledge_bases_user_id ON knowledge_bases(user_id);

-- ============================================================
-- 7. documents
-- ============================================================
CREATE TABLE documents (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id         UUID        NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename      VARCHAR(500) NOT NULL,
    file_type     VARCHAR(10)  NOT NULL,
    file_size     BIGINT       NOT NULL,
    status        VARCHAR(20)  NOT NULL DEFAULT 'pending',
    chunk_count   INTEGER      DEFAULT 0,
    error_message TEXT,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT chk_documents_file_type CHECK (file_type IN ('pdf', 'md', 'txt', 'docx')),
    CONSTRAINT chk_documents_status    CHECK (status IN ('pending', 'processing', 'completed', 'error'))
);

CREATE INDEX idx_documents_kb_id  ON documents(kb_id);
CREATE INDEX idx_documents_status ON documents(kb_id, status);

-- ============================================================
-- 8. agents
-- ============================================================
CREATE TABLE agents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            VARCHAR(200) NOT NULL,
    description     TEXT,
    system_prompt   TEXT,
    model_config_id UUID        REFERENCES model_configs(id) ON DELETE SET NULL,
    strategy        VARCHAR(30) NOT NULL DEFAULT 'function_calling',
    max_iterations  INTEGER     NOT NULL DEFAULT 10,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_agents_strategy       CHECK (strategy IN ('react', 'function_calling')),
    CONSTRAINT chk_agents_max_iterations CHECK (max_iterations > 0)
);

CREATE INDEX idx_agents_user_id ON agents(user_id);

-- ============================================================
-- 9. agent_tools
-- ============================================================
CREATE TABLE agent_tools (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id    UUID        NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    tool_type   VARCHAR(30) NOT NULL,
    tool_config JSONB       DEFAULT '{}'::jsonb,
    is_enabled  BOOLEAN     NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_agent_tools_type CHECK (
        tool_type IN ('web_search', 'calculator', 'code_runner', 'http_request', 'knowledge_retrieval')
    )
);

CREATE INDEX idx_agent_tools_agent_id ON agent_tools(agent_id);

-- ============================================================
-- 10. agent_knowledge_bases
-- ============================================================
CREATE TABLE agent_knowledge_bases (
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kb_id    UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,

    PRIMARY KEY (agent_id, kb_id)
);

-- ============================================================
-- 11. workflows
-- ============================================================
CREATE TABLE workflows (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         VARCHAR(200) NOT NULL,
    description  TEXT,
    graph_data   JSONB,
    is_published BOOLEAN     NOT NULL DEFAULT false,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_workflows_user_id ON workflows(user_id);

-- ============================================================
-- 12. workflow_nodes
-- ============================================================
CREATE TABLE workflow_nodes (
    id          UUID             PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID             NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    node_type   VARCHAR(30)      NOT NULL,
    label       VARCHAR(100),
    position_x  DOUBLE PRECISION NOT NULL,
    position_y  DOUBLE PRECISION NOT NULL,
    config      JSONB            DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ      NOT NULL DEFAULT now(),

    CONSTRAINT chk_workflow_nodes_type CHECK (
        node_type IN ('start', 'llm', 'knowledge_retrieval', 'condition', 'code', 'http', 'end')
    )
);

CREATE INDEX idx_workflow_nodes_workflow_id ON workflow_nodes(workflow_id);

-- ============================================================
-- 13. workflow_edges
-- ============================================================
CREATE TABLE workflow_edges (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id    UUID        NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    source_node_id UUID        NOT NULL REFERENCES workflow_nodes(id) ON DELETE CASCADE,
    target_node_id UUID        NOT NULL REFERENCES workflow_nodes(id) ON DELETE CASCADE,
    source_handle  VARCHAR(50),
    target_handle  VARCHAR(50),
    condition      JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_workflow_edges_workflow_id ON workflow_edges(workflow_id);

-- ============================================================
-- 14. apps
-- ============================================================
CREATE TABLE apps (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         VARCHAR(200) NOT NULL,
    description  TEXT,
    app_type     VARCHAR(20) NOT NULL,
    config       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    is_published BOOLEAN     NOT NULL DEFAULT false,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_apps_type CHECK (app_type IN ('chatbot', 'completion', 'workflow'))
);

CREATE INDEX idx_apps_user_id ON apps(user_id);

-- ============================================================
-- 15. app_api_keys
-- ============================================================
CREATE TABLE app_api_keys (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    app_id       UUID        NOT NULL REFERENCES apps(id) ON DELETE CASCADE,
    key_hash     VARCHAR(64) NOT NULL,
    key_prefix   VARCHAR(12) NOT NULL,
    name         VARCHAR(100),
    is_active    BOOLEAN     NOT NULL DEFAULT true,
    rate_limit   INTEGER     NOT NULL DEFAULT 60,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ
);

CREATE INDEX idx_app_api_keys_key_hash ON app_api_keys(key_hash);
CREATE INDEX idx_app_api_keys_app_id   ON app_api_keys(app_id);

-- ============================================================
-- 16. conversations
-- ============================================================
CREATE TABLE conversations (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    app_id           UUID        NOT NULL REFERENCES apps(id) ON DELETE CASCADE,
    external_user_id VARCHAR(100),
    title            VARCHAR(200),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_conversations_app_id    ON conversations(app_id);
CREATE INDEX idx_conversations_app_user  ON conversations(app_id, external_user_id);

-- ============================================================
-- 17. messages
-- ============================================================
CREATE TABLE messages (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID          NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            VARCHAR(20)   NOT NULL,
    content         TEXT          NOT NULL,
    token_count     INTEGER,
    model_name      VARCHAR(100),
    cost            DECIMAL(10,6),
    metadata        JSONB         DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT now(),

    CONSTRAINT chk_messages_role CHECK (role IN ('system', 'user', 'assistant', 'tool'))
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at      ON messages(conversation_id, created_at);

-- ============================================================
-- updated_at 自动更新触发器
-- ============================================================
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为所有含 updated_at 字段的表创建触发器
DO $$
DECLARE
    t TEXT;
BEGIN
    FOR t IN
        SELECT unnest(ARRAY[
            'users', 'providers', 'prompts', 'knowledge_bases',
            'documents', 'agents', 'workflows', 'apps', 'conversations'
        ])
    LOOP
        EXECUTE format(
            'CREATE TRIGGER set_updated_at BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at()',
            t
        );
    END LOOP;
END;
$$;
```
