# Mini-Dify API 设计文档

## 1. API 总览

### 1.1 Base URL

| 类型 | Base URL | 用途 |
|------|----------|------|
| 内部 API | `/api/v1` | 前端管理后台调用 |
| Gateway API | `/api/gateway` | 外部应用通过 API Key 调用 |

### 1.2 认证方式

#### 内部 API — JWT

所有 `/api/v1/*` 端点需要在请求头中携带 JWT：

```
Authorization: Bearer <jwt_token>
```

JWT 由 NextAuth OAuth 回调签发，包含 `user_id`、`email` 等声明。

#### Gateway API — App API Key

所有 `/api/gateway/*` 端点需要在请求头中携带应用 API Key：

```
Authorization: Bearer sk-mini-xxx
```

API Key 在「应用管理 > 发布」时生成，前缀统一为 `sk-mini-`。

### 1.3 通用响应格式

#### 成功响应

```json
{
  "code": 0,
  "message": "success",
  "data": { ... }
}
```

#### 分页格式

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [ ... ],
    "total": 100,
    "page": 1,
    "page_size": 20,
    "total_pages": 5
  }
}
```

通用分页查询参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | int | 1 | 页码，从 1 开始 |
| `page_size` | int | 20 | 每页条数，最大 100 |

#### 错误响应

```json
{
  "code": 40001,
  "message": "Invalid provider API key",
  "details": "OpenAI API returned 401 Unauthorized"
}
```

---

## 2. 详细 API 端点

### 2.1 认证模块

#### POST /api/auth/callback — OAuth 回调

由 NextAuth 处理，前端无需直接调用。OAuth Provider 回调到此端点完成用户登录/注册，签发 JWT。

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/auth/callback` |
| 认证 | 无（OAuth Provider 回调） |

---

### 2.2 模型管理（Model Hub）

#### GET /api/v1/providers — 获取供应商列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/providers` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "provider_001",
        "name": "My OpenAI",
        "provider_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_set": true,
        "is_verified": true,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/providers?page=1&page_size=20' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/providers — 添加供应商

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/providers` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "My OpenAI",
  "provider_type": "openai",
  "api_key": "sk-xxxxxxxxxxxxxxxx",
  "base_url": "https://api.openai.com/v1"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 供应商显示名称 |
| `provider_type` | string | 是 | 供应商类型：`openai` / `anthropic` / `google` / `custom` |
| `api_key` | string | 是 | API 密钥 |
| `base_url` | string | 否 | 自定义接口地址（custom 类型必填） |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "provider_001",
    "name": "My OpenAI",
    "provider_type": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key_set": true,
    "is_verified": false,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/providers' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "My OpenAI",
    "provider_type": "openai",
    "api_key": "sk-xxxxxxxxxxxxxxxx",
    "base_url": "https://api.openai.com/v1"
  }'
```

---

#### PUT /api/v1/providers/{id} — 更新供应商

| 项目 | 说明 |
|------|------|
| Method | PUT |
| URL | `/api/v1/providers/{id}` |
| 认证 | JWT |

**Path Parameters:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `id` | string | 供应商 ID |

**Request Body:**

```json
{
  "name": "My OpenAI (Updated)",
  "api_key": "sk-new-key",
  "base_url": "https://api.openai.com/v1"
}
```

所有字段均为可选，仅更新传入的字段。

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "provider_001",
    "name": "My OpenAI (Updated)",
    "provider_type": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key_set": true,
    "is_verified": true,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X PUT 'http://localhost:8000/api/v1/providers/provider_001' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{"name": "My OpenAI (Updated)", "api_key": "sk-new-key"}'
```

---

#### DELETE /api/v1/providers/{id} — 删除供应商

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/providers/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/providers/provider_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/providers/{id}/verify — 验证连通性

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/providers/{id}/verify` |
| 认证 | JWT |

**Request Body:** 无

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "is_valid": true,
    "latency_ms": 230,
    "error": null
  }
}
```

验证失败时：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "is_valid": false,
    "latency_ms": null,
    "error": "Authentication failed: invalid API key"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/providers/provider_001/verify' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### GET /api/v1/providers/{id}/models — 获取供应商可用模型列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/providers/{id}/models` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "model_id": "gpt-4o",
      "model_name": "GPT-4o",
      "model_type": "chat",
      "max_tokens": 128000
    },
    {
      "model_id": "gpt-4o-mini",
      "model_name": "GPT-4o Mini",
      "model_type": "chat",
      "max_tokens": 128000
    },
    {
      "model_id": "text-embedding-3-small",
      "model_name": "Text Embedding 3 Small",
      "model_type": "embedding",
      "max_tokens": 8191
    }
  ]
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/providers/provider_001/models' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/model-configs — 创建模型参数预设

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/model-configs` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "GPT-4o Creative",
  "provider_id": "provider_001",
  "model_id": "gpt-4o",
  "parameters": {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 4096,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 预设名称 |
| `provider_id` | string | 是 | 关联供应商 ID |
| `model_id` | string | 是 | 模型标识 |
| `parameters` | object | 是 | 模型参数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "mc_001",
    "name": "GPT-4o Creative",
    "provider_id": "provider_001",
    "model_id": "gpt-4o",
    "parameters": {
      "temperature": 0.9,
      "top_p": 0.95,
      "max_tokens": 4096,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    },
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/model-configs' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "GPT-4o Creative",
    "provider_id": "provider_001",
    "model_id": "gpt-4o",
    "parameters": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 4096}
  }'
```

---

#### GET /api/v1/model-configs — 获取模型参数预设列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/model-configs` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `provider_id` | string | 否 | 按供应商筛选 |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "mc_001",
        "name": "GPT-4o Creative",
        "provider_id": "provider_001",
        "model_id": "gpt-4o",
        "parameters": {
          "temperature": 0.9,
          "top_p": 0.95,
          "max_tokens": 4096
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/model-configs?provider_id=provider_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

### 2.3 Prompt Studio

#### GET /api/v1/prompts — Prompt 列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/prompts` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `keyword` | string | 否 | 搜索关键词（匹配名称） |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "prompt_001",
        "name": "翻译助手",
        "description": "中英互译 Prompt",
        "current_version": 3,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-03T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/prompts?page=1&page_size=20' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/prompts — 创建 Prompt

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/prompts` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "翻译助手",
  "description": "中英互译 Prompt",
  "system_prompt": "你是一个专业翻译。如果输入是中文，翻译为英文；如果输入是英文，翻译为中文。",
  "user_prompt_template": "请翻译以下文本：\n\n{{text}}",
  "variables": [
    {
      "name": "text",
      "type": "string",
      "description": "待翻译文本",
      "required": true
    }
  ]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | Prompt 名称 |
| `description` | string | 否 | Prompt 描述 |
| `system_prompt` | string | 否 | 系统提示词 |
| `user_prompt_template` | string | 是 | 用户提示词模板，支持 `{{variable}}` 占位符 |
| `variables` | array | 否 | 变量定义列表 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "prompt_001",
    "name": "翻译助手",
    "description": "中英互译 Prompt",
    "system_prompt": "你是一个专业翻译...",
    "user_prompt_template": "请翻译以下文本：\n\n{{text}}",
    "variables": [
      {
        "name": "text",
        "type": "string",
        "description": "待翻译文本",
        "required": true
      }
    ],
    "current_version": 1,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/prompts' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "翻译助手",
    "description": "中英互译 Prompt",
    "system_prompt": "你是一个专业翻译...",
    "user_prompt_template": "请翻译以下文本：\n\n{{text}}",
    "variables": [{"name": "text", "type": "string", "required": true}]
  }'
```

---

#### GET /api/v1/prompts/{id} — 获取 Prompt 详情

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/prompts/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "prompt_001",
    "name": "翻译助手",
    "description": "中英互译 Prompt",
    "system_prompt": "你是一个专业翻译...",
    "user_prompt_template": "请翻译以下文本：\n\n{{text}}",
    "variables": [
      {
        "name": "text",
        "type": "string",
        "description": "待翻译文本",
        "required": true
      }
    ],
    "current_version": 3,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-03T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/prompts/prompt_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### PUT /api/v1/prompts/{id} — 更新 Prompt

更新 Prompt 内容时自动创建新版本。

| 项目 | 说明 |
|------|------|
| Method | PUT |
| URL | `/api/v1/prompts/{id}` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "翻译助手 v2",
  "system_prompt": "你是一个资深翻译专家，注重信达雅。",
  "user_prompt_template": "请将以下文本翻译为{{target_lang}}：\n\n{{text}}",
  "variables": [
    {"name": "text", "type": "string", "required": true},
    {"name": "target_lang", "type": "string", "required": true, "description": "目标语言"}
  ]
}
```

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "prompt_001",
    "name": "翻译助手 v2",
    "system_prompt": "你是一个资深翻译专家，注重信达雅。",
    "user_prompt_template": "请将以下文本翻译为{{target_lang}}：\n\n{{text}}",
    "variables": [
      {"name": "text", "type": "string", "required": true},
      {"name": "target_lang", "type": "string", "required": true, "description": "目标语言"}
    ],
    "current_version": 4,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-04T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X PUT 'http://localhost:8000/api/v1/prompts/prompt_001' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "翻译助手 v2",
    "system_prompt": "你是一个资深翻译专家，注重信达雅。",
    "user_prompt_template": "请将以下文本翻译为{{target_lang}}：\n\n{{text}}"
  }'
```

---

#### DELETE /api/v1/prompts/{id} — 删除 Prompt

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/prompts/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/prompts/prompt_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### GET /api/v1/prompts/{id}/versions — 版本历史列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/prompts/{id}/versions` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "version_id": "ver_003",
      "version_number": 3,
      "system_prompt": "你是一个资深翻译专家...",
      "user_prompt_template": "请将以下文本翻译为{{target_lang}}：\n\n{{text}}",
      "variables": [...],
      "created_at": "2026-01-03T00:00:00Z"
    },
    {
      "version_id": "ver_002",
      "version_number": 2,
      "system_prompt": "你是一个专业翻译...",
      "user_prompt_template": "请翻译以下文本：\n\n{{text}}",
      "variables": [...],
      "created_at": "2026-01-02T00:00:00Z"
    },
    {
      "version_id": "ver_001",
      "version_number": 1,
      "system_prompt": "你是一个翻译。",
      "user_prompt_template": "翻译：{{text}}",
      "variables": [...],
      "created_at": "2026-01-01T00:00:00Z"
    }
  ]
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/prompts/prompt_001/versions' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/prompts/{id}/versions/{version_id}/rollback — 回滚到指定版本

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/prompts/{id}/versions/{version_id}/rollback` |
| 认证 | JWT |

**Request Body:** 无

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "prompt_001",
    "name": "翻译助手",
    "current_version": 4,
    "system_prompt": "你是一个专业翻译...",
    "user_prompt_template": "请翻译以下文本：\n\n{{text}}",
    "updated_at": "2026-01-04T12:00:00Z"
  }
}
```

> 说明：回滚操作会基于目标版本的内容创建一个新版本（version +1），而非覆盖当前版本。

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/prompts/prompt_001/versions/ver_002/rollback' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/prompts/{id}/test — 测试 Prompt（SSE 流式）

支持同时指定多个 model_config_id，对比不同模型的输出效果。

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/prompts/{id}/test` |
| 认证 | JWT |
| 响应类型 | `text/event-stream` (SSE) |

**Request Body:**

```json
{
  "model_config_ids": ["mc_001", "mc_002"],
  "variables": {
    "text": "人工智能正在改变世界"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_config_ids` | array[string] | 是 | 模型参数预设 ID 列表，支持多个以便对比 |
| `variables` | object | 否 | 变量键值对，用于替换模板中的占位符 |

**SSE 事件流:**

```
event: message_delta
data: {"model_config_id": "mc_001", "content": "Artificial"}

event: message_delta
data: {"model_config_id": "mc_001", "content": " intelligence"}

event: message_delta
data: {"model_config_id": "mc_002", "content": "AI is"}

event: message_done
data: {"model_config_id": "mc_001", "total_tokens": 45, "latency_ms": 1200}

event: message_done
data: {"model_config_id": "mc_002", "total_tokens": 52, "latency_ms": 980}

event: done
data: {}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/prompts/prompt_001/test' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model_config_ids": ["mc_001"],
    "variables": {"text": "人工智能正在改变世界"}
  }'
```

---

### 2.4 知识库（Knowledge Base）

#### GET /api/v1/knowledge-bases — 知识库列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/knowledge-bases` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `keyword` | string | 否 | 搜索关键词 |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "kb_001",
        "name": "产品文档库",
        "description": "公司产品帮助文档",
        "document_count": 12,
        "embedding_model": "text-embedding-3-small",
        "chunk_strategy": {
          "chunk_size": 500,
          "chunk_overlap": 50
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-05T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/knowledge-bases?page=1' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/knowledge-bases — 创建知识库

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/knowledge-bases` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "产品文档库",
  "description": "公司产品帮助文档",
  "embedding_provider_id": "provider_001",
  "embedding_model_id": "text-embedding-3-small",
  "chunk_strategy": {
    "chunk_size": 500,
    "chunk_overlap": 50
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 知识库名称 |
| `description` | string | 否 | 描述 |
| `embedding_provider_id` | string | 是 | Embedding 模型供应商 ID |
| `embedding_model_id` | string | 是 | Embedding 模型标识 |
| `chunk_strategy` | object | 否 | 分块策略 |
| `chunk_strategy.chunk_size` | int | 否 | 分块大小（字符数），默认 500 |
| `chunk_strategy.chunk_overlap` | int | 否 | 分块重叠大小，默认 50 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "kb_001",
    "name": "产品文档库",
    "description": "公司产品帮助文档",
    "embedding_provider_id": "provider_001",
    "embedding_model_id": "text-embedding-3-small",
    "document_count": 0,
    "chunk_strategy": {
      "chunk_size": 500,
      "chunk_overlap": 50
    },
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/knowledge-bases' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "产品文档库",
    "description": "公司产品帮助文档",
    "embedding_provider_id": "provider_001",
    "embedding_model_id": "text-embedding-3-small",
    "chunk_strategy": {"chunk_size": 500, "chunk_overlap": 50}
  }'
```

---

#### GET /api/v1/knowledge-bases/{id} — 知识库详情

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/knowledge-bases/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "kb_001",
    "name": "产品文档库",
    "description": "公司产品帮助文档",
    "embedding_provider_id": "provider_001",
    "embedding_model_id": "text-embedding-3-small",
    "document_count": 12,
    "total_chunks": 248,
    "chunk_strategy": {
      "chunk_size": 500,
      "chunk_overlap": 50
    },
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-05T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/knowledge-bases/kb_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### PUT /api/v1/knowledge-bases/{id} — 更新知识库配置

| 项目 | 说明 |
|------|------|
| Method | PUT |
| URL | `/api/v1/knowledge-bases/{id}` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "产品文档库（更新）",
  "description": "更新后的描述",
  "chunk_strategy": {
    "chunk_size": 800,
    "chunk_overlap": 100
  }
}
```

> 注意：更新 chunk_strategy 不会自动重新分块已有文档，需要删除后重新上传。

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "kb_001",
    "name": "产品文档库（更新）",
    "description": "更新后的描述",
    "chunk_strategy": {
      "chunk_size": 800,
      "chunk_overlap": 100
    },
    "updated_at": "2026-01-06T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X PUT 'http://localhost:8000/api/v1/knowledge-bases/kb_001' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{"name": "产品文档库（更新）", "chunk_strategy": {"chunk_size": 800, "chunk_overlap": 100}}'
```

---

#### DELETE /api/v1/knowledge-bases/{id} — 删除知识库

删除知识库及其所有文档、向量数据。

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/knowledge-bases/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/knowledge-bases/kb_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/knowledge-bases/{id}/documents — 上传文档

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/knowledge-bases/{id}/documents` |
| 认证 | JWT |
| Content-Type | `multipart/form-data` |

**Request Body (form-data):**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 上传文件，支持 `.txt`、`.md`、`.pdf`、`.docx` |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "document_id": "doc_001",
    "filename": "product-guide.pdf",
    "file_size": 1048576,
    "status": "processing",
    "chunk_count": 0,
    "created_at": "2026-01-05T00:00:00Z"
  }
}
```

> 文档上传后进入异步处理流程（解析 -> 分块 -> Embedding -> 入库），`status` 字段变化：`processing` -> `completed` / `failed`。

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/knowledge-bases/kb_001/documents' \
  -H 'Authorization: Bearer <jwt_token>' \
  -F 'file=@/path/to/product-guide.pdf'
```

---

#### GET /api/v1/knowledge-bases/{id}/documents — 文档列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/knowledge-bases/{id}/documents` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "document_id": "doc_001",
        "filename": "product-guide.pdf",
        "file_size": 1048576,
        "status": "completed",
        "chunk_count": 24,
        "created_at": "2026-01-05T00:00:00Z"
      },
      {
        "document_id": "doc_002",
        "filename": "faq.md",
        "file_size": 20480,
        "status": "completed",
        "chunk_count": 8,
        "created_at": "2026-01-05T10:00:00Z"
      }
    ],
    "total": 2,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/knowledge-bases/kb_001/documents' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### DELETE /api/v1/knowledge-bases/{id}/documents/{doc_id} — 删除文档

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/knowledge-bases/{id}/documents/{doc_id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/knowledge-bases/kb_001/documents/doc_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/knowledge-bases/{id}/search — 检索测试

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/knowledge-bases/{id}/search` |
| 认证 | JWT |

**Request Body:**

```json
{
  "query": "如何重置密码？",
  "top_k": 5
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 检索查询文本 |
| `top_k` | int | 否 | 返回最相似的 Top K 条结果，默认 5 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "results": [
      {
        "chunk_id": "chunk_042",
        "document_id": "doc_001",
        "filename": "product-guide.pdf",
        "content": "重置密码步骤：1. 点击登录页面的"忘记密码"链接...",
        "score": 0.92
      },
      {
        "chunk_id": "chunk_015",
        "document_id": "doc_002",
        "filename": "faq.md",
        "content": "Q: 忘记密码怎么办？A: 您可以通过邮箱验证重置密码...",
        "score": 0.87
      }
    ]
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/knowledge-bases/kb_001/search' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{"query": "如何重置密码？", "top_k": 5}'
```

---

### 2.5 Agent Builder

#### GET /api/v1/agents — Agent 列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/agents` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `keyword` | string | 否 | 搜索关键词 |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "agent_001",
        "name": "客服助手",
        "description": "智能客服 Agent，可查询知识库并调用工具",
        "model_config_id": "mc_001",
        "tools": ["knowledge_search", "web_search"],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-05T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/agents?page=1' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/agents — 创建 Agent

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/agents` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "客服助手",
  "description": "智能客服 Agent",
  "model_config_id": "mc_001",
  "system_prompt": "你是一个专业客服...",
  "tools": [
    {
      "type": "knowledge_search",
      "config": {
        "knowledge_base_id": "kb_001",
        "top_k": 3
      }
    },
    {
      "type": "web_search",
      "config": {}
    }
  ],
  "max_iterations": 10
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | Agent 名称 |
| `description` | string | 否 | 描述 |
| `model_config_id` | string | 是 | 模型参数预设 ID |
| `system_prompt` | string | 否 | Agent 系统提示词 |
| `tools` | array | 否 | 工具配置列表 |
| `tools[].type` | string | 是 | 工具类型：`knowledge_search` / `web_search` / `code_interpreter` |
| `tools[].config` | object | 否 | 工具配置参数 |
| `max_iterations` | int | 否 | 最大推理循环次数，默认 10 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "agent_001",
    "name": "客服助手",
    "description": "智能客服 Agent",
    "model_config_id": "mc_001",
    "system_prompt": "你是一个专业客服...",
    "tools": [...],
    "max_iterations": 10,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/agents' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "客服助手",
    "description": "智能客服 Agent",
    "model_config_id": "mc_001",
    "system_prompt": "你是一个专业客服...",
    "tools": [{"type": "knowledge_search", "config": {"knowledge_base_id": "kb_001", "top_k": 3}}],
    "max_iterations": 10
  }'
```

---

#### GET /api/v1/agents/{id} — Agent 详情

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/agents/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "agent_001",
    "name": "客服助手",
    "description": "智能客服 Agent",
    "model_config_id": "mc_001",
    "system_prompt": "你是一个专业客服...",
    "tools": [
      {
        "type": "knowledge_search",
        "config": {
          "knowledge_base_id": "kb_001",
          "top_k": 3
        }
      }
    ],
    "max_iterations": 10,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-05T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/agents/agent_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### PUT /api/v1/agents/{id} — 更新 Agent

| 项目 | 说明 |
|------|------|
| Method | PUT |
| URL | `/api/v1/agents/{id}` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "客服助手 v2",
  "system_prompt": "你是一个高级客服专家...",
  "tools": [
    {"type": "knowledge_search", "config": {"knowledge_base_id": "kb_001", "top_k": 5}},
    {"type": "web_search", "config": {}}
  ]
}
```

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "agent_001",
    "name": "客服助手 v2",
    "system_prompt": "你是一个高级客服专家...",
    "tools": [...],
    "updated_at": "2026-01-06T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X PUT 'http://localhost:8000/api/v1/agents/agent_001' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{"name": "客服助手 v2", "system_prompt": "你是一个高级客服专家..."}'
```

---

#### DELETE /api/v1/agents/{id} — 删除 Agent

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/agents/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/agents/agent_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/agents/{id}/chat — Agent 对话（SSE 流式）

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/agents/{id}/chat` |
| 认证 | JWT |
| 响应类型 | `text/event-stream` (SSE) |

**Request Body:**

```json
{
  "message": "我的订单什么时候发货？",
  "conversation_id": "conv_001"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 用户消息 |
| `conversation_id` | string | 否 | 对话 ID，不传则创建新对话 |

**SSE 事件流:**

```
event: thought
data: {"content": "用户在询问订单发货时间，我需要先查询知识库..."}

event: tool_call
data: {"tool": "knowledge_search", "input": {"query": "订单发货时间"}}

event: tool_result
data: {"tool": "knowledge_search", "output": "标准订单将在付款后48小时内发货..."}

event: message_delta
data: {"content": "根据"}

event: message_delta
data: {"content": "我们的政策，"}

event: message_delta
data: {"content": "标准订单将在付款后48小时内发货。"}

event: message_done
data: {"conversation_id": "conv_001", "message_id": "msg_003", "total_tokens": 320, "latency_ms": 2500}

event: done
data: {}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/agents/agent_001/chat' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{"message": "我的订单什么时候发货？", "conversation_id": "conv_001"}'
```

---

### 2.6 工作流引擎（Workflow Engine）

#### GET /api/v1/workflows — 工作流列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/workflows` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `keyword` | string | 否 | 搜索关键词 |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "wf_001",
        "name": "智能摘要工作流",
        "description": "输入长文本，自动生成结构化摘要",
        "node_count": 5,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-05T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/workflows?page=1' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/workflows — 创建工作流

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/workflows` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "智能摘要工作流",
  "description": "输入长文本，自动生成结构化摘要",
  "graph": {
    "nodes": [
      {
        "id": "node_start",
        "type": "start",
        "position": {"x": 100, "y": 200},
        "data": {
          "inputs": [
            {"name": "text", "type": "string", "required": true}
          ]
        }
      },
      {
        "id": "node_llm_1",
        "type": "llm",
        "position": {"x": 400, "y": 200},
        "data": {
          "model_config_id": "mc_001",
          "system_prompt": "你是一个摘要专家",
          "user_prompt_template": "请为以下文本生成摘要：\n\n{{text}}"
        }
      },
      {
        "id": "node_end",
        "type": "end",
        "position": {"x": 700, "y": 200},
        "data": {
          "output_variable": "node_llm_1.output"
        }
      }
    ],
    "edges": [
      {"id": "e1", "source": "node_start", "target": "node_llm_1"},
      {"id": "e2", "source": "node_llm_1", "target": "node_end"}
    ]
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 工作流名称 |
| `description` | string | 否 | 描述 |
| `graph` | object | 是 | React Flow 图定义 |
| `graph.nodes` | array | 是 | 节点列表 |
| `graph.edges` | array | 是 | 边列表 |

节点类型 (`type`) 支持：`start`、`end`、`llm`、`knowledge_search`、`condition`、`code`、`template`

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "wf_001",
    "name": "智能摘要工作流",
    "description": "输入长文本，自动生成结构化摘要",
    "graph": { ... },
    "node_count": 3,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/workflows' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "智能摘要工作流",
    "description": "输入长文本，自动生成结构化摘要",
    "graph": {
      "nodes": [
        {"id": "node_start", "type": "start", "position": {"x": 100, "y": 200}, "data": {"inputs": [{"name": "text", "type": "string", "required": true}]}},
        {"id": "node_llm_1", "type": "llm", "position": {"x": 400, "y": 200}, "data": {"model_config_id": "mc_001", "system_prompt": "你是一个摘要专家", "user_prompt_template": "请为以下文本生成摘要：\n\n{{text}}"}},
        {"id": "node_end", "type": "end", "position": {"x": 700, "y": 200}, "data": {"output_variable": "node_llm_1.output"}}
      ],
      "edges": [
        {"id": "e1", "source": "node_start", "target": "node_llm_1"},
        {"id": "e2", "source": "node_llm_1", "target": "node_end"}
      ]
    }
  }'
```

---

#### GET /api/v1/workflows/{id} — 工作流详情

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/workflows/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "wf_001",
    "name": "智能摘要工作流",
    "description": "输入长文本，自动生成结构化摘要",
    "graph": {
      "nodes": [...],
      "edges": [...]
    },
    "node_count": 3,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-05T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/workflows/wf_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### PUT /api/v1/workflows/{id} — 更新工作流

保存 React Flow 的完整状态（节点位置、连线、配置）。

| 项目 | 说明 |
|------|------|
| Method | PUT |
| URL | `/api/v1/workflows/{id}` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "智能摘要工作流 v2",
  "graph": {
    "nodes": [...],
    "edges": [...]
  }
}
```

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "wf_001",
    "name": "智能摘要工作流 v2",
    "graph": { ... },
    "node_count": 5,
    "updated_at": "2026-01-06T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X PUT 'http://localhost:8000/api/v1/workflows/wf_001' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{"name": "智能摘要工作流 v2", "graph": {"nodes": [...], "edges": [...]}}'
```

---

#### DELETE /api/v1/workflows/{id} — 删除工作流

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/workflows/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/workflows/wf_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/workflows/{id}/run — 执行工作流（SSE 流式）

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/workflows/{id}/run` |
| 认证 | JWT |
| 响应类型 | `text/event-stream` (SSE) |

**Request Body:**

```json
{
  "inputs": {
    "text": "一篇很长的文章内容..."
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `inputs` | object | 是 | Start 节点定义的输入变量键值对 |

**SSE 事件流:**

```
event: node_started
data: {"node_id": "node_start", "node_type": "start"}

event: node_completed
data: {"node_id": "node_start", "node_type": "start", "outputs": {"text": "一篇很长的文章内容..."}}

event: node_started
data: {"node_id": "node_llm_1", "node_type": "llm"}

event: node_completed
data: {"node_id": "node_llm_1", "node_type": "llm", "outputs": {"output": "摘要内容..."}, "tokens": 250, "latency_ms": 1800}

event: node_started
data: {"node_id": "node_end", "node_type": "end"}

event: node_completed
data: {"node_id": "node_end", "node_type": "end", "outputs": {"result": "摘要内容..."}}

event: done
data: {"total_tokens": 250, "total_latency_ms": 2100}
```

节点执行出错时：

```
event: node_error
data: {"node_id": "node_llm_1", "node_type": "llm", "error": "Model API request timeout"}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/workflows/wf_001/run' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{"inputs": {"text": "一篇很长的文章内容..."}}'
```

---

### 2.7 应用管理（App Management）

#### GET /api/v1/apps — 应用列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/apps` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `app_type` | string | 否 | 应用类型筛选：`chatbot` / `completion` / `workflow` |
| `keyword` | string | 否 | 搜索关键词 |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "app_001",
        "name": "智能客服",
        "description": "基于知识库的客服聊天机器人",
        "app_type": "chatbot",
        "is_published": true,
        "agent_id": "agent_001",
        "workflow_id": null,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-05T00:00:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
    "total_pages": 1
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/apps?app_type=chatbot&page=1' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/apps — 创建应用

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/apps` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "智能客服",
  "description": "基于知识库的客服聊天机器人",
  "app_type": "chatbot",
  "agent_id": "agent_001"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 应用名称 |
| `description` | string | 否 | 描述 |
| `app_type` | string | 是 | 应用类型：`chatbot` / `completion` / `workflow` |
| `agent_id` | string | 条件 | 关联 Agent ID（`chatbot` 和 `completion` 类型必填） |
| `workflow_id` | string | 条件 | 关联工作流 ID（`workflow` 类型必填） |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "app_001",
    "name": "智能客服",
    "description": "基于知识库的客服聊天机器人",
    "app_type": "chatbot",
    "is_published": false,
    "agent_id": "agent_001",
    "workflow_id": null,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/apps' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "智能客服",
    "description": "基于知识库的客服聊天机器人",
    "app_type": "chatbot",
    "agent_id": "agent_001"
  }'
```

---

#### GET /api/v1/apps/{id} — 应用详情

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/apps/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "app_001",
    "name": "智能客服",
    "description": "基于知识库的客服聊天机器人",
    "app_type": "chatbot",
    "is_published": true,
    "agent_id": "agent_001",
    "workflow_id": null,
    "api_key_count": 2,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-05T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/apps/app_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### PUT /api/v1/apps/{id} — 更新应用

| 项目 | 说明 |
|------|------|
| Method | PUT |
| URL | `/api/v1/apps/{id}` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "智能客服 v2",
  "description": "升级版客服机器人",
  "agent_id": "agent_002"
}
```

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "app_001",
    "name": "智能客服 v2",
    "description": "升级版客服机器人",
    "agent_id": "agent_002",
    "updated_at": "2026-01-06T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X PUT 'http://localhost:8000/api/v1/apps/app_001' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{"name": "智能客服 v2", "agent_id": "agent_002"}'
```

---

#### DELETE /api/v1/apps/{id} — 删除应用

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/apps/{id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/apps/app_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/apps/{id}/publish — 发布应用

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/apps/{id}/publish` |
| 认证 | JWT |

**Request Body:** 无

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "app_001",
    "is_published": true,
    "published_at": "2026-01-05T00:00:00Z"
  }
}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/apps/app_001/publish' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### POST /api/v1/apps/{id}/api-keys — 生成 API Key

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/v1/apps/{id}/api-keys` |
| 认证 | JWT |

**Request Body:**

```json
{
  "name": "Production Key"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 否 | API Key 备注名称 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "key_001",
    "name": "Production Key",
    "api_key": "sk-mini-a1b2c3d4e5f6g7h8i9j0xxxxxxxxxxxx",
    "created_at": "2026-01-05T00:00:00Z"
  }
}
```

> **重要提示：** 完整的 `api_key` 仅在创建时返回一次，后续查询只能看到脱敏版本。请提醒用户妥善保存。

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/v1/apps/app_001/api-keys' \
  -H 'Authorization: Bearer <jwt_token>' \
  -H 'Content-Type: application/json' \
  -d '{"name": "Production Key"}'
```

---

#### GET /api/v1/apps/{id}/api-keys — API Key 列表

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/apps/{id}/api-keys` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": "key_001",
      "name": "Production Key",
      "api_key_preview": "sk-mini-a1b2...xxxx",
      "created_at": "2026-01-05T00:00:00Z",
      "last_used_at": "2026-01-10T12:00:00Z"
    },
    {
      "id": "key_002",
      "name": "Development Key",
      "api_key_preview": "sk-mini-x9y8...zzzz",
      "created_at": "2026-01-06T00:00:00Z",
      "last_used_at": null
    }
  ]
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/apps/app_001/api-keys' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### DELETE /api/v1/apps/{id}/api-keys/{key_id} — 吊销 API Key

| 项目 | 说明 |
|------|------|
| Method | DELETE |
| URL | `/api/v1/apps/{id}/api-keys/{key_id}` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

> 吊销后，使用该 Key 的请求将立即返回 `401 Unauthorized`。

**curl 示例:**

```bash
curl -X DELETE 'http://localhost:8000/api/v1/apps/app_001/api-keys/key_001' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

### 2.8 监控（Monitoring）

#### GET /api/v1/monitoring/conversations — 对话日志

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/monitoring/conversations` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `app_id` | string | 否 | 按应用 ID 筛选 |
| `start_date` | string | 否 | 起始日期（ISO 8601 格式） |
| `end_date` | string | 否 | 截止日期（ISO 8601 格式） |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "conv_001",
        "app_id": "app_001",
        "app_name": "智能客服",
        "message_count": 6,
        "total_tokens": 1250,
        "created_at": "2026-01-10T10:00:00Z",
        "last_message_at": "2026-01-10T10:05:00Z"
      }
    ],
    "total": 100,
    "page": 1,
    "page_size": 20,
    "total_pages": 5
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/monitoring/conversations?app_id=app_001&start_date=2026-01-01&end_date=2026-01-31&page=1' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### GET /api/v1/monitoring/conversations/{id}/messages — 对话消息详情

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/monitoring/conversations/{id}/messages` |
| 认证 | JWT |

**Response Body:**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "conversation_id": "conv_001",
    "messages": [
      {
        "id": "msg_001",
        "role": "user",
        "content": "我的订单什么时候发货？",
        "created_at": "2026-01-10T10:00:00Z"
      },
      {
        "id": "msg_002",
        "role": "assistant",
        "content": "根据我们的政策，标准订单将在付款后48小时内发货。",
        "metadata": {
          "model": "gpt-4o",
          "tokens": {"prompt": 120, "completion": 45, "total": 165},
          "latency_ms": 1200,
          "tool_calls": [
            {"tool": "knowledge_search", "query": "订单发货时间"}
          ]
        },
        "created_at": "2026-01-10T10:00:03Z"
      }
    ]
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/monitoring/conversations/conv_001/messages' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### GET /api/v1/monitoring/stats/tokens — Token 用量统计

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/monitoring/stats/tokens` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `group_by` | string | 否 | 聚合维度：`model` / `app` / `day`，默认 `day` |
| `start_date` | string | 否 | 起始日期 |
| `end_date` | string | 否 | 截止日期 |
| `app_id` | string | 否 | 按应用筛选 |

**Response Body (group_by=day):**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "stats": [
      {
        "date": "2026-01-10",
        "prompt_tokens": 15000,
        "completion_tokens": 8000,
        "total_tokens": 23000,
        "request_count": 120
      },
      {
        "date": "2026-01-09",
        "prompt_tokens": 12000,
        "completion_tokens": 6500,
        "total_tokens": 18500,
        "request_count": 95
      }
    ],
    "summary": {
      "total_prompt_tokens": 27000,
      "total_completion_tokens": 14500,
      "total_tokens": 41500,
      "total_requests": 215
    }
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/monitoring/stats/tokens?group_by=day&start_date=2026-01-01&end_date=2026-01-31' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

#### GET /api/v1/monitoring/stats/costs — 成本统计

| 项目 | 说明 |
|------|------|
| Method | GET |
| URL | `/api/v1/monitoring/stats/costs` |
| 认证 | JWT |

**Query Parameters:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `group_by` | string | 否 | 聚合维度：`model` / `app` / `day`，默认 `day` |
| `start_date` | string | 否 | 起始日期 |
| `end_date` | string | 否 | 截止日期 |

**Response Body (group_by=model):**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "stats": [
      {
        "model": "gpt-4o",
        "prompt_tokens": 50000,
        "completion_tokens": 25000,
        "total_tokens": 75000,
        "cost_usd": 1.875
      },
      {
        "model": "gpt-4o-mini",
        "prompt_tokens": 100000,
        "completion_tokens": 50000,
        "total_tokens": 150000,
        "cost_usd": 0.075
      }
    ],
    "summary": {
      "total_cost_usd": 1.95
    }
  }
}
```

**curl 示例:**

```bash
curl -X GET 'http://localhost:8000/api/v1/monitoring/stats/costs?group_by=model&start_date=2026-01-01&end_date=2026-01-31' \
  -H 'Authorization: Bearer <jwt_token>'
```

---

### 2.9 Gateway API（外部调用）

Gateway API 供外部应用通过 API Key 调用已发布的应用。

#### POST /api/gateway/{app_id}/chat — Chatbot 对话（SSE）

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/gateway/{app_id}/chat` |
| 认证 | App API Key |
| 响应类型 | `text/event-stream` (SSE) |

**Request Body:**

```json
{
  "message": "你好，请问如何重置密码？",
  "conversation_id": "ext_conv_001",
  "stream": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 用户消息 |
| `conversation_id` | string | 否 | 对话 ID，不传则创建新对话 |
| `stream` | bool | 否 | 是否流式返回，默认 `true` |

**SSE 事件流 (stream=true):**

```
event: message_delta
data: {"content": "您可以"}

event: message_delta
data: {"content": "通过以下步骤重置密码："}

event: message_done
data: {"conversation_id": "ext_conv_001", "message_id": "msg_010", "total_tokens": 180}

event: done
data: {}
```

**非流式响应 (stream=false):**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "conversation_id": "ext_conv_001",
    "message_id": "msg_010",
    "content": "您可以通过以下步骤重置密码：...",
    "total_tokens": 180
  }
}
```

**curl 示例:**

```bash
# 流式
curl -X POST 'http://localhost:8000/api/gateway/app_001/chat' \
  -H 'Authorization: Bearer sk-mini-a1b2c3d4e5f6g7h8i9j0xxxxxxxxxxxx' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{"message": "你好，请问如何重置密码？", "stream": true}'

# 非流式
curl -X POST 'http://localhost:8000/api/gateway/app_001/chat' \
  -H 'Authorization: Bearer sk-mini-a1b2c3d4e5f6g7h8i9j0xxxxxxxxxxxx' \
  -H 'Content-Type: application/json' \
  -d '{"message": "你好，请问如何重置密码？", "stream": false}'
```

---

#### POST /api/gateway/{app_id}/completion — 文本生成（SSE）

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/gateway/{app_id}/completion` |
| 认证 | App API Key |
| 响应类型 | `text/event-stream` (SSE) |

**Request Body:**

```json
{
  "inputs": {
    "text": "人工智能的发展趋势"
  },
  "stream": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `inputs` | object | 是 | Prompt 模板变量键值对 |
| `stream` | bool | 否 | 是否流式返回，默认 `true` |

**SSE 事件流:**

```
event: message_delta
data: {"content": "人工智能的发展趋势主要体现在"}

event: message_delta
data: {"content": "以下几个方面：..."}

event: message_done
data: {"message_id": "msg_020", "total_tokens": 350}

event: done
data: {}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/gateway/app_001/completion' \
  -H 'Authorization: Bearer sk-mini-a1b2c3d4e5f6g7h8i9j0xxxxxxxxxxxx' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{"inputs": {"text": "人工智能的发展趋势"}, "stream": true}'
```

---

#### POST /api/gateway/{app_id}/workflow/run — 执行工作流（SSE）

| 项目 | 说明 |
|------|------|
| Method | POST |
| URL | `/api/gateway/{app_id}/workflow/run` |
| 认证 | App API Key |
| 响应类型 | `text/event-stream` (SSE) |

**Request Body:**

```json
{
  "inputs": {
    "text": "一篇很长的文章..."
  },
  "stream": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `inputs` | object | 是 | 工作流 Start 节点输入变量 |
| `stream` | bool | 否 | 是否流式返回，默认 `true` |

**SSE 事件流:**

```
event: node_started
data: {"node_id": "node_start", "node_type": "start"}

event: node_completed
data: {"node_id": "node_start", "node_type": "start"}

event: node_started
data: {"node_id": "node_llm_1", "node_type": "llm"}

event: node_completed
data: {"node_id": "node_llm_1", "node_type": "llm", "outputs": {"output": "摘要内容..."}}

event: done
data: {"outputs": {"result": "摘要内容..."}, "total_tokens": 400}
```

**curl 示例:**

```bash
curl -X POST 'http://localhost:8000/api/gateway/app_001/workflow/run' \
  -H 'Authorization: Bearer sk-mini-a1b2c3d4e5f6g7h8i9j0xxxxxxxxxxxx' \
  -H 'Content-Type: application/json' \
  -N \
  -d '{"inputs": {"text": "一篇很长的文章..."}, "stream": true}'
```

---

## 3. SSE 流式协议

### 3.1 协议说明

Mini-Dify 使用 Server-Sent Events (SSE) 协议实现流式响应。客户端通过 HTTP 请求建立长连接，服务端逐步推送事件。

**HTTP 响应头：**

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

### 3.2 事件格式

每个 SSE 事件由 `event` 和 `data` 两行组成，事件之间用空行分隔：

```
event: <event_type>
data: <json_payload>

```

### 3.3 事件类型

#### Agent 对话事件

| 事件类型 | 说明 | data 结构 |
|----------|------|-----------|
| `thought` | Agent 思考过程 | `{"content": "思考内容..."}` |
| `tool_call` | Agent 调用工具 | `{"tool": "工具名", "input": {...}}` |
| `tool_result` | 工具返回结果 | `{"tool": "工具名", "output": "结果内容"}` |
| `message_delta` | 消息内容增量 | `{"content": "增量文本"}` |
| `message_done` | 消息生成完成 | `{"conversation_id": "...", "message_id": "...", "total_tokens": 320, "latency_ms": 2500}` |
| `done` | 整个请求完成 | `{}` |
| `error` | 发生错误 | `{"code": 50001, "message": "错误描述"}` |

#### 工作流执行事件

| 事件类型 | 说明 | data 结构 |
|----------|------|-----------|
| `node_started` | 节点开始执行 | `{"node_id": "...", "node_type": "llm"}` |
| `node_completed` | 节点执行完成 | `{"node_id": "...", "node_type": "llm", "outputs": {...}, "tokens": 250, "latency_ms": 1800}` |
| `node_error` | 节点执行出错 | `{"node_id": "...", "node_type": "llm", "error": "错误描述"}` |
| `done` | 工作流执行完成 | `{"total_tokens": 500, "total_latency_ms": 3000}` |
| `error` | 全局错误 | `{"code": 50001, "message": "错误描述"}` |

#### Prompt 测试事件

| 事件类型 | 说明 | data 结构 |
|----------|------|-----------|
| `message_delta` | 模型输出增量 | `{"model_config_id": "...", "content": "增量文本"}` |
| `message_done` | 单个模型生成完成 | `{"model_config_id": "...", "total_tokens": 45, "latency_ms": 1200}` |
| `done` | 所有模型生成完成 | `{}` |
| `error` | 发生错误 | `{"model_config_id": "...", "code": 50001, "message": "错误描述"}` |

### 3.4 客户端处理示例

**JavaScript (EventSource):**

```javascript
const eventSource = new EventSource('/api/v1/agents/agent_001/chat', {
  headers: { 'Authorization': 'Bearer <jwt_token>' }
});

eventSource.addEventListener('thought', (e) => {
  const data = JSON.parse(e.data);
  console.log('Agent 思考:', data.content);
});

eventSource.addEventListener('message_delta', (e) => {
  const data = JSON.parse(e.data);
  appendToChat(data.content);
});

eventSource.addEventListener('done', () => {
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  console.error('错误:', data.message);
  eventSource.close();
});
```

**Python (httpx-sse):**

```python
import httpx
from httpx_sse import connect_sse

with httpx.Client() as client:
    with connect_sse(
        client, "POST",
        "http://localhost:8000/api/v1/agents/agent_001/chat",
        json={"message": "你好"},
        headers={"Authorization": "Bearer <jwt_token>"}
    ) as event_source:
        for sse in event_source.iter_sse():
            if sse.event == "message_delta":
                data = json.loads(sse.data)
                print(data["content"], end="", flush=True)
            elif sse.event == "done":
                break
```

---

## 4. 错误码

### 4.1 通用错误码

| 错误码 | HTTP 状态码 | 说明 |
|--------|-------------|------|
| 40000 | 400 | 请求参数错误 |
| 40001 | 401 | 未认证 / JWT 过期 |
| 40002 | 401 | API Key 无效或已吊销 |
| 40003 | 403 | 无权限访问该资源 |
| 40004 | 404 | 资源不存在 |
| 40005 | 409 | 资源冲突（如名称重复） |
| 40006 | 413 | 上传文件过大 |
| 40007 | 422 | 请求参数校验失败 |
| 40008 | 429 | 请求频率超限 |

### 4.2 业务错误码

| 错误码 | HTTP 状态码 | 说明 |
|--------|-------------|------|
| 41001 | 400 | 供应商 API Key 无效 |
| 41002 | 400 | 供应商连接超时 |
| 41003 | 400 | 不支持的供应商类型 |
| 42001 | 400 | Prompt 模板变量缺失 |
| 42002 | 400 | Prompt 模板语法错误 |
| 43001 | 400 | 不支持的文档格式 |
| 43002 | 400 | 文档解析失败 |
| 43003 | 400 | Embedding 处理失败 |
| 43004 | 400 | 知识库检索失败 |
| 44001 | 400 | Agent 工具配置无效 |
| 44002 | 400 | Agent 推理循环超出限制 |
| 45001 | 400 | 工作流图结构无效（存在环或未连接节点） |
| 45002 | 400 | 工作流节点配置错误 |
| 45003 | 400 | 工作流执行超时 |
| 46001 | 400 | 应用未发布，无法通过 Gateway 调用 |
| 46002 | 400 | 应用类型与调用端点不匹配 |

### 4.3 服务端错误码

| 错误码 | HTTP 状态码 | 说明 |
|--------|-------------|------|
| 50001 | 500 | 服务器内部错误 |
| 50002 | 500 | LLM 调用失败 |
| 50003 | 500 | 向量数据库异常 |
| 50004 | 503 | 服务暂不可用 |

---

## 5. 附录

### 5.1 通用请求头

| Header | 值 | 说明 |
|--------|---|------|
| `Content-Type` | `application/json` | 请求体格式（上传文件除外） |
| `Authorization` | `Bearer <token>` | 认证令牌（JWT 或 API Key） |
| `Accept` | `text/event-stream` | SSE 流式端点建议携带 |

### 5.2 ID 格式

所有资源 ID 使用 UUID v4 格式，示例中使用简短 ID 仅为便于阅读。

### 5.3 时间格式

所有时间字段统一使用 ISO 8601 格式，时区为 UTC：`2026-01-01T00:00:00Z`

### 5.4 文件上传限制

| 文件类型 | 最大大小 |
|----------|----------|
| `.txt` | 10 MB |
| `.md` | 10 MB |
| `.pdf` | 50 MB |
| `.docx` | 50 MB |
