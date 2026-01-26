# DocuMind AI - API 接口设计文档

> 版本: v1.0  
> 更新日期: 2026-01-26  
> 规范: OpenAPI 3.0

---

## 1. API 概览

### 1.1 基础信息

| 项目     | 值                             |
| -------- | ------------------------------ |
| Base URL | `http://localhost:8000/api/v1` |
| 协议     | HTTP/HTTPS                     |
| 数据格式 | JSON                           |
| 认证     | 无（MVP 版本）                 |

### 1.2 接口分类

| 模块   | 前缀         | 说明           |
| ------ | ------------ | -------------- |
| 知识库 | `/kb`        | 知识库 CRUD    |
| 文档   | `/documents` | 文档上传与管理 |
| 问答   | `/chat`      | 对话问答       |
| 系统   | `/system`    | 健康检查等     |

### 1.3 通用响应格式

```json
// 成功响应
{
  "code": 0,
  "message": "success",
  "data": { ... }
}

// 错误响应
{
  "code": 40001,
  "message": "文档格式不支持",
  "data": null
}
```

### 1.4 错误码定义

| 错误码 | 说明           |
| ------ | -------------- |
| 0      | 成功           |
| 40001  | 请求参数错误   |
| 40002  | 文件格式不支持 |
| 40003  | 文件大小超限   |
| 40004  | 知识库不存在   |
| 40005  | 文档不存在     |
| 50001  | 服务器内部错误 |
| 50002  | 模型推理错误   |

---

## 2. 知识库接口

### 2.1 创建知识库

**POST** `/api/v1/kb`

创建一个新的知识库。

**请求体**:

```json
{
  "name": "技术文档库",
  "description": "存放公司技术相关文档"
}
```

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "kb_a1b2c3d4",
    "name": "技术文档库",
    "description": "存放公司技术相关文档",
    "document_count": 0,
    "created_at": "2026-01-26T10:30:00Z",
    "updated_at": "2026-01-26T10:30:00Z"
  }
}
```

---

### 2.2 获取知识库列表

**GET** `/api/v1/kb`

获取所有知识库列表。

**查询参数**:

| 参数      | 类型 | 必填 | 说明              |
| --------- | ---- | ---- | ----------------- |
| page      | int  | 否   | 页码，默认 1      |
| page_size | int  | 否   | 每页数量，默认 20 |

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "kb_a1b2c3d4",
        "name": "技术文档库",
        "description": "存放公司技术相关文档",
        "document_count": 5,
        "total_size": 10485760,
        "created_at": "2026-01-26T10:30:00Z",
        "updated_at": "2026-01-26T15:45:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20
  }
}
```

---

### 2.3 获取知识库详情

**GET** `/api/v1/kb/{kb_id}`

获取指定知识库的详细信息。

**路径参数**:

| 参数  | 类型   | 说明      |
| ----- | ------ | --------- |
| kb_id | string | 知识库 ID |

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "kb_a1b2c3d4",
    "name": "技术文档库",
    "description": "存放公司技术相关文档",
    "document_count": 5,
    "chunk_count": 128,
    "total_size": 10485760,
    "created_at": "2026-01-26T10:30:00Z",
    "updated_at": "2026-01-26T15:45:00Z"
  }
}
```

---

### 2.4 更新知识库

**PUT** `/api/v1/kb/{kb_id}`

更新知识库信息。

**请求体**:

```json
{
  "name": "技术文档库（更新）",
  "description": "更新后的描述"
}
```

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "kb_a1b2c3d4",
    "name": "技术文档库（更新）",
    "description": "更新后的描述",
    "updated_at": "2026-01-26T16:00:00Z"
  }
}
```

---

### 2.5 删除知识库

**DELETE** `/api/v1/kb/{kb_id}`

删除知识库及其所有关联文档和向量索引。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "deleted_documents": 5,
    "deleted_chunks": 128
  }
}
```

---

## 3. 文档接口

### 3.1 上传文档

**POST** `/api/v1/documents/upload`

上传文档到指定知识库。

**请求头**:

| Header       | 值                  |
| ------------ | ------------------- |
| Content-Type | multipart/form-data |

**请求参数**:

| 参数  | 类型   | 必填 | 说明      |
| ----- | ------ | ---- | --------- |
| kb_id | string | 是   | 知识库 ID |
| file  | file   | 是   | 文档文件  |

**支持格式**: PDF, DOCX, TXT, MD

**大小限制**: 50MB

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "doc_e5f6g7h8",
    "kb_id": "kb_a1b2c3d4",
    "filename": "技术手册.pdf",
    "file_type": "pdf",
    "file_size": 2097152,
    "status": "pending",
    "created_at": "2026-01-26T11:00:00Z"
  }
}
```

**文档状态**:

| 状态      | 说明       |
| --------- | ---------- |
| pending   | 等待处理   |
| parsing   | 正在解析   |
| chunking  | 正在分块   |
| embedding | 正在向量化 |
| completed | 处理完成   |
| failed    | 处理失败   |

---

### 3.2 批量上传文档

**POST** `/api/v1/documents/upload/batch`

批量上传多个文档。

**请求参数**:

| 参数  | 类型   | 必填 | 说明                     |
| ----- | ------ | ---- | ------------------------ |
| kb_id | string | 是   | 知识库 ID                |
| files | file[] | 是   | 文档文件列表（最多10个） |

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "uploaded": 3,
    "documents": [
      { "id": "doc_001", "filename": "文档1.pdf", "status": "pending" },
      { "id": "doc_002", "filename": "文档2.docx", "status": "pending" },
      { "id": "doc_003", "filename": "文档3.md", "status": "pending" }
    ]
  }
}
```

---

### 3.3 获取文档列表

**GET** `/api/v1/documents`

获取知识库下的文档列表。

**查询参数**:

| 参数      | 类型   | 必填 | 说明       |
| --------- | ------ | ---- | ---------- |
| kb_id     | string | 是   | 知识库 ID  |
| status    | string | 否   | 按状态筛选 |
| page      | int    | 否   | 页码       |
| page_size | int    | 否   | 每页数量   |

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "doc_e5f6g7h8",
        "filename": "技术手册.pdf",
        "file_type": "pdf",
        "file_size": 2097152,
        "chunk_count": 45,
        "status": "completed",
        "created_at": "2026-01-26T11:00:00Z"
      }
    ],
    "total": 5,
    "page": 1,
    "page_size": 20
  }
}
```

---

### 3.4 获取文档详情

**GET** `/api/v1/documents/{doc_id}`

获取文档详细信息。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "doc_e5f6g7h8",
    "kb_id": "kb_a1b2c3d4",
    "filename": "技术手册.pdf",
    "file_type": "pdf",
    "file_size": 2097152,
    "chunk_count": 45,
    "status": "completed",
    "error_message": null,
    "created_at": "2026-01-26T11:00:00Z",
    "processed_at": "2026-01-26T11:00:30Z"
  }
}
```

---

### 3.5 获取文档处理状态

**GET** `/api/v1/documents/{doc_id}/status`

轮询获取文档处理进度。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "doc_e5f6g7h8",
    "status": "embedding",
    "progress": 75,
    "current_step": "正在生成向量嵌入...",
    "steps": [
      { "name": "parsing", "status": "completed" },
      { "name": "chunking", "status": "completed" },
      { "name": "embedding", "status": "in_progress" }
    ]
  }
}
```

---

### 3.6 获取文档分块

**GET** `/api/v1/documents/{doc_id}/chunks`

获取文档的分块内容（用于调试和预览）。

**查询参数**:

| 参数      | 类型 | 说明               |
| --------- | ---- | ------------------ |
| page      | int  | 页码               |
| page_size | int  | 每页数量（最大50） |

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "chunk_001",
        "index": 0,
        "content": "这是第一个分块的内容...",
        "metadata": {
          "page": 1,
          "position": "top"
        }
      },
      {
        "id": "chunk_002",
        "index": 1,
        "content": "这是第二个分块的内容...",
        "metadata": {
          "page": 1,
          "position": "middle"
        }
      }
    ],
    "total": 45,
    "page": 1,
    "page_size": 20
  }
}
```

---

### 3.7 删除文档

**DELETE** `/api/v1/documents/{doc_id}`

删除文档及其向量索引。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "deleted_chunks": 45
  }
}
```

---

## 4. 问答接口

### 4.1 发起对话（流式）

**POST** `/api/v1/chat/stream`

发起问答请求，流式返回结果。

**请求体**:

```json
{
  "kb_id": "kb_a1b2c3d4",
  "conversation_id": "conv_123456", // 可选，用于多轮对话
  "query": "这个系统支持哪些文件格式？",
  "options": {
    "top_k": 5, // 检索数量
    "temperature": 0.7, // 生成温度
    "max_tokens": 512 // 最大生成长度
  }
}
```

**响应**（SSE 流式）:

```
event: chunk
data: {"type": "chunk", "content": "根据"}

event: chunk
data: {"type": "chunk", "content": "文档"}

event: chunk
data: {"type": "chunk", "content": "内容"}

...

event: sources
data: {"type": "sources", "sources": [
  {"doc_id": "doc_001", "filename": "PRD.md", "chunk_index": 5, "content": "支持PDF、DOCX、TXT、MD格式...", "score": 0.89},
  {"doc_id": "doc_002", "filename": "用户手册.pdf", "chunk_index": 12, "content": "文档上传支持多种格式...", "score": 0.76}
]}

event: done
data: {"type": "done", "message_id": "msg_abc123", "conversation_id": "conv_123456"}
```

---

### 4.2 发起对话（非流式）

**POST** `/api/v1/chat`

发起问答请求，一次性返回完整结果。

**请求体**:

```json
{
  "kb_id": "kb_a1b2c3d4",
  "conversation_id": "conv_123456",
  "query": "这个系统支持哪些文件格式？"
}
```

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "message_id": "msg_abc123",
    "conversation_id": "conv_123456",
    "answer": "根据文档内容，系统支持以下文件格式：\n\n1. **PDF** - 便携式文档格式\n2. **DOCX** - Microsoft Word 文档\n3. **TXT** - 纯文本文件\n4. **MD** - Markdown 文件\n\n[来源: PRD.md, 段落5]",
    "sources": [
      {
        "doc_id": "doc_001",
        "filename": "PRD.md",
        "chunk_index": 5,
        "content": "格式支持 | PDF, DOCX, TXT, MD | P0",
        "score": 0.89
      },
      {
        "doc_id": "doc_002",
        "filename": "用户手册.pdf",
        "chunk_index": 12,
        "content": "系统支持常见的文档格式，包括PDF文件、Word文档...",
        "score": 0.76
      }
    ],
    "usage": {
      "prompt_tokens": 856,
      "completion_tokens": 128,
      "total_tokens": 984
    },
    "created_at": "2026-01-26T12:30:00Z"
  }
}
```

---

### 4.3 获取对话历史

**GET** `/api/v1/chat/conversations/{conversation_id}`

获取对话历史记录。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "conversation_id": "conv_123456",
    "kb_id": "kb_a1b2c3d4",
    "title": "关于文件格式的咨询",
    "messages": [
      {
        "id": "msg_001",
        "role": "user",
        "content": "这个系统支持哪些文件格式？",
        "created_at": "2026-01-26T12:30:00Z"
      },
      {
        "id": "msg_002",
        "role": "assistant",
        "content": "根据文档内容，系统支持以下文件格式...",
        "sources": [...],
        "created_at": "2026-01-26T12:30:05Z"
      }
    ],
    "created_at": "2026-01-26T12:30:00Z",
    "updated_at": "2026-01-26T12:30:05Z"
  }
}
```

---

### 4.4 获取对话列表

**GET** `/api/v1/chat/conversations`

获取知识库的对话列表。

**查询参数**:

| 参数      | 类型   | 说明      |
| --------- | ------ | --------- |
| kb_id     | string | 知识库 ID |
| page      | int    | 页码      |
| page_size | int    | 每页数量  |

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "conversation_id": "conv_123456",
        "title": "关于文件格式的咨询",
        "message_count": 4,
        "created_at": "2026-01-26T12:30:00Z",
        "updated_at": "2026-01-26T12:35:00Z"
      }
    ],
    "total": 10,
    "page": 1,
    "page_size": 20
  }
}
```

---

### 4.5 删除对话

**DELETE** `/api/v1/chat/conversations/{conversation_id}`

删除对话记录。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "deleted_messages": 6
  }
}
```

---

### 4.6 提交反馈

**POST** `/api/v1/chat/feedback`

对回答提交反馈。

**请求体**:

```json
{
  "message_id": "msg_abc123",
  "rating": "good", // good | bad
  "comment": "回答准确，引用清晰"
}
```

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "feedback_id": "fb_001"
  }
}
```

---

## 5. 系统接口

### 5.1 健康检查

**GET** `/api/v1/system/health`

检查系统健康状态。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "components": {
      "database": "healthy",
      "vector_store": "healthy",
      "embedding_model": "healthy",
      "llm": "healthy"
    },
    "uptime": 86400
  }
}
```

---

### 5.2 获取系统信息

**GET** `/api/v1/system/info`

获取系统配置信息。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "version": "1.0.0",
    "models": {
      "embedding": "BAAI/bge-large-zh-v1.5",
      "llm": "Qwen/Qwen2.5-7B-Instruct"
    },
    "limits": {
      "max_file_size": 52428800,
      "supported_formats": ["pdf", "docx", "txt", "md"],
      "max_batch_upload": 10
    }
  }
}
```

---

### 5.3 获取统计信息

**GET** `/api/v1/system/stats`

获取系统使用统计。

**响应**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "knowledge_bases": 3,
    "documents": 25,
    "chunks": 1280,
    "conversations": 50,
    "messages": 320,
    "storage_used": 104857600
  }
}
```

---

## 6. WebSocket 接口

### 6.1 实时对话

**WS** `/api/v1/ws/chat`

WebSocket 实时对话接口（可选实现）。

**连接**:

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/ws/chat");
```

**发送消息**:

```json
{
  "type": "query",
  "kb_id": "kb_a1b2c3d4",
  "conversation_id": "conv_123456",
  "query": "这个系统支持哪些文件格式？"
}
```

**接收消息**:

```json
{
  "type": "chunk",
  "content": "根据"
}

{
  "type": "sources",
  "sources": [...]
}

{
  "type": "done",
  "message_id": "msg_abc123"
}
```

---

## 附录

### A. 请求示例

#### cURL 示例

```bash
# 创建知识库
curl -X POST http://localhost:8000/api/v1/kb \
  -H "Content-Type: application/json" \
  -d '{"name": "测试知识库", "description": "测试描述"}'

# 上传文档
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "kb_id=kb_a1b2c3d4" \
  -F "file=@/path/to/document.pdf"

# 发起对话
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"kb_id": "kb_a1b2c3d4", "query": "文档主要讲了什么？"}'
```

#### Python 示例

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 创建知识库
response = requests.post(f"{BASE_URL}/kb", json={
    "name": "测试知识库",
    "description": "测试描述"
})
kb_id = response.json()["data"]["id"]

# 上传文档
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/documents/upload",
        data={"kb_id": kb_id},
        files={"file": f}
    )

# 发起对话
response = requests.post(f"{BASE_URL}/chat", json={
    "kb_id": kb_id,
    "query": "文档主要讲了什么？"
})
print(response.json()["data"]["answer"])
```

### B. Pydantic Schema 定义

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    document_count: int
    created_at: datetime
    updated_at: datetime

class DocumentResponse(BaseModel):
    id: str
    kb_id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    created_at: datetime

class ChatRequest(BaseModel):
    kb_id: str
    conversation_id: Optional[str] = None
    query: str = Field(..., min_length=1, max_length=2000)
    options: Optional[dict] = None

class SourceReference(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    content: str
    score: float

class ChatResponse(BaseModel):
    message_id: str
    conversation_id: str
    answer: str
    sources: List[SourceReference]
    created_at: datetime
```
