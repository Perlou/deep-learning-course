---
name: add-api-route
description: 为 DocuMind AI 添加新的 FastAPI 路由接口
---

# 添加 API 路由技能

此技能用于为 DocuMind AI 项目添加新的 FastAPI 路由接口。

## 项目结构

```
projects/capstone-rag-qa-system/src/api/
├── main.py                 # 应用入口，注册路由
├── dependencies.py         # 依赖注入
├── schemas/
│   └── __init__.py         # Pydantic 模型定义
└── routes/
    ├── __init__.py         # 路由导出
    ├── knowledge_base.py   # 知识库路由
    ├── documents.py        # 文档路由
    ├── chat.py             # 问答路由
    └── system.py           # 系统路由
```

## 创建新路由步骤

### 1. 在 `schemas/__init__.py` 添加 Pydantic 模型

```python
# 请求模型
class NewFeatureCreate(BaseModel):
    """创建请求"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None

# 响应模型
class NewFeatureResponse(BaseModel):
    """响应模型"""
    id: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True
```

### 2. 创建路由文件 `routes/new_feature.py`

```python
"""
DocuMind AI - 新功能路由
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.api.schemas import (
    NewFeatureCreate,
    NewFeatureResponse,
    ResponseModel,
)
from src.models import NewFeature
from src.utils import generate_id

router = APIRouter(prefix="/new-feature", tags=["新功能"])


@router.post("", response_model=ResponseModel[NewFeatureResponse])
async def create_feature(
    data: NewFeatureCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """创建新功能"""
    feature = NewFeature(
        id=generate_id("nf"),
        name=data.name,
        description=data.description,
    )
    db.add(feature)
    await db.commit()
    await db.refresh(feature)

    return ResponseModel(data=NewFeatureResponse.model_validate(feature))


@router.get("/{feature_id}", response_model=ResponseModel[NewFeatureResponse])
async def get_feature(
    feature_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """获取功能详情"""
    query = select(NewFeature).where(NewFeature.id == feature_id)
    result = await db.execute(query)
    feature = result.scalar_one_or_none()

    if not feature:
        raise HTTPException(status_code=404, detail="功能不存在")

    return ResponseModel(data=NewFeatureResponse.model_validate(feature))
```

### 3. 在 `routes/__init__.py` 导出路由

```python
from .new_feature import router as new_feature_router

__all__ = [
    # ... 已有路由
    "new_feature_router",
]
```

### 4. 在 `main.py` 注册路由

```python
from src.api.routes import new_feature_router

# 在 create_app() 函数中添加
app.include_router(new_feature_router, prefix="/api/v1")
```

## 路由命名规范

| HTTP 方法 | 路径模式 | 函数名       | 用途     |
| --------- | -------- | ------------ | -------- |
| POST      | `/`      | `create_xxx` | 创建资源 |
| GET       | `/`      | `list_xxx`   | 列表查询 |
| GET       | `/{id}`  | `get_xxx`    | 获取详情 |
| PUT       | `/{id}`  | `update_xxx` | 更新资源 |
| DELETE    | `/{id}`  | `delete_xxx` | 删除资源 |

## 响应格式规范

所有接口使用统一响应格式：

```python
{
    "code": 0,          # 0 表示成功，非 0 表示错误
    "message": "success",
    "data": { ... }     # 实际数据
}
```

## 错误处理

```python
from fastapi import HTTPException

# 资源不存在
raise HTTPException(status_code=404, detail="资源不存在")

# 参数错误
raise HTTPException(status_code=400, detail="参数错误: xxx")

# 权限不足
raise HTTPException(status_code=403, detail="权限不足")
```

## 测试新路由

在 `tests/test_api.py` 添加测试：

```python
class TestNewFeatureAPI:
    """新功能接口测试"""

    def test_create_feature(self):
        response = client.post(
            "/api/v1/new-feature",
            json={"name": "测试功能"}
        )
        assert response.status_code == 200
        assert response.json()["code"] == 0
```
