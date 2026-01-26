---
name: add-react-page
description: 为 DocuMind AI 添加新的 React 页面
---

# 添加 React 页面技能

此技能用于为 DocuMind AI 前端项目添加新的页面。

## 项目结构

```
projects/capstone-rag-qa-system/frontend/src/
├── pages/
│   ├── ChatPage.tsx         # 对话页面
│   ├── DocumentsPage.tsx    # 文档管理页面
│   ├── SettingsPage.tsx     # 设置页面
│   └── NewPage.tsx          # 新页面
├── App.tsx                  # 路由配置
└── main.tsx                 # 入口文件
```

## 创建新页面步骤

### 1. 创建页面组件

```tsx
// src/pages/NewPage.tsx
import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";

export function NewPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [data, setData] = useState<DataType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await api.getData(id);
        setData(response.data);
      } catch (error) {
        console.error("加载失败:", error);
      } finally {
        setLoading(false);
      }
    }

    if (id) {
      fetchData();
    }
  }, [id]);

  if (loading) {
    return <PageSkeleton />;
  }

  return (
    <div className="flex flex-col h-full">
      {/* 页面头部 */}
      <header className="flex items-center justify-between p-4 border-b border-border">
        <h1 className="text-xl font-semibold">页面标题</h1>
        <Button onClick={() => navigate(-1)} variant="outline">
          返回
        </Button>
      </header>

      {/* 页面内容 */}
      <main className="flex-1 overflow-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>内容区域</CardTitle>
            </CardHeader>
            <CardContent>{/* 页面主要内容 */}</CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

function PageSkeleton() {
  return (
    <div className="p-4 space-y-4">
      <Skeleton className="h-8 w-48" />
      <Skeleton className="h-64 w-full" />
    </div>
  );
}
```

### 2. 添加路由

```tsx
// src/App.tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { ChatPage } from "@/pages/ChatPage";
import { DocumentsPage } from "@/pages/DocumentsPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { NewPage } from "@/pages/NewPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<ChatPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="new/:id?" element={<NewPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

### 3. 更新导航

```tsx
// src/components/Sidebar.tsx
const navItems = [
  { path: "/", label: "对话", icon: MessageSquare },
  { path: "/documents", label: "文档", icon: FileText },
  { path: "/new", label: "新页面", icon: Plus }, // 新增
  { path: "/settings", label: "设置", icon: Settings },
];
```

## 页面布局模板

### 列表页面

```tsx
export function ListPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(true);

  return (
    <div className="flex flex-col h-full">
      {/* 工具栏 */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h1 className="text-xl font-semibold">列表标题</h1>
        <Button onClick={handleCreate}>
          <Plus className="w-4 h-4 mr-2" />
          新建
        </Button>
      </div>

      {/* 列表内容 */}
      <ScrollArea className="flex-1">
        <div className="p-4 grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => (
            <ItemCard key={item.id} item={item} />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
```

### 详情页面

```tsx
export function DetailPage() {
  const { id } = useParams();
  const [data, setData] = useState<Detail | null>(null);

  return (
    <div className="flex flex-col h-full">
      {/* 面包屑 */}
      <nav className="p-4 border-b border-border">
        <Breadcrumb>
          <BreadcrumbItem href="/">首页</BreadcrumbItem>
          <BreadcrumbItem href="/list">列表</BreadcrumbItem>
          <BreadcrumbItem>{data?.title}</BreadcrumbItem>
        </Breadcrumb>
      </nav>

      {/* 详情内容 */}
      <main className="flex-1 overflow-auto p-4">
        <div className="max-w-3xl mx-auto">{/* 详情卡片 */}</div>
      </main>
    </div>
  );
}
```

### 表单页面

```tsx
export function FormPage() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<FormData>({
    name: "",
    description: "",
  });
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      await api.create(formData);
      navigate("/list");
    } catch (error) {
      console.error("提交失败:", error);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>创建新项目</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">名称</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    name: e.target.value,
                  }))
                }
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">描述</Label>
              <Textarea
                id="description"
                value={formData.description}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    description: e.target.value,
                  }))
                }
              />
            </div>

            <div className="flex gap-2">
              <Button type="submit" disabled={submitting}>
                {submitting ? "提交中..." : "提交"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => navigate(-1)}
              >
                取消
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
```

## 页面状态管理

### 使用 Context

```tsx
// src/contexts/PageContext.tsx
import { createContext, useContext, useState, ReactNode } from "react";

interface PageState {
  title: string;
  setTitle: (title: string) => void;
}

const PageContext = createContext<PageState | null>(null);

export function PageProvider({ children }: { children: ReactNode }) {
  const [title, setTitle] = useState("DocuMind AI");

  return (
    <PageContext.Provider value={{ title, setTitle }}>
      {children}
    </PageContext.Provider>
  );
}

export function usePageContext() {
  const context = useContext(PageContext);
  if (!context) {
    throw new Error("usePageContext must be used within PageProvider");
  }
  return context;
}
```

### 使用 URL 参数

```tsx
import { useSearchParams } from "react-router-dom";

export function FilteredListPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const status = searchParams.get("status") || "all";
  const page = parseInt(searchParams.get("page") || "1");

  const handleFilterChange = (newStatus: string) => {
    setSearchParams({ status: newStatus, page: "1" });
  };

  return (
    // ...
  );
}
```

## 注意事项

1. **路由参数**：使用 `useParams()` 获取动态路由参数
2. **导航**：使用 `useNavigate()` 进行编程式导航
3. **滚动区域**：使用 `ScrollArea` 组件处理溢出内容
4. **响应式**：使用 Tailwind 的响应式断点 (`md:`, `lg:`)
5. **加载状态**：始终处理加载和错误状态
