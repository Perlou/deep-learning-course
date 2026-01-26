---
name: setup-react-frontend
description: 初始化和配置 React 前端项目
---

# 初始化 React 前端项目技能

此技能用于初始化 DocuMind AI 的 React 前端项目。

## 技术栈

| 技术         | 版本   | 用途     |
| ------------ | ------ | -------- |
| React        | 18.x   | UI 框架  |
| TypeScript   | 5.x    | 类型安全 |
| Vite         | 5.x    | 构建工具 |
| Tailwind CSS | 3.x    | 样式框架 |
| shadcn/ui    | latest | 组件库   |
| React Router | 6.x    | 路由管理 |
| Lucide React | latest | 图标库   |

## 初始化步骤

### 1. 创建 Vite 项目

```bash
cd projects/capstone-rag-qa-system

# 创建项目
npm create vite@latest frontend -- --template react-ts

cd frontend
npm install
```

### 2. 安装 Tailwind CSS

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

配置 `tailwind.config.js`：

```js
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        border: "hsl(var(--border))",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
```

### 3. 设置全局样式

更新 `src/index.css`：

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 47% 11%;
    --card-foreground: 210 40% 98%;
    --primary: 217.2 91.2% 59.8%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --radius: 0.5rem;
  }

  * {
    @apply border-border;
  }

  body {
    @apply bg-background text-foreground;
    font-family: "Inter", system-ui, sans-serif;
  }
}
```

### 4. 初始化 shadcn/ui

```bash
npx shadcn-ui@latest init
```

选择配置：

- Style: New York
- Base color: Slate
- CSS variables: Yes

安装常用组件：

```bash
npx shadcn-ui@latest add button card input dialog \
  avatar badge scroll-area separator dropdown-menu \
  accordion tabs textarea tooltip skeleton
```

### 5. 安装其他依赖

```bash
# 路由
npm install react-router-dom

# 图标
npm install lucide-react

# HTTP 客户端
npm install axios

# 拖拽上传
npm install react-dropzone

# Markdown 渲染
npm install react-markdown remark-gfm

# 类名合并工具 (shadcn 已包含)
npm install clsx tailwind-merge
```

### 6. 配置路径别名

更新 `tsconfig.json`：

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

更新 `vite.config.ts`：

```ts
import path from "path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

### 7. 创建项目结构

```bash
mkdir -p src/{components,pages,hooks,lib,types}
mkdir -p src/components/ui
```

### 8. 创建工具函数

`src/lib/utils.ts`：

```ts
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

### 9. 创建 API 客户端

`src/lib/api.ts`：

```ts
const API_BASE = import.meta.env.VITE_API_URL || "/api/v1";

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export const api = {
  // 知识库
  getKnowledgeBases: () => request<ApiResponse<KnowledgeBase[]>>("/kb"),
  createKnowledgeBase: (data: CreateKBRequest) =>
    request<ApiResponse<KnowledgeBase>>("/kb", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // 文档
  uploadDocument: async (kbId: string, file: File) => {
    const formData = new FormData();
    formData.append("kb_id", kbId);
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    return response.json();
  },

  // 聊天
  chat: (data: ChatRequest) =>
    request<ApiResponse<ChatResponse>>("/chat", {
      method: "POST",
      body: JSON.stringify(data),
    }),
};
```

### 10. 创建类型定义

`src/types/index.ts`：

```ts
export interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string;
  document_count: number;
  chunk_count: number;
  created_at: string;
}

export interface Document {
  id: string;
  kb_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status:
    | "pending"
    | "parsing"
    | "chunking"
    | "embedding"
    | "completed"
    | "failed";
  created_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  created_at: string;
}

export interface Source {
  doc_id: string;
  filename: string;
  content: string;
  score: number;
}
```

## 验证安装

```bash
# 启动开发服务器
npm run dev

# 访问 http://localhost:5173
```

## 目录结构

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn 组件
│   │   ├── Layout.tsx
│   │   ├── Sidebar.tsx
│   │   └── ...
│   ├── pages/
│   │   ├── ChatPage.tsx
│   │   ├── DocumentsPage.tsx
│   │   └── SettingsPage.tsx
│   ├── hooks/
│   │   └── useKnowledgeBases.ts
│   ├── lib/
│   │   ├── api.ts
│   │   └── utils.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## 注意事项

1. **环境变量**：使用 `VITE_` 前缀定义环境变量
2. **代理配置**：开发时通过 Vite 代理 API 请求
3. **类型安全**：始终定义 TypeScript 接口
4. **响应式**：使用 Tailwind 的响应式断点
5. **深色主题**：默认使用深色主题变量
