---
name: add-react-component
description: 为 DocuMind AI 添加新的 React 组件
---

# 添加 React 组件技能

此技能用于为 DocuMind AI 前端项目添加新的 React 组件。

## 项目结构

```
projects/capstone-rag-qa-system/frontend/
├── src/
│   ├── components/
│   │   ├── ui/                 # shadcn/ui 基础组件
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   └── ...
│   │   ├── Sidebar.tsx         # 侧边栏
│   │   ├── ChatMessage.tsx     # 消息组件
│   │   ├── FileUploader.tsx    # 文件上传
│   │   └── ...
│   ├── pages/
│   │   ├── ChatPage.tsx
│   │   ├── DocumentsPage.tsx
│   │   └── SettingsPage.tsx
│   ├── hooks/                  # 自定义 Hooks
│   ├── lib/                    # 工具函数
│   └── types/                  # 类型定义
```

## 创建新组件步骤

### 1. 定义组件类型

```tsx
// src/types/index.ts
export interface NewComponentProps {
  title: string;
  description?: string;
  onAction?: () => void;
  children?: React.ReactNode;
}
```

### 2. 创建组件文件

```tsx
// src/components/NewComponent.tsx
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface NewComponentProps {
  title: string;
  description?: string;
  onAction?: () => void;
  className?: string;
  children?: React.ReactNode;
}

export function NewComponent({
  title,
  description,
  onAction,
  className,
  children,
}: NewComponentProps) {
  return (
    <Card className={cn("bg-card border-border", className)}>
      <CardHeader>
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
        {description && (
          <p className="text-sm text-muted-foreground">{description}</p>
        )}
      </CardHeader>
      <CardContent>
        {children}
        {onAction && (
          <Button onClick={onAction} className="mt-4">
            执行操作
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
```

### 3. 导出组件

```tsx
// src/components/index.ts
export { NewComponent } from "./NewComponent";
```

## shadcn/ui 组件使用

### 添加 shadcn 组件

```bash
# 添加单个组件
npx shadcn-ui@latest add button

# 添加多个组件
npx shadcn-ui@latest add card dialog input textarea

# 常用组件列表
npx shadcn-ui@latest add \
  avatar badge button card dialog dropdown-menu \
  input label scroll-area separator sheet \
  skeleton tabs textarea tooltip
```

### 常用 shadcn 组件

| 组件         | 用途       |
| ------------ | ---------- |
| Button       | 按钮       |
| Card         | 卡片容器   |
| Dialog       | 模态对话框 |
| Input        | 输入框     |
| ScrollArea   | 滚动区域   |
| Accordion    | 手风琴     |
| Avatar       | 头像       |
| Badge        | 标签/徽章  |
| DropdownMenu | 下拉菜单   |
| Skeleton     | 加载骨架屏 |

## 组件模式

### 带状态的组件

```tsx
import { useState, useEffect } from "react";

export function StatefulComponent() {
  const [data, setData] = useState<DataType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await api.getData();
        setData(response.data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "未知错误");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) return <Skeleton className="h-32" />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  return <DataDisplay data={data} />;
}
```

### 受控组件

```tsx
interface ControlledInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function ControlledInput({
  value,
  onChange,
  placeholder,
}: ControlledInputProps) {
  return (
    <Input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="bg-background border-border"
    />
  );
}
```

### 带 ref 的组件

```tsx
import { forwardRef } from "react";

interface CustomInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export const CustomInput = forwardRef<HTMLInputElement, CustomInputProps>(
  ({ label, className, ...props }, ref) => {
    return (
      <div className="space-y-2">
        {label && <label className="text-sm font-medium">{label}</label>}
        <input
          ref={ref}
          className={cn(
            "w-full px-3 py-2 rounded-lg bg-background border border-border",
            className,
          )}
          {...props}
        />
      </div>
    );
  },
);

CustomInput.displayName = "CustomInput";
```

## 样式规范

### Tailwind 类名顺序

```tsx
className={cn(
  // 1. 布局
  "flex items-center justify-between",
  // 2. 尺寸
  "w-full h-12 px-4 py-2",
  // 3. 背景/边框
  "bg-card border border-border rounded-lg",
  // 4. 文字
  "text-sm font-medium text-foreground",
  // 5. 交互
  "hover:bg-accent cursor-pointer",
  // 6. 过渡
  "transition-colors duration-200",
  // 7. 条件样式
  isActive && "bg-primary text-primary-foreground",
  // 8. 外部 className
  className
)}
```

### 深色主题颜色

```tsx
// 背景色
"bg-background"; // #0f172a
"bg-card"; // #1e293b
"bg-accent"; // hover 状态

// 文字色
"text-foreground"; // 主文字
"text-muted-foreground"; // 次要文字

// 边框色
"border-border"; // #334155

// 主题色
"bg-primary"; // #3b82f6
"text-primary"; // 蓝色文字
```

## 自定义 Hook

```tsx
// src/hooks/useKnowledgeBases.ts
import { useState, useEffect } from "react";
import { api } from "@/lib/api";

export function useKnowledgeBases() {
  const [data, setData] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      setLoading(true);
      const response = await api.getKnowledgeBases();
      setData(response.data.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  return { data, loading, error, refresh };
}
```

## 测试组件

```tsx
// src/components/__tests__/NewComponent.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { NewComponent } from "../NewComponent";

describe("NewComponent", () => {
  it("renders title correctly", () => {
    render(<NewComponent title="测试标题" />);
    expect(screen.getByText("测试标题")).toBeInTheDocument();
  });

  it("calls onAction when button clicked", () => {
    const handleAction = jest.fn();
    render(<NewComponent title="测试" onAction={handleAction} />);

    fireEvent.click(screen.getByText("执行操作"));
    expect(handleAction).toHaveBeenCalledTimes(1);
  });
});
```

## 注意事项

1. **类型安全**：始终定义 Props 接口
2. **cn() 工具**：使用 `cn()` 合并类名
3. **响应式**：使用 Tailwind 响应式前缀 (`sm:`, `md:`, `lg:`)
4. **可访问性**：添加适当的 ARIA 属性
5. **性能**：使用 `memo`, `useMemo`, `useCallback` 优化渲染
