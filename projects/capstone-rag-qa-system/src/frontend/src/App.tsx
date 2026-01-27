import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useEffect, useState } from "react";
import {
  Loader2,
  Database,
  FileText,
  MessageSquare,
  HardDrive,
} from "lucide-react";
import { Layout } from "./components";
import { KnowledgeBasePage, DocumentsPage, ChatPage } from "./pages";
import {
  systemApi,
  type SystemInfo,
  type SystemStats,
  type SystemHealth,
} from "./lib/api";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<KnowledgeBasePage />} />
          <Route path="kb/:kbId/documents" element={<DocumentsPage />} />
          <Route path="chat/:kbId" element={<ChatPage />} />
          <Route path="chat" element={<Navigate to="/" replace />} />
          <Route path="documents" element={<Navigate to="/" replace />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

// 设置页面 - 动态获取系统信息
function SettingsPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadSystemData = async () => {
      try {
        const [healthRes, infoRes, statsRes] = await Promise.all([
          systemApi.health(),
          systemApi.info(),
          systemApi.stats(),
        ]);
        setHealth(healthRes.data);
        setInfo(infoRes.data);
        setStats(statsRes.data);
      } catch (error) {
        console.error("Failed to load system data:", error);
      } finally {
        setLoading(false);
      }
    };
    loadSystemData();
  }, []);

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024)
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    if (days > 0) return `${days}天 ${hours}小时`;
    if (hours > 0) return `${hours}小时 ${mins}分钟`;
    return `${mins}分钟`;
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 overflow-auto">
      <h1 className="text-2xl font-bold mb-6">设置</h1>

      <div className="grid gap-6 max-w-4xl">
        {/* 系统状态 */}
        <div className="card">
          <h2 className="font-semibold mb-4">系统状态</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-background rounded-lg">
              <p className="text-sm text-muted-foreground">状态</p>
              <p
                className={`font-medium ${
                  health?.status === "healthy" ? "text-success" : "text-warning"
                }`}
              >
                {health?.status === "healthy" ? "正常" : "降级"}
              </p>
            </div>
            <div className="p-3 bg-background rounded-lg">
              <p className="text-sm text-muted-foreground">版本</p>
              <p className="font-medium">{health?.version || "N/A"}</p>
            </div>
            <div className="p-3 bg-background rounded-lg">
              <p className="text-sm text-muted-foreground">运行时间</p>
              <p className="font-medium">
                {health ? formatUptime(health.uptime) : "N/A"}
              </p>
            </div>
            <div className="p-3 bg-background rounded-lg">
              <p className="text-sm text-muted-foreground">数据库</p>
              <p
                className={`font-medium ${
                  health?.components.database === "healthy"
                    ? "text-success"
                    : "text-destructive"
                }`}
              >
                {health?.components.database === "healthy" ? "正常" : "异常"}
              </p>
            </div>
          </div>
        </div>

        {/* 统计数据 */}
        {stats && (
          <div className="card">
            <h2 className="font-semibold mb-4">统计数据</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="p-3 bg-background rounded-lg text-center">
                <Database className="w-6 h-6 mx-auto mb-2 text-primary" />
                <p className="text-2xl font-bold">{stats.knowledge_bases}</p>
                <p className="text-sm text-muted-foreground">知识库</p>
              </div>
              <div className="p-3 bg-background rounded-lg text-center">
                <FileText className="w-6 h-6 mx-auto mb-2 text-primary" />
                <p className="text-2xl font-bold">{stats.documents}</p>
                <p className="text-sm text-muted-foreground">文档</p>
              </div>
              <div className="p-3 bg-background rounded-lg text-center">
                <div className="w-6 h-6 mx-auto mb-2 text-primary text-sm font-bold">
                  Σ
                </div>
                <p className="text-2xl font-bold">{stats.chunks}</p>
                <p className="text-sm text-muted-foreground">分块</p>
              </div>
              <div className="p-3 bg-background rounded-lg text-center">
                <MessageSquare className="w-6 h-6 mx-auto mb-2 text-primary" />
                <p className="text-2xl font-bold">{stats.messages}</p>
                <p className="text-sm text-muted-foreground">消息</p>
              </div>
              <div className="p-3 bg-background rounded-lg text-center">
                <HardDrive className="w-6 h-6 mx-auto mb-2 text-primary" />
                <p className="text-2xl font-bold">
                  {formatBytes(stats.storage_used)}
                </p>
                <p className="text-sm text-muted-foreground">存储</p>
              </div>
            </div>
          </div>
        )}

        {/* 模型配置 */}
        {info && (
          <div className="card">
            <h2 className="font-semibold mb-4">模型配置</h2>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Embedding 模型</span>
                <span className="font-mono">{info.models.embedding}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">LLM 模型</span>
                <span className="font-mono">{info.models.llm}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">最大文件大小</span>
                <span>{formatBytes(info.limits.max_file_size)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">支持格式</span>
                <span>{info.limits.supported_formats.join(", ")}</span>
              </div>
            </div>
          </div>
        )}

        {/* API 配置 */}
        <div className="card">
          <h2 className="font-semibold mb-4">API 配置</h2>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">API 地址</span>
              <span className="font-mono text-xs">
                {import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
