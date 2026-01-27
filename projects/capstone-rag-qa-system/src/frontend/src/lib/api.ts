import axios, { type AxiosInstance } from "axios";

// ============================================
// Axios 实例配置
// ============================================

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const api: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message =
      error.response?.data?.message || error.message || "请求失败";
    console.error("API Error:", message);
    return Promise.reject(new Error(message));
  },
);

// ============================================
// 类型定义
// ============================================

export interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

// 知识库
export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
  document_count: number;
  chunk_count: number;
  total_size: number;
  created_at: string;
  updated_at: string;
}

// 文档
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
  error_message: string | null;
  chunk_count: number;
  created_at: string;
  processed_at: string | null;
}

export interface DocumentStatus {
  id: string;
  status: string;
  progress: number;
  current_step: string | null;
  steps: Array<{ step: string; status: string }>;
}

export interface DocumentChunk {
  id: string;
  index: number;
  content: string;
  metadata: Record<string, unknown> | null;
}

// 问答
export interface SourceReference {
  doc_id: string;
  filename: string;
  chunk_index: number;
  content: string;
  score: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceReference[];
  created_at: string;
}

export interface ChatResponse {
  message_id: string;
  conversation_id: string;
  answer: string;
  sources: SourceReference[];
  created_at: string;
}

export interface Conversation {
  id: string;
  kb_id: string;
  title: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail {
  id: string;
  kb_id: string;
  title: string | null;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

// 系统
export interface SystemHealth {
  status: string;
  version: string;
  components: {
    database: string;
    vector_store: string;
    embedding_model: string;
    llm: string;
  };
  uptime: number;
}

export interface SystemInfo {
  version: string;
  models: {
    embedding: string;
    llm: string;
  };
  limits: {
    max_file_size: number;
    supported_formats: string[];
    max_batch_upload: number;
  };
}

export interface SystemStats {
  knowledge_bases: number;
  documents: number;
  chunks: number;
  conversations: number;
  messages: number;
  storage_used: number;
}

// ============================================
// 知识库 API
// ============================================

export const knowledgeBaseApi = {
  // 获取知识库列表
  list: (page = 1, pageSize = 20) =>
    api.get<unknown, ApiResponse<PaginatedResponse<KnowledgeBase>>>(
      `/kb?page=${page}&page_size=${pageSize}`,
    ),

  // 获取单个知识库
  get: (id: string) =>
    api.get<unknown, ApiResponse<KnowledgeBase>>(`/kb/${id}`),

  // 创建知识库
  create: (name: string, description?: string) =>
    api.post<unknown, ApiResponse<KnowledgeBase>>("/kb", { name, description }),

  // 更新知识库
  update: (id: string, name?: string, description?: string) =>
    api.put<unknown, ApiResponse<KnowledgeBase>>(`/kb/${id}`, {
      name,
      description,
    }),

  // 删除知识库
  delete: (id: string) => api.delete<unknown, ApiResponse<null>>(`/kb/${id}`),
};

// ============================================
// 文档 API
// ============================================

export const documentApi = {
  // 获取文档列表
  list: (kbId: string, page = 1, pageSize = 20, status?: string) => {
    let url = `/documents?kb_id=${kbId}&page=${page}&page_size=${pageSize}`;
    if (status) url += `&status=${status}`;
    return api.get<unknown, ApiResponse<PaginatedResponse<Document>>>(url);
  },

  // 获取单个文档
  get: (docId: string) =>
    api.get<unknown, ApiResponse<Document>>(`/documents/${docId}`),

  // 获取文档状态
  getStatus: (docId: string) =>
    api.get<unknown, ApiResponse<DocumentStatus>>(`/documents/${docId}/status`),

  // 获取文档分块
  getChunks: (docId: string, page = 1, pageSize = 20) =>
    api.get<unknown, ApiResponse<PaginatedResponse<DocumentChunk>>>(
      `/documents/${docId}/chunks?page=${page}&page_size=${pageSize}`,
    ),

  // 上传文档
  upload: async (kbId: string, file: File) => {
    const formData = new FormData();
    formData.append("kb_id", kbId);
    formData.append("file", file);

    return api.post<unknown, ApiResponse<Document>>(
      "/documents/upload",
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
      },
    );
  },

  // 批量上传文档
  uploadBatch: async (kbId: string, files: File[]) => {
    const formData = new FormData();
    formData.append("kb_id", kbId);
    files.forEach((file) => formData.append("files", file));

    return api.post<
      unknown,
      ApiResponse<{ uploaded: Document[]; failed: string[] }>
    >("/documents/upload/batch", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },

  // 删除文档
  delete: (docId: string) =>
    api.delete<unknown, ApiResponse<null>>(`/documents/${docId}`),
};

// ============================================
// 问答 API
// ============================================

export const chatApi = {
  // 发送问答请求 (非流式)
  send: (
    kbId: string,
    query: string,
    conversationId?: string,
    options?: {
      top_k?: number;
      temperature?: number;
      max_tokens?: number;
    },
  ) =>
    api.post<unknown, ApiResponse<ChatResponse>>("/chat", {
      kb_id: kbId,
      query,
      conversation_id: conversationId,
      options,
    }),

  // 获取对话列表
  listConversations: (kbId: string, page = 1, pageSize = 20) =>
    api.get<unknown, ApiResponse<PaginatedResponse<Conversation>>>(
      `/chat/conversations?kb_id=${kbId}&page=${page}&page_size=${pageSize}`,
    ),

  // 获取对话详情
  getConversation: (conversationId: string) =>
    api.get<unknown, ApiResponse<ConversationDetail>>(
      `/chat/conversations/${conversationId}`,
    ),

  // 删除对话
  deleteConversation: (conversationId: string) =>
    api.delete<unknown, ApiResponse<null>>(
      `/chat/conversations/${conversationId}`,
    ),

  // 提交反馈
  submitFeedback: (
    messageId: string,
    rating: "good" | "bad",
    comment?: string,
  ) =>
    api.post<unknown, ApiResponse<{ feedback_id: string }>>("/chat/feedback", {
      message_id: messageId,
      rating,
      comment,
    }),
};

// 流式问答 (使用 fetch + SSE)
export function streamChat(
  kbId: string,
  query: string,
  conversationId: string | undefined,
  options:
    | {
        top_k?: number;
        temperature?: number;
        max_tokens?: number;
      }
    | undefined,
  callbacks: {
    onChunk: (text: string) => void;
    onSources: (sources: SourceReference[]) => void;
    onDone: (data: { message_id: string; conversation_id: string }) => void;
    onError: (error: string) => void;
  },
): AbortController {
  const controller = new AbortController();

  const body = JSON.stringify({
    kb_id: kbId,
    query,
    conversation_id: conversationId,
    options,
  });

  fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError("Failed to create reader");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === "chunk") {
                callbacks.onChunk(data.content);
              } else if (data.type === "sources") {
                callbacks.onSources(data.sources);
              } else if (data.type === "done") {
                callbacks.onDone(data);
              } else if (data.type === "error") {
                callbacks.onError(data.error);
              }
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
    })
    .catch((error) => {
      if (error.name !== "AbortError") {
        callbacks.onError(error.message);
      }
    });

  return controller;
}

// ============================================
// 系统 API
// ============================================

export const systemApi = {
  // 健康检查
  health: () => api.get<unknown, ApiResponse<SystemHealth>>("/system/health"),

  // 系统信息
  info: () => api.get<unknown, ApiResponse<SystemInfo>>("/system/info"),

  // 统计信息
  stats: () => api.get<unknown, ApiResponse<SystemStats>>("/system/stats"),
};

// ============================================
// 便捷导出
// ============================================

export default api;
