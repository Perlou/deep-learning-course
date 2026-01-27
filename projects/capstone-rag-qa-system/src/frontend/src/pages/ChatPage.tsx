import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Loader2,
  History,
  ThumbsUp,
  ThumbsDown,
} from "lucide-react";
import { ChatMessage, ChatInput } from "../components";
import {
  streamChat,
  knowledgeBaseApi,
  chatApi,
  type KnowledgeBase,
  type SourceReference,
  type Conversation,
} from "../lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceReference[];
}

export function ChatPage() {
  const { kbId } = useParams<{ kbId: string }>();
  const navigate = useNavigate();
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const loadKb = useCallback(async () => {
    if (!kbId) return;
    try {
      const response = await knowledgeBaseApi.get(kbId);
      setKb(response.data);
    } catch (error) {
      console.error("Failed to load knowledge base:", error);
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  const loadConversations = useCallback(async () => {
    if (!kbId) return;
    try {
      const response = await chatApi.listConversations(kbId);
      setConversations(response.data.items);
    } catch (error) {
      console.error("Failed to load conversations:", error);
    }
  }, [kbId]);

  useEffect(() => {
    loadKb();
    loadConversations();
  }, [loadKb, loadConversations]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadConversation = async (convId: string) => {
    try {
      const response = await chatApi.getConversation(convId);
      const conv = response.data;
      setConversationId(conv.id);
      setMessages(
        conv.messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          sources: m.sources,
        })),
      );
      setShowHistory(false);
    } catch (error) {
      console.error("Failed to load conversation:", error);
    }
  };

  const handleNewConversation = () => {
    setConversationId(undefined);
    setMessages([]);
    setShowHistory(false);
  };

  const handleSend = async (message: string) => {
    if (!kbId || generating) return;

    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: message,
    };
    setMessages((prev) => [...prev, userMessage]);

    // Add placeholder for assistant message
    const assistantId = `assistant-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "" },
    ]);

    setGenerating(true);

    // Stream response
    abortControllerRef.current = streamChat(
      kbId,
      message,
      conversationId,
      undefined,
      {
        onChunk: (text) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: msg.content + text }
                : msg,
            ),
          );
        },
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId ? { ...msg, sources } : msg,
            ),
          );
        },
        onDone: (data) => {
          setConversationId(data.conversation_id);
          setGenerating(false);
          loadConversations(); // 刷新对话列表
        },
        onError: (error) => {
          console.error("Chat error:", error);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: `错误: ${error}` }
                : msg,
            ),
          );
          setGenerating(false);
        },
      },
    );
  };

  const handleFeedback = async (messageId: string, rating: "good" | "bad") => {
    try {
      await chatApi.submitFeedback(messageId, rating);
      // 可以在这里添加 UI 反馈
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="h-14 border-b border-border flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(`/kb/${kbId}/documents`)}
            className="p-2 hover:bg-card rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="font-semibold">{kb?.name || "对话"}</h1>
            <p className="text-xs text-muted-foreground">向知识库提问</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="btn-secondary flex items-center gap-2"
          >
            <History className="w-4 h-4" />
            历史
          </button>
          <button onClick={handleNewConversation} className="btn-primary">
            新对话
          </button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Conversation History Sidebar */}
        {showHistory && (
          <div className="w-64 border-r border-border p-4 overflow-auto">
            <h3 className="font-medium mb-3">对话历史</h3>
            {conversations.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无历史对话</p>
            ) : (
              <div className="space-y-2">
                {conversations.map((conv) => (
                  <button
                    key={conv.id}
                    onClick={() => loadConversation(conv.id)}
                    className={`w-full text-left p-2 rounded-lg text-sm transition-colors ${
                      conversationId === conv.id
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-card"
                    }`}
                  >
                    <p className="truncate">{conv.title || "新对话"}</p>
                    <p className="text-xs text-muted-foreground">
                      {conv.message_count} 条消息
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-auto">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-accent flex items-center justify-center mb-4">
                  <span className="text-2xl text-white">AI</span>
                </div>
                <h2 className="text-xl font-semibold mb-2">
                  欢迎使用 DocuMind AI
                </h2>
                <p className="text-muted-foreground max-w-md">
                  我已分析您的文档，您可以问我任何关于
                  <span className="text-primary">"{kb?.name}"</span>
                  知识库的问题。
                </p>
              </div>
            ) : (
              <div className="max-w-4xl mx-auto py-4">
                {messages.map((msg) => (
                  <div key={msg.id}>
                    <ChatMessage
                      role={msg.role}
                      content={msg.content}
                      sources={msg.sources}
                    />
                    {/* Feedback buttons for assistant messages */}
                    {msg.role === "assistant" && msg.content && !generating && (
                      <div className="flex gap-2 px-4 pb-2 ml-11">
                        <button
                          onClick={() => handleFeedback(msg.id, "good")}
                          className="p-1 text-muted-foreground hover:text-success transition-colors"
                          title="有帮助"
                        >
                          <ThumbsUp className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleFeedback(msg.id, "bad")}
                          className="p-1 text-muted-foreground hover:text-destructive transition-colors"
                          title="没帮助"
                        >
                          <ThumbsDown className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <ChatInput
            onSend={handleSend}
            disabled={generating}
            placeholder={`向 ${kb?.name || "知识库"} 提问...`}
          />
        </div>
      </div>
    </div>
  );
}
