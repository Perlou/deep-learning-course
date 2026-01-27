import { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import {
  ArrowLeft,
  Loader2,
  Upload,
  MessageSquare,
  RefreshCw,
} from "lucide-react";
import { FileUploader, FileCard } from "../components";
import {
  documentApi,
  knowledgeBaseApi,
  type Document,
  type KnowledgeBase,
} from "../lib/api";

export function DocumentsPage() {
  const { kbId } = useParams<{ kbId: string }>();
  const navigate = useNavigate();
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);

  const loadData = useCallback(async () => {
    if (!kbId) return;

    try {
      // 获取知识库信息
      const kbResponse = await knowledgeBaseApi.get(kbId);
      setKb(kbResponse.data);

      // 获取文档列表
      const docsResponse = await documentApi.list(kbId);
      setDocuments(docsResponse.data.items);
    } catch (error) {
      console.error("Failed to load documents:", error);
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // 轮询处理中的文档状态
  useEffect(() => {
    const processingDocs = documents.filter(
      (d) => d.status !== "completed" && d.status !== "failed",
    );

    if (processingDocs.length === 0) return;

    const interval = setInterval(() => {
      loadData();
    }, 3000);

    return () => clearInterval(interval);
  }, [documents, loadData]);

  const handleUpload = async (files: File[]) => {
    if (!kbId) return;

    setUploading(true);
    try {
      for (const file of files) {
        await documentApi.upload(kbId, file);
      }
      setShowUploadModal(false);
      loadData();
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    if (!confirm("确定要删除这个文档吗？")) return;

    try {
      await documentApi.delete(docId);
      loadData();
    } catch (error) {
      console.error("Failed to delete document:", error);
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
      <header className="h-16 border-b border-border/50 bg-card/30 backdrop-blur-sm flex items-center justify-between px-6 lg:px-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate("/")}
            className="p-2 hover:bg-card rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold">
              Knowledge Base: {kb?.name || "..."}
            </h1>
            <span className="px-2.5 py-1 bg-primary/20 text-primary text-xs font-medium rounded-full">
              {documents.length} Documents
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadData}
            className="p-2.5 rounded-lg hover:bg-card transition-colors text-muted-foreground"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowUploadModal(true)}
            className="btn-primary flex items-center gap-2 text-sm"
          >
            <Upload className="w-4 h-4" />
            Upload Documents
          </button>
          <Link
            to={`/chat/${kbId}`}
            className="btn-secondary flex items-center gap-2 text-sm"
          >
            <MessageSquare className="w-4 h-4" />
            Start Chat
          </Link>
        </div>
      </header>

      {/* Toolbar */}
      <div className="px-6 lg:px-8 py-4 border-b border-border/50 flex items-center gap-4">
        <div className="flex-1 relative">
          <input
            type="text"
            placeholder="Search documents..."
            className="input pl-10 py-2"
          />
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Filter:</span>
          <select className="input py-2 px-3 text-sm min-w-[140px]">
            <option>All Status</option>
            <option>Completed</option>
            <option>Processing</option>
            <option>Failed</option>
          </select>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-6 lg:p-8 overflow-auto">
        {documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
              <Upload className="w-10 h-10 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">No documents yet</h3>
            <p className="text-muted-foreground mb-6 max-w-md">
              Upload documents to start building your knowledge base. We support
              PDF, DOCX, TXT, and Markdown files.
            </p>
            <button
              onClick={() => setShowUploadModal(true)}
              className="btn-primary"
            >
              Upload Your First Document
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-5xl">
            {documents.map((doc) => (
              <FileCard
                key={doc.id}
                filename={doc.filename}
                fileType={doc.file_type}
                fileSize={doc.file_size}
                status={doc.status}
                createdAt={doc.created_at}
                onDelete={() => handleDelete(doc.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-2xl p-6 w-full max-w-md shadow-modal">
            <h2 className="text-xl font-semibold mb-4">上传文档</h2>

            <FileUploader onUpload={handleUpload} uploading={uploading} />

            <div className="flex justify-end mt-6">
              <button
                onClick={() => setShowUploadModal(false)}
                className="btn-secondary"
                disabled={uploading}
              >
                关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
