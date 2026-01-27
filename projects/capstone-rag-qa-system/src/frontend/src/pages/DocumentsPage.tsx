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

  const processingCount = documents.filter(
    (d) => d.status !== "completed" && d.status !== "failed",
  ).length;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="h-14 border-b border-border flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate("/")}
            className="p-2 hover:bg-card rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="font-semibold">{kb?.name || "知识库"}</h1>
            <p className="text-xs text-muted-foreground">
              {documents.length} 个文档
              {processingCount > 0 && (
                <span className="ml-2 text-warning">
                  ({processingCount} 处理中)
                </span>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadData}
            className="btn-secondary flex items-center gap-2"
            title="刷新"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowUploadModal(true)}
            className="btn-primary flex items-center gap-2"
          >
            <Upload className="w-4 h-4" />
            上传文档
          </button>
          <Link
            to={`/chat/${kbId}`}
            className="btn-secondary flex items-center gap-2"
          >
            <MessageSquare className="w-4 h-4" />
            开始对话
          </Link>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 p-6 overflow-auto">
        {documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <Upload className="w-16 h-16 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">暂无文档</h3>
            <p className="text-muted-foreground mb-4">
              上传文档开始构建您的知识库
            </p>
            <button
              onClick={() => setShowUploadModal(true)}
              className="btn-primary"
            >
              上传文档
            </button>
          </div>
        ) : (
          <div className="space-y-3 max-w-3xl">
            {documents.map((doc) => (
              <FileCard
                key={doc.id}
                filename={doc.filename}
                fileType={doc.file_type}
                fileSize={doc.file_size}
                status={doc.status}
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
