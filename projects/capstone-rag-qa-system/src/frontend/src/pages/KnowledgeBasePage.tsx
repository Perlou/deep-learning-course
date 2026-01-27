import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, Loader2 } from "lucide-react";
import { KnowledgeBaseCard, CreateKnowledgeBaseCard } from "../components";
import { knowledgeBaseApi, type KnowledgeBase } from "../lib/api";

export function KnowledgeBasePage() {
  const navigate = useNavigate();
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newKbName, setNewKbName] = useState("");
  const [newKbDescription, setNewKbDescription] = useState("");

  const loadData = useCallback(async () => {
    try {
      const response = await knowledgeBaseApi.list();
      setKnowledgeBases(response.data.items);
    } catch (error) {
      console.error("Failed to load knowledge bases:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreate = async () => {
    if (!newKbName.trim()) return;

    setCreating(true);
    try {
      await knowledgeBaseApi.create(newKbName, newKbDescription || undefined);
      setShowCreateModal(false);
      setNewKbName("");
      setNewKbDescription("");
      loadData();
    } catch (error) {
      console.error("Failed to create knowledge base:", error);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("确定要删除这个知识库吗？")) return;

    try {
      await knowledgeBaseApi.delete(id);
      loadData();
    } catch (error) {
      console.error("Failed to delete knowledge base:", error);
    }
  };

  return (
    <div className="flex-1 p-6 lg:p-8 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-1">Knowledge Bases</h1>
          <p className="text-muted-foreground">
            Manage your document collections
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Create New
        </button>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-6">
          {knowledgeBases.map((kb) => (
            <KnowledgeBaseCard
              key={kb.id}
              id={kb.id}
              name={kb.name}
              description={kb.description}
              documentCount={kb.document_count}
              chunkCount={kb.chunk_count}
              updatedAt={kb.updated_at}
              onClick={() => navigate(`/kb/${kb.id}/documents`)}
              onDelete={() => handleDelete(kb.id)}
              onChat={() => navigate(`/chat/${kb.id}`)}
            />
          ))}
          <CreateKnowledgeBaseCard onClick={() => setShowCreateModal(true)} />
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-2xl p-6 w-full max-w-md shadow-modal">
            <h2 className="text-xl font-semibold mb-4">创建知识库</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">名称</label>
                <input
                  type="text"
                  value={newKbName}
                  onChange={(e) => setNewKbName(e.target.value)}
                  placeholder="输入知识库名称"
                  className="input"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">描述</label>
                <textarea
                  value={newKbDescription}
                  onChange={(e) => setNewKbDescription(e.target.value)}
                  placeholder="输入知识库描述（可选）"
                  className="input resize-none"
                  rows={3}
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="btn-secondary"
              >
                取消
              </button>
              <button
                onClick={handleCreate}
                disabled={!newKbName.trim() || creating}
                className="btn-primary flex items-center gap-2"
              >
                {creating && <Loader2 className="w-4 h-4 animate-spin" />}
                创建
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
