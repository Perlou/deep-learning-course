import { Database, FileText, MoreHorizontal, Trash2 } from "lucide-react";
import { useState } from "react";

interface KnowledgeBaseCardProps {
  id: string;
  name: string;
  description: string | null;
  documentCount: number;
  chunkCount: number;
  updatedAt: string;
  onClick: () => void;
  onDelete: () => void;
}

export function KnowledgeBaseCard({
  name,
  description,
  documentCount,
  chunkCount,
  updatedAt,
  onClick,
  onDelete,
}: KnowledgeBaseCardProps) {
  const [showMenu, setShowMenu] = useState(false);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("zh-CN", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="card cursor-pointer group relative" onClick={onClick}>
      {/* Icon */}
      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-4">
        <Database className="w-7 h-7 text-primary" />
      </div>

      {/* Title & Description */}
      <h3 className="font-semibold text-lg mb-1 truncate">{name}</h3>
      <p className="text-sm text-muted-foreground line-clamp-2 mb-4 min-h-[40px]">
        {description || "暂无描述"}
      </p>

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm text-muted-foreground mb-3">
        <div className="flex items-center gap-1">
          <FileText className="w-4 h-4" />
          <span>{documentCount} 文档</span>
        </div>
        <div className="flex items-center gap-1">
          <Database className="w-4 h-4" />
          <span>{chunkCount} 分块</span>
        </div>
      </div>

      {/* Updated Time */}
      <p className="text-xs text-muted">更新于: {formatDate(updatedAt)}</p>

      {/* Menu Button */}
      <div className="absolute top-4 right-4">
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowMenu(!showMenu);
          }}
          className="p-1.5 rounded-lg opacity-0 group-hover:opacity-100 hover:bg-border transition-all"
        >
          <MoreHorizontal className="w-4 h-4" />
        </button>

        {showMenu && (
          <>
            <div
              className="fixed inset-0 z-10"
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(false);
              }}
            />
            <div className="absolute right-0 top-8 z-20 bg-card border border-border rounded-lg shadow-modal p-1 min-w-[120px]">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(false);
                  onDelete();
                }}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-destructive hover:bg-destructive/10 rounded-md transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                删除
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// Create New Card
interface CreateCardProps {
  onClick: () => void;
}

export function CreateKnowledgeBaseCard({ onClick }: CreateCardProps) {
  return (
    <div
      className="card cursor-pointer border-dashed hover:border-primary flex flex-col items-center justify-center min-h-[200px]"
      onClick={onClick}
    >
      <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
        <span className="text-3xl text-primary">+</span>
      </div>
      <h3 className="font-medium">创建知识库</h3>
      <p className="text-sm text-muted-foreground mt-1">开始导入您的文档</p>
    </div>
  );
}
