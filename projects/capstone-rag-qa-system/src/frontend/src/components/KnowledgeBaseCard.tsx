import {
  FolderOpen,
  FileText,
  Layers,
  MoreHorizontal,
  Trash2,
} from "lucide-react";
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
  onChat?: () => void;
}

export function KnowledgeBaseCard({
  name,
  description,
  documentCount,
  chunkCount,
  updatedAt,
  onClick,
  onDelete,
  onChat,
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
    <div
      className="group relative bg-card/60 backdrop-blur-xl border border-border/50 rounded-2xl p-5 cursor-pointer transition-all duration-300 hover:border-primary/50 hover:shadow-glow hover:translate-y-[-2px]"
      onClick={onClick}
    >
      {/* Gradient Folder Icon */}
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 via-primary/30 to-accent/20 flex items-center justify-center mb-4 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-primary/10 to-transparent" />
        <FolderOpen className="w-8 h-8 text-primary relative z-10" />
      </div>

      {/* Title */}
      <h3 className="font-semibold text-lg mb-2 truncate text-foreground">
        {name}
      </h3>

      {/* Description */}
      <p className="text-sm text-muted-foreground line-clamp-2 mb-4 min-h-[40px]">
        {description || "暂无描述"}
      </p>

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm text-muted-foreground mb-3">
        <div className="flex items-center gap-1.5">
          <FileText className="w-4 h-4" />
          <span>{documentCount} Documents</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Layers className="w-4 h-4" />
          <span>{chunkCount} Chunks</span>
        </div>
      </div>

      {/* Updated Time */}
      <p className="text-xs text-muted">
        Last updated: {formatDate(updatedAt)}
      </p>

      {/* Action Buttons - Show on Hover */}
      <div className="absolute bottom-4 right-4 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        {onChat && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onChat();
            }}
            className="px-3 py-1.5 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 transition-colors"
          >
            Open
          </button>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="px-3 py-1.5 rounded-lg bg-destructive/10 text-destructive text-sm font-medium hover:bg-destructive/20 transition-colors"
        >
          Delete
        </button>
      </div>

      {/* Menu Button */}
      <div className="absolute top-4 right-4">
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowMenu(!showMenu);
          }}
          className="p-1.5 rounded-lg opacity-0 group-hover:opacity-100 hover:bg-border/50 transition-all"
        >
          <MoreHorizontal className="w-4 h-4 text-muted-foreground" />
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
            <div className="absolute right-0 top-8 z-20 bg-card border border-border rounded-xl shadow-modal p-1.5 min-w-[140px]">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(false);
                  onDelete();
                }}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-destructive hover:bg-destructive/10 rounded-lg transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                删除知识库
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
      className="group relative bg-gradient-to-br from-primary/5 to-accent/5 border-2 border-dashed border-border/50 rounded-2xl p-5 cursor-pointer transition-all duration-300 hover:border-primary/50 hover:bg-primary/5 flex flex-col items-center justify-center min-h-[240px]"
      onClick={onClick}
    >
      {/* Plus Icon */}
      <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
          <span className="text-2xl text-white font-medium">+</span>
        </div>
      </div>

      <h3 className="font-semibold text-lg mb-1">Create your first</h3>
      <h3 className="font-semibold text-lg mb-2">knowledge base</h3>
      <p className="text-sm text-muted-foreground text-center">
        Start by importing documents or
        <br />
        connecting data sources.
      </p>
    </div>
  );
}
