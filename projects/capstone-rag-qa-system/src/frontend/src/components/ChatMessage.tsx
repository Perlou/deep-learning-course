import { cn } from "../lib/utils";
import type { SourceReference } from "../lib/api";
import { ChevronDown, ChevronRight, FileText, BookOpen } from "lucide-react";
import { useState } from "react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: SourceReference[];
  isStreaming?: boolean;
}

export function ChatMessage({
  role,
  content,
  sources,
  isStreaming,
}: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false);
  const validSources = sources?.filter((s) => s.score > 0) || [];

  return (
    <div
      className={cn(
        "flex gap-3 p-4",
        role === "user" ? "flex-row-reverse" : "",
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 text-sm font-medium",
          role === "user"
            ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white"
            : "bg-gradient-to-br from-primary to-accent text-white",
        )}
      >
        {role === "user" ? "U" : "AI"}
      </div>

      {/* Message Content */}
      <div
        className={cn(
          "flex flex-col gap-2",
          role === "user" ? "items-end" : "items-start",
          "max-w-[85%]",
        )}
      >
        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            role === "user"
              ? "bg-primary text-white rounded-tr-sm"
              : "bg-card/80 backdrop-blur-sm text-foreground rounded-tl-sm border border-border/50",
          )}
        >
          <p className="whitespace-pre-wrap leading-relaxed">
            {content}
            {isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-primary/80 animate-pulse rounded-sm" />
            )}
          </p>
        </div>

        {/* Sources Toggle - Only show after streaming is done and we have sources */}
        {validSources.length > 0 && !isStreaming && (
          <div className="w-full">
            {/* Toggle Button */}
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors rounded-lg hover:bg-card/50"
            >
              <BookOpen className="w-4 h-4" />
              <span>引用来源 ({validSources.length})</span>
              {showSources ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>

            {/* Sources List - Collapsed by default */}
            {showSources && (
              <div className="mt-2 space-y-2 animate-in fade-in slide-in-from-top-2 duration-200">
                {validSources.slice(0, 3).map((source, index) => (
                  <SourceCard key={index} source={source} index={index + 1} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface SourceCardProps {
  source: SourceReference;
  index: number;
}

function SourceCard({ source }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const relevancePercent = Math.round(source.score * 100);

  return (
    <div
      className="bg-card/40 backdrop-blur-sm border border-border/30 rounded-lg p-3 cursor-pointer hover:border-primary/30 transition-all"
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <FileText className="w-4 h-4 text-primary/70 shrink-0" />
          <span className="text-sm text-foreground/80 truncate">
            {source.filename}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span
            className={cn(
              "text-xs px-2 py-0.5 rounded-full",
              relevancePercent >= 50
                ? "bg-green-500/20 text-green-400"
                : relevancePercent >= 30
                  ? "bg-yellow-500/20 text-yellow-400"
                  : "bg-gray-500/20 text-gray-400",
            )}
          >
            {relevancePercent}%
          </span>
          <ChevronDown
            className={cn(
              "w-4 h-4 text-muted-foreground transition-transform",
              expanded && "rotate-180",
            )}
          />
        </div>
      </div>

      {/* Content preview - Only show when expanded */}
      {expanded && (
        <p className="mt-2 text-sm text-muted-foreground border-t border-border/30 pt-2">
          {source.content}
        </p>
      )}
    </div>
  );
}
