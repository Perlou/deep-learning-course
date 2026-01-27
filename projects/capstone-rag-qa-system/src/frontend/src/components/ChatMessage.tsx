import { cn } from "../lib/utils";
import type { SourceReference } from "../lib/api";
import { ChevronDown, FileText } from "lucide-react";
import { useState } from "react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: SourceReference[];
}

export function ChatMessage({ role, content, sources }: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false);

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
          "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
          role === "user"
            ? "bg-primary text-white"
            : "bg-gradient-to-br from-primary to-accent text-white",
        )}
      >
        {role === "user" ? "U" : "AI"}
      </div>

      {/* Message Content */}
      <div className="flex flex-col gap-2 max-w-[80%]">
        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            role === "user"
              ? "bg-primary text-primary-foreground rounded-tr-sm"
              : "bg-card text-card-foreground rounded-tl-sm",
          )}
        >
          <p className="whitespace-pre-wrap">{content}</p>
        </div>

        {/* Sources */}
        {sources && sources.length > 0 && (
          <div className="space-y-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ChevronDown
                className={cn(
                  "w-4 h-4 transition-transform",
                  showSources && "rotate-180",
                )}
              />
              <span>{sources.length} 个来源</span>
            </button>

            {showSources && (
              <div className="space-y-2">
                {sources.map((source, index) => (
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

function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="bg-card/50 border border-border rounded-lg p-3 cursor-pointer hover:border-primary/30 transition-colors"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2">
        <FileText className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm font-medium flex-1 truncate">
          Source {index}: {source.filename}
        </span>
        <span className="text-xs text-muted-foreground">
          相关度: {(source.score * 100).toFixed(0)}%
        </span>
      </div>

      {expanded && (
        <p className="mt-2 text-sm text-muted-foreground line-clamp-3">
          {source.content}
        </p>
      )}
    </div>
  );
}
