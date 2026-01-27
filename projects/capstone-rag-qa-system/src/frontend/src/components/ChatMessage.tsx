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
          <p className="whitespace-pre-wrap leading-relaxed">{content}</p>
        </div>

        {/* Sources */}
        {sources && sources.length > 0 && (
          <div className="w-full space-y-2">
            {sources.slice(0, 3).map((source, index) => (
              <SourceCard key={index} source={source} index={index + 1} />
            ))}
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
      className="bg-card/60 backdrop-blur-sm border border-border/50 rounded-xl p-3 cursor-pointer hover:border-primary/30 transition-all group"
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-1">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <FileText className="w-4 h-4 text-primary shrink-0" />
          <span className="text-sm font-medium truncate">
            Source {index}: {source.filename}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            Relevance: {(source.score * 100).toFixed(0)}%
          </span>
          <ChevronDown
            className={cn(
              "w-4 h-4 text-muted-foreground transition-transform",
              expanded && "rotate-180",
            )}
          />
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-border/50 rounded-full overflow-hidden mb-2">
        <div
          className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all"
          style={{ width: `${source.score * 100}%` }}
        />
      </div>

      {/* Content preview */}
      <p
        className={cn(
          "text-sm text-muted-foreground transition-all",
          expanded ? "line-clamp-none" : "line-clamp-2",
        )}
      >
        ...{source.content}...
      </p>
    </div>
  );
}
