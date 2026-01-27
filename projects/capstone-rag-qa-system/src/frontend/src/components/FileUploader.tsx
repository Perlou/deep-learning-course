import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X } from "lucide-react";
import { cn } from "../lib/utils";

interface FileUploaderProps {
  onUpload: (files: File[]) => void;
  uploading?: boolean;
  accept?: Record<string, string[]>;
}

const defaultAccept = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    ".docx",
  ],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
};

export function FileUploader({
  onUpload,
  uploading = false,
  accept = defaultAccept,
}: FileUploaderProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles);
      }
    },
    [onUpload],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    disabled: uploading,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all",
        isDragActive
          ? "border-primary bg-primary/10"
          : "border-border hover:border-primary/50",
        uploading && "opacity-50 cursor-not-allowed",
      )}
    >
      <input {...getInputProps()} />
      <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
      <p className="text-foreground font-medium">
        {isDragActive ? "释放文件以上传" : "拖拽文件到此处，或点击选择"}
      </p>
      <p className="mt-2 text-sm text-muted-foreground">
        支持格式: PDF, DOCX, TXT, MD
      </p>
      {uploading && (
        <p className="mt-4 text-primary animate-pulse">上传中...</p>
      )}
    </div>
  );
}

interface FileCardProps {
  filename: string;
  fileType: string;
  fileSize: number;
  status:
    | "pending"
    | "parsing"
    | "chunking"
    | "embedding"
    | "completed"
    | "failed";
  createdAt?: string;
  onDelete?: () => void;
}

export function FileCard({
  filename,
  fileType,
  fileSize,
  status,
  createdAt,
  onDelete,
}: FileCardProps) {
  const statusText: Record<string, string> = {
    pending: "Pending",
    parsing: "Parsing",
    chunking: "Chunking",
    embedding: "Embedding",
    completed: "Completed",
    failed: "Failed",
  };

  const statusClass: Record<string, string> = {
    pending: "badge-warning",
    parsing: "badge-warning",
    chunking: "badge-warning",
    embedding: "badge-warning",
    completed: "badge-success",
    failed: "badge-error",
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "";
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  // File type specific styling
  const getFileTypeStyle = () => {
    switch (fileType.toLowerCase()) {
      case "pdf":
        return { bg: "bg-red-500/20", text: "text-red-400", label: "PDF" };
      case "docx":
        return { bg: "bg-blue-500/20", text: "text-blue-400", label: "DOCX" };
      case "txt":
        return { bg: "bg-gray-500/20", text: "text-gray-400", label: "TXT" };
      case "md":
        return { bg: "bg-purple-500/20", text: "text-purple-400", label: "MD" };
      default:
        return {
          bg: "bg-primary/20",
          text: "text-primary",
          label: fileType.toUpperCase(),
        };
    }
  };

  const fileStyle = getFileTypeStyle();
  const isProcessing = ["pending", "parsing", "chunking", "embedding"].includes(
    status,
  );

  return (
    <div className="bg-card/60 backdrop-blur-sm border border-border/50 rounded-xl p-4 flex items-center gap-4 group hover:border-primary/30 transition-all">
      {/* File Type Icon */}
      <div
        className={cn(
          "w-12 h-12 rounded-xl flex items-center justify-center font-bold text-xs",
          fileStyle.bg,
        )}
      >
        <span className={fileStyle.text}>{fileStyle.label}</span>
      </div>

      {/* File Info */}
      <div className="flex-1 min-w-0">
        <p className="font-medium truncate text-foreground">{filename}</p>
        <p className="text-sm text-muted-foreground">
          {formatSize(fileSize)} • {createdAt && formatDate(createdAt)}
        </p>
      </div>

      {/* Status Badge */}
      <div className="flex items-center gap-3">
        <span
          className={cn("badge flex items-center gap-1.5", statusClass[status])}
        >
          {isProcessing && (
            <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
          )}
          {statusText[status]}
        </span>

        {/* Delete Button */}
        {onDelete && (
          <button
            onClick={onDelete}
            className="opacity-0 group-hover:opacity-100 p-2 rounded-lg hover:bg-destructive/10 text-destructive transition-all"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
