import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, FileText, File } from "lucide-react";
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
  onDelete?: () => void;
}

export function FileCard({
  filename,
  fileType,
  fileSize,
  status,
  onDelete,
}: FileCardProps) {
  const statusText: Record<string, string> = {
    pending: "等待中",
    parsing: "解析中",
    chunking: "分块中",
    embedding: "向量化",
    completed: "已完成",
    failed: "失败",
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

  return (
    <div className="card flex items-center gap-4 group">
      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
        {fileType === "pdf" ? (
          <FileText className="w-5 h-5 text-primary" />
        ) : (
          <File className="w-5 h-5 text-primary" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        <p className="font-medium truncate">{filename}</p>
        <p className="text-sm text-muted-foreground">
          {formatSize(fileSize)} • {fileType.toUpperCase()}
        </p>
      </div>

      <span className={cn("badge", statusClass[status])}>
        {statusText[status]}
      </span>

      {onDelete && (
        <button
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-destructive/10 text-destructive transition-all"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
