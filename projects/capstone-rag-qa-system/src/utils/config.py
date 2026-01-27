"""
DocuMind AI - 配置管理模块

加载和管理应用配置
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """应用基础配置"""

    name: str = "DocuMind AI"
    version: str = "1.0.0"
    debug: bool = True
    description: str = "智能文档问答系统"


class ServerConfig(BaseModel):
    """服务器配置"""

    host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_port: int = 8501


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""

    name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    dimension: int = 1024  # BGE-large 默认维度


class LLMConfig(BaseModel):
    """大语言模型配置"""

    name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "auto"
    quantization: str = "none"  # none, int4, int8
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class ModelsConfig(BaseModel):
    """模型配置"""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


class RetrievalConfig(BaseModel):
    """检索配置"""

    top_k: int = 5
    score_threshold: float = 0.3
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-large"


class ChunkingConfig(BaseModel):
    """分块配置"""

    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )


class StorageConfig(BaseModel):
    """存储配置"""

    upload_dir: str = "./data/uploads"
    index_dir: str = "./data/indices"
    db_path: str = "./data/db/documind.db"
    max_file_size: int = 52428800  # 50MB


class LoggingConfig(BaseModel):
    """日志配置"""

    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    rotation: str = "10 MB"
    retention: str = "7 days"
    log_dir: str = "./logs"


class Settings(BaseModel):
    """全局配置"""

    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    supported_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt", "md"]
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，默认为 configs/config.yaml

    Returns:
        Settings 配置对象
    """
    if config_path is None:
        # 默认配置文件路径
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return Settings(**config_dict)
    else:
        # 使用默认配置
        return Settings()


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def init_directories():
    """初始化必要的目录"""
    settings = get_settings()
    project_root = get_project_root()

    directories = [
        project_root / settings.storage.upload_dir,
        project_root / settings.storage.index_dir,
        project_root / Path(settings.storage.db_path).parent,
        project_root / settings.logging.log_dir,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
