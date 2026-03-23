"""Configuration management for Power BI RAG Assistant."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration."""
    vector_db_type: str = "chromadb"
    vector_db_path: str = "./chroma_db"
    cache_db_path: str = "./data/cache.db"


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class UIConfig(BaseModel):
    """UI configuration."""
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 5
    retrieval_mode: str = "hybrid"
    hybrid_dense_weight: float = 0.6
    hybrid_lexical_weight: float = 0.4
    min_relevance_score: float = 0.3
    temperature: float = 0.1
    max_tokens_per_request: int = 4000


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model_dev: str = "claude-3-haiku-20240307"
    llm_model_prod: str = "claude-3-5-sonnet-20241022"
    
    # Application Settings
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Cost Controls
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()
    ui: UIConfig = UIConfig()
    processing: ProcessingConfig = ProcessingConfig()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    @property
    def llm_model(self) -> str:
        """Get the appropriate LLM model based on environment."""
        return self.llm_model_dev if self.is_development else self.llm_model_prod


# Global settings instance
settings = Settings()
