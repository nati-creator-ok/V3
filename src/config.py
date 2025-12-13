"""
Configuration management for the Table Extraction System.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = "table-extraction-system"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/table_extraction"
    database_pool_size: int = 10
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # Storage
    storage_type: Literal["local", "s3", "minio"] = "local"
    storage_path: Path = Path("./data/uploads")
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    minio_endpoint: str = "localhost:9000"
    
    # Model Settings
    model_device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    table_detection_model: str = "yolov8"
    table_detection_weights: Path = Path("./models/weights/table_detector.pt")
    structure_model: str = "tableformer"
    structure_weights: Path = Path("./models/weights/structure_model.pt")
    ocr_engine: Literal["easyocr", "tesseract", "paddle"] = "easyocr"
    ocr_languages: List[str] = Field(default=["ko", "en"])
    
    # Processing
    max_image_size: int = 4096
    batch_size: int = 4
    num_workers: int = 4
    confidence_threshold: float = 0.5
    
    # Monitoring
    prometheus_port: int = 9090
    enable_sentry: bool = False
    sentry_dsn: str = ""
    
    # Security
    secret_key: str = "your-super-secret-key-change-in-production"
    api_key_enabled: bool = False
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    
    @field_validator("ocr_languages", mode="before")
    @classmethod
    def parse_languages(cls, v):
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
    @property
    def use_gpu(self) -> bool:
        return self.model_device in ("cuda", "mps")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
