"""Application configuration settings."""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/exoplanet.db"
    
    # Model paths
    MODEL_CHECKPOINT_PATH: str = "./models/checkpoints"
    DEFAULT_MODEL: str = "spectrum_recovery_v1"
    
    # Data paths
    RAW_DATA_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"
    SYNTHETIC_DATA_PATH: str = "./data/synthetic"
    
    # GPU
    USE_GPU: bool = True
    CUDA_VISIBLE_DEVICES: str = "0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()
