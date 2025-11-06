import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API settings
    llm_api_key: Optional[str] = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Data settings
    data_file: str = "servers.json"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    class Config:
        env_prefix = "LLM_"
        case_sensitive = False

# Global settings instance
settings = Settings()