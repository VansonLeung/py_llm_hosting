from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API settings
    llm_api_key: str | None = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Data settings
    data_file: str = "servers.json"

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    # External services
    tool_service_url: str | None = None

    class Config:
        env_prefix = "LLM_"
        case_sensitive = False

# Global settings instance
settings = Settings()
