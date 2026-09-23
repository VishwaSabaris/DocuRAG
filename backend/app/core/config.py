from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


BASE_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    app_name: str = "DocuRAG"
    app_version: str = "0.1.0"

    environment: str = "development"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8000

    ollama_base_url: str = (
        "http://localhost:11434"
    )
    ollama_model: str = (
        "gemma3:4b-it-q4_K_M"
    )

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "docurag"
    postgres_user: str = "docurag"
    postgres_password: str = (
        "docurag_dev_password"
    )

    retrieval_min_score: float = 0.30

    max_context_characters: int = 6000

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
