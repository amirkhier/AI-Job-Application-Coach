"""
Centralised application configuration via Pydantic BaseSettings.

All environment variables are loaded here once, validated at import time,
and exposed through the module-level ``settings`` singleton.  Every other
module should ``from app.config import settings`` instead of calling
``os.getenv`` directly.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — sourced from environment variables / .env file."""

    # ── Environment ──────────────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "text"  # json for prod, text for dev

    # ── API ───────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: str = "*"  # comma-separated in production
    API_KEY: Optional[str] = None  # set in production to require auth

    # ── Database (MySQL) ──────────────────────────────────────────────────
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "job_coach"

    # ── Redis / Celery ────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # ── ChromaDB ──────────────────────────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIRECTORY: str = "./chroma"

    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""

    # ── Rate limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── Feature flags ─────────────────────────────────────────────────────
    USE_LANGGRAPH: bool = False

    # ── Derived helpers ───────────────────────────────────────────────────
    @property
    def cors_origins(self) -> List[str]:
        """Parse comma-separated ALLOWED_ORIGINS into a list."""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()
