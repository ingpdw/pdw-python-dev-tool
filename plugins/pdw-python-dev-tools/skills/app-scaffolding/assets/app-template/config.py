"""Application configuration via pydantic-settings.

Settings are loaded from environment variables and an optional `.env` file.
Environment variables take precedence over values in the `.env` file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration.

    All fields map to environment variables. Nested prefixes are not used so
    that each variable name is explicit and easy to grep in deployment configs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Application --
    app_name: str = "FastAPI App"
    app_version: str = "0.1.0"
    debug: bool = False

    # -- Server --
    host: str = "0.0.0.0"
    port: int = 8000

    # -- Database --
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/appdb"

    # -- Authentication --
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 30

    # -- CORS --
    cors_origins: list[str] = []

    # -- External services --
    redis_url: str = "redis://localhost:6379/0"
