from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://search:search@localhost:5432/search_engine"
    redis_url: str = "redis://localhost:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    rate_limit_per_minute: int = 120
    cache_ttl_seconds: int = 300
    data_dir: str = "./data"
    models_dir: str = "./models"

    bm25_top_k: int = 50
    semantic_top_k: int = 50
    rrf_k: int = 60
    final_top_k: int = 10


settings = Settings()
