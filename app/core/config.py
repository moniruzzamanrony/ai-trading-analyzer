from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_title: str = "Trading AI Analyzer API"
    app_version: str = "2.0.0"
    log_level: str = "INFO"
    log_dir: str = "logs"

    binance_base_url: str = "https://api.binance.com"
    kline_interval: str = "5m"
    kline_limit: int = 300

    models_dir: Path = Path(".")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
