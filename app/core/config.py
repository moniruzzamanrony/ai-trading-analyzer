from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_title: str = "Trading AI Analyzer API"
    app_version: str = "2.1.0"
    log_level: str = "INFO"
    log_dir: str = "logs"

    binance_base_url: str = "https://api.binance.com"
    kline_interval: str = "15m"
    kline_limit: int = 300

    models_dir: Path = Path(".")

    # Round-trip trading fee (% of notional). Net `profitPercentage` returned
    # to the bot subtracts this. Override via TRADING_FEE_PCT to match the
    # bot's actual fee tier (e.g. 0.15 when paying in BNB).
    trading_fee_pct: float = 0.2

    mailjet_api_key: str = ""
    mailjet_api_secret: str = ""
    mailjet_from_email: str = ""
    mailjet_error_recipient: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
