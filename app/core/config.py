"""
Application configuration loaded from environment variables.
Uses pydantic-settings for validation and .env file support.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    app_title: str = "Multi-Symbol Trading Signal API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    log_dir: str = "logs"

    # ── Symbols watched by the background worker ──────────────────────────────
    # Comma-separated list, overridable via SYMBOLS env var
    symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT"

    # ── Worker schedule ───────────────────────────────────────────────────────
    worker_interval_minutes: int = 5  # how often signals are refreshed

    # ── Market data source ────────────────────────────────────────────────────
    binance_base_url: str = "https://api.binance.com"
    kline_interval: str = "5m"        # candle timeframe
    kline_limit: int = 300            # history depth; ≥200 so all indicators warm up

    # ── ML model files ────────────────────────────────────────────────────────
    # Directory that holds <symbol_lower>_xgb_model.pkl and <symbol_lower>_features.pkl
    models_dir: Path = Path(".")

    # ── Signal thresholds ─────────────────────────────────────────────────────
    # Minimum model confidence to emit BUY or SELL (below → HOLD)
    min_signal_confidence: float = 0.40

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def symbol_list(self) -> list[str]:
        """Return symbols as an uppercased Python list."""
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]


# Module-level singleton – import this everywhere
settings = Settings()
