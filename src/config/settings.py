from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    binance_api_key: str | None = os.getenv("BINANCE_API_KEY")
    binance_api_secret: str | None = os.getenv("BINANCE_API_SECRET")


settings = Settings()