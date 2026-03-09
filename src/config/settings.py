from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class IndicatorSettings:
    rsi_period: int
    ema_fast_period: int
    ema_slow_period: int
    macd_fast_period: int
    macd_slow_period: int
    macd_signal_period: int
    atr_period: int


@dataclass(frozen=True)
class RiskSettings:
    entry_timeframe: str
    atr_column: str
    stop_atr_multiplier: float
    take_profit_atr_multiplier: float
    min_risk_reward_ratio: float


@dataclass(frozen=True)
class AISettings:
    model: str
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float
    environment: str
    run_every_n_cycles: int


@dataclass(frozen=True)
class PipelineSettings:
    symbols: tuple[str, ...]
    send_telegram: bool

    @property
    def default_symbol(self) -> str:
        """Backward-compatible primary symbol accessor."""
        return self.symbols[0]


@dataclass(frozen=True)
class SchedulerSettings:
    interval_minutes: int


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables and TOML."""

    binance_api_key: str | None
    binance_api_secret: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None

    pipeline: PipelineSettings
    indicators: IndicatorSettings
    risk: RiskSettings
    ai: AISettings
    scheduler: SchedulerSettings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_toml_config() -> dict[str, Any]:
    config_path = _project_root() / "config" / "trading.toml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Create config/trading.toml first."
        )

    with config_path.open("rb") as file:
        return tomllib.load(file)


def _parse_pipeline_symbols(pipeline_cfg: dict[str, Any]) -> tuple[str, ...]:
    raw_symbols = pipeline_cfg.get("symbols")

    if raw_symbols is None:
        default_symbol = str(pipeline_cfg.get("default_symbol", "BTCUSDT")).strip()
        symbols = [default_symbol]
    elif isinstance(raw_symbols, list):
        symbols = [str(item).strip() for item in raw_symbols]
    elif isinstance(raw_symbols, str):
        symbols = [item.strip() for item in raw_symbols.split(",")]
    else:
        raise ValueError("pipeline.symbols must be a list[str] or comma-separated string")

    cleaned = tuple(symbol for symbol in symbols if symbol)
    if not cleaned:
        raise ValueError("pipeline.symbols must include at least one symbol")

    return cleaned


def _build_settings() -> Settings:
    config = _load_toml_config()

    pipeline_cfg = config.get("pipeline", {})
    indicator_cfg = config.get("indicators", {})
    risk_cfg = config.get("risk", {})
    ai_cfg = config.get("ai", {})
    scheduler_cfg = config.get("scheduler", {})

    interval_minutes = int(scheduler_cfg.get("interval_minutes", 5))
    if interval_minutes <= 0:
        raise ValueError("scheduler.interval_minutes must be greater than 0")

    run_every_n_cycles = int(ai_cfg.get("run_every_n_cycles", 3))
    if run_every_n_cycles <= 0:
        raise ValueError("ai.run_every_n_cycles must be greater than 0")

    return Settings(
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_api_secret=os.getenv("BINANCE_API_SECRET"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        pipeline=PipelineSettings(
            symbols=_parse_pipeline_symbols(pipeline_cfg),
            send_telegram=bool(pipeline_cfg.get("send_telegram", True)),
        ),
        indicators=IndicatorSettings(
            rsi_period=int(indicator_cfg.get("rsi_period", 14)),
            ema_fast_period=int(indicator_cfg.get("ema_fast_period", 20)),
            ema_slow_period=int(indicator_cfg.get("ema_slow_period", 50)),
            macd_fast_period=int(indicator_cfg.get("macd_fast_period", 12)),
            macd_slow_period=int(indicator_cfg.get("macd_slow_period", 26)),
            macd_signal_period=int(indicator_cfg.get("macd_signal_period", 9)),
            atr_period=int(indicator_cfg.get("atr_period", 14)),
        ),
        risk=RiskSettings(
            entry_timeframe=str(risk_cfg.get("entry_timeframe", "5m")),
            atr_column=str(risk_cfg.get("atr_column", "atr_14")),
            stop_atr_multiplier=float(risk_cfg.get("stop_atr_multiplier", 1.5)),
            take_profit_atr_multiplier=float(
                risk_cfg.get("take_profit_atr_multiplier", 2.0)
            ),
            min_risk_reward_ratio=float(
                risk_cfg.get("min_risk_reward_ratio", 1.0)
            ),
        ),
        ai=AISettings(
            model=str(ai_cfg.get("model", "gpt-4.1-mini")),
            timeout_seconds=int(ai_cfg.get("timeout_seconds", 30)),
            max_retries=int(ai_cfg.get("max_retries", 2)),
            retry_backoff_seconds=float(ai_cfg.get("retry_backoff_seconds", 1.5)),
            environment=str(ai_cfg.get("environment", "paper")),
            run_every_n_cycles=run_every_n_cycles,
        ),
        scheduler=SchedulerSettings(
            interval_minutes=interval_minutes,
        ),
    )


settings = _build_settings()
