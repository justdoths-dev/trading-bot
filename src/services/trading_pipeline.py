from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.ai.ai_service import AIService, AIServiceConfig
from src.config.settings import settings
from src.data.multi_timeframe_loader import MultiTimeframeLoader
from src.execution.execution_engine import ExecutionEngine
from src.exchange.binance_client import BinanceMarketDataClient
from src.indicators.indicator_engine import IndicatorEngine
from src.risk.risk_manager import RiskManager
from src.storage.trade_analysis_logger import (
    TradeAnalysisLogger,
    TradeAnalysisLoggerConfig,
)
from src.strategy.strategy_engine import StrategyEngine
from src.telegram.telegram_formatter import TelegramFormatter
from src.telegram.telegram_sender import TelegramSender


@dataclass
class TimeframeConfig:
    """Configuration for one timeframe data request."""
    timeframe: str
    limit: int


@dataclass
class TradingPipelineConfig:
    """Configuration for the trading pipeline."""
    symbol: str = "BTCUSDT"
    send_telegram: bool = True


class TradingPipeline:
    """Run the full trading analysis pipeline end-to-end."""

    def __init__(self, config: TradingPipelineConfig | None = None) -> None:
        self.config = config or TradingPipelineConfig()

        self.client = BinanceMarketDataClient()
        self.loader = MultiTimeframeLoader(client=self.client)

        self.indicator_engine = IndicatorEngine()
        self.strategy_engine = StrategyEngine()
        self.risk_manager = RiskManager()

        self.execution_engine = ExecutionEngine(
            symbol=self.config.symbol,
            execution_mode="paper",
        )

        self.ai_service = AIService(
            config=AIServiceConfig(symbol=self.config.symbol)
        )

        self.logger = TradeAnalysisLogger(
            config=TradeAnalysisLoggerConfig()
        )

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline and return all outputs."""
        symbol = self.config.symbol

        if not settings.binance_api_key or not settings.binance_api_secret:
            print("BINANCE API key missing (public data still works).")

        timeframe_configs = self._build_timeframe_configs()

        raw_data = self.loader.load(
            symbol=symbol,
            configs=timeframe_configs,
        )

        enriched_data = self.indicator_engine.enrich(raw_data)

        strategy_result = self.strategy_engine.evaluate(enriched_data)
        risk_result = self.risk_manager.evaluate(strategy_result, enriched_data)

        execution_result = self.execution_engine.create_plan(
            strategy_result,
            risk_result,
        )

        ai_output = self.ai_service.run(
            enriched_data=enriched_data,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
        )

        log_record = self.logger.log(
            symbol=symbol,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_output["result"],
        )

        telegram_formatter = TelegramFormatter(
            symbol=symbol,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_output["result"],
        )
        telegram_message = telegram_formatter.format_message()

        telegram_send_result = self._send_telegram_if_enabled(telegram_message)

        return {
            "symbol": symbol,
            "raw_data": raw_data,
            "enriched_data": enriched_data,
            "strategy_result": strategy_result,
            "risk_result": risk_result,
            "execution_result": execution_result,
            "ai_output": ai_output,
            "log_record": log_record,
            "telegram_message": telegram_message,
            "telegram_send_result": telegram_send_result,
        }

    def _send_telegram_if_enabled(self, telegram_message: str) -> dict[str, Any]:
        """
        Send Telegram message if enabled and credentials are available.

        Telegram failures should not break the full trading pipeline.
        """
        if not self.config.send_telegram:
            return {
                "sent": False,
                "reason": "Telegram sending disabled by pipeline config.",
            }

        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return {
                "sent": False,
                "reason": "Telegram credentials missing.",
            }

        try:
            sender = TelegramSender(
                bot_token=settings.telegram_bot_token,
                chat_id=settings.telegram_chat_id,
            )
            response = sender.send_message(telegram_message)
            return {
                "sent": True,
                "reason": "Telegram message sent successfully.",
                "response": response,
            }
        except Exception as exc:
            return {
                "sent": False,
                "reason": f"Telegram send failed: {exc}",
            }

    @staticmethod
    def _build_timeframe_configs() -> list[TimeframeConfig]:
        """Return the default multi-timeframe configuration set."""
        return [
            TimeframeConfig(timeframe="1m", limit=100),
            TimeframeConfig(timeframe="5m", limit=100),
            TimeframeConfig(timeframe="15m", limit=100),
            TimeframeConfig(timeframe="1h", limit=100),
            TimeframeConfig(timeframe="4h", limit=100),
            TimeframeConfig(timeframe="1d", limit=100),
        ]