from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ai.ai_service import AIService, AIServiceConfig
from src.config.settings import settings
from src.data.multi_timeframe_loader import MultiTimeframeLoader
from src.execution.execution_engine import ExecutionEngine
from src.exchange.binance_client import BinanceMarketDataClient
from src.indicators.indicator_engine import IndicatorEngine
from src.notifications.trading_notifier import TradingNotifier
from src.risk.risk_manager import RiskManager
from src.storage.trade_analysis_logger import (
    TradeAnalysisLogger,
    TradeAnalysisLoggerConfig,
)
from src.strategy.strategy_engine import StrategyEngine


@dataclass
class TimeframeConfig:
    timeframe: str
    limit: int


@dataclass
class TradingPipelineConfig:
    symbol: str
    send_telegram: bool


class TradingPipeline:

    def __init__(
        self,
        config: TradingPipelineConfig | None = None,
        trading_notifier: TradingNotifier | None = None,
    ) -> None:

        self.config = config or TradingPipelineConfig(
            symbol=settings.pipeline.default_symbol,
            send_telegram=settings.pipeline.send_telegram,
        )

        self.client = BinanceMarketDataClient()
        self.loader = MultiTimeframeLoader(client=self.client)

        self.indicator_engine = IndicatorEngine(
            rsi_period=settings.indicators.rsi_period,
            ema_fast_period=settings.indicators.ema_fast_period,
            ema_slow_period=settings.indicators.ema_slow_period,
            macd_fast_period=settings.indicators.macd_fast_period,
            macd_slow_period=settings.indicators.macd_slow_period,
            macd_signal_period=settings.indicators.macd_signal_period,
            atr_period=settings.indicators.atr_period,
        )

        self.strategy_engine = StrategyEngine()

        self.risk_manager = RiskManager(
            entry_timeframe=settings.risk.entry_timeframe,
            atr_column=settings.risk.atr_column,
            stop_atr_multiplier=settings.risk.stop_atr_multiplier,
            take_profit_atr_multiplier=settings.risk.take_profit_atr_multiplier,
            min_risk_reward_ratio=settings.risk.min_risk_reward_ratio,
        )

        self.execution_engine = ExecutionEngine(
            symbol=self.config.symbol,
            execution_mode=settings.ai.environment,
        )

        self.ai_service = AIService(
            config=AIServiceConfig(
                model=settings.ai.model,
                timeout_seconds=settings.ai.timeout_seconds,
                max_retries=settings.ai.max_retries,
                retry_backoff_seconds=settings.ai.retry_backoff_seconds,
                environment=settings.ai.environment,
                symbol=self.config.symbol,
            )
        )

        self.logger = TradeAnalysisLogger(
            config=TradeAnalysisLoggerConfig()
        )

        self.trading_notifier = trading_notifier or TradingNotifier()

    def run(
        self,
        run_ai: bool = True,
        ai_result_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        symbol = self.config.symbol

        timeframe_configs = self._build_timeframe_configs()

        raw_data = self.loader.load(
            symbol=symbol,
            configs=timeframe_configs,
        )

        enriched_data = self.indicator_engine.enrich(raw_data)

        strategy_result = self.strategy_engine.evaluate(enriched_data)
        selected_result = strategy_result["selected_result"]

        risk_result = self.risk_manager.evaluate(selected_result, enriched_data)

        execution_result = self.execution_engine.create_plan(
            selected_result,
            risk_result,
        )

        ai_output = self._build_ai_output(
            enriched_data=enriched_data,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            run_ai=run_ai,
            ai_result_override=ai_result_override,
        )

        log_record = self.logger.log(
            symbol=symbol,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_output["result"],
        )

        telegram_send_result = self._maybe_send_telegram(
            symbol=symbol,
            strategy_result=selected_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_output["result"],
        )

        return {
            "symbol": symbol,
            "raw_data": raw_data,
            "enriched_data": enriched_data,
            "strategy_result": strategy_result,
            "selected_result": selected_result,
            "risk_result": risk_result,
            "execution_result": execution_result,
            "ai_output": ai_output,
            "log_record": log_record,
            "telegram_send_result": telegram_send_result,
        }

    def _maybe_send_telegram(
        self,
        symbol: str,
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
        ai_result: dict[str, Any],
    ) -> dict[str, Any]:

        if not self.config.send_telegram:
            return {
                "sent": False,
                "reason": "Telegram sending disabled by pipeline config.",
            }

        execution_allowed = execution_result.get("execution_allowed", False)

        if execution_allowed is not True:
            return {
                "sent": False,
                "reason": "Execution not allowed, Telegram skipped.",
            }

        return self.trading_notifier.send_pipeline_alert(
            symbol=symbol,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_result,
        )

    @staticmethod
    def _build_timeframe_configs() -> list[TimeframeConfig]:

        return [
            TimeframeConfig(timeframe="1m", limit=100),
            TimeframeConfig(timeframe="5m", limit=100),
            TimeframeConfig(timeframe="15m", limit=100),
            TimeframeConfig(timeframe="1h", limit=100),
            TimeframeConfig(timeframe="4h", limit=100),
            TimeframeConfig(timeframe="1d", limit=100),
        ]
