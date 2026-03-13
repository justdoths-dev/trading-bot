from __future__ import annotations

import logging
import threading
from typing import Any

from src.notifications.trading_notifier import TradingNotifier
from src.services.trading_pipeline import TradingPipeline, TradingPipelineConfig

logger = logging.getLogger(__name__)


class TradingPipelineService:
    """Service wrapper for one full trading cycle execution."""

    def __init__(self) -> None:
        self._pipelines: dict[str, TradingPipeline] = {}
        self._pipeline_send_telegram_flags: dict[str, bool] = {}
        self._last_ai_results: dict[str, dict[str, Any]] = {}

        self._lock = threading.Lock()

        # NEW: notifier
        self._trading_notifier = TradingNotifier()

    def run(self, symbol: str, run_ai: bool, send_telegram: bool) -> dict[str, Any]:
        """Run one trading cycle and return the full pipeline result payload."""
        if not symbol:
            raise ValueError("symbol is required")

        logger.debug(
            "TradingPipelineService.run called: symbol=%s run_ai=%s send_telegram=%s",
            symbol,
            run_ai,
            send_telegram,
        )

        pipeline = self._get_pipeline(symbol=symbol, send_telegram=send_telegram)

        ai_result_override = None
        with self._lock:
            if run_ai is False:
                ai_result_override = self._last_ai_results.get(symbol)

        result = pipeline.run(
            run_ai=run_ai,
            ai_result_override=ai_result_override,
        )

        if run_ai is True:
            ai_result = result.get("ai_output", {}).get("result")
            if isinstance(ai_result, dict):
                with self._lock:
                    self._last_ai_results[symbol] = ai_result

        # -------- Trading Alert --------
        try:

            strategy_result = result.get("strategy_result", {})
            execution_result = result.get("execution_result", {})

            execution_allowed = execution_result.get("execution_allowed", False)

            if execution_allowed:

                self._trading_notifier.send_trading_alert(
                    symbol=symbol,
                    strategy=strategy_result.get("selected_strategy", "unknown"),
                    bias=strategy_result.get("bias", "unknown"),
                    entry_price=execution_result.get("entry_price"),
                    stop_loss=execution_result.get("stop_loss"),
                    take_profit=execution_result.get("take_profit"),
                    reason=strategy_result.get("reason", "n/a"),
                    mode="PAPER",
                )

        except Exception:
            logger.exception("TradingNotifier failed.")

        logger.debug(
            "TradingPipelineService.run completed: symbol=%s run_ai=%s",
            symbol,
            run_ai,
        )

        return result

    def _get_pipeline(self, symbol: str, send_telegram: bool) -> TradingPipeline:

        with self._lock:

            pipeline = self._pipelines.get(symbol)
            cached_send_telegram = self._pipeline_send_telegram_flags.get(symbol)

            if pipeline is None or cached_send_telegram != send_telegram:

                logger.info(
                    "Creating TradingPipeline instance: symbol=%s send_telegram=%s",
                    symbol,
                    send_telegram,
                )

                pipeline = TradingPipeline(
                    config=TradingPipelineConfig(
                        symbol=symbol,
                        send_telegram=send_telegram,
                    )
                )

                self._pipelines[symbol] = pipeline
                self._pipeline_send_telegram_flags[symbol] = send_telegram

        return pipeline