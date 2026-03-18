from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from src.research.edge_selection_engine import run_edge_selection_engine
from src.research.edge_selection_input_mapper import map_edge_selection_input
from src.research.edge_selection_shadow_writer import write_edge_selection_shadow_output
from src.services.trading_pipeline import TradingPipeline, TradingPipelineConfig

logger = logging.getLogger(__name__)

RESEARCH_REPORTS_DIR = Path("logs/research_reports")


class TradingPipelineService:
    """Service wrapper for one full trading cycle execution."""

    def __init__(self) -> None:
        self._pipelines: dict[str, TradingPipeline] = {}
        self._pipeline_send_telegram_flags: dict[str, bool] = {}
        self._last_ai_results: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

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

        self._run_shadow_observation(symbol=symbol)

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

    def _run_shadow_observation(self, symbol: str) -> None:
        """
        Produce a shadow edge-selection output after a trading cycle.

        This path must never interrupt the main trading pipeline. Any exception is
        logged and suppressed so runtime trading behavior remains unchanged.
        """
        try:
            mapped_payload = map_edge_selection_input(RESEARCH_REPORTS_DIR)
            shadow_result = run_edge_selection_engine(mapped_payload)
            output_path = write_edge_selection_shadow_output(shadow_result)

            logger.info(
                "Shadow observation output written: symbol=%s status=%s candidates=%s path=%s",
                symbol,
                shadow_result.get("selection_status"),
                shadow_result.get("candidates_considered"),
                output_path,
            )
        except Exception:
            logger.exception(
                "Shadow observation generation failed after trading cycle: symbol=%s",
                symbol,
            )