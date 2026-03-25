from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from src.notifications.shadow_event_notifier import ShadowEventNotifier
from src.research.edge_selection_engine import run_edge_selection_engine
from src.research.edge_selection_input_mapper import map_edge_selection_input
from src.research.edge_selection_shadow_writer import write_edge_selection_shadow_output
from src.services.trading_pipeline import TradingPipeline, TradingPipelineConfig

logger = logging.getLogger(__name__)

RESEARCH_REPORTS_DIR = Path("logs/research_reports")
EDGE_SELECTION_MAPPER_VERSION = "edge_selection_input_mapper_v1"
EDGE_SELECTION_ENGINE_VERSION = "edge_selection_engine_v1"
FORCE_SHADOW_FAILURE_ENV_VAR = "EDGE_SELECTION_FORCE_SHADOW_FAILURE"


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

        shadow_context = self._run_shadow_observation(trigger_symbol=symbol)
        enriched_record = self._enrich_trade_analysis_log_record(
            pipeline=pipeline,
            symbol=symbol,
            log_record=result.get("log_record"),
            shadow_context=shadow_context,
        )
        if enriched_record is not None:
            result["log_record"] = enriched_record

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

    def _run_shadow_observation(self, trigger_symbol: str) -> dict[str, Any]:
        """
        Produce a shadow edge-selection output after a trading cycle.

        This path must never interrupt the main trading pipeline. Any exception is
        logged and suppressed so runtime trading behavior remains unchanged.

        Note:
        The shadow observation is generated from the research reports directory,
        so `trigger_symbol` identifies the trading cycle that triggered this work,
        not necessarily the symbol identity of the resulting shadow selection.

        Returns a context dict in both success and failure cases so the trade-analysis
        row can always be enriched with replay-related metadata.
        """
        base_metadata: dict[str, Any] = {
            "mapper_version": EDGE_SELECTION_MAPPER_VERSION,
            "engine_version": EDGE_SELECTION_ENGINE_VERSION,
            "replay_ready": False,
            "shadow_status": "not_started",
            "trigger_symbol": trigger_symbol,
            "reports_dir": str(RESEARCH_REPORTS_DIR),
            "shadow_output_path": None,
            "error_type": None,
            "error_message": None,
        }

        try:
            if self._is_forced_shadow_failure_enabled():
                raise RuntimeError(
                    f"Forced shadow observation failure via env var {FORCE_SHADOW_FAILURE_ENV_VAR}"
                )

            mapped_payload = map_edge_selection_input(RESEARCH_REPORTS_DIR)
            shadow_result = run_edge_selection_engine(mapped_payload)
            output_path = write_edge_selection_shadow_output(shadow_result)

            logger.info(
                "Shadow observation output written: trigger_symbol=%s status=%s candidates=%s path=%s",
                trigger_symbol,
                shadow_result.get("selection_status"),
                shadow_result.get("candidates_considered"),
                output_path,
            )

            self._notify_shadow_events(
                trigger_symbol=trigger_symbol,
                shadow_output_path=output_path,
            )

            metadata = dict(base_metadata)
            metadata.update(
                {
                    "replay_ready": True,
                    "shadow_status": "success",
                    "shadow_output_path": str(output_path),
                    "mapper_generated_at": mapped_payload.get("generated_at")
                    if isinstance(mapped_payload, dict)
                    else None,
                    "selection_status": shadow_result.get("selection_status")
                    if isinstance(shadow_result, dict)
                    else None,
                }
            )

            return {
                "edge_selection_mapper_payload": mapped_payload,
                "edge_selection_output": shadow_result,
                "edge_selection_metadata": metadata,
            }
        except Exception as exc:
            logger.exception(
                "Shadow observation generation failed after trading cycle: trigger_symbol=%s",
                trigger_symbol,
            )

            metadata = dict(base_metadata)
            metadata.update(
                {
                    "shadow_status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            return {
                "edge_selection_mapper_payload": None,
                "edge_selection_output": None,
                "edge_selection_metadata": metadata,
            }

    def _enrich_trade_analysis_log_record(
        self,
        *,
        pipeline: TradingPipeline,
        symbol: str,
        log_record: Any,
        shadow_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(log_record, dict):
            return None

        if not isinstance(shadow_context, dict):
            return None

        logged_at = log_record.get("logged_at")
        if not isinstance(logged_at, str) or not logged_at.strip():
            return None

        try:
            enriched_record = pipeline.logger.enrich_latest_record(
                symbol=symbol,
                logged_at=logged_at,
                edge_selection_mapper_payload=shadow_context.get(
                    "edge_selection_mapper_payload"
                ),
                edge_selection_output=shadow_context.get("edge_selection_output"),
                edge_selection_metadata=shadow_context.get("edge_selection_metadata"),
            )
            logger.debug(
                "Trade-analysis record enriched with replay-ready edge-selection context: symbol=%s logged_at=%s",
                symbol,
                logged_at,
            )
            return enriched_record
        except Exception:
            logger.exception(
                "Trade-analysis record enrichment failed after shadow observation: symbol=%s logged_at=%s",
                symbol,
                logged_at,
            )
            return None

    def _notify_shadow_events(
        self,
        *,
        trigger_symbol: str,
        shadow_output_path: Path,
    ) -> None:
        """Attempt observer-only shadow event notifications without affecting runtime."""
        try:
            result = ShadowEventNotifier(
                shadow_output_path=shadow_output_path
            ).notify_latest_events()

            if result.get("event_count", 0) > 0 or result.get("failure_count", 0) > 0:
                logger.info(
                    "Shadow event notification result: trigger_symbol=%s events=%s sent=%s failures=%s",
                    trigger_symbol,
                    result.get("event_count", 0),
                    result.get("sent_count", 0),
                    result.get("failure_count", 0),
                )
        except Exception:
            logger.exception(
                "Shadow event notification failed after trading cycle: trigger_symbol=%s",
                trigger_symbol,
            )

    def _is_forced_shadow_failure_enabled(self) -> bool:
        value = os.getenv(FORCE_SHADOW_FAILURE_ENV_VAR, "")
        return value.strip().lower() in {"1", "true", "yes", "on"}