"""Periodic scheduler for running the trading pipeline."""

from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from src.config.settings import settings
from src.notifications.alert_notifier import AlertNotifier
from src.services.trading_pipeline_service import TradingPipelineService

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Run TradingPipeline on a fixed minute interval."""

    def __init__(
        self,
        symbol: str,
        interval_minutes: int,
        send_telegram: bool,
    ) -> None:

        if not symbol:
            raise ValueError("symbol is required")

        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than 0")

        self.symbol = symbol
        self.interval_minutes = interval_minutes
        self.send_telegram = send_telegram

        self._pipeline_service = TradingPipelineService()
        self._alert_notifier = AlertNotifier()

        self._scheduler = BackgroundScheduler()
        self._job_id = f"trading_pipeline_{self.symbol.lower()}"
        self._run_counter = 0

    def start(self) -> None:
        """Start periodic trading pipeline execution."""

        if self._scheduler.running:
            logger.info("TradingScheduler is already running.")
            return

        logger.info(
            "Running initial trading pipeline immediately: symbol=%s send_telegram=%s",
            self.symbol,
            self.send_telegram,
        )

        # Run immediately once
        self._run_job()

        # Schedule periodic runs
        self._scheduler.add_job(
            self._run_job,
            trigger="interval",
            minutes=self.interval_minutes,
            id=self._job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )

        self._scheduler.start()

        logger.info(
            "TradingScheduler started: symbol=%s interval=%smin send_telegram=%s ai_every_n_cycles=%s",
            self.symbol,
            self.interval_minutes,
            self.send_telegram,
            settings.ai.run_every_n_cycles,
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown scheduler safely."""

        if not self._scheduler.running:
            logger.info("TradingScheduler is not running.")
            return

        self._scheduler.shutdown(wait=wait)

        logger.info("TradingScheduler stopped.")

    def _run_job(self) -> None:

        self._run_counter += 1
        run_ai = self._should_run_ai()

        logger.info(
            "Trading pipeline run started: symbol=%s cycle=%s run_ai=%s",
            self.symbol,
            self._run_counter,
            run_ai,
        )

        try:

            result: dict[str, Any] = self._pipeline_service.run(
                symbol=self.symbol,
                run_ai=run_ai,
                send_telegram=self.send_telegram,
            )

            strategy_result = result.get("strategy_result", {})
            execution_result = result.get("execution_result", {})
            ai_result = result.get("ai_output", {}).get("result", {})
            ai_analysis = ai_result.get("analysis", {})
            telegram_send_result = result.get("telegram_send_result", {})

            signal = strategy_result.get("signal", "unknown")
            action = execution_result.get("action", "unknown")
            ai_stance = ai_analysis.get("final_stance", "unknown")
            telegram_sent = telegram_send_result.get("sent", False)

            logger.info(
                "Trading pipeline run finished: symbol=%s cycle=%s signal=%s action=%s ai_stance=%s telegram_sent=%s",
                self.symbol,
                self._run_counter,
                signal,
                action,
                ai_stance,
                telegram_sent,
            )

        except Exception as e:

            logger.exception(
                "Trading pipeline run failed: symbol=%s cycle=%s",
                self.symbol,
                self._run_counter,
            )

            # Send alert to Telegram
            try:

                self._alert_notifier.send_error_alert(
                    source="trading_pipeline",
                    message="Trading pipeline execution failed",
                    details=str(e),
                )

            except Exception:
                logger.exception("AlertNotifier failed.")

    def _should_run_ai(self) -> bool:

        if self._run_counter == 1:
            return True

        return self._run_counter % settings.ai.run_every_n_cycles == 0
