from __future__ import annotations

import logging
import time
from typing import Any

from src.notifications.alert_notifier import AlertNotifier

LOGGER = logging.getLogger(__name__)


class CronHealthReporter:
    """Small helper for cron-style job success/failure reporting."""

    def __init__(self, job_name: str) -> None:
        self.job_name = job_name
        self.started_at = time.time()
        self.alert_notifier = AlertNotifier()

    def success(self, details: dict[str, Any] | None = None) -> dict[str, Any]:
        duration_seconds = round(time.time() - self.started_at, 3)

        payload: dict[str, Any] = {
            "job": self.job_name,
            "status": "success",
            "duration_seconds": duration_seconds,
        }

        if details:
            payload.update(details)

        LOGGER.info("Cron job success: %s", payload)
        return payload

    def failure(self, error: Exception, message: str) -> dict[str, Any]:
        duration_seconds = round(time.time() - self.started_at, 3)

        payload = {
            "job": self.job_name,
            "status": "error",
            "duration_seconds": duration_seconds,
            "error": str(error),
        }

        LOGGER.exception("Cron job failed: %s", self.job_name)

        try:
            self.alert_notifier.send_error_alert(
                source=self.job_name,
                message=message,
                details=str(error),
            )
        except Exception:
            LOGGER.exception("AlertNotifier failed while reporting cron job failure.")

        return payload