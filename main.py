from __future__ import annotations

import logging
import time

from src.config.settings import settings
from src.services.trading_scheduler import TradingScheduler

STARTUP_STAGGER_SECONDS = 150


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger = logging.getLogger(__name__)

    interval_seconds = settings.scheduler.interval_minutes * 60
    if interval_seconds <= 0:
        raise ValueError("scheduler.interval_minutes must be greater than 0")

    schedulers: list[TradingScheduler] = []
    used_offsets: dict[int, str] = {}

    for index, symbol in enumerate(settings.pipeline.symbols):
        raw_offset_seconds = index * STARTUP_STAGGER_SECONDS
        offset_seconds = raw_offset_seconds % interval_seconds

        if offset_seconds in used_offsets:
            logger.warning(
                "Scheduler offset collision detected: symbol=%s offset=%ss collides with symbol=%s. "
                "This means two symbols will share the same aligned schedule slot.",
                symbol,
                offset_seconds,
                used_offsets[offset_seconds],
            )
        else:
            used_offsets[offset_seconds] = symbol

        schedulers.append(
            TradingScheduler(
                symbol=symbol,
                interval_minutes=settings.scheduler.interval_minutes,
                send_telegram=settings.pipeline.send_telegram,
                offset_seconds=offset_seconds,
            )
        )

    for scheduler in schedulers:
        logger.info(
            "Starting scheduler for symbol=%s with aligned offset=%ss",
            scheduler.symbol,
            scheduler.offset_seconds,
        )
        scheduler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        for scheduler in reversed(schedulers):
            scheduler.shutdown()


if __name__ == "__main__":
    main()
