from __future__ import annotations

import logging
import time

from src.config.settings import settings
from src.services.trading_scheduler import TradingScheduler


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger = logging.getLogger(__name__)

    schedulers = [
        TradingScheduler(
            symbol=symbol,
            interval_minutes=settings.scheduler.interval_minutes,
            send_telegram=settings.pipeline.send_telegram,
        )
        for symbol in settings.pipeline.symbols
    ]

    for scheduler in schedulers:
        logger.info("Starting scheduler for symbol=%s", scheduler.symbol)
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
