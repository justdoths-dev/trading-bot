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

    scheduler = TradingScheduler(
        symbol=settings.pipeline.default_symbol,
        interval_minutes=settings.scheduler.interval_minutes,
        send_telegram=settings.pipeline.send_telegram,
    )

    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutdown signal received.")
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
