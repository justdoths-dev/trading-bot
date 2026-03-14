from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.notifications.alert_notifier import AlertNotifier


def send_pipeline_failure_alert(
    *,
    job_name: str,
    failed_step: str,
    exit_code: int,
    log_file: Path | None = None,
    details: str | None = None,
) -> dict[str, Any]:
    status = f"Pipeline failure at step={failed_step} (exit_code={exit_code})"

    detail_lines = [
        f"failed_step={failed_step}",
        f"exit_code={exit_code}",
    ]
    if log_file is not None:
        detail_lines.append(f"log_file={log_file}")
    if details:
        detail_lines.append(details)

    notifier = AlertNotifier()
    sent = notifier.send_cron_alert(
        job_name=job_name,
        status=status,
        details="\n".join(detail_lines),
    )

    return {
        "job_name": job_name,
        "failed_step": failed_step,
        "exit_code": exit_code,
        "log_file": str(log_file) if log_file is not None else None,
        "sent": bool(sent),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a Telegram cron alert for research pipeline failures"
    )
    parser.add_argument(
        "--job-name",
        default="research_pipeline",
        help="Logical cron job name for the alert",
    )
    parser.add_argument(
        "--failed-step",
        required=True,
        help="Pipeline step name that failed",
    )
    parser.add_argument(
        "--exit-code",
        type=int,
        required=True,
        help="Process exit code for the failed step",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional path to the cron log file",
    )
    parser.add_argument(
        "--details",
        help="Optional additional failure details",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = send_pipeline_failure_alert(
        job_name=args.job_name,
        failed_step=args.failed_step,
        exit_code=args.exit_code,
        log_file=args.log_file,
        details=args.details,
    )
    print(json.dumps(summary, indent=2))

    if not summary["sent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
