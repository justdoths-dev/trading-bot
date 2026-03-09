from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.research_metrics import HORIZONS, calculate_research_metrics


def load_jsonl_records(input_path: Path) -> list[dict[str, Any]]:
    """Load JSONL records while ignoring blank lines and validating JSON."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            content = line.strip()
            if not content:
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {input_path} at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected JSON object in {input_path} at line {line_number}, got {type(parsed).__name__}"
                )

            records.append(parsed)

    return records


def write_summary_files(
    metrics: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write summary.json and summary.md into output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    summary_json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_md_path.write_text(_build_markdown(metrics), encoding="utf-8")

    return summary_json_path, summary_md_path


def run_research_analyzer(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run full analyzer flow: load records, calculate metrics, and write reports."""
    records = load_jsonl_records(input_path)
    metrics = calculate_research_metrics(records)
    write_summary_files(metrics, output_dir)
    return metrics


def _build_markdown(metrics: dict[str, Any]) -> str:
    overview = metrics.get("dataset_overview", {}) or {}
    horizons = metrics.get("horizon_summary", {}) or {}
    by_symbol = metrics.get("by_symbol", {}) or {}
    by_strategy = metrics.get("by_strategy", {}) or {}

    lines: list[str] = []
    lines.append("# Research Summary")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(UTC).isoformat()}")
    lines.append("")

    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- total_records: {overview.get('total_records', 0)}")
    lines.append(
        f"- records_with_any_future_label: {overview.get('records_with_any_future_label', 0)}"
    )
    lines.append(
        f"- label_coverage_any_horizon_pct: {_fmt_metric(overview.get('label_coverage_any_horizon_pct'))}"
    )

    date_range = overview.get("date_range", {}) or {}
    lines.append(f"- date_range.start: {date_range.get('start', 'unknown')}")
    lines.append(f"- date_range.end: {date_range.get('end', 'unknown')}")
    lines.append("")

    lines.extend(_markdown_distribution("symbols_distribution", overview.get("symbols_distribution", {})))
    lines.extend(
        _markdown_distribution(
            "selected_strategies_distribution",
            overview.get("selected_strategies_distribution", {}),
        )
    )
    lines.extend(_markdown_distribution("bias_distribution", overview.get("bias_distribution", {})))
    lines.extend(
        _markdown_distribution(
            "ai_execution_distribution",
            overview.get("ai_execution_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "alignment_distribution",
            overview.get("alignment_distribution", {}),
        )
    )

    lines.append("## Horizon Summary")
    lines.append("")
    for horizon in HORIZONS:
        lines.extend(_markdown_horizon_block(horizon, horizons.get(horizon, {}) or {}, heading_level=3))

    lines.append("## By Symbol")
    lines.append("")
    if by_symbol:
        for symbol, data in by_symbol.items():
            lines.append(f"### {symbol}")
            lines.append(f"- total_records: {data.get('total_records', 0)}")
            lines.append(
                f"- records_with_any_future_label: {data.get('records_with_any_future_label', 0)}"
            )
            lines.append(
                f"- label_coverage_any_horizon_pct: {_fmt_metric(data.get('label_coverage_any_horizon_pct'))}"
            )
            lines.append("")

            group_horizons = data.get("horizon_summary", {}) or {}
            for horizon in HORIZONS:
                lines.extend(
                    _markdown_horizon_block(
                        horizon,
                        group_horizons.get(horizon, {}) or {},
                        heading_level=4,
                    )
                )
    else:
        lines.append("No symbol groups available.")
        lines.append("")

    lines.append("## By Strategy")
    lines.append("")
    if by_strategy:
        for strategy, data in by_strategy.items():
            lines.append(f"### {strategy}")
            lines.append(f"- total_records: {data.get('total_records', 0)}")
            lines.append(
                f"- records_with_any_future_label: {data.get('records_with_any_future_label', 0)}"
            )
            lines.append(
                f"- label_coverage_any_horizon_pct: {_fmt_metric(data.get('label_coverage_any_horizon_pct'))}"
            )
            lines.append("")

            group_horizons = data.get("horizon_summary", {}) or {}
            for horizon in HORIZONS:
                lines.extend(
                    _markdown_horizon_block(
                        horizon,
                        group_horizons.get(horizon, {}) or {},
                        heading_level=4,
                    )
                )
    else:
        lines.append("No strategy groups available.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _markdown_horizon_block(
    horizon: str,
    horizon_data: dict[str, Any],
    heading_level: int,
) -> list[str]:
    heading = "#" * heading_level
    lines: list[str] = []
    lines.append(f"{heading} {horizon}")
    lines.append("")
    lines.append(f"- labeled_records: {horizon_data.get('labeled_records', 0)}")

    label_dist = horizon_data.get("label_distribution", {}) or {}
    lines.append(
        "- label_distribution: "
        f"up={label_dist.get('up', 0)}, "
        f"down={label_dist.get('down', 0)}, "
        f"flat={label_dist.get('flat', 0)}"
    )

    lines.append(f"- avg_future_return_pct: {_fmt_metric(horizon_data.get('avg_future_return_pct'))}")
    lines.append(
        f"- median_future_return_pct: {_fmt_metric(horizon_data.get('median_future_return_pct'))}"
    )
    lines.append(f"- positive_rate_pct: {_fmt_metric(horizon_data.get('positive_rate_pct'))}")
    lines.append(f"- negative_rate_pct: {_fmt_metric(horizon_data.get('negative_rate_pct'))}")
    lines.append(f"- flat_rate_pct: {_fmt_metric(horizon_data.get('flat_rate_pct'))}")

    bias_vs_label = horizon_data.get("bias_vs_label", {}) or {}
    lines.append(
        "- bias_vs_label: "
        f"match={bias_vs_label.get('match', 0)}, "
        f"mismatch={bias_vs_label.get('mismatch', 0)}, "
        f"unknown={bias_vs_label.get('unknown', 0)}, "
        f"match_rate_pct={_fmt_metric(bias_vs_label.get('match_rate_pct'))}"
    )

    signal_vs_label = horizon_data.get("signal_vs_label", {}) or {}
    lines.append(
        "- signal_vs_label: "
        f"match={signal_vs_label.get('match', 0)}, "
        f"mismatch={signal_vs_label.get('mismatch', 0)}, "
        f"unknown={signal_vs_label.get('unknown', 0)}, "
        f"match_rate_pct={_fmt_metric(signal_vs_label.get('match_rate_pct'))}"
    )

    lines.append("")
    return lines


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _markdown_distribution(title: str, values: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"### {title}")
    lines.append("")

    if not values:
        lines.append("No data.")
        lines.append("")
        return lines

    for key, count in values.items():
        lines.append(f"- {key}: {count}")

    lines.append("")
    return lines


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "research_reports" / "latest"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trade_analysis.jsonl research metrics")
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory to write summary.json and summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = run_research_analyzer(input_path=args.input, output_dir=args.output_dir)

    print(f"Records analyzed: {metrics.get('dataset_overview', {}).get('total_records', 0)}")
    print(f"Summary JSON: {(args.output_dir / 'summary.json').resolve()}")
    print(f"Summary MD: {(args.output_dir / 'summary.md').resolve()}")


if __name__ == "__main__":
    main()
