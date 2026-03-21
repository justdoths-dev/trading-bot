from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.strategy_lab.dataset_builder import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LATEST_MAX_ROWS,
    DEFAULT_LATEST_WINDOW_HOURS,
    build_dataset,
    load_jsonl_records_with_metadata,
)

DEFAULT_JSON_OUTPUT = (
    Path("logs/research_reports/latest")
    / "future_return_distribution_diagnosis_summary.json"
)
DEFAULT_MD_OUTPUT = (
    Path("logs/research_reports/latest")
    / "future_return_distribution_diagnosis_summary.md"
)

REPORT_TITLE = "Future Return Distribution Diagnosis"
TARGET_HORIZONS = ("15m", "1h", "4h")
HORIZON_RETURN_FIELDS = {
    "15m": "future_return_15m",
    "1h": "future_return_1h",
    "4h": "future_return_4h",
}
THRESHOLD_SWEEP_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CURRENT_FLAT_THRESHOLD_PCT = 0.20
MARKDOWN_TOP_SYMBOL_ROWS = 10


@dataclass(frozen=True)
class ReturnSeries:
    """Normalized return series for a single horizon."""

    horizon: str
    values: list[float]


def _safe_float(value: Any) -> float | None:
    """Convert common scalar inputs to float while ignoring invalid values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _format_pct(value: float | None) -> str:
    """Format percentage-like numeric values for markdown output."""
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _ratio_pct(numerator: int, denominator: int) -> float:
    """Return a rounded percentage ratio."""
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _avg(values: list[float]) -> float | None:
    """Return the arithmetic mean for a non-empty series."""
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _median(values: list[float]) -> float | None:
    """Return the median for a sorted or unsorted numeric series."""
    return _percentile(values, 50.0)


def _percentile(values: list[float], percentile: float) -> float | None:
    """Compute a percentile with linear interpolation and defensive fallbacks."""
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 6)

    bounded = min(max(percentile, 0.0), 100.0)
    position = (len(sorted_values) - 1) * (bounded / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)

    if lower_index == upper_index:
        return round(sorted_values[lower_index], 6)

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    interpolated = lower_value + ((upper_value - lower_value) * weight)
    return round(interpolated, 6)


def _count_flat_band(values: list[float], threshold_pct: float) -> int:
    """Count returns that fall inside the absolute flat band threshold."""
    return sum(1 for value in values if abs(value) <= threshold_pct)


def _threshold_sweep(values: list[float]) -> dict[str, dict[str, float | int]]:
    """Build flat-band counts and ratios across the configured threshold sweep."""
    sweep: dict[str, dict[str, float | int]] = {}
    for threshold in THRESHOLD_SWEEP_VALUES:
        flat_count = _count_flat_band(values, threshold)
        sweep[f"{threshold:.2f}"] = {
            "flat_band_count": flat_count,
            "flat_band_ratio_pct": _ratio_pct(flat_count, len(values)),
        }
    return sweep


def _normalize_symbol(value: Any) -> str | None:
    """Return a clean symbol identifier or None when absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_return_series(dataset: list[dict[str, Any]]) -> dict[str, ReturnSeries]:
    """Collect valid future return series by horizon from normalized dataset rows."""
    by_horizon: dict[str, ReturnSeries] = {}
    for horizon in TARGET_HORIZONS:
        field_name = HORIZON_RETURN_FIELDS[horizon]
        values = [
            numeric
            for row in dataset
            if (numeric := _safe_float(row.get(field_name))) is not None
        ]
        by_horizon[horizon] = ReturnSeries(horizon=horizon, values=values)
    return by_horizon


def _build_horizon_overview(series: ReturnSeries) -> dict[str, float | int | None]:
    """Summarize distribution statistics for a horizon."""
    values = series.values
    sample_count = len(values)
    flat_count = _count_flat_band(values, CURRENT_FLAT_THRESHOLD_PCT)
    positive_count = sum(1 for value in values if value > 0.0)
    non_positive_count = sum(1 for value in values if value <= 0.0)

    if not values:
        return {
            "sample_count": 0,
            "min_return_pct": None,
            "max_return_pct": None,
            "avg_return_pct": None,
            "median_return_pct": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p40": None,
            "p50": None,
            "p60": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "current_flat_threshold_pct": CURRENT_FLAT_THRESHOLD_PCT,
            "current_flat_band_count": 0,
            "current_flat_band_ratio_pct": 0.0,
            "positive_return_ratio_pct": 0.0,
            "non_positive_return_ratio_pct": 0.0,
        }

    return {
        "sample_count": sample_count,
        "min_return_pct": round(min(values), 6),
        "max_return_pct": round(max(values), 6),
        "avg_return_pct": _avg(values),
        "median_return_pct": _median(values),
        "p05": _percentile(values, 5.0),
        "p10": _percentile(values, 10.0),
        "p25": _percentile(values, 25.0),
        "p40": _percentile(values, 40.0),
        "p50": _percentile(values, 50.0),
        "p60": _percentile(values, 60.0),
        "p75": _percentile(values, 75.0),
        "p90": _percentile(values, 90.0),
        "p95": _percentile(values, 95.0),
        "current_flat_threshold_pct": CURRENT_FLAT_THRESHOLD_PCT,
        "current_flat_band_count": flat_count,
        "current_flat_band_ratio_pct": _ratio_pct(flat_count, sample_count),
        "positive_return_ratio_pct": _ratio_pct(positive_count, sample_count),
        "non_positive_return_ratio_pct": _ratio_pct(non_positive_count, sample_count),
    }


def _build_by_symbol(
    dataset: list[dict[str, Any]],
) -> dict[str, list[dict[str, float | int | str | None]]]:
    """Build per-symbol return distributions by horizon."""
    by_horizon: dict[str, list[dict[str, float | int | str | None]]] = {}

    for horizon in TARGET_HORIZONS:
        field_name = HORIZON_RETURN_FIELDS[horizon]
        symbol_buckets: dict[str, list[float]] = {}

        for row in dataset:
            symbol = _normalize_symbol(row.get("symbol"))
            value = _safe_float(row.get(field_name))
            if symbol is None or value is None:
                continue
            symbol_buckets.setdefault(symbol, []).append(value)

        rows: list[dict[str, float | int | str | None]] = []
        for symbol, values in symbol_buckets.items():
            flat_count = _count_flat_band(values, CURRENT_FLAT_THRESHOLD_PCT)
            rows.append(
                {
                    "symbol": symbol,
                    "sample_count": len(values),
                    "avg_return_pct": _avg(values),
                    "median_return_pct": _median(values),
                    "p25": _percentile(values, 25.0),
                    "p75": _percentile(values, 75.0),
                    "current_flat_band_ratio_pct": _ratio_pct(flat_count, len(values)),
                }
            )

        rows.sort(
            key=lambda item: (
                -float(item["current_flat_band_ratio_pct"]),
                -int(item["sample_count"]),
                str(item["symbol"]),
            )
        )
        by_horizon[horizon] = rows

    return by_horizon


def _build_observations(
    horizon_overview: dict[str, dict[str, float | int | None]],
    by_symbol: dict[str, list[dict[str, float | int | str | None]]],
) -> list[str]:
    """Attach lightweight diagnosis labels derived from the observed distributions."""
    labels: list[str] = []

    ratio_15m = float(horizon_overview["15m"]["current_flat_band_ratio_pct"])
    ratio_1h = float(horizon_overview["1h"]["current_flat_band_ratio_pct"])
    ratio_4h = float(horizon_overview["4h"]["current_flat_band_ratio_pct"])

    if ratio_15m >= 35.0 or (ratio_15m - ratio_4h) >= 12.0:
        labels.append("current_threshold_likely_too_wide_for_15m")

    symbol_flat_concentration_detected = False
    for horizon in TARGET_HORIZONS:
        horizon_ratio = float(horizon_overview[horizon]["current_flat_band_ratio_pct"])
        for row in by_symbol[horizon][:5]:
            if int(row["sample_count"]) < 20:
                continue
            if float(row["current_flat_band_ratio_pct"]) >= max(
                horizon_ratio + 15.0,
                45.0,
            ):
                symbol_flat_concentration_detected = True
                break
        if symbol_flat_concentration_detected:
            break

    if symbol_flat_concentration_detected:
        labels.append("symbol_level_flat_concentration_detected")

    broad_flat_pressure = all(
        float(horizon_overview[horizon]["current_flat_band_ratio_pct"]) >= 25.0
        for horizon in TARGET_HORIZONS
    )
    if broad_flat_pressure:
        labels.append("flat_pressure_is_broad_across_horizons")

    horizon_spread = max(ratio_15m, ratio_1h, ratio_4h) - min(
        ratio_15m,
        ratio_1h,
        ratio_4h,
    )
    if horizon_spread >= 10.0:
        labels.append("horizon_specific_threshold_review_recommended")

    if not labels:
        labels.append("future_return_distribution_looks_balanced_at_current_threshold")

    return labels


def build_future_return_distribution_diagnosis_report(
    input_path: Path,
    *,
    rotation_aware: bool = False,
    max_age_hours: int | None = DEFAULT_LATEST_WINDOW_HOURS,
    max_rows: int | None = DEFAULT_LATEST_MAX_ROWS,
) -> dict[str, Any]:
    """Build a diagnosis-only report for raw future return distributions."""
    source_metadata_input = load_jsonl_records_with_metadata(
        path=input_path,
        rotation_aware=rotation_aware,
        max_age_hours=max_age_hours,
        max_rows=max_rows,
    )
    _, source_metadata = source_metadata_input
    dataset = build_dataset(
        path=input_path,
        rotation_aware=rotation_aware,
        max_age_hours=max_age_hours,
        max_rows=max_rows,
    )

    return_series = _build_return_series(dataset)
    horizon_overview = {
        horizon: _build_horizon_overview(return_series[horizon])
        for horizon in TARGET_HORIZONS
    }
    threshold_sweep = {
        horizon: _threshold_sweep(return_series[horizon].values)
        for horizon in TARGET_HORIZONS
    }
    by_symbol = _build_by_symbol(dataset)
    observations = _build_observations(
        horizon_overview=horizon_overview,
        by_symbol=by_symbol,
    )

    total_rows = len(dataset)

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_path": str(input_path),
        },
        "source_targeting": {
            "dataset_source": str(input_path),
            "rotation_aware": bool(source_metadata.get("rotation_aware", False)),
            "source_files": source_metadata.get("source_files", []),
            "source_file_count": source_metadata.get("source_file_count", 0),
            "raw_record_count": source_metadata.get("raw_record_count", 0),
            "windowed_record_count": source_metadata.get("windowed_record_count", 0),
            "normalized_dataset_rows": total_rows,
        },
        "dataset_instrumentation": {
            "max_age_hours": source_metadata.get("max_age_hours"),
            "max_rows": source_metadata.get("max_rows"),
            "source_row_counts": source_metadata.get("source_row_counts", {}),
            "future_return_field_names": {
                horizon: HORIZON_RETURN_FIELDS[horizon]
                for horizon in TARGET_HORIZONS
            },
            "current_flat_threshold_pct": CURRENT_FLAT_THRESHOLD_PCT,
            "threshold_sweep_values_pct": list(THRESHOLD_SWEEP_VALUES),
            "ignored_rows_missing_symbol_count": sum(
                1 for row in dataset if _normalize_symbol(row.get("symbol")) is None
            ),
            "missing_or_non_numeric_return_count_by_horizon": {
                horizon: max(0, total_rows - len(return_series[horizon].values))
                for horizon in TARGET_HORIZONS
            },
        },
        "horizon_overview": horizon_overview,
        "threshold_sweep": threshold_sweep,
        "by_symbol": by_symbol,
        "observations": observations,
    }


def build_future_return_distribution_diagnosis_markdown(summary: dict[str, Any]) -> str:
    """Render the diagnosis report in the compact markdown style used by peer modules."""
    lines: list[str] = []
    lines.append(REPORT_TITLE)
    lines.append(f"Generated: {summary['metadata']['generated_at']}")
    lines.append("")
    lines.append("Source Targeting")
    for key, value in summary["source_targeting"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Dataset Instrumentation")
    for key, value in summary["dataset_instrumentation"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Horizon Overview")
    for horizon in TARGET_HORIZONS:
        payload = summary["horizon_overview"][horizon]
        lines.append(f"- {horizon}:")
        for key, value in payload.items():
            if isinstance(value, float) or value is None:
                lines.append(f"  - {key}: {_format_pct(value)}")
            else:
                lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("Threshold Sweep")
    for horizon in TARGET_HORIZONS:
        lines.append(f"- {horizon}:")
        for threshold, payload in summary["threshold_sweep"][horizon].items():
            lines.append(
                "  - "
                f"threshold_pct={threshold}, "
                f"flat_band_count={payload['flat_band_count']}, "
                f"flat_band_ratio_pct={_format_pct(float(payload['flat_band_ratio_pct']))}"
            )
    lines.append("")
    lines.append("By Symbol")
    for horizon in TARGET_HORIZONS:
        lines.append(f"- {horizon}:")
        rows = summary["by_symbol"][horizon]
        if not rows:
            lines.append("  - none")
            continue
        for row in rows[:MARKDOWN_TOP_SYMBOL_ROWS]:
            lines.append(
                "  - "
                f"symbol={row['symbol']}, "
                f"sample_count={row['sample_count']}, "
                f"avg_return_pct={_format_pct(row['avg_return_pct'])}, "
                f"median_return_pct={_format_pct(row['median_return_pct'])}, "
                f"p25={_format_pct(row['p25'])}, "
                f"p75={_format_pct(row['p75'])}, "
                f"current_flat_band_ratio_pct={_format_pct(float(row['current_flat_band_ratio_pct']))}"
            )
        if len(rows) > MARKDOWN_TOP_SYMBOL_ROWS:
            lines.append(
                f"  - ... truncated_to_top_{MARKDOWN_TOP_SYMBOL_ROWS}_of_{len(rows)}_rows"
            )
    lines.append("")
    lines.append("Observations")
    for label in summary["observations"]:
        lines.append(f"- {label}")
    lines.append("")
    return "\n".join(lines)


def write_future_return_distribution_diagnosis_report(
    summary: dict[str, Any],
    json_output_path: Path,
    markdown_output_path: Path,
) -> dict[str, str]:
    """Write JSON and Markdown outputs for the diagnosis report."""
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_output_path.write_text(
        build_future_return_distribution_diagnosis_markdown(summary),
        encoding="utf-8",
    )

    return {
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for future return distribution diagnosis generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the trade analysis JSONL dataset.",
    )
    parser.add_argument(
        "--rotation-aware",
        action="store_true",
        help="Read rotated trade_analysis.jsonl files within the default recent window.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=DEFAULT_LATEST_WINDOW_HOURS,
        help="Maximum record age for rotation-aware dataset loading.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_LATEST_MAX_ROWS,
        help="Maximum rows retained for rotation-aware dataset loading.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Output path for JSON summary.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_OUTPUT,
        help="Output path for Markdown summary.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for future return distribution diagnosis generation."""
    args = parse_args()
    summary = build_future_return_distribution_diagnosis_report(
        input_path=args.input_path,
        rotation_aware=args.rotation_aware,
        max_age_hours=args.max_age_hours,
        max_rows=args.max_rows,
    )
    outputs = write_future_return_distribution_diagnosis_report(
        summary=summary,
        json_output_path=args.json_output,
        markdown_output_path=args.md_output,
    )
    print(
        json.dumps(
            {
                **outputs,
                "source_targeting": summary["source_targeting"],
                "observations": summary["observations"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
