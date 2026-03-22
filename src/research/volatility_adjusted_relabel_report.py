from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_INPUT_PATH = Path(
    "logs/experiments/trade_analysis_relabel_candidate_b_vol_adjusted.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_b_vol_adjusted/volatility_adjusted_relabel_summary.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_b_vol_adjusted/volatility_adjusted_relabel_summary.md"
)
TARGET_HORIZONS = ("15m", "1h", "4h")
TARGET_LABELS = ("up", "down", "flat")
LABELING_METHOD = "candidate_b_volatility_adjusted_v1"


def _safe_dict(value: Any) -> dict[str, Any]:
    """Return the input when it is a dict; otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    """Safely coerce supported scalar values to float."""
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


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a rounded ratio, guarding against division by zero."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _format_pct(value: float | None) -> str:
    """Format a numeric ratio/percent-like value for markdown output."""
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset safely into a list of dict rows."""
    if not path.exists() or not path.is_file():
        return []

    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)

    return records


def build_volatility_adjusted_relabel_summary(
    records: list[dict[str, Any]],
    *,
    input_path: Path,
) -> dict[str, Any]:
    """Build a summary report for the Candidate B volatility-adjusted dataset."""
    candidate_b_records = [
        row
        for row in records
        if _safe_dict(row.get("experimental_labeling")).get("labeling_method") == LABELING_METHOD
    ]

    threshold_values: dict[str, list[float]] = {horizon: [] for horizon in TARGET_HORIZONS}
    label_counts: dict[str, dict[str, int]] = {
        horizon: {label: 0 for label in TARGET_LABELS}
        for horizon in TARGET_HORIZONS
    }
    fallback_row_count = 0

    for row in candidate_b_records:
        metadata = _safe_dict(row.get("experimental_labeling"))
        if metadata.get("used_fallback_atr_pct") is True:
            fallback_row_count += 1

        thresholds = _safe_dict(metadata.get("thresholds"))
        for horizon in TARGET_HORIZONS:
            threshold_value = _safe_float(thresholds.get(horizon))
            if threshold_value is not None:
                threshold_values[horizon].append(threshold_value)

            label_value = row.get(f"future_label_{horizon}")
            if isinstance(label_value, str) and label_value in TARGET_LABELS:
                label_counts[horizon][label_value] += 1

    threshold_statistics: dict[str, dict[str, float | None]] = {}
    label_distribution_counts: dict[str, dict[str, int]] = {}
    label_distribution_ratios: dict[str, dict[str, float]] = {}

    for horizon in TARGET_HORIZONS:
        values = sorted(threshold_values[horizon])
        threshold_statistics[horizon] = {
            "min": round(min(values), 6) if values else None,
            "max": round(max(values), 6) if values else None,
            "mean": round(mean(values), 6) if values else None,
            "median": round(median(values), 6) if values else None,
        }
        label_distribution_counts[horizon] = dict(label_counts[horizon])
        label_distribution_ratios[horizon] = {
            label: _safe_ratio(label_counts[horizon][label], len(candidate_b_records))
            for label in TARGET_LABELS
        }

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_path": str(input_path),
            "labeling_method": LABELING_METHOD,
        },
        "dataset_overview": {
            "total_row_count": len(candidate_b_records),
            "fallback_row_count": fallback_row_count,
            "fallback_row_ratio": _safe_ratio(fallback_row_count, len(candidate_b_records)),
        },
        "threshold_statistics_by_horizon": threshold_statistics,
        "label_distribution_counts_by_horizon": label_distribution_counts,
        "label_distribution_ratios_by_horizon": label_distribution_ratios,
    }


def build_volatility_adjusted_relabel_markdown(summary: dict[str, Any]) -> str:
    """Render the Candidate B relabel summary as Markdown."""
    metadata = _safe_dict(summary.get("metadata"))
    overview = _safe_dict(summary.get("dataset_overview"))

    lines = [
        "# Volatility-Adjusted Relabel Summary",
        "",
        "## Dataset Overview",
        f"- input_path: {metadata.get('input_path', 'n/a')}",
        f"- labeling_method: {metadata.get('labeling_method', 'n/a')}",
        f"- total_row_count: {overview.get('total_row_count', 0)}",
        f"- fallback_row_count: {overview.get('fallback_row_count', 0)}",
        f"- fallback_row_ratio: {_format_pct(_safe_float(overview.get('fallback_row_ratio')))}",
        "",
        "## Threshold Statistics By Horizon",
    ]

    threshold_stats = _safe_dict(summary.get("threshold_statistics_by_horizon"))
    for horizon in TARGET_HORIZONS:
        payload_dict = _safe_dict(threshold_stats.get(horizon))
        lines.append(
            f"- {horizon}: min={_format_pct(_safe_float(payload_dict.get('min')))}, "
            f"max={_format_pct(_safe_float(payload_dict.get('max')))}, "
            f"mean={_format_pct(_safe_float(payload_dict.get('mean')))}, "
            f"median={_format_pct(_safe_float(payload_dict.get('median')))}"
        )

    lines.extend(["", "## Label Distribution Counts By Horizon"])
    label_counts = _safe_dict(summary.get("label_distribution_counts_by_horizon"))
    for horizon in TARGET_HORIZONS:
        payload_dict = _safe_dict(label_counts.get(horizon))
        lines.append(
            f"- {horizon}: up={payload_dict.get('up', 0)}, "
            f"down={payload_dict.get('down', 0)}, "
            f"flat={payload_dict.get('flat', 0)}"
        )

    lines.extend(["", "## Label Distribution Ratios By Horizon"])
    label_ratios = _safe_dict(summary.get("label_distribution_ratios_by_horizon"))
    for horizon in TARGET_HORIZONS:
        payload_dict = _safe_dict(label_ratios.get(horizon))
        lines.append(
            f"- {horizon}: up={_format_pct(_safe_float(payload_dict.get('up')))}, "
            f"down={_format_pct(_safe_float(payload_dict.get('down')))}, "
            f"flat={_format_pct(_safe_float(payload_dict.get('flat')))}"
        )

    return "\n".join(lines).strip() + "\n"


def run_volatility_adjusted_relabel_report(
    input_path: Path = DEFAULT_INPUT_PATH,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
) -> dict[str, Any]:
    """Build, render, and persist the Candidate B relabel summary report."""
    records = load_jsonl_records(input_path)
    summary = build_volatility_adjusted_relabel_summary(records, input_path=input_path)
    markdown = build_volatility_adjusted_relabel_markdown(summary)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "markdown": markdown,
        "json_output_path": json_output_path,
        "markdown_output_path": markdown_output_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Candidate B relabel summary report."""
    parser = argparse.ArgumentParser(
        description="Build a summary report for the Candidate B volatility-adjusted experimental relabel dataset"
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the Candidate B relabel summary report."""
    args = parse_args()
    result = run_volatility_adjusted_relabel_report(
        input_path=args.input_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
