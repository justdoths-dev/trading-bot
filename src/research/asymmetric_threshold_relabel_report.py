from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.experimental_labeling.asymmetric_threshold_config import (
    DEFAULT_VARIANT_NAME,
    TARGET_HORIZONS,
    get_asymmetric_threshold_variant,
)
from src.research.experimental_labeling.asymmetric_threshold_relabeler import (
    LABELING_METHOD,
    build_default_output_path,
)
from src.research.experimental_labeling.asymmetric_thresholds import build_threshold_map

TARGET_LABELS = ("up", "down", "flat")


def build_default_json_output_path(variant_name: str = DEFAULT_VARIANT_NAME) -> Path:
    return Path(
        f"logs/research_reports/experiments/candidate_c_asymmetric/{variant_name}/asymmetric_threshold_relabel_summary.json"
    )


def build_default_markdown_output_path(variant_name: str = DEFAULT_VARIANT_NAME) -> Path:
    return Path(
        f"logs/research_reports/experiments/candidate_c_asymmetric/{variant_name}/asymmetric_threshold_relabel_summary.md"
    )


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_bool(value: Any) -> bool:
    return value is True


def _safe_float(value: Any) -> float | None:
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
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def load_jsonl_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    instrumentation = {
        "blank_line_count": 0,
        "invalid_json_line_count": 0,
        "non_object_line_count": 0,
    }

    if not path.exists() or not path.is_file():
        return [], instrumentation

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    instrumentation["blank_line_count"] += 1
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    instrumentation["invalid_json_line_count"] += 1
                    continue
                if not isinstance(payload, dict):
                    instrumentation["non_object_line_count"] += 1
                    continue
                records.append(payload)
    except OSError:
        return [], instrumentation

    return records, instrumentation


def build_asymmetric_threshold_relabel_summary(
    records: list[dict[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
    variant_name: str = DEFAULT_VARIANT_NAME,
    parser_instrumentation: dict[str, int] | None = None,
) -> dict[str, Any]:
    config = get_asymmetric_threshold_variant(variant_name)
    variant_thresholds = build_threshold_map(config)

    candidate_c_records = [
        row
        for row in records
        if _safe_dict(row.get("experimental_labeling")).get("labeling_method") == LABELING_METHOD
        and _safe_dict(row.get("experimental_labeling")).get("variant") == config.variant_name
    ]

    rebuilt_row_counts_by_horizon = {horizon: 0 for horizon in TARGET_HORIZONS}
    label_counts = {
        horizon: {label: 0 for label in TARGET_LABELS}
        for horizon in TARGET_HORIZONS
    }

    for row in candidate_c_records:
        experimental_labeling = _safe_dict(row.get("experimental_labeling"))
        rebuilt_flags = _safe_dict(
            experimental_labeling.get("label_rebuilt_by_horizon")
        )

        for horizon in TARGET_HORIZONS:
            if not _safe_bool(rebuilt_flags.get(horizon)):
                continue

            rebuilt_row_counts_by_horizon[horizon] += 1
            label_value = row.get(f"future_label_{horizon}")
            if isinstance(label_value, str) and label_value in TARGET_LABELS:
                label_counts[horizon][label_value] += 1

    label_ratios = {
        horizon: {
            label: _safe_ratio(
                label_counts[horizon][label],
                rebuilt_row_counts_by_horizon[horizon],
            )
            for label in TARGET_LABELS
        }
        for horizon in TARGET_HORIZONS
    }

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "labeling_method": LABELING_METHOD,
            "variant": config.variant_name,
        },
        "parser_instrumentation": parser_instrumentation or {},
        "dataset_overview": {
            "total_row_count": len(candidate_c_records),
            "rebuilt_row_count_by_horizon": rebuilt_row_counts_by_horizon,
        },
        "thresholds_by_horizon": variant_thresholds,
        "label_distribution_counts_by_horizon": label_counts,
        "label_distribution_ratios_by_horizon": label_ratios,
    }


def build_asymmetric_threshold_relabel_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    overview = _safe_dict(summary.get("dataset_overview"))
    rebuilt_row_count_by_horizon = _safe_dict(overview.get("rebuilt_row_count_by_horizon"))
    parser_instrumentation = _safe_dict(summary.get("parser_instrumentation"))
    thresholds = _safe_dict(summary.get("thresholds_by_horizon"))
    label_counts = _safe_dict(summary.get("label_distribution_counts_by_horizon"))
    label_ratios = _safe_dict(summary.get("label_distribution_ratios_by_horizon"))

    lines = [
        "# Asymmetric Threshold Relabel Summary",
        "",
        "## Dataset Overview",
        f"- input_path: {metadata.get('input_path', 'n/a')}",
        f"- output_path: {metadata.get('output_path', 'n/a')}",
        f"- labeling_method: {metadata.get('labeling_method', 'n/a')}",
        f"- variant: {metadata.get('variant', 'n/a')}",
        f"- total_row_count: {overview.get('total_row_count', 0)}",
        "",
        "## Parser Instrumentation",
        f"- blank_line_count: {parser_instrumentation.get('blank_line_count', 0)}",
        f"- invalid_json_line_count: {parser_instrumentation.get('invalid_json_line_count', 0)}",
        f"- non_object_line_count: {parser_instrumentation.get('non_object_line_count', 0)}",
        "",
        "## Thresholds By Horizon",
    ]

    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(thresholds.get(horizon))
        lines.append(
            f"- {horizon}: up={_format_pct(_safe_float(payload.get('up')))}, "
            f"down={_format_pct(_safe_float(payload.get('down')))}"
        )

    lines.extend(["", "## Rebuilt Row Count By Horizon"])
    for horizon in TARGET_HORIZONS:
        lines.append(f"- {horizon}: {rebuilt_row_count_by_horizon.get(horizon, 0)}")

    lines.extend(["", "## Label Distribution Counts By Horizon"])
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(label_counts.get(horizon))
        lines.append(
            f"- {horizon}: up={payload.get('up', 0)}, down={payload.get('down', 0)}, flat={payload.get('flat', 0)}"
        )

    lines.extend(["", "## Label Distribution Ratios By Horizon"])
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(label_ratios.get(horizon))
        lines.append(
            f"- {horizon}: up={_format_pct(_safe_float(payload.get('up')))}, "
            f"down={_format_pct(_safe_float(payload.get('down')))}, "
            f"flat={_format_pct(_safe_float(payload.get('flat')))}"
        )

    return "\n".join(lines).strip() + "\n"


def run_asymmetric_threshold_relabel_report(
    input_path: Path | None = None,
    json_output_path: Path | None = None,
    markdown_output_path: Path | None = None,
    variant_name: str = DEFAULT_VARIANT_NAME,
) -> dict[str, Any]:
    config = get_asymmetric_threshold_variant(variant_name)
    resolved_input_path = input_path or build_default_output_path(config.variant_name)
    resolved_json_output_path = json_output_path or build_default_json_output_path(
        config.variant_name
    )
    resolved_markdown_output_path = (
        markdown_output_path
        or build_default_markdown_output_path(config.variant_name)
    )

    records, parser_instrumentation = load_jsonl_records(resolved_input_path)
    summary = build_asymmetric_threshold_relabel_summary(
        records,
        input_path=resolved_input_path,
        output_path=resolved_json_output_path,
        variant_name=config.variant_name,
        parser_instrumentation=parser_instrumentation,
    )
    markdown = build_asymmetric_threshold_relabel_markdown(summary)

    resolved_json_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    resolved_markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "markdown": markdown,
        "json_output_path": resolved_json_output_path,
        "markdown_output_path": resolved_markdown_output_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a summary report for the Candidate C asymmetric-threshold experimental relabel dataset"
    )
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--variant", type=str, default=DEFAULT_VARIANT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_asymmetric_threshold_relabel_report(
        input_path=args.input_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
        variant_name=args.variant,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
