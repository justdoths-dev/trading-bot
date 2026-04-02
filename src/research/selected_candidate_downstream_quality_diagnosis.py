from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPORT_TYPE = "selected_candidate_downstream_quality_diagnosis"
REPORT_TITLE = "Selected Candidate Downstream Quality Diagnosis"
DEFAULT_PRIMARY_INPUT = Path("logs/trade_analysis.jsonl")
DEFAULT_FALLBACK_INPUT = Path("logs/trade_analysis_cumulative.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
RETURN_FIELDS = (
    "future_return_15m",
    "future_return_1h",
    "future_return_4h",
)
MIN_ROWS_FOR_VERDICT = 12
OUTLIER_DIVERGENCE_THRESHOLD_PCT = 0.25


@dataclass(frozen=True)
class SelectedRow:
    line_number: int
    logged_at: str | None
    selected_symbol: str | None
    selected_strategy: str | None
    selected_horizon: str | None
    selection_reason: str | None
    top_score: float | None
    returns: dict[str, float | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether recovered shadow-selected candidates are showing "
            "useful downstream quality in trade-analysis logs."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help=(
            "Optional JSONL input path. Defaults to logs/trade_analysis.jsonl and "
            "falls back to logs/trade_analysis_cumulative.jsonl."
        ),
    )
    parser.add_argument("--symbol", type=str, default=None, help="Filter by selected_symbol.")
    parser.add_argument("--strategy", type=str, default=None, help="Filter by selected_strategy.")
    parser.add_argument("--horizon", type=str, default=None, help="Filter by selected_horizon.")
    parser.add_argument(
        "--recent-selected",
        type=int,
        default=None,
        help=(
            "Limit to the most recent selected rows after applying selected-row and "
            "candidate filters."
        ),
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown report copies into the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and Markdown output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_input = resolve_input_path(args.input_path)
    report = build_report(
        input_path=resolved_input,
        symbol=args.symbol,
        strategy=args.strategy,
        horizon=args.horizon,
        recent_selected=args.recent_selected,
    )

    written_paths: dict[str, str] = {}
    if args.write_latest_copy:
        written_paths = write_report_files(report, args.output_dir)

    metadata = report["metadata"]
    summary = {
        "report_type": REPORT_TYPE,
        "input_path_used": metadata["input_path_used"],
        "selected_rows_seen": metadata["selected_rows_seen"],
        "filtered_selected_row_count_before_recent_limit": metadata[
            "filtered_selected_row_count_before_recent_limit"
        ],
        "selected_row_count_used": metadata["selected_row_count_used"],
        "downstream_return_fields_available": metadata["downstream_return_fields_available"],
        "downstream_non_null_total_count": metadata["downstream_non_null_total_count"],
        "verdict": report["verdict"]["label"],
        "written_paths": written_paths,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def resolve_input_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {explicit_path}")
        return explicit_path

    for candidate in (DEFAULT_PRIMARY_INPUT, DEFAULT_FALLBACK_INPUT):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a trade-analysis JSONL input path. Checked: "
        f"{DEFAULT_PRIMARY_INPUT} and {DEFAULT_FALLBACK_INPUT}"
    )


def build_report(
    *,
    input_path: Path,
    symbol: str | None,
    strategy: str | None,
    horizon: str | None,
    recent_selected: int | None,
) -> dict[str, Any]:
    total_rows_read = 0
    malformed_row_count = 0
    skipped_non_selected_count = 0
    skipped_filter_mismatch_count = 0
    selected_rows_seen = 0
    selected_rows_all: list[SelectedRow] = []
    selected_identities: set[tuple[str, str, str]] = set()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            total_rows_read += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed_row_count += 1
                continue

            if not isinstance(payload, dict):
                malformed_row_count += 1
                continue

            edge_output = _safe_dict(payload.get("edge_selection_output"))
            if edge_output is None:
                malformed_row_count += 1
                continue

            selection_status = _clean_text(edge_output.get("selection_status"))
            if selection_status != "selected":
                skipped_non_selected_count += 1
                continue

            selected_rows_seen += 1
            row = _build_selected_row(payload, edge_output, line_number)
            selected_rows_all.append(row)

            identity = (
                row.selected_symbol or "<missing-symbol>",
                row.selected_strategy or "<missing-strategy>",
                row.selected_horizon or "<missing-horizon>",
            )
            selected_identities.add(identity)

            if not _matches_filters(row, symbol=symbol, strategy=strategy, horizon=horizon):
                skipped_filter_mismatch_count += 1

    filtered_rows = [
        row
        for row in selected_rows_all
        if _matches_filters(row, symbol=symbol, strategy=strategy, horizon=horizon)
    ]
    filtered_selected_row_count_before_recent_limit = len(filtered_rows)
    limited_rows = _apply_recent_selected_limit(filtered_rows, recent_selected)

    returns_summary = {
        field: _compute_return_summary([row.returns.get(field) for row in limited_rows])
        for field in RETURN_FIELDS
    }
    score_summary = _compute_score_summary([row.top_score for row in limited_rows])

    downstream_non_null_total_count = sum(
        int(returns_summary[field].get("non_null_count", 0)) for field in RETURN_FIELDS
    )
    downstream_return_fields_available = downstream_non_null_total_count > 0

    interpretation_notes, verdict_label, verdict_explanation = _interpret_report(
        selected_row_count=len(limited_rows),
        returns_summary=returns_summary,
        downstream_return_fields_available=downstream_return_fields_available,
    )

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "input_path_used": str(input_path),
        "total_rows_read": total_rows_read,
        "selected_rows_seen": selected_rows_seen,
        "filtered_selected_row_count_before_recent_limit": filtered_selected_row_count_before_recent_limit,
        "selected_row_count_used": len(limited_rows),
        "malformed_row_count": malformed_row_count,
        "skipped_non_selected_count": skipped_non_selected_count,
        "skipped_filter_mismatch_count": skipped_filter_mismatch_count,
        "unique_selected_candidate_count": len(selected_identities),
        "downstream_return_fields_available": downstream_return_fields_available,
        "downstream_non_null_total_count": downstream_non_null_total_count,
        "filters": {
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
            "recent_selected_limit": recent_selected,
        },
    }

    return {
        "metadata": metadata,
        "score_summary": score_summary,
        "return_quality": returns_summary,
        "interpretation_notes": interpretation_notes,
        "verdict": {
            "label": verdict_label,
            "explanation": verdict_explanation,
        },
        "filtered_selected_rows": [_selected_row_to_dict(row) for row in limited_rows],
    }


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{REPORT_TYPE}.json"
    md_path = output_dir / f"{REPORT_TYPE}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report.get("metadata", {})
    filters = metadata.get("filters", {})
    score_summary = report.get("score_summary", {})
    interpretation_notes = report.get("interpretation_notes", [])
    verdict = report.get("verdict", {})
    return_quality = report.get("return_quality", {})

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Run",
        "",
        f"- generated_at: {metadata.get('generated_at')}",
        f"- input_path: {metadata.get('input_path_used')}",
        f"- total_rows_read: {metadata.get('total_rows_read', 0)}",
        f"- selected_rows_seen: {metadata.get('selected_rows_seen', 0)}",
        f"- filtered_selected_row_count_before_recent_limit: {metadata.get('filtered_selected_row_count_before_recent_limit', 0)}",
        f"- selected_row_count_used: {metadata.get('selected_row_count_used', 0)}",
        f"- malformed_rows: {metadata.get('malformed_row_count', 0)}",
        f"- skipped_non_selected_count: {metadata.get('skipped_non_selected_count', 0)}",
        f"- skipped_filter_mismatch_count: {metadata.get('skipped_filter_mismatch_count', 0)}",
        f"- unique_selected_candidate_count: {metadata.get('unique_selected_candidate_count', 0)}",
        f"- downstream_return_fields_available: {metadata.get('downstream_return_fields_available')}",
        f"- downstream_non_null_total_count: {metadata.get('downstream_non_null_total_count', 0)}",
        "",
        "## Candidate Filter",
        "",
        f"- symbol: {filters.get('symbol') or 'all'}",
        f"- strategy: {filters.get('strategy') or 'all'}",
        f"- horizon: {filters.get('horizon') or 'all'}",
        f"- recent_selected_limit: {filters.get('recent_selected_limit') if filters.get('recent_selected_limit') is not None else 'none'}",
        "",
        "## Score Summary",
        "",
        f"- top_score_count: {score_summary.get('top_score_count', 0)}",
        f"- mean: {_format_float(score_summary.get('top_score_mean'))}",
        f"- median: {_format_float(score_summary.get('top_score_median'))}",
        f"- min: {_format_float(score_summary.get('top_score_min'))}",
        f"- max: {_format_float(score_summary.get('top_score_max'))}",
        "",
        "## Downstream Return Quality",
        "",
    ]

    for field in RETURN_FIELDS:
        heading = field.replace("future_return_", "")
        metrics = return_quality.get(field, {})
        lines.extend(
            [
                f"### {heading}",
                "",
                f"- non_null_count: {metrics.get('non_null_count', 0)}",
                f"- positive_count: {metrics.get('positive_count', 0)}",
                f"- positive_rate_pct: {_format_pct(metrics.get('positive_rate_pct'))}",
                f"- median_return_pct: {_format_pct(metrics.get('median_return_pct'))}",
                f"- mean_return_pct: {_format_pct(metrics.get('mean_return_pct'))}",
                f"- min_return_pct: {_format_pct(metrics.get('min_return_pct'))}",
                f"- max_return_pct: {_format_pct(metrics.get('max_return_pct'))}",
                f"- p10_return_pct: {_format_pct(metrics.get('p10_return_pct'))}",
                f"- p25_return_pct: {_format_pct(metrics.get('p25_return_pct'))}",
                f"- p75_return_pct: {_format_pct(metrics.get('p75_return_pct'))}",
                f"- p90_return_pct: {_format_pct(metrics.get('p90_return_pct'))}",
                f"- mean_minus_median_pct: {_format_pct(metrics.get('mean_minus_median_pct'))}",
                "",
            ]
        )

    lines.extend(["## Interpretation Notes", ""])
    if interpretation_notes:
        lines.extend([f"- {note}" for note in interpretation_notes])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- verdict_label: {verdict.get('label')}",
            f"- explanation: {verdict.get('explanation')}",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_text(value: str | None) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    return cleaned.casefold()


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _safe_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _build_selected_row(
    payload: dict[str, Any],
    edge_output: dict[str, Any],
    line_number: int,
) -> SelectedRow:
    ranking = edge_output.get("ranking")
    top_score = None
    if isinstance(ranking, list) and ranking:
        first_entry = _safe_dict(ranking[0])
        if first_entry is not None:
            top_score = _safe_float(first_entry.get("score")) or _safe_float(
                first_entry.get("selection_score")
            )

    returns = {field: _safe_float(payload.get(field)) for field in RETURN_FIELDS}

    return SelectedRow(
        line_number=line_number,
        logged_at=_clean_text(payload.get("logged_at")),
        selected_symbol=_clean_text(edge_output.get("selected_symbol")),
        selected_strategy=_clean_text(edge_output.get("selected_strategy")),
        selected_horizon=_clean_text(edge_output.get("selected_horizon")),
        selection_reason=_clean_text(edge_output.get("reason")),
        top_score=top_score,
        returns=returns,
    )


def _matches_filters(
    row: SelectedRow,
    *,
    symbol: str | None,
    strategy: str | None,
    horizon: str | None,
) -> bool:
    row_symbol = _normalize_text(row.selected_symbol)
    row_strategy = _normalize_text(row.selected_strategy)
    row_horizon = _normalize_text(row.selected_horizon)

    filter_symbol = _normalize_text(symbol)
    filter_strategy = _normalize_text(strategy)
    filter_horizon = _normalize_text(horizon)

    if filter_symbol is not None and row_symbol != filter_symbol:
        return False
    if filter_strategy is not None and row_strategy != filter_strategy:
        return False
    if filter_horizon is not None and row_horizon != filter_horizon:
        return False
    return True


def _apply_recent_selected_limit(rows: list[SelectedRow], recent_selected: int | None) -> list[SelectedRow]:
    if recent_selected is None:
        return rows
    if recent_selected <= 0:
        return []

    ordered = sorted(
        rows,
        key=lambda row: (
            _safe_datetime(row.logged_at) or datetime.min.replace(tzinfo=UTC),
            row.line_number,
        ),
    )
    return ordered[-recent_selected:]


def _compute_return_summary(values: list[float | None]) -> dict[str, int | float | None]:
    cleaned = [value for value in values if value is not None]
    positive_count = sum(1 for value in cleaned if value > 0.0)
    mean_value = _mean(cleaned)
    median_value = _percentile(cleaned, 50.0)

    return {
        "non_null_count": len(cleaned),
        "positive_count": positive_count,
        "positive_rate_pct": _ratio_pct(positive_count, len(cleaned)),
        "median_return_pct": median_value,
        "mean_return_pct": mean_value,
        "min_return_pct": round(min(cleaned), 6) if cleaned else None,
        "max_return_pct": round(max(cleaned), 6) if cleaned else None,
        "p10_return_pct": _percentile(cleaned, 10.0),
        "p25_return_pct": _percentile(cleaned, 25.0),
        "p75_return_pct": _percentile(cleaned, 75.0),
        "p90_return_pct": _percentile(cleaned, 90.0),
        "mean_minus_median_pct": (
            round(mean_value - median_value, 6)
            if mean_value is not None and median_value is not None
            else None
        ),
    }


def _compute_score_summary(values: list[float | None]) -> dict[str, int | float | None]:
    cleaned = [value for value in values if value is not None]
    return {
        "top_score_count": len(cleaned),
        "top_score_mean": _mean(cleaned),
        "top_score_median": _percentile(cleaned, 50.0),
        "top_score_min": round(min(cleaned), 6) if cleaned else None,
        "top_score_max": round(max(cleaned), 6) if cleaned else None,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _ratio_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)

    bounded = min(max(percentile, 0.0), 100.0)
    position = (len(ordered) - 1) * (bounded / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    if lower_index == upper_index:
        return round(ordered[lower_index], 6)

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return round(lower_value + ((upper_value - lower_value) * weight), 6)


def _interpret_report(
    *,
    selected_row_count: int,
    returns_summary: dict[str, dict[str, int | float | None]],
    downstream_return_fields_available: bool,
) -> tuple[list[str], str, str]:
    metrics_15m = returns_summary["future_return_15m"]
    metrics_1h = returns_summary["future_return_1h"]
    metrics_4h = returns_summary["future_return_4h"]
    notes = [
        (
            "15m: "
            f"positive_rate={_format_pct(metrics_15m.get('positive_rate_pct'))}, "
            f"median={_format_pct(metrics_15m.get('median_return_pct'))}, "
            f"mean={_format_pct(metrics_15m.get('mean_return_pct'))}"
        ),
        (
            "1h: "
            f"positive_rate={_format_pct(metrics_1h.get('positive_rate_pct'))}, "
            f"median={_format_pct(metrics_1h.get('median_return_pct'))}, "
            f"mean={_format_pct(metrics_1h.get('mean_return_pct'))}"
        ),
        (
            "4h: "
            f"positive_rate={_format_pct(metrics_4h.get('positive_rate_pct'))}, "
            f"median={_format_pct(metrics_4h.get('median_return_pct'))}, "
            f"mean={_format_pct(metrics_4h.get('mean_return_pct'))}"
        ),
    ]

    if selected_row_count < MIN_ROWS_FOR_VERDICT:
        return (
            notes + ["Selected sample is still small, so the recovery signal is not yet robust."],
            "insufficient_selected_rows",
            "There are too few filtered selected rows to make a stable downstream quality call.",
        )

    if not downstream_return_fields_available:
        return (
            notes
            + [
                "Selected rows are present, but downstream future-return fields are not yet populated in the input rows.",
                "This input appears to be unlabeled or not yet matured for realized downstream quality evaluation.",
            ],
            "downstream_returns_not_yet_available",
            "Selected rows exist, but downstream future-return labels are not yet available, so realized quality cannot be evaluated yet.",
        )

    if _is_outlier_driven(metrics_15m) or _is_outlier_driven(metrics_1h) or _is_outlier_driven(metrics_4h):
        return (
            notes
            + [
                "Mean returns are materially above median returns on at least one horizon, "
                "suggesting outlier dependence."
            ],
            "outlier_driven_recovery",
            "Recovered selection appears to rely on a small number of outsized winners rather than broad quality.",
        )

    if _healthy(metrics_1h) and _healthy(metrics_4h, require_positive_rate=False):
        if _healthy(metrics_4h):
            return (
                notes + ["Both 1h and 4h horizons look broadly supportive, not just short-term reactive."],
                "broad_multi_horizon_edge",
                "Recovered selected candidates are showing healthy downstream quality across multiple horizons.",
            )
        return (
            notes + ["1h looks healthy and 4h is not materially bad, but the longer-horizon sample still needs time."],
            "encouraging_but_early",
            "Recovered selection quality looks promising, though the longer-horizon confirmation is still early.",
        )

    if _healthy(metrics_15m) and not _healthy(metrics_1h, require_positive_rate=False):
        return (
            notes + ["Short-horizon returns look better than 1h and 4h, so the edge may decay quickly downstream."],
            "short_horizon_only_edge",
            "Recovered selection appears to help mainly at 15m, without clear follow-through at 1h or 4h.",
        )

    if _weak(metrics_15m) and _weak(metrics_1h) and _weak(metrics_4h):
        return (
            notes + ["Most measured horizons are weak or negative, which is a poor sign for downstream quality."],
            "selection_quality_concerning",
            "Recovered selection is producing candidates, but their downstream return profile is concerning.",
        )

    return (
        notes + ["Selections are flowing again, but the downstream return profile is still mixed rather than clearly strong."],
        "selection_recovery_without_clear_edge",
        "Selection recovery is visible, but the downstream quality is not yet clearly strong or broad-based.",
    )


def _healthy(
    metrics: dict[str, int | float | None],
    *,
    require_positive_rate: bool = True,
) -> bool:
    median_return = _optional_float(metrics.get("median_return_pct"))
    mean_return = _optional_float(metrics.get("mean_return_pct"))
    positive_rate = _optional_float(metrics.get("positive_rate_pct"))
    non_null_count = int(metrics.get("non_null_count", 0))

    if non_null_count < MIN_ROWS_FOR_VERDICT:
        return False
    if median_return is None or mean_return is None:
        return False
    if median_return <= 0.0 or mean_return <= 0.0:
        return False
    if require_positive_rate and (positive_rate is None or positive_rate < 55.0):
        return False
    return True


def _weak(metrics: dict[str, int | float | None]) -> bool:
    mean_return = _optional_float(metrics.get("mean_return_pct"))
    median_return = _optional_float(metrics.get("median_return_pct"))
    positive_rate = _optional_float(metrics.get("positive_rate_pct"))
    non_null_count = int(metrics.get("non_null_count", 0))

    if non_null_count < MIN_ROWS_FOR_VERDICT:
        return False
    if mean_return is None or median_return is None:
        return False
    if mean_return < 0.0 and median_return <= 0.0:
        return True
    return positive_rate is not None and positive_rate < 45.0 and median_return <= 0.0


def _is_outlier_driven(metrics: dict[str, int | float | None]) -> bool:
    mean_minus_median = _optional_float(metrics.get("mean_minus_median_pct"))
    mean_return = _optional_float(metrics.get("mean_return_pct"))
    median_return = _optional_float(metrics.get("median_return_pct"))
    non_null_count = int(metrics.get("non_null_count", 0))

    if non_null_count < MIN_ROWS_FOR_VERDICT:
        return False
    if mean_minus_median is None or mean_return is None or median_return is None:
        return False
    return (
        mean_return > 0.0
        and median_return <= 0.0
        and mean_minus_median >= OUTLIER_DIVERGENCE_THRESHOLD_PCT
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _selected_row_to_dict(row: SelectedRow) -> dict[str, Any]:
    return {
        "line_number": row.line_number,
        "logged_at": row.logged_at,
        "selected_symbol": row.selected_symbol,
        "selected_strategy": row.selected_strategy,
        "selected_horizon": row.selected_horizon,
        "selection_reason": row.selection_reason,
        "top_score": row.top_score,
        "future_return_15m": row.returns.get("future_return_15m"),
        "future_return_1h": row.returns.get("future_return_1h"),
        "future_return_4h": row.returns.get("future_return_4h"),
    }


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()