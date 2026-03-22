from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.experimental_candidate_comparison_matrix import (
    CANDIDATE_A_DEFAULT_PATH,
    CANDIDATE_B_DEFAULT_PATH,
    CANDIDATE_B_LABELING_METHOD,
    TARGET_HORIZONS,
    _build_delta_a_to_b,
    _candidate_summary,
    _format_pct,
    _group_highlights,
    _safe_dict,
    _safe_float,
    _safe_text,
    load_jsonl_records,
)

DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_b_intersection_comparison.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_b_intersection_comparison.md"
)

MATCH_KEY_FIELDS = (
    "logged_at",
    "symbol",
    "selected_strategy",
    "future_return_15m",
    "future_return_1h",
    "future_return_4h",
)
KEY_MISSING = "__missing__"


def _normalize_key_text(value: Any) -> str:
    text = _safe_text(value)
    return text if text is not None else KEY_MISSING


def _normalize_key_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return KEY_MISSING
    return f"{number:.12g}"


def build_row_match_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Build a deterministic composite key for Candidate A/B row alignment."""
    return (
        _normalize_key_text(row.get("logged_at")),
        _normalize_key_text(row.get("symbol")),
        _normalize_key_text(row.get("selected_strategy") or row.get("strategy")),
        _normalize_key_number(row.get("future_return_15m")),
        _normalize_key_number(row.get("future_return_1h")),
        _normalize_key_number(row.get("future_return_4h")),
    )


def _filter_candidate_b_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in records:
        metadata = _safe_dict(row.get("experimental_labeling"))
        if metadata.get("labeling_method") == CANDIDATE_B_LABELING_METHOD:
            filtered.append(row)
    return filtered


def build_intersection_datasets(
    candidate_a_rows: list[dict[str, Any]],
    candidate_b_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    indexed_a: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    indexed_b: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in candidate_a_rows:
        indexed_a[build_row_match_key(row)].append(row)
    for row in candidate_b_rows:
        indexed_b[build_row_match_key(row)].append(row)

    candidate_a_intersection_rows: list[dict[str, Any]] = []
    candidate_b_intersection_rows: list[dict[str, Any]] = []

    all_keys = sorted(set(indexed_a) | set(indexed_b))
    candidate_a_only_count = 0
    candidate_b_only_count = 0

    for key in all_keys:
        a_bucket = indexed_a.get(key, [])
        b_bucket = indexed_b.get(key, [])
        pair_count = min(len(a_bucket), len(b_bucket))

        if pair_count:
            candidate_a_intersection_rows.extend(a_bucket[:pair_count])
            candidate_b_intersection_rows.extend(b_bucket[:pair_count])

        if len(a_bucket) > pair_count:
            candidate_a_only_count += len(a_bucket) - pair_count
        if len(b_bucket) > pair_count:
            candidate_b_only_count += len(b_bucket) - pair_count

    summary = {
        "candidate_a_total_rows": len(candidate_a_rows),
        "candidate_b_total_rows": len(candidate_b_rows),
        "candidate_a_intersection_count": len(candidate_a_intersection_rows),
        "candidate_b_intersection_count": len(candidate_b_intersection_rows),
        "candidate_a_only_count": candidate_a_only_count,
        "candidate_b_only_count": candidate_b_only_count,
    }
    return candidate_a_intersection_rows, candidate_b_intersection_rows, summary


def _bucket_quality_score(horizon_delta: dict[str, Any]) -> tuple[int, int]:
    """
    Score whether Candidate B looks better or worse on one horizon.

    Interpretation:
    - up bucket: higher median / higher positive-rate is better
    - down bucket: more negative median / lower positive-rate is better
    - flat bucket: median closer to 0 is better
    - flat bucket positive-rate should be closer to 0.5 (neutral / mixed), not 0.0
    """
    favorable = 0
    adverse = 0

    up_bucket_median = _safe_float(horizon_delta.get("up_bucket_median_change"))
    up_bucket_positive = _safe_float(horizon_delta.get("up_bucket_positive_rate_change"))
    down_bucket_median = _safe_float(horizon_delta.get("down_bucket_median_change"))
    down_bucket_positive = _safe_float(horizon_delta.get("down_bucket_positive_rate_change"))
    flat_bucket_median = _safe_float(horizon_delta.get("flat_bucket_median_change"))
    flat_bucket_positive = _safe_float(horizon_delta.get("flat_bucket_positive_rate_change"))

    if up_bucket_median is not None:
        favorable += int(up_bucket_median > 0.0)
        adverse += int(up_bucket_median < 0.0)

    if up_bucket_positive is not None:
        favorable += int(up_bucket_positive > 0.0)
        adverse += int(up_bucket_positive < 0.0)

    if down_bucket_median is not None:
        favorable += int(down_bucket_median < 0.0)
        adverse += int(down_bucket_median > 0.0)

    if down_bucket_positive is not None:
        favorable += int(down_bucket_positive < 0.0)
        adverse += int(down_bucket_positive > 0.0)

    if flat_bucket_median is not None:
        favorable += int(abs(flat_bucket_median) < 0.01)
        adverse += int(abs(flat_bucket_median) >= 0.03)

    if flat_bucket_positive is not None:
        favorable += int(abs(flat_bucket_positive) < 0.05)
        adverse += int(abs(flat_bucket_positive) >= 0.15)

    return favorable, adverse


def _build_final_summary(delta_a_to_b_on_intersection: dict[str, Any]) -> dict[str, Any]:
    notes: list[str] = []
    reduced_flat_without_quality_harm = 0
    increased_flat_but_improved_purity = 0
    clearly_worse = 0

    for horizon in TARGET_HORIZONS:
        horizon_delta = _safe_dict(
            _safe_dict(delta_a_to_b_on_intersection.get("by_horizon")).get(horizon)
        )
        flat_change = _safe_float(horizon_delta.get("flat_ratio_change"))
        favorable_quality, adverse_quality = _bucket_quality_score(horizon_delta)

        if (
            flat_change is not None
            and flat_change < 0.0
            and favorable_quality >= adverse_quality
        ):
            reduced_flat_without_quality_harm += 1
            notes.append(
                f"{horizon}: Candidate B reduced flat share on the shared intersection without a clear bucket-quality regression."
            )
            continue

        if (
            flat_change is not None
            and flat_change > 0.0
            and favorable_quality >= adverse_quality + 2
        ):
            increased_flat_but_improved_purity += 1
            notes.append(
                f"{horizon}: Candidate B used more flat labels on common rows, but bucket-quality metrics improved enough to keep the result directionally interesting."
            )
            continue

        if adverse_quality >= favorable_quality + 2:
            clearly_worse += 1
            notes.append(
                f"{horizon}: Candidate B looked weaker on common rows because bucket-quality regressions outweighed any distribution benefit."
            )

    if reduced_flat_without_quality_harm > 0 and clearly_worse == 0:
        primary_finding = "candidate_b_reduces_flat_without_hurting_bucket_quality"
        secondary_finding = "common_intersection_supports_candidate_b"
    elif increased_flat_but_improved_purity > 0 and clearly_worse == 0:
        primary_finding = "candidate_b_increases_flat_but_materially_improves_bucket_purity"
        secondary_finding = "candidate_b_may_be_trading_coverage_for_purity"
    elif clearly_worse > 0 and reduced_flat_without_quality_harm == 0:
        primary_finding = "candidate_b_looks_worse_on_common_intersection"
        secondary_finding = "bucket_quality_regression_detected"
    else:
        primary_finding = "candidate_b_remains_mixed_on_common_intersection"
        secondary_finding = "intersection_follow_up_still_required"

    if not notes:
        notes.append(
            "The common-row comparison did not reveal a clean structural winner; bucket-quality and distribution changes remained mixed."
        )
    notes.append(
        "This summary is intentionally intersection-only so labeling-policy effects are not conflated with dataset population differences."
    )

    return {
        "primary_finding": primary_finding,
        "secondary_finding": secondary_finding,
        "notes": notes,
    }


def build_experimental_candidate_intersection_comparison(
    candidate_a_records: list[dict[str, Any]],
    candidate_b_records: list[dict[str, Any]],
    *,
    candidate_a_path: Path,
    candidate_b_path: Path,
    candidate_a_instrumentation: dict[str, int] | None = None,
    candidate_b_instrumentation: dict[str, int] | None = None,
) -> dict[str, Any]:
    filtered_candidate_b_records = _filter_candidate_b_records(candidate_b_records)
    (
        candidate_a_intersection_rows,
        candidate_b_intersection_rows,
        intersection_summary,
    ) = build_intersection_datasets(candidate_a_records, filtered_candidate_b_records)

    candidate_a_intersection = _candidate_summary(
        candidate_a_intersection_rows,
        include_volatility_metadata=False,
    )
    candidate_b_intersection = _candidate_summary(
        candidate_b_intersection_rows,
        include_volatility_metadata=True,
    )
    delta_a_to_b_on_intersection = _build_delta_a_to_b(
        candidate_a_intersection,
        candidate_b_intersection,
    )

    strategy_highlights = _group_highlights(
        candidate_a_intersection["positive_rate_by_strategy"],
        candidate_b_intersection["positive_rate_by_strategy"],
        category="strategy",
    )
    symbol_highlights = _group_highlights(
        candidate_a_intersection["positive_rate_by_symbol"],
        candidate_b_intersection["positive_rate_by_symbol"],
        category="symbol",
    )
    final_summary = _build_final_summary(delta_a_to_b_on_intersection)

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "comparison_name": "candidate_a_vs_b_intersection",
            "report_type": "experimental_candidate_intersection_comparison",
        },
        "inputs": {
            "candidate_a_path": str(candidate_a_path),
            "candidate_b_path": str(candidate_b_path),
            "candidate_a_parser_instrumentation": candidate_a_instrumentation or {},
            "candidate_b_parser_instrumentation": candidate_b_instrumentation or {},
            "candidate_a_raw_total_rows": len(candidate_a_records),
            "candidate_b_raw_total_rows": len(candidate_b_records),
            "candidate_b_filtered_row_count": len(filtered_candidate_b_records),
            "candidate_b_required_labeling_method": CANDIDATE_B_LABELING_METHOD,
            "match_key_fields": list(MATCH_KEY_FIELDS),
        },
        "intersection_summary": intersection_summary,
        "candidate_a_intersection": candidate_a_intersection,
        "candidate_b_intersection": candidate_b_intersection,
        "delta_a_to_b_on_intersection": delta_a_to_b_on_intersection,
        "highlights": {
            "strategy_level": strategy_highlights,
            "symbol_level": symbol_highlights,
        },
        "final_summary": final_summary,
    }


def build_experimental_candidate_intersection_markdown(summary: dict[str, Any]) -> str:
    intersection_summary = _safe_dict(summary.get("intersection_summary"))
    candidate_b_intersection = _safe_dict(summary.get("candidate_b_intersection"))
    delta_by_horizon = _safe_dict(
        _safe_dict(summary.get("delta_a_to_b_on_intersection")).get("by_horizon")
    )
    final_summary = _safe_dict(summary.get("final_summary"))
    highlights = _safe_dict(summary.get("highlights"))

    lines = [
        "# Candidate A vs Candidate B Intersection Comparison",
        "",
        "## Intersection Summary",
        f"- Candidate A total_rows: {intersection_summary.get('candidate_a_total_rows', 0)}",
        f"- Candidate B total_rows: {intersection_summary.get('candidate_b_total_rows', 0)}",
        f"- Candidate A intersection_count: {intersection_summary.get('candidate_a_intersection_count', 0)}",
        f"- Candidate B intersection_count: {intersection_summary.get('candidate_b_intersection_count', 0)}",
        f"- Candidate A only_count: {intersection_summary.get('candidate_a_only_count', 0)}",
        f"- Candidate B only_count: {intersection_summary.get('candidate_b_only_count', 0)}",
        "",
        "## Candidate B Volatility Metadata On Intersection",
        f"- fallback_row_count: {_safe_dict(candidate_b_intersection.get('volatility_metadata')).get('fallback_row_count', 0)}",
        f"- fallback_row_ratio: {_format_pct(_safe_float(_safe_dict(candidate_b_intersection.get('volatility_metadata')).get('fallback_row_ratio')))}",
        "",
        "## Horizon Delta On Shared Rows",
    ]

    for horizon in TARGET_HORIZONS:
        delta = _safe_dict(delta_by_horizon.get(horizon))
        lines.append(
            f"- {horizon}: "
            f"flat_ratio_change={_format_pct(_safe_float(delta.get('flat_ratio_change')))}, "
            f"up_ratio_change={_format_pct(_safe_float(delta.get('up_ratio_change')))}, "
            f"down_ratio_change={_format_pct(_safe_float(delta.get('down_ratio_change')))}, "
            f"up_bucket_median_change={_format_pct(_safe_float(delta.get('up_bucket_median_change')))}, "
            f"down_bucket_median_change={_format_pct(_safe_float(delta.get('down_bucket_median_change')))}, "
            f"flat_bucket_median_change={_format_pct(_safe_float(delta.get('flat_bucket_median_change')))}, "
            f"up_bucket_positive_rate_change={_format_pct(_safe_float(delta.get('up_bucket_positive_rate_change')))}, "
            f"down_bucket_positive_rate_change={_format_pct(_safe_float(delta.get('down_bucket_positive_rate_change')))}, "
            f"flat_bucket_positive_rate_change={_format_pct(_safe_float(delta.get('flat_bucket_positive_rate_change')))}"
        )

    lines.extend(["", "## Strategy-Level Highlights"])
    for row in highlights.get("strategy_level", []):
        row_payload = _safe_dict(row)
        lines.append(
            f"- {row_payload.get('group', 'unknown')}: impact_score={_format_pct(_safe_float(row_payload.get('impact_score')))}"
        )

    lines.extend(["", "## Symbol-Level Highlights"])
    for row in highlights.get("symbol_level", []):
        row_payload = _safe_dict(row)
        lines.append(
            f"- {row_payload.get('group', 'unknown')}: impact_score={_format_pct(_safe_float(row_payload.get('impact_score')))}"
        )

    lines.extend(
        [
            "",
            "## Final Summary",
            f"- primary_finding: {final_summary.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {final_summary.get('secondary_finding', 'unknown')}",
        ]
    )
    for note in final_summary.get("notes", []):
        lines.append(f"- note: {note}")

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_intersection_comparison(
    candidate_a_path: Path = CANDIDATE_A_DEFAULT_PATH,
    candidate_b_path: Path = CANDIDATE_B_DEFAULT_PATH,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
) -> dict[str, Any]:
    candidate_a_records, candidate_a_instrumentation = load_jsonl_records(candidate_a_path)
    candidate_b_records, candidate_b_instrumentation = load_jsonl_records(candidate_b_path)

    summary = build_experimental_candidate_intersection_comparison(
        candidate_a_records,
        candidate_b_records,
        candidate_a_path=candidate_a_path,
        candidate_b_path=candidate_b_path,
        candidate_a_instrumentation=candidate_a_instrumentation,
        candidate_b_instrumentation=candidate_b_instrumentation,
    )
    markdown = build_experimental_candidate_intersection_markdown(summary)

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
    parser = argparse.ArgumentParser(
        description="Build an intersection-aware research-only comparison for Candidate A and Candidate B"
    )
    parser.add_argument("--candidate-a-path", type=Path, default=CANDIDATE_A_DEFAULT_PATH)
    parser.add_argument("--candidate-b-path", type=Path, default=CANDIDATE_B_DEFAULT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_intersection_comparison(
        candidate_a_path=args.candidate_a_path,
        candidate_b_path=args.candidate_b_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
