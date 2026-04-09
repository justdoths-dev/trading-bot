from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.experimental_candidate_comparison_matrix import (
    CANDIDATE_A_DEFAULT_PATH,
    TARGET_HORIZONS,
    TARGET_LABELS,
    _candidate_summary,
    _format_pct,
    _safe_float,
    _safe_text,
    load_jsonl_records,
)
from src.research.experimental_candidate_comparison_utils import load_summary_json
from src.research.experimental_candidate_intersection_utils import (
    MATCH_KEY_FIELDS,
    build_intersection_datasets,
    build_row_match_key,
    filter_candidate_c_records,
)

DEFAULT_CANDIDATE_C_DATASET = Path(
    "logs/experiments/trade_analysis_relabel_candidate_c_c2_moderate.jsonl"
)
DEFAULT_CANDIDATE_A_SUMMARY = Path(
    "logs/research_reports/experiments/candidate_a/latest/summary.json"
)
DEFAULT_CANDIDATE_C_SUMMARY = Path(
    "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate/summary.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_c_intersection_comparison.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_c_intersection_comparison.md"
)
DEFAULT_CANDIDATE_C_VARIANT = "c2_moderate"
MEANINGFUL_COVERAGE_DELTA = -0.03
MEANINGFUL_PURITY_RATE_DELTA = 0.03
MEANINGFUL_PURITY_RETURN_DELTA = 0.02


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _compare_numeric(candidate_a: float | int | None, candidate_c: float | int | None) -> dict[str, Any]:
    a_value = None if candidate_a is None else float(candidate_a)
    c_value = None if candidate_c is None else float(candidate_c)
    delta = None
    if a_value is not None and c_value is not None:
        delta = round(c_value - a_value, 6)
    return {
        "candidate_a": a_value,
        "candidate_c": c_value,
        "delta": delta,
    }


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(median(values), 6)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(mean(values), 6)


def _valid_label(value: Any) -> str | None:
    text = _safe_text(value)
    if text in TARGET_LABELS:
        return text
    return None


def _extract_label_ratio(summary: dict[str, Any], horizon: str, label: str) -> float | None:
    distribution = _safe_dict(summary.get("label_distribution_by_horizon"))
    horizon_payload = _safe_dict(distribution.get(horizon))
    return _safe_float(_safe_dict(horizon_payload.get(label)).get("ratio"))


def _extract_bucket_count(summary: dict[str, Any], horizon: str, label: str) -> int:
    conditional_positive = _safe_dict(summary.get("label_conditional_positive_rate_by_horizon"))
    horizon_payload = _safe_dict(conditional_positive.get(horizon))
    return int(_safe_dict(horizon_payload.get(label)).get("row_count", 0))


def _extract_bucket_positive_rate(summary: dict[str, Any], horizon: str, label: str) -> float | None:
    conditional_positive = _safe_dict(summary.get("label_conditional_positive_rate_by_horizon"))
    horizon_payload = _safe_dict(conditional_positive.get(horizon))
    return _safe_float(_safe_dict(horizon_payload.get(label)).get("positive_rate"))


def _extract_bucket_median(summary: dict[str, Any], horizon: str, label: str) -> float | None:
    conditional_median = _safe_dict(summary.get("label_conditional_median_future_return_by_horizon"))
    horizon_payload = _safe_dict(conditional_median.get(horizon))
    return _safe_float(horizon_payload.get(label))


def _format_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.6f}"


def _format_count_delta(payload: dict[str, Any]) -> str:
    return _format_number(_safe_dict(payload).get("delta"))


def _build_shared_row_horizon_comparison(
    candidate_a_summary: dict[str, Any],
    candidate_c_summary: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        result[horizon] = {
            "flat_share": _compare_numeric(
                _extract_label_ratio(candidate_a_summary, horizon, "flat"),
                _extract_label_ratio(candidate_c_summary, horizon, "flat"),
            ),
            "up_share": _compare_numeric(
                _extract_label_ratio(candidate_a_summary, horizon, "up"),
                _extract_label_ratio(candidate_c_summary, horizon, "up"),
            ),
            "down_share": _compare_numeric(
                _extract_label_ratio(candidate_a_summary, horizon, "down"),
                _extract_label_ratio(candidate_c_summary, horizon, "down"),
            ),
        }
    return result


def _build_bucket_return_stats(
    records: list[dict[str, Any]],
    *,
    horizon: str,
    label: str,
) -> dict[str, Any]:
    returns: list[float] = []
    for row in records:
        if _valid_label(row.get(f"future_label_{horizon}")) != label:
            continue
        future_return = _safe_float(row.get(f"future_return_{horizon}"))
        if future_return is None:
            continue
        returns.append(future_return)

    positive_count = sum(1 for value in returns if value > 0.0)
    return {
        "row_count": len(returns),
        "positive_rate": _safe_ratio(positive_count, len(returns)) if returns else None,
        "median_return": _median_or_none(returns),
        "mean_return": _mean_or_none(returns),
    }


def _build_purity_delta_on_shared_rows(
    candidate_a_rows: list[dict[str, Any]],
    candidate_c_rows: list[dict[str, Any]],
    candidate_a_summary: dict[str, Any],
    candidate_c_summary: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        a_up = _build_bucket_return_stats(candidate_a_rows, horizon=horizon, label="up")
        c_up = _build_bucket_return_stats(candidate_c_rows, horizon=horizon, label="up")
        a_down = _build_bucket_return_stats(candidate_a_rows, horizon=horizon, label="down")
        c_down = _build_bucket_return_stats(candidate_c_rows, horizon=horizon, label="down")

        result[horizon] = {
            "up_bucket_count": _compare_numeric(
                _extract_bucket_count(candidate_a_summary, horizon, "up"),
                _extract_bucket_count(candidate_c_summary, horizon, "up"),
            ),
            "up_bucket_positive_rate": _compare_numeric(
                _extract_bucket_positive_rate(candidate_a_summary, horizon, "up"),
                _extract_bucket_positive_rate(candidate_c_summary, horizon, "up"),
            ),
            "up_bucket_median_return": _compare_numeric(
                _extract_bucket_median(candidate_a_summary, horizon, "up"),
                _extract_bucket_median(candidate_c_summary, horizon, "up"),
            ),
            "up_bucket_mean_return": _compare_numeric(
                a_up.get("mean_return"),
                c_up.get("mean_return"),
            ),
            "down_bucket_count": _compare_numeric(
                _extract_bucket_count(candidate_a_summary, horizon, "down"),
                _extract_bucket_count(candidate_c_summary, horizon, "down"),
            ),
            "down_bucket_positive_rate": _compare_numeric(
                _extract_bucket_positive_rate(candidate_a_summary, horizon, "down"),
                _extract_bucket_positive_rate(candidate_c_summary, horizon, "down"),
            ),
            "down_bucket_impurity_rate": _compare_numeric(
                _extract_bucket_positive_rate(candidate_a_summary, horizon, "down"),
                _extract_bucket_positive_rate(candidate_c_summary, horizon, "down"),
            ),
            "down_bucket_median_return": _compare_numeric(
                _extract_bucket_median(candidate_a_summary, horizon, "down"),
                _extract_bucket_median(candidate_c_summary, horizon, "down"),
            ),
            "down_bucket_mean_return": _compare_numeric(
                a_down.get("mean_return"),
                c_down.get("mean_return"),
            ),
        }
    return result


def _build_exclusive_row_counts_by_horizon(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        counts = {label: 0 for label in TARGET_LABELS}
        labeled_rows = 0
        for row in records:
            label = _valid_label(row.get(f"future_label_{horizon}"))
            if label is None:
                continue
            counts[label] += 1
            labeled_rows += 1

        result[horizon] = {
            "exclusive_rows": len(records),
            "labeled_rows": labeled_rows,
            "up": counts["up"],
            "down": counts["down"],
            "flat": counts["flat"],
        }
    return result


def _build_coverage_structure(
    candidate_a_only_rows: list[dict[str, Any]],
    candidate_c_only_rows: list[dict[str, Any]],
    shared_row_horizon_comparison: dict[str, Any],
) -> dict[str, Any]:
    a_only_by_horizon = _build_exclusive_row_counts_by_horizon(candidate_a_only_rows)
    c_only_by_horizon = _build_exclusive_row_counts_by_horizon(candidate_c_only_rows)

    interpretation: list[str] = []
    if len(candidate_c_only_rows) > len(candidate_a_only_rows):
        interpretation.append(
            "Candidate C contributes more exclusive rows than Candidate A, so its coverage expansion is not just a shared-row relabel effect."
        )
    elif len(candidate_c_only_rows) < len(candidate_a_only_rows):
        interpretation.append(
            "Candidate C contributes fewer exclusive rows than Candidate A, so any shared-row coverage recovery should be read as relabel redistribution more than outright row expansion."
        )
    else:
        interpretation.append(
            "Candidate A and Candidate C contribute the same number of exclusive rows, so the main difference is how shared rows are relabeled."
        )

    for horizon in TARGET_HORIZONS:
        a_payload = _safe_dict(a_only_by_horizon.get(horizon))
        c_payload = _safe_dict(c_only_by_horizon.get(horizon))
        shared_payload = _safe_dict(shared_row_horizon_comparison.get(horizon))
        flat_delta = _safe_float(_safe_dict(shared_payload.get("flat_share")).get("delta"))
        c_seed_like = int(c_payload.get("up", 0)) + int(c_payload.get("down", 0))
        a_seed_like = int(a_payload.get("up", 0)) + int(a_payload.get("down", 0))

        if c_seed_like > a_seed_like:
            interpretation.append(
                f"{horizon}: Candidate C has more exclusive directional rows ({c_seed_like} vs {a_seed_like}), which supports a coverage-expansion reading."
            )
        elif c_seed_like < a_seed_like:
            interpretation.append(
                f"{horizon}: Candidate C has fewer exclusive directional rows ({c_seed_like} vs {a_seed_like}), so shared-row purity matters more than raw expansion."
            )

        if flat_delta is not None and flat_delta <= MEANINGFUL_COVERAGE_DELTA:
            interpretation.append(
                f"{horizon}: shared-row flat share is lower for Candidate C, which suggests coverage recovery on the common row set."
            )

    interpretation.append(
        "Exclusive-row counts are derived from the existing row match key. If duplicate rows share the same match key, exclusive-row assignment can be order-sensitive, so purity conclusions should still be anchored primarily on shared rows."
    )

    return {
        "candidate_a_only_total_rows": len(candidate_a_only_rows),
        "candidate_c_only_total_rows": len(candidate_c_only_rows),
        "exclusive_row_counts_by_horizon": {
            "candidate_a_only": a_only_by_horizon,
            "candidate_c_only": c_only_by_horizon,
        },
        "interpretation": interpretation,
    }


def _build_final_diagnosis(
    shared_row_horizon_comparison: dict[str, Any],
    purity_delta_on_shared_rows: dict[str, Any],
    coverage_structure: dict[str, Any],
) -> dict[str, Any]:
    coverage_recovery = 0
    material_purity_giveback = 0
    notes: list[str] = []

    for horizon in TARGET_HORIZONS:
        directional = _safe_dict(shared_row_horizon_comparison.get(horizon))
        purity = _safe_dict(purity_delta_on_shared_rows.get(horizon))

        flat_delta = _safe_float(_safe_dict(directional.get("flat_share")).get("delta"))
        up_positive_delta = _safe_float(_safe_dict(purity.get("up_bucket_positive_rate")).get("delta"))
        up_median_delta = _safe_float(_safe_dict(purity.get("up_bucket_median_return")).get("delta"))
        down_positive_delta = _safe_float(_safe_dict(purity.get("down_bucket_positive_rate")).get("delta"))
        down_impurity_delta = _safe_float(_safe_dict(purity.get("down_bucket_impurity_rate")).get("delta"))
        down_median_delta = _safe_float(_safe_dict(purity.get("down_bucket_median_return")).get("delta"))
        down_mean_delta = _safe_float(_safe_dict(purity.get("down_bucket_mean_return")).get("delta"))

        if flat_delta is not None and flat_delta <= MEANINGFUL_COVERAGE_DELTA:
            coverage_recovery += 1

        horizon_loss = False
        if up_positive_delta is not None and up_positive_delta <= -MEANINGFUL_PURITY_RATE_DELTA:
            horizon_loss = True
        if up_median_delta is not None and up_median_delta <= -MEANINGFUL_PURITY_RETURN_DELTA:
            horizon_loss = True
        if down_positive_delta is not None and down_positive_delta >= MEANINGFUL_PURITY_RATE_DELTA:
            horizon_loss = True
        if down_median_delta is not None and down_median_delta >= MEANINGFUL_PURITY_RETURN_DELTA:
            horizon_loss = True
        if down_mean_delta is not None and down_mean_delta >= MEANINGFUL_PURITY_RETURN_DELTA:
            horizon_loss = True

        material_purity_giveback += int(horizon_loss)
        notes.append(
            f"{horizon}: flat_delta={flat_delta if flat_delta is not None else 'n/a'}, "
            f"up_positive_delta={up_positive_delta if up_positive_delta is not None else 'n/a'}, "
            f"down_positive_delta={down_positive_delta if down_positive_delta is not None else 'n/a'}, "
            f"down_impurity_delta={down_impurity_delta if down_impurity_delta is not None else 'n/a'}."
        )

    c_only_total = int(coverage_structure.get("candidate_c_only_total_rows", 0))
    a_only_total = int(coverage_structure.get("candidate_a_only_total_rows", 0))
    c_expands = c_only_total > a_only_total

    if coverage_recovery > 0 and material_purity_giveback == 0:
        primary_finding = "candidate_c_recovers_shared_row_coverage_without_clear_purity_giveback"
        secondary_finding = "candidate_c_looks_better_suited_for_seed_starved_edge_selection"
        recommendation = "Prefer Candidate C2 for further seed-availability experiments while continuing to monitor down-bucket contamination."
    elif coverage_recovery > 0 and material_purity_giveback > 0:
        primary_finding = "candidate_c_recovers_shared_row_coverage_but_gives_back_some_purity"
        secondary_finding = "c2_tradeoff_is_real_and_should_be_judged_by_tolerance_for_seed_noise"
        recommendation = "Treat Candidate C2 as viable only if the observed purity give-back is acceptable for the downstream edge-selection gate."
    elif coverage_recovery == 0 and material_purity_giveback == 0:
        primary_finding = "candidate_c_does_not_show_clear_shared_row_coverage_recovery_vs_candidate_a"
        secondary_finding = "shared_rows_do_not_yet_justify_switching_on_seed_starvation_grounds"
        recommendation = "Keep Candidate A as the cleaner default and use Candidate C2 only for targeted follow-up where coverage scarcity is the binding issue."
    else:
        primary_finding = "candidate_c_looks_weaker_than_candidate_a_on_shared_rows"
        secondary_finding = "coverage_recovery_is_not_large_enough_to_offset_purity_giveback"
        recommendation = "Do not prefer Candidate C2 for the seed-starved path unless additional filtering recovers purity."

    summary = (
        "Shared-row interpretation prioritizes whether Candidate C reduces flat share while keeping directional bucket quality intact enough for a seed-starved edge-selection system."
    )
    if c_expands:
        summary += " Candidate C also adds more exclusive rows than Candidate A, so some of its benefit comes from broader row coverage."
    elif c_only_total < a_only_total:
        summary += " Candidate C adds fewer exclusive rows than Candidate A, so the decision should lean more heavily on shared-row purity."
    else:
        summary += " Exclusive-row totals are balanced, so the main question is shared-row purity versus shared-row coverage."

    return {
        "primary_finding": primary_finding,
        "secondary_finding": secondary_finding,
        "recommendation": recommendation,
        "summary": summary,
        "notes": notes,
    }


def _build_exclusive_rows(
    source_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_counts: dict[tuple[str, ...], int] = {}
    shared_counts: dict[tuple[str, ...], int] = {}

    for row in source_rows:
        key = build_row_match_key(row)
        source_counts[key] = source_counts.get(key, 0) + 1
    for row in shared_rows:
        key = build_row_match_key(row)
        shared_counts[key] = shared_counts.get(key, 0) + 1

    exclusive_rows: list[dict[str, Any]] = []
    emitted_counts: dict[tuple[str, ...], int] = {}
    for row in source_rows:
        key = build_row_match_key(row)
        allowed = source_counts.get(key, 0) - shared_counts.get(key, 0)
        emitted = emitted_counts.get(key, 0)
        if emitted >= allowed:
            continue
        exclusive_rows.append(row)
        emitted_counts[key] = emitted + 1

    return exclusive_rows


def build_experimental_candidate_ac_intersection_comparison_report(
    candidate_a_records: list[dict[str, Any]],
    candidate_c_records: list[dict[str, Any]],
    *,
    candidate_a_path: Path,
    candidate_c_path: Path,
    candidate_a_instrumentation: dict[str, int] | None = None,
    candidate_c_instrumentation: dict[str, int] | None = None,
    candidate_a_summary_path: Path = DEFAULT_CANDIDATE_A_SUMMARY,
    candidate_c_summary_path: Path = DEFAULT_CANDIDATE_C_SUMMARY,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
) -> dict[str, Any]:
    filtered_candidate_c_records = filter_candidate_c_records(
        candidate_c_records,
        variant_name=candidate_c_variant,
    )
    candidate_a_shared_rows, candidate_c_shared_rows, intersection_overview = build_intersection_datasets(
        candidate_a_records,
        filtered_candidate_c_records,
    )
    candidate_a_only_rows = _build_exclusive_rows(candidate_a_records, candidate_a_shared_rows)
    candidate_c_only_rows = _build_exclusive_rows(filtered_candidate_c_records, candidate_c_shared_rows)

    candidate_a_shared_summary = _candidate_summary(
        candidate_a_shared_rows,
        include_volatility_metadata=False,
    )
    candidate_c_shared_summary = _candidate_summary(
        candidate_c_shared_rows,
        include_volatility_metadata=False,
    )

    # These analyzer summaries are loaded only as reference metadata inputs.
    # The actual shared-row comparison metrics in this report are computed directly
    # from the intersection-aligned rows above, not from the analyzer summaries.
    candidate_a_summary_metadata, candidate_a_summary_loaded = load_summary_json(candidate_a_summary_path)
    candidate_c_summary_metadata, candidate_c_summary_loaded = load_summary_json(candidate_c_summary_path)

    shared_row_horizon_comparison = _build_shared_row_horizon_comparison(
        candidate_a_shared_summary,
        candidate_c_shared_summary,
    )
    purity_delta_on_shared_rows = _build_purity_delta_on_shared_rows(
        candidate_a_shared_rows,
        candidate_c_shared_rows,
        candidate_a_shared_summary,
        candidate_c_shared_summary,
    )
    coverage_structure = _build_coverage_structure(
        candidate_a_only_rows,
        candidate_c_only_rows,
        shared_row_horizon_comparison,
    )
    final_diagnosis = _build_final_diagnosis(
        shared_row_horizon_comparison,
        purity_delta_on_shared_rows,
        coverage_structure,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "comparison_name": "candidate_a_vs_c_intersection",
            "report_type": "experimental_candidate_ac_intersection_comparison_report",
        },
        "inputs": {
            "candidate_a_path": str(candidate_a_path),
            "candidate_c_path": str(candidate_c_path),
            "candidate_a_parser_instrumentation": candidate_a_instrumentation or {},
            "candidate_c_parser_instrumentation": candidate_c_instrumentation or {},
            "candidate_a_raw_total_rows": len(candidate_a_records),
            "candidate_c_raw_total_rows": len(candidate_c_records),
            "candidate_c_filtered_row_count": len(filtered_candidate_c_records),
            "candidate_c_variant": candidate_c_variant,
            "row_key_definition_summary": (
                "Shared rows are aligned with the existing experimental intersection key: "
                + ", ".join(MATCH_KEY_FIELDS)
                + "."
            ),
            "relabel_inputs": {
                "candidate_a_dataset_path": str(candidate_a_path),
                "candidate_c_dataset_path": str(candidate_c_path),
            },
            "analyzer_summary_inputs": {
                "role": "reference_metadata_only",
                "note": (
                    "Analyzer summaries are loaded only to record the upstream summary context. "
                    "Shared-row comparison metrics in this report are computed directly from intersection-aligned rows."
                ),
                "candidate_a_summary_path": str(candidate_a_summary_path),
                "candidate_c_summary_path": str(candidate_c_summary_path),
                "candidate_a_summary_loaded": candidate_a_summary_loaded,
                "candidate_c_summary_loaded": candidate_c_summary_loaded,
                "candidate_a_summary_report_type": candidate_a_summary_metadata.get("report_type"),
                "candidate_c_summary_report_type": candidate_c_summary_metadata.get("report_type"),
            },
        },
        "alignment": {
            "candidate_a_rows": int(intersection_overview.get("baseline_total_rows", 0)),
            "candidate_c_rows": int(intersection_overview.get("experiment_total_rows", 0)),
            "shared_rows": int(intersection_overview.get("shared_row_count", 0)),
            "candidate_a_only_rows": int(intersection_overview.get("baseline_only_row_count", 0)),
            "candidate_c_only_rows": int(intersection_overview.get("experiment_only_row_count", 0)),
            "shared_row_ratio_from_a": _safe_float(intersection_overview.get("shared_ratio_vs_baseline")),
            "shared_row_ratio_from_c": _safe_float(intersection_overview.get("shared_ratio_vs_experiment")),
        },
        "shared_row_horizon_comparison": shared_row_horizon_comparison,
        "purity_delta_on_shared_rows": purity_delta_on_shared_rows,
        "coverage_structure": coverage_structure,
        "final_diagnosis": final_diagnosis,
    }


def build_experimental_candidate_ac_intersection_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    inputs = _safe_dict(summary.get("inputs"))
    alignment = _safe_dict(summary.get("alignment"))
    shared = _safe_dict(summary.get("shared_row_horizon_comparison"))
    purity = _safe_dict(summary.get("purity_delta_on_shared_rows"))
    coverage = _safe_dict(summary.get("coverage_structure"))
    diagnosis = _safe_dict(summary.get("final_diagnosis"))
    analyzer_inputs = _safe_dict(inputs.get("analyzer_summary_inputs"))
    exclusive = _safe_dict(coverage.get("exclusive_row_counts_by_horizon"))
    a_only = _safe_dict(exclusive.get("candidate_a_only"))
    c_only = _safe_dict(exclusive.get("candidate_c_only"))

    lines = [
        "# Candidate A vs Candidate C Intersection Comparison",
        "",
        "## Metadata",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- comparison_name: {metadata.get('comparison_name', 'n/a')}",
        f"- report_type: {metadata.get('report_type', 'n/a')}",
        "",
        "## Inputs",
        f"- candidate_a_path: {inputs.get('candidate_a_path', 'n/a')}",
        f"- candidate_c_path: {inputs.get('candidate_c_path', 'n/a')}",
        f"- candidate_a_parser_instrumentation: {json.dumps(inputs.get('candidate_a_parser_instrumentation', {}), ensure_ascii=False, sort_keys=True)}",
        f"- candidate_c_parser_instrumentation: {json.dumps(inputs.get('candidate_c_parser_instrumentation', {}), ensure_ascii=False, sort_keys=True)}",
        f"- row_key_definition_summary: {inputs.get('row_key_definition_summary', 'n/a')}",
        "- relabel_inputs: separated from analyzer_summary_inputs to avoid mixing row-alignment facts with analyzer-level summary context.",
        f"- analyzer_summary_inputs_role: {analyzer_inputs.get('role', 'n/a')}",
        f"- analyzer_summary_inputs_note: {analyzer_inputs.get('note', 'n/a')}",
        f"- analyzer_summary_inputs_loaded: candidate_a_loaded={analyzer_inputs.get('candidate_a_summary_loaded', False)}, candidate_c_loaded={analyzer_inputs.get('candidate_c_summary_loaded', False)}",
        "",
        "## Alignment",
        f"- candidate_a_rows: {alignment.get('candidate_a_rows', 0)}",
        f"- candidate_c_rows: {alignment.get('candidate_c_rows', 0)}",
        f"- shared_rows: {alignment.get('shared_rows', 0)}",
        f"- candidate_a_only_rows: {alignment.get('candidate_a_only_rows', 0)}",
        f"- candidate_c_only_rows: {alignment.get('candidate_c_only_rows', 0)}",
        f"- shared_row_ratio_from_a: {_format_pct(_safe_float(alignment.get('shared_row_ratio_from_a')))}",
        f"- shared_row_ratio_from_c: {_format_pct(_safe_float(alignment.get('shared_row_ratio_from_c')))}",
        "",
        "## Shared-Row Horizon Comparison",
    ]

    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(shared.get(horizon))
        lines.append(
            f"- {horizon}: "
            f"flat A={_format_pct(_safe_float(_safe_dict(payload.get('flat_share')).get('candidate_a')))}, "
            f"C={_format_pct(_safe_float(_safe_dict(payload.get('flat_share')).get('candidate_c')))}, "
            f"delta={_format_pct(_safe_float(_safe_dict(payload.get('flat_share')).get('delta')))}; "
            f"up A={_format_pct(_safe_float(_safe_dict(payload.get('up_share')).get('candidate_a')))}, "
            f"C={_format_pct(_safe_float(_safe_dict(payload.get('up_share')).get('candidate_c')))}, "
            f"delta={_format_pct(_safe_float(_safe_dict(payload.get('up_share')).get('delta')))}; "
            f"down A={_format_pct(_safe_float(_safe_dict(payload.get('down_share')).get('candidate_a')))}, "
            f"C={_format_pct(_safe_float(_safe_dict(payload.get('down_share')).get('candidate_c')))}, "
            f"delta={_format_pct(_safe_float(_safe_dict(payload.get('down_share')).get('delta')))}"
        )

    lines.extend(["", "## Purity Delta On Shared Rows"])
    lines.append(
        "- note: down_bucket_positive_rate and down_bucket_impurity_rate refer to the same underlying metric here. "
        "For down buckets, a higher positive rate means worse purity because more supposedly-down rows ended up positive."
    )
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(purity.get(horizon))
        lines.append(
            f"- {horizon}: "
            f"up_count_delta={_format_count_delta(_safe_dict(payload.get('up_bucket_count')))}, "
            f"up_positive_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('up_bucket_positive_rate')).get('delta')))}, "
            f"up_median_delta={_format_pct(_safe_float(_safe_dict(payload.get('up_bucket_median_return')).get('delta')))}, "
            f"up_mean_delta={_format_pct(_safe_float(_safe_dict(payload.get('up_bucket_mean_return')).get('delta')))}, "
            f"down_count_delta={_format_count_delta(_safe_dict(payload.get('down_bucket_count')))}, "
            f"down_positive_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('down_bucket_positive_rate')).get('delta')))}, "
            f"down_impurity_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('down_bucket_impurity_rate')).get('delta')))}, "
            f"down_median_delta={_format_pct(_safe_float(_safe_dict(payload.get('down_bucket_median_return')).get('delta')))}, "
            f"down_mean_delta={_format_pct(_safe_float(_safe_dict(payload.get('down_bucket_mean_return')).get('delta')))}"
        )

    lines.extend(["", "## Coverage Structure"])
    for horizon in TARGET_HORIZONS:
        a_payload = _safe_dict(a_only.get(horizon))
        c_payload = _safe_dict(c_only.get(horizon))
        lines.append(
            f"- {horizon}: "
            f"A_only labeled={a_payload.get('labeled_rows', 0)} (up={a_payload.get('up', 0)}, down={a_payload.get('down', 0)}, flat={a_payload.get('flat', 0)}); "
            f"C_only labeled={c_payload.get('labeled_rows', 0)} (up={c_payload.get('up', 0)}, down={c_payload.get('down', 0)}, flat={c_payload.get('flat', 0)})"
        )
    for note in coverage.get("interpretation", []):
        lines.append(f"- interpretation: {note}")

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            f"- primary_finding: {diagnosis.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {diagnosis.get('secondary_finding', 'unknown')}",
            f"- recommendation: {diagnosis.get('recommendation', 'unknown')}",
            f"- summary: {diagnosis.get('summary', 'unknown')}",
        ]
    )
    for note in diagnosis.get("notes", []):
        lines.append(f"- note: {note}")

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_ac_intersection_comparison_report(
    candidate_a_path: Path = CANDIDATE_A_DEFAULT_PATH,
    candidate_c_path: Path = DEFAULT_CANDIDATE_C_DATASET,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
    candidate_a_summary_path: Path = DEFAULT_CANDIDATE_A_SUMMARY,
    candidate_c_summary_path: Path = DEFAULT_CANDIDATE_C_SUMMARY,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
) -> dict[str, Any]:
    candidate_a_records, candidate_a_instrumentation = load_jsonl_records(candidate_a_path)
    candidate_c_records, candidate_c_instrumentation = load_jsonl_records(candidate_c_path)

    summary = build_experimental_candidate_ac_intersection_comparison_report(
        candidate_a_records,
        candidate_c_records,
        candidate_a_path=candidate_a_path,
        candidate_c_path=candidate_c_path,
        candidate_a_instrumentation=candidate_a_instrumentation,
        candidate_c_instrumentation=candidate_c_instrumentation,
        candidate_a_summary_path=candidate_a_summary_path,
        candidate_c_summary_path=candidate_c_summary_path,
        candidate_c_variant=candidate_c_variant,
    )
    markdown = build_experimental_candidate_ac_intersection_markdown(summary)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an intersection-aware Candidate A vs Candidate C comparison report"
    )
    parser.add_argument("--candidate-a-path", type=Path, default=CANDIDATE_A_DEFAULT_PATH)
    parser.add_argument("--candidate-c-path", type=Path, default=DEFAULT_CANDIDATE_C_DATASET)
    parser.add_argument("--candidate-a-summary", type=Path, default=DEFAULT_CANDIDATE_A_SUMMARY)
    parser.add_argument("--candidate-c-summary", type=Path, default=DEFAULT_CANDIDATE_C_SUMMARY)
    parser.add_argument("--candidate-c-variant", default=DEFAULT_CANDIDATE_C_VARIANT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_ac_intersection_comparison_report(
        candidate_a_path=args.candidate_a_path,
        candidate_c_path=args.candidate_c_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
        candidate_a_summary_path=args.candidate_a_summary,
        candidate_c_summary_path=args.candidate_c_summary,
        candidate_c_variant=args.candidate_c_variant,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()