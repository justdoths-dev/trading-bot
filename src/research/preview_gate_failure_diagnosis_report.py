from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HORIZONS = ("15m", "1h", "4h")
CANDIDATE_KEYS = ("top_strategy", "top_symbol", "top_alignment_state")

DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "research_reports"
    / "latest"
    / "summary.json"
)

MIN_EDGE_CANDIDATE_SAMPLE_COUNT = 30
ABSOLUTE_MIN_MEDIAN_RETURN_GT = 0.0
ABSOLUTE_MIN_POSITIVE_RATE_PCT = 50.0
MAX_EXAMPLES_PER_BUCKET = 3


def run_preview_gate_failure_diagnosis_report(
    input_path: Path | None = None,
) -> dict[str, Any]:
    resolved = input_path or DEFAULT_INPUT_PATH
    summary = load_summary_json(resolved)
    diagnosis = build_preview_gate_failure_diagnosis_summary(summary, resolved)
    markdown = render_preview_gate_failure_diagnosis_markdown(diagnosis)

    return {
        "input_path": str(resolved),
        "summary": diagnosis,
        "markdown": markdown,
    }


def load_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def build_preview_gate_failure_diagnosis_summary(
    summary: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    edge_candidates_preview = _safe_dict(summary.get("edge_candidates_preview"))
    by_horizon = _safe_dict(edge_candidates_preview.get("by_horizon"))
    strategy_lab = _safe_dict(summary.get("strategy_lab"))
    ranking = _safe_dict(strategy_lab.get("ranking"))
    edge = _safe_dict(strategy_lab.get("edge"))
    schema_validation = _safe_dict(summary.get("schema_validation"))

    total_candidate_slots = 0
    visible_candidate_slots = 0
    insufficient_candidate_slots = 0

    candidate_strength_distribution: Counter[str] = Counter()
    gate_reason_distribution: Counter[str] = Counter()
    category_reason_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    horizon_reason_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    category_visible_distribution: Counter[str] = Counter()
    horizon_visible_distribution: Counter[str] = Counter()

    edge_presence_by_horizon: dict[str, dict[str, int]] = {}
    candidate_details_by_horizon: dict[str, dict[str, Any]] = {}
    representative_examples: dict[str, list[dict[str, Any]]] = {
        "sample_gate_failures": [],
        "median_gate_failures": [],
        "positive_rate_failures": [],
        "passed_sample_but_failed_quality": [],
        "visible_candidates": [],
    }

    for horizon in HORIZONS:
        horizon_preview = _safe_dict(by_horizon.get(horizon))
        horizon_rank = _safe_dict(ranking.get(horizon))
        horizon_edge = _safe_dict(edge.get(horizon))

        candidate_details_by_horizon[horizon] = {}
        edge_presence_by_horizon[horizon] = {
            "symbol_edges": _count_edge_findings(horizon_edge.get("by_symbol")),
            "strategy_edges": _count_edge_findings(horizon_edge.get("by_strategy")),
            "alignment_state_edges": _count_edge_findings(
                horizon_edge.get("by_alignment_state")
            ),
            "ai_execution_state_edges": _count_edge_findings(
                horizon_edge.get("by_ai_execution_state")
            ),
        }

        for candidate_key in CANDIDATE_KEYS:
            total_candidate_slots += 1

            preview_candidate = horizon_preview.get(candidate_key)
            preview_candidate_dict = (
                preview_candidate if isinstance(preview_candidate, dict) else {}
            )

            preview_strength = _text(preview_candidate_dict.get("candidate_strength")) or "insufficient_data"
            candidate_strength_distribution[preview_strength] += 1

            ranked_report = horizon_rank.get(_ranking_bucket_name(candidate_key))
            top_ranked = _extract_top_ranked_candidate(ranked_report)
            analyzed = _analyze_candidate_gate(
                top_ranked=top_ranked,
                preview_candidate=preview_candidate_dict,
                horizon=horizon,
                candidate_key=candidate_key,
            )
            candidate_details_by_horizon[horizon][candidate_key] = analyzed

            if analyzed["preview_visible"]:
                visible_candidate_slots += 1
                category_visible_distribution[candidate_key] += 1
                horizon_visible_distribution[horizon] += 1
                _push_example(representative_examples["visible_candidates"], analyzed)
            else:
                insufficient_candidate_slots += 1

            for reason in analyzed["gate_failure_reasons"]:
                gate_reason_distribution[reason] += 1
                category_reason_distribution[candidate_key][reason] += 1
                horizon_reason_distribution[horizon][reason] += 1

            if "sample_count_below_minimum" in analyzed["gate_failure_reasons"]:
                _push_example(representative_examples["sample_gate_failures"], analyzed)
            if "median_future_return_non_positive_or_missing" in analyzed["gate_failure_reasons"]:
                _push_example(representative_examples["median_gate_failures"], analyzed)
            if "positive_rate_below_minimum_or_missing" in analyzed["gate_failure_reasons"]:
                _push_example(representative_examples["positive_rate_failures"], analyzed)
            if analyzed["sample_gate_passed"] and not analyzed["quality_gate_passed"]:
                _push_example(
                    representative_examples["passed_sample_but_failed_quality"],
                    analyzed,
                )

    diagnosis = _build_diagnosis(
        total_candidate_slots=total_candidate_slots,
        visible_candidate_slots=visible_candidate_slots,
        insufficient_candidate_slots=insufficient_candidate_slots,
        gate_reason_distribution=gate_reason_distribution,
        edge_presence_by_horizon=edge_presence_by_horizon,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "preview_gate_failure_diagnosis_report",
            "input_path": str(input_path),
            "schema_validation": schema_validation,
        },
        "preview_overview": {
            "total_candidate_slots": total_candidate_slots,
            "visible_candidate_slots": visible_candidate_slots,
            "insufficient_candidate_slots": insufficient_candidate_slots,
            "visible_candidate_ratio": _safe_ratio(visible_candidate_slots, total_candidate_slots),
            "insufficient_candidate_ratio": _safe_ratio(insufficient_candidate_slots, total_candidate_slots),
            "candidate_strength_distribution": _counter_rows(candidate_strength_distribution),
            "visible_distribution_by_category": _counter_rows(category_visible_distribution),
            "visible_distribution_by_horizon": _counter_rows(horizon_visible_distribution),
        },
        "gate_configuration": {
            "absolute_minimums": {
                "sample_count_min": MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
                "median_future_return_gt": ABSOLUTE_MIN_MEDIAN_RETURN_GT,
                "positive_rate_pct_min": ABSOLUTE_MIN_POSITIVE_RATE_PCT,
            }
        },
        "gate_failures": {
            "overall_reason_distribution": _counter_rows(gate_reason_distribution),
            "by_category": {
                key: _counter_rows(counter)
                for key, counter in sorted(category_reason_distribution.items())
            },
            "by_horizon": {
                key: _counter_rows(counter)
                for key, counter in sorted(horizon_reason_distribution.items())
            },
        },
        "edge_context": edge_presence_by_horizon,
        "candidate_details_by_horizon": candidate_details_by_horizon,
        "diagnosis": diagnosis,
        "representative_examples": representative_examples,
    }


def render_preview_gate_failure_diagnosis_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    preview_overview = _safe_dict(summary.get("preview_overview"))
    gate_configuration = _safe_dict(summary.get("gate_configuration"))
    gate_failures = _safe_dict(summary.get("gate_failures"))
    diagnosis = _safe_dict(summary.get("diagnosis"))
    edge_context = _safe_dict(summary.get("edge_context"))

    lines = [
        "# Preview Gate Failure Diagnosis Report",
        "",
        "## Executive Summary",
        f"- Input path: {metadata.get('input_path', 'unknown')}",
        f"- Visible candidate ratio: {_format_ratio(preview_overview.get('visible_candidate_ratio'))}",
        f"- Insufficient candidate ratio: {_format_ratio(preview_overview.get('insufficient_candidate_ratio'))}",
        f"- Primary issue: {diagnosis.get('primary_issue', 'unknown')}",
        f"- Secondary issue: {diagnosis.get('secondary_issue', 'unknown')}",
        f"- Summary: {diagnosis.get('summary', 'n/a')}",
        "",
        "## Gate Configuration",
        f"- sample_count_min: {_safe_dict(gate_configuration.get('absolute_minimums')).get('sample_count_min', 'n/a')}",
        f"- median_future_return_gt: {_safe_dict(gate_configuration.get('absolute_minimums')).get('median_future_return_gt', 'n/a')}",
        f"- positive_rate_pct_min: {_safe_dict(gate_configuration.get('absolute_minimums')).get('positive_rate_pct_min', 'n/a')}",
        "",
        "## Preview Overview",
        _markdown_distribution_block(
            "Candidate strength distribution",
            preview_overview.get("candidate_strength_distribution"),
        ),
        _markdown_distribution_block(
            "Visible distribution by category",
            preview_overview.get("visible_distribution_by_category"),
        ),
        _markdown_distribution_block(
            "Visible distribution by horizon",
            preview_overview.get("visible_distribution_by_horizon"),
        ),
        "",
        "## Overall Gate Failure Reasons",
        _markdown_distribution_block(
            "Overall failure reasons",
            gate_failures.get("overall_reason_distribution"),
        ),
        "",
        "## Gate Failure Reasons by Category",
        _markdown_nested_distribution_block(gate_failures.get("by_category")),
        "",
        "## Gate Failure Reasons by Horizon",
        _markdown_nested_distribution_block(gate_failures.get("by_horizon")),
        "",
        "## Edge Context",
        _markdown_edge_context(edge_context),
        "",
        "## Final Diagnosis",
        f"- Primary issue: {diagnosis.get('primary_issue', 'unknown')}",
        f"- Secondary issue: {diagnosis.get('secondary_issue', 'unknown')}",
        f"- Summary: {diagnosis.get('summary', 'n/a')}",
    ]
    return "\n".join(lines).strip() + "\n"


def _analyze_candidate_gate(
    *,
    top_ranked: dict[str, Any] | None,
    preview_candidate: dict[str, Any],
    horizon: str,
    candidate_key: str,
) -> dict[str, Any]:
    ranked_metrics = _safe_dict(_safe_dict(top_ranked).get("metrics"))

    sample_count = _to_float(ranked_metrics.get("sample_count"))
    labeled_count = _to_float(ranked_metrics.get("labeled_count"))
    coverage_pct = _to_float(ranked_metrics.get("coverage_pct"))
    median_future_return_pct = _to_float(ranked_metrics.get("median_future_return_pct"))
    positive_rate_pct = _to_float(
        ranked_metrics.get("positive_rate_pct", ranked_metrics.get("up_rate_pct"))
    )
    signal_match_rate_pct = _to_float(
        ranked_metrics.get("signal_match_rate_pct", ranked_metrics.get("signal_match_rate"))
    )
    bias_match_rate_pct = _to_float(
        ranked_metrics.get("bias_match_rate_pct", ranked_metrics.get("bias_match_rate"))
    )
    robustness_label, robustness_value = _select_robustness_signal(ranked_metrics)

    sample_gate_passed = (
        sample_count is not None
        and sample_count >= MIN_EDGE_CANDIDATE_SAMPLE_COUNT
        and median_future_return_pct is not None
        and median_future_return_pct > ABSOLUTE_MIN_MEDIAN_RETURN_GT
        and positive_rate_pct is not None
        and positive_rate_pct >= ABSOLUTE_MIN_POSITIVE_RATE_PCT
    )

    preview_strength = _text(preview_candidate.get("candidate_strength")) or "insufficient_data"
    quality_gate_value = _text(preview_candidate.get("quality_gate")) or "failed"
    preview_visible = preview_strength != "insufficient_data"
    quality_gate_passed = quality_gate_value == "passed"

    failure_reasons: list[str] = []
    if sample_count is None or sample_count < MIN_EDGE_CANDIDATE_SAMPLE_COUNT:
        failure_reasons.append("sample_count_below_minimum")
    if median_future_return_pct is None or median_future_return_pct <= ABSOLUTE_MIN_MEDIAN_RETURN_GT:
        failure_reasons.append("median_future_return_non_positive_or_missing")
    if positive_rate_pct is None or positive_rate_pct < ABSOLUTE_MIN_POSITIVE_RATE_PCT:
        failure_reasons.append("positive_rate_below_minimum_or_missing")
    if sample_gate_passed and not quality_gate_passed:
        failure_reasons.append("quality_gate_not_passed_after_sample_gate")
    if top_ranked is None:
        failure_reasons.append("no_ranked_candidate_available")

    return {
        "horizon": horizon,
        "candidate_key": candidate_key,
        "group": _safe_group(top_ranked),
        "rank": _safe_rank(top_ranked),
        "score": _safe_score(top_ranked),
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": coverage_pct,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "signal_match_rate_pct": signal_match_rate_pct,
        "bias_match_rate_pct": bias_match_rate_pct,
        "robustness_signal": robustness_label,
        "robustness_signal_pct": robustness_value,
        "sample_gate_passed": sample_gate_passed,
        "quality_gate_value": quality_gate_value,
        "quality_gate_passed": quality_gate_passed,
        "preview_candidate_strength": preview_strength,
        "preview_visible": preview_visible,
        "gate_failure_reasons": failure_reasons,
    }


def _build_diagnosis(
    *,
    total_candidate_slots: int,
    visible_candidate_slots: int,
    insufficient_candidate_slots: int,
    gate_reason_distribution: Counter[str],
    edge_presence_by_horizon: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if total_candidate_slots == 0:
        return {
            "primary_issue": "no_data",
            "secondary_issue": "no_data",
            "summary": "No preview candidate slots were available for diagnosis.",
        }

    dominant_reason = "none"
    if gate_reason_distribution:
        dominant_reason = max(
            gate_reason_distribution.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

    strategy_edges_total = sum(
        _safe_int(_safe_dict(edge_presence_by_horizon.get(h)).get("strategy_edges"))
        for h in HORIZONS
    )
    alignment_edges_total = sum(
        _safe_int(_safe_dict(edge_presence_by_horizon.get(h)).get("alignment_state_edges"))
        for h in HORIZONS
    )
    any_edge_detected = strategy_edges_total > 0 or alignment_edges_total > 0

    if insufficient_candidate_slots == total_candidate_slots and dominant_reason == "positive_rate_below_minimum_or_missing":
        primary_issue = "positive_rate_absolute_minimum_is_dominant_preview_blocker"
        secondary_issue = (
            "preview_gate_may_be_misaligned_with_flat_heavy_or_mixed_label_distribution"
        )
    elif insufficient_candidate_slots == total_candidate_slots and dominant_reason == "median_future_return_non_positive_or_missing":
        primary_issue = "median_return_absolute_minimum_is_dominant_preview_blocker"
        secondary_issue = "preview_gate_filters_out_all_candidates_before_visibility"
    elif insufficient_candidate_slots == total_candidate_slots and any_edge_detected:
        primary_issue = "preview_gate_blocks_visibility_despite_detected_edge_activity"
        secondary_issue = "edge_detector_and_preview_gate_are_not_aligned"
    elif insufficient_candidate_slots == total_candidate_slots:
        primary_issue = "preview_gate_blocks_all_candidate_slots"
        secondary_issue = "top_ranked_groups_fail_absolute_minimum_preview_requirements"
    else:
        primary_issue = "preview_gate_partially_blocks_candidate_slots"
        secondary_issue = "some_candidate_visibility_exists_but_gate_failures_remain"

    return {
        "primary_issue": primary_issue,
        "secondary_issue": secondary_issue,
        "dominant_failure_reason": dominant_reason,
        "visible_candidate_ratio": _safe_ratio(visible_candidate_slots, total_candidate_slots),
        "insufficient_candidate_ratio": _safe_ratio(insufficient_candidate_slots, total_candidate_slots),
        "summary": (
            f"Primary issue={primary_issue}; secondary issue={secondary_issue}; "
            f"dominant_failure_reason={dominant_reason}; "
            f"visible_candidate_ratio={_safe_ratio(visible_candidate_slots, total_candidate_slots):.2%}; "
            f"insufficient_candidate_ratio={_safe_ratio(insufficient_candidate_slots, total_candidate_slots):.2%}."
        ),
    }


def _extract_top_ranked_candidate(report: Any) -> dict[str, Any] | None:
    ranked = _extract_ranked_groups(report)
    if not ranked:
        return None
    return ranked[0]


def _extract_ranked_groups(report: Any) -> list[dict[str, Any]]:
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]

    if not isinstance(report, dict):
        return []

    for key in ("ranked_groups", "rankings", "results"):
        items = report.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    return []


def _ranking_bucket_name(candidate_key: str) -> str:
    return {
        "top_symbol": "by_symbol",
        "top_strategy": "by_strategy",
        "top_alignment_state": "by_alignment_state",
    }[candidate_key]


def _select_robustness_signal(metrics: dict[str, Any]) -> tuple[str, float | None]:
    for output_label, keys in (
        ("signal_match_rate_pct", ("signal_match_rate_pct", "signal_match_rate")),
        ("bias_match_rate_pct", ("bias_match_rate_pct", "bias_match_rate")),
        ("coverage_pct", ("coverage_pct",)),
    ):
        for key in keys:
            value = _to_float(metrics.get(key))
            if value is not None:
                return output_label, value
    return "n/a", None


def _count_edge_findings(report: Any) -> int:
    payload = _safe_dict(report)
    findings = payload.get("edge_findings")
    if not isinstance(findings, list):
        return 0
    return len(findings)


def _safe_group(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return "n/a"
    return str(candidate.get("group", "n/a"))


def _safe_rank(candidate: dict[str, Any] | None) -> int | None:
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("rank")
    return value if isinstance(value, int) else None


def _safe_score(candidate: dict[str, Any] | None) -> float | None:
    if not isinstance(candidate, dict):
        return None
    return _to_float(candidate.get("score"))


def _counter_rows(counter: Counter[Any]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "value": value,
            "count": count,
            "ratio": _safe_ratio(count, total),
        }
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def _push_example(target: list[dict[str, Any]], example: dict[str, Any]) -> None:
    target.append(example)
    target.sort(
        key=lambda item: (
            str(item.get("horizon", "")),
            str(item.get("candidate_key", "")),
            str(item.get("group", "")),
        )
    )
    del target[MAX_EXAMPLES_PER_BUCKET:]


def _markdown_distribution_block(title: str, rows: Any) -> str:
    lines = [f"- {title}:"]
    if not isinstance(rows, list) or not rows:
        lines.append("  - none")
        return "\n".join(lines)

    for row in rows:
        if isinstance(row, dict):
            lines.append(
                f"  - {row.get('value', 'n/a')}: {row.get('count', 0)} ({_format_ratio(row.get('ratio'))})"
            )
    return "\n".join(lines)


def _markdown_nested_distribution_block(data: Any) -> str:
    nested = data if isinstance(data, dict) else {}
    if not nested:
        return "- none"

    lines: list[str] = []
    for key, rows in sorted(nested.items()):
        lines.append(f"- {key}:")
        if not isinstance(rows, list) or not rows:
            lines.append("  - none")
            continue
        for row in rows:
            if isinstance(row, dict):
                lines.append(
                    f"  - {row.get('value', 'n/a')}: {row.get('count', 0)} ({_format_ratio(row.get('ratio'))})"
                )
    return "\n".join(lines)


def _markdown_edge_context(edge_context: Any) -> str:
    context = edge_context if isinstance(edge_context, dict) else {}
    if not context:
        return "- none"

    lines: list[str] = []
    for horizon in HORIZONS:
        payload = _safe_dict(context.get(horizon))
        lines.append(f"- {horizon}:")
        lines.append(f"  - symbol_edges: {payload.get('symbol_edges', 0)}")
        lines.append(f"  - strategy_edges: {payload.get('strategy_edges', 0)}")
        lines.append(f"  - alignment_state_edges: {payload.get('alignment_state_edges', 0)}")
        lines.append(f"  - ai_execution_state_edges: {payload.get('ai_execution_state_edges', 0)}")
    return "\n".join(lines)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _format_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "0.00%"


def _default_output_paths(input_path: Path) -> tuple[Path, Path]:
    output_dir = input_path.parent
    json_path = output_dir / "preview_gate_failure_diagnosis_summary.json"
    markdown_path = output_dir / "preview_gate_failure_diagnosis_summary.md"
    return json_path, markdown_path


def main() -> None:
    result = run_preview_gate_failure_diagnosis_report()
    summary = result["summary"]
    markdown = result["markdown"]

    input_path = Path(result["input_path"])
    json_path, markdown_path = _default_output_paths(input_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "json_summary_path": str(json_path),
                "markdown_summary_path": str(markdown_path),
                "primary_issue": _safe_dict(summary.get("diagnosis")).get("primary_issue", "unknown"),
                "secondary_issue": _safe_dict(summary.get("diagnosis")).get("secondary_issue", "unknown"),
                "visible_candidate_ratio": _safe_dict(summary.get("preview_overview")).get("visible_candidate_ratio", 0.0),
                "insufficient_candidate_ratio": _safe_dict(summary.get("preview_overview")).get("insufficient_candidate_ratio", 0.0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()