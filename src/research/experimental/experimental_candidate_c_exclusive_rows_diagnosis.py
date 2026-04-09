from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.research.experimental_candidate_comparison_matrix import (
    CANDIDATE_A_DEFAULT_PATH,
    TARGET_HORIZONS,
    TARGET_LABELS,
    _safe_float,
    _safe_text,
    load_jsonl_records,
)
from src.research.experimental_candidate_intersection_utils import (
    MATCH_KEY_FIELDS,
    build_intersection_datasets,
    build_row_match_key,
    filter_candidate_c_records,
)

DEFAULT_CANDIDATE_C_DATASET = Path(
    "logs/experiments/trade_analysis_relabel_candidate_c_c2_moderate.jsonl"
)
DEFAULT_CANDIDATE_C_VARIANT = "c2_moderate"
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate/candidate_c_exclusive_rows_diagnosis.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate/candidate_c_exclusive_rows_diagnosis.md"
)
TOP_DISTRIBUTION_LIMIT = 15
OPTIONAL_FIELD_PATTERNS = ("regime", "vol", "atr")
MEANINGFUL_DIRECTIONAL_RATIO = 0.60
HIGH_CONCENTRATION_RATIO = 0.60
MODERATE_CONCENTRATION_RATIO = 0.40


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _format_pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.6f}"


def _format_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.6f}"


def _valid_label(value: Any) -> str | None:
    text = _safe_text(value)
    if text in TARGET_LABELS:
        return text
    return None


def _strategy_value(row: dict[str, Any]) -> str:
    return _safe_text(row.get("selected_strategy") or row.get("strategy")) or "unknown"


def _symbol_value(row: dict[str, Any]) -> str:
    return _safe_text(row.get("symbol")) or "unknown"


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

    emitted_counts: dict[tuple[str, ...], int] = {}
    exclusive_rows: list[dict[str, Any]] = []
    for row in source_rows:
        key = build_row_match_key(row)
        allowed = source_counts.get(key, 0) - shared_counts.get(key, 0)
        emitted = emitted_counts.get(key, 0)
        if emitted >= allowed:
            continue
        exclusive_rows.append(row)
        emitted_counts[key] = emitted + 1

    return exclusive_rows


def _build_horizon_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_rows = len(rows)
    for horizon in TARGET_HORIZONS:
        counts = {label: 0 for label in TARGET_LABELS}
        labeled_rows = 0
        for row in rows:
            label = _valid_label(row.get(f"future_label_{horizon}"))
            if label is None:
                continue
            counts[label] += 1
            labeled_rows += 1

        directional_count = counts["up"] + counts["down"]
        unlabeled_rows = total_rows - labeled_rows
        result[horizon] = {
            "labeled_rows": labeled_rows,
            "directional_counts": counts,
            "directional_ratios": {
                label: _safe_ratio(counts[label], labeled_rows) if labeled_rows else 0.0
                for label in TARGET_LABELS
            },
            "directional_total": directional_count,
            "directional_ratio": _safe_ratio(directional_count, labeled_rows) if labeled_rows else 0.0,
            "directional_ratio_vs_total_rows": _safe_ratio(directional_count, total_rows) if total_rows else 0.0,
            "unlabeled_rows": unlabeled_rows,
            "unlabeled_ratio_vs_total_rows": _safe_ratio(unlabeled_rows, total_rows) if total_rows else 0.0,
        }
    return result


def _top_distribution(counter: Counter[str], total: int) -> dict[str, Any]:
    rows = []
    for name, count in counter.most_common(TOP_DISTRIBUTION_LIMIT):
        rows.append(
            {
                "name": name,
                "count": count,
                "ratio": _safe_ratio(count, total),
            }
        )
    top_ratio = _safe_float(rows[0]["ratio"]) if rows else None
    return {
        "unique_count": len(counter),
        "top": rows,
        "top_ratio": top_ratio,
    }


def _build_symbol_strategy_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[f"{_symbol_value(row)} | {_strategy_value(row)}"] += 1
    return _top_distribution(counter, len(rows))


def _candidate_optional_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        for key, value in row.items():
            key_text = _safe_text(key)
            if key_text is None:
                continue
            if not any(pattern in key_text.lower() for pattern in OPTIONAL_FIELD_PATTERNS):
                continue
            if _safe_text(value) is None and _safe_float(value) is None:
                continue
            fields.add(key_text)
    return sorted(fields)


def _build_optional_field_breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_name in _candidate_optional_fields(rows):
        counter: Counter[str] = Counter()
        for row in rows:
            value = row.get(field_name)
            if isinstance(value, dict):
                continue
            text_value = _safe_text(value)
            if text_value is None:
                numeric_value = _safe_float(value)
                if numeric_value is None:
                    continue
                text_value = _format_number(numeric_value)
            counter[text_value] += 1
        if counter:
            result[field_name] = _top_distribution(counter, len(rows))
    return result


def _build_distribution_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    symbol_counter = Counter(_symbol_value(row) for row in rows)
    strategy_counter = Counter(_strategy_value(row) for row in rows)
    return {
        "symbol_distribution": _top_distribution(symbol_counter, len(rows)),
        "selected_strategy_distribution": _top_distribution(strategy_counter, len(rows)),
        "symbol_strategy_distribution": _build_symbol_strategy_distribution(rows),
        "optional_field_distributions": _build_optional_field_breakdowns(rows),
    }


def _interpret_seed_supply(
    profile: dict[str, Any],
    distributions: dict[str, Any],
    total_rows: int,
) -> dict[str, Any]:
    notes: list[str] = []
    directional_support_horizons: list[str] = []
    flat_heavy_horizons: list[str] = []
    unlabeled_heavy_horizons: list[str] = []

    if total_rows <= 0:
        return {
            "directional_support_horizons": [],
            "flat_heavy_horizons": [],
            "unlabeled_heavy_horizons": [],
            "notes": [
                "No Candidate C2-exclusive rows were available in the loaded datasets, so seed-supply usefulness could not be assessed from data."
            ],
            "breadth_snapshot": {
                "unique_symbols": 0,
                "unique_strategies": 0,
                "top_symbol_ratio": None,
                "top_strategy_ratio": None,
            },
            "primary_finding": "c2_exclusive_rows_not_observed_in_loaded_inputs",
            "secondary_finding": "seed_supply_diagnosis_is_blocked_without_exclusive_rows",
            "recommendation": "Re-run this diagnosis against the actual experimental relabel outputs so the exclusive-row structure can be assessed.",
            "summary": "This diagnosis cannot judge seed-supply usefulness because no Candidate C2-exclusive rows were present in the loaded inputs.",
        }

    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(profile.get(horizon))
        labeled_rows = int(payload.get("labeled_rows", 0))
        unlabeled_rows = int(payload.get("unlabeled_rows", 0))
        directional_ratio = _safe_float(payload.get("directional_ratio")) or 0.0
        directional_ratio_vs_total_rows = _safe_float(payload.get("directional_ratio_vs_total_rows")) or 0.0
        flat_ratio = _safe_float(_safe_dict(payload.get("directional_ratios")).get("flat")) or 0.0
        unlabeled_ratio_vs_total_rows = _safe_float(payload.get("unlabeled_ratio_vs_total_rows")) or 0.0

        if directional_ratio >= MEANINGFUL_DIRECTIONAL_RATIO:
            directional_support_horizons.append(horizon)
        if flat_ratio >= 0.50:
            flat_heavy_horizons.append(horizon)
        if unlabeled_ratio_vs_total_rows >= 0.40:
            unlabeled_heavy_horizons.append(horizon)

        notes.append(
            f"{horizon}: labeled={labeled_rows}, directional_ratio={_format_pct(directional_ratio)}, "
            f"directional_ratio_vs_total_rows={_format_pct(directional_ratio_vs_total_rows)}, "
            f"flat_ratio={_format_pct(flat_ratio)}, unlabeled_rows={unlabeled_rows}, "
            f"unlabeled_ratio_vs_total_rows={_format_pct(unlabeled_ratio_vs_total_rows)}."
        )

    symbol_distribution = _safe_dict(distributions.get("symbol_distribution"))
    strategy_distribution = _safe_dict(distributions.get("selected_strategy_distribution"))
    top_symbol = _safe_dict((_safe_list(symbol_distribution.get("top")) or [{}])[0])
    top_strategy = _safe_dict((_safe_list(strategy_distribution.get("top")) or [{}])[0])
    top_symbol_ratio = _safe_float(top_symbol.get("ratio")) or 0.0
    top_strategy_ratio = _safe_float(top_strategy.get("ratio")) or 0.0
    unique_symbols = int(symbol_distribution.get("unique_count", 0))
    unique_strategies = int(strategy_distribution.get("unique_count", 0))

    if top_symbol_ratio >= HIGH_CONCENTRATION_RATIO:
        notes.append(
            f"Symbol concentration is high: {_safe_text(top_symbol.get('name')) or 'unknown'} accounts for {_format_pct(top_symbol_ratio)} of exclusive rows."
        )
    elif top_symbol_ratio >= MODERATE_CONCENTRATION_RATIO:
        notes.append(
            f"Symbol concentration is moderate: {_safe_text(top_symbol.get('name')) or 'unknown'} accounts for {_format_pct(top_symbol_ratio)} of exclusive rows."
        )
    else:
        notes.append(
            f"Exclusive rows are spread across {unique_symbols} symbols rather than collapsing into a single symbol pocket."
        )

    if top_strategy_ratio >= HIGH_CONCENTRATION_RATIO:
        notes.append(
            f"Strategy concentration is high: {_safe_text(top_strategy.get('name')) or 'unknown'} accounts for {_format_pct(top_strategy_ratio)} of exclusive rows."
        )
    elif top_strategy_ratio >= MODERATE_CONCENTRATION_RATIO:
        notes.append(
            f"Strategy concentration is moderate: {_safe_text(top_strategy.get('name')) or 'unknown'} accounts for {_format_pct(top_strategy_ratio)} of exclusive rows."
        )
    else:
        notes.append(
            f"Exclusive rows span {unique_strategies} strategies, which is more consistent with seed-supply expansion than a single-strategy edge case."
        )

    notes.append(
        "Exclusive-row counts are derived from the existing row match key. If duplicate rows share the same match key, exclusive-row assignment can be order-sensitive, so downstream conclusions should be validated against seed-generation behavior as well."
    )

    if directional_support_horizons:
        primary = "c2_exclusive_rows_look_directionally_material_for_seed_supply"
    elif flat_heavy_horizons or unlabeled_heavy_horizons:
        primary = "c2_exclusive_rows_look_weak_for_seed_supply"
    else:
        primary = "c2_exclusive_rows_look_mixed_for_seed_supply"

    if top_symbol_ratio >= HIGH_CONCENTRATION_RATIO or top_strategy_ratio >= HIGH_CONCENTRATION_RATIO:
        secondary = "exclusive_rows_are_structurally_concentrated"
    elif unique_symbols >= 3 and unique_strategies >= 2:
        secondary = "exclusive_rows_expand_across_multiple_symbols_and_strategies"
    else:
        secondary = "exclusive_rows_have_limited_breadth"

    if (
        primary == "c2_exclusive_rows_look_directionally_material_for_seed_supply"
        and secondary == "exclusive_rows_expand_across_multiple_symbols_and_strategies"
    ):
        recommendation = "Prioritize downstream testing on whether C2-exclusive rows increase candidate availability after edge-selection filtering."
    elif primary == "c2_exclusive_rows_look_weak_for_seed_supply":
        recommendation = "Treat C2-exclusive rows cautiously and verify whether they survive downstream filters before investing further in coverage-expansion tuning."
    else:
        recommendation = "Keep C2-exclusive rows in follow-up diagnostics, but measure whether the added rows are broad and durable enough to improve downstream seed supply."

    summary = (
        "This diagnosis does not claim profitability; it evaluates whether C2-only rows look directional and broad enough to matter for candidate availability in a seed-starved system."
    )
    if directional_support_horizons:
        summary += f" Directional support is present on {', '.join(directional_support_horizons)}."
    if flat_heavy_horizons:
        summary += f" Flat-heavy behavior appears on {', '.join(flat_heavy_horizons)}."
    if unlabeled_heavy_horizons:
        summary += f" Unlabeled share remains material on {', '.join(unlabeled_heavy_horizons)}."

    return {
        "directional_support_horizons": directional_support_horizons,
        "flat_heavy_horizons": flat_heavy_horizons,
        "unlabeled_heavy_horizons": unlabeled_heavy_horizons,
        "notes": notes,
        "breadth_snapshot": {
            "unique_symbols": unique_symbols,
            "unique_strategies": unique_strategies,
            "top_symbol_ratio": top_symbol_ratio,
            "top_strategy_ratio": top_strategy_ratio,
        },
        "primary_finding": primary,
        "secondary_finding": secondary,
        "recommendation": recommendation,
        "summary": summary,
    }


def build_experimental_candidate_c_exclusive_rows_diagnosis(
    candidate_a_records: list[dict[str, Any]],
    candidate_c_records: list[dict[str, Any]],
    *,
    candidate_a_path: Path,
    candidate_c_path: Path,
    candidate_a_instrumentation: dict[str, int] | None = None,
    candidate_c_instrumentation: dict[str, int] | None = None,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
) -> dict[str, Any]:
    filtered_candidate_c_records = filter_candidate_c_records(
        candidate_c_records,
        variant_name=candidate_c_variant,
    )
    _, candidate_c_shared_rows, intersection_overview = build_intersection_datasets(
        candidate_a_records,
        filtered_candidate_c_records,
    )
    candidate_c_only_rows = _build_exclusive_rows(filtered_candidate_c_records, candidate_c_shared_rows)
    profile = _build_horizon_profile(candidate_c_only_rows)
    distributions = _build_distribution_breakdown(candidate_c_only_rows)
    interpretation = _interpret_seed_supply(profile, distributions, len(candidate_c_only_rows))

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "experimental_candidate_c_exclusive_rows_diagnosis",
            "report_name": "candidate_c_exclusive_rows_diagnosis",
        },
        "inputs": {
            "candidate_a_path": str(candidate_a_path),
            "candidate_c_path": str(candidate_c_path),
            "candidate_c_variant": candidate_c_variant,
            "candidate_a_parser_instrumentation": candidate_a_instrumentation or {},
            "candidate_c_parser_instrumentation": candidate_c_instrumentation or {},
            "candidate_a_raw_total_rows": len(candidate_a_records),
            "candidate_c_raw_total_rows": len(candidate_c_records),
            "candidate_c_filtered_row_count": len(filtered_candidate_c_records),
            "row_key_definition_summary": (
                "Exclusive rows are derived from the existing experimental intersection key: "
                + ", ".join(MATCH_KEY_FIELDS)
                + "."
            ),
        },
        "alignment_overview": {
            "candidate_a_total_rows": int(intersection_overview.get("baseline_total_rows", 0)),
            "candidate_c_total_rows": int(intersection_overview.get("experiment_total_rows", 0)),
            "shared_rows": int(intersection_overview.get("shared_row_count", 0)),
            "candidate_a_only_rows": int(intersection_overview.get("baseline_only_row_count", 0)),
            "candidate_c_only_rows": int(intersection_overview.get("experiment_only_row_count", 0)),
            "shared_row_ratio_from_a": _safe_float(
                intersection_overview.get("shared_ratio_vs_baseline")
            ),
            "shared_row_ratio_from_c": _safe_float(
                intersection_overview.get("shared_ratio_vs_experiment")
            ),
        },
        "c2_exclusive_row_profile": {
            "total_exclusive_rows": len(candidate_c_only_rows),
            "by_horizon": profile,
        },
        "distribution_breakdown": distributions,
        "seed_supply_interpretation": {
            "directional_support_horizons": interpretation.get("directional_support_horizons", []),
            "flat_heavy_horizons": interpretation.get("flat_heavy_horizons", []),
            "unlabeled_heavy_horizons": interpretation.get("unlabeled_heavy_horizons", []),
            "breadth_snapshot": interpretation.get("breadth_snapshot", {}),
            "notes": interpretation.get("notes", []),
        },
        "final_diagnosis": {
            "primary_finding": interpretation.get("primary_finding"),
            "secondary_finding": interpretation.get("secondary_finding"),
            "recommendation": interpretation.get("recommendation"),
            "summary": interpretation.get("summary"),
        },
    }


def build_experimental_candidate_c_exclusive_rows_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    inputs = _safe_dict(summary.get("inputs"))
    alignment = _safe_dict(summary.get("alignment_overview"))
    profile = _safe_dict(summary.get("c2_exclusive_row_profile"))
    by_horizon = _safe_dict(profile.get("by_horizon"))
    distributions = _safe_dict(summary.get("distribution_breakdown"))
    interpretation = _safe_dict(summary.get("seed_supply_interpretation"))
    diagnosis = _safe_dict(summary.get("final_diagnosis"))
    breadth = _safe_dict(interpretation.get("breadth_snapshot"))

    lines = [
        "# Candidate C2 Exclusive Rows Diagnosis",
        "",
        "## Metadata",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- report_type: {metadata.get('report_type', 'n/a')}",
        f"- report_name: {metadata.get('report_name', 'n/a')}",
        "",
        "## Inputs",
        f"- candidate_a_path: {inputs.get('candidate_a_path', 'n/a')}",
        f"- candidate_c_path: {inputs.get('candidate_c_path', 'n/a')}",
        f"- candidate_c_variant: {inputs.get('candidate_c_variant', 'n/a')}",
        f"- candidate_a_parser_instrumentation: {json.dumps(inputs.get('candidate_a_parser_instrumentation', {}), ensure_ascii=False, sort_keys=True)}",
        f"- candidate_c_parser_instrumentation: {json.dumps(inputs.get('candidate_c_parser_instrumentation', {}), ensure_ascii=False, sort_keys=True)}",
        f"- candidate_a_raw_total_rows: {inputs.get('candidate_a_raw_total_rows', 0)}",
        f"- candidate_c_raw_total_rows: {inputs.get('candidate_c_raw_total_rows', 0)}",
        f"- candidate_c_filtered_row_count: {inputs.get('candidate_c_filtered_row_count', 0)}",
        f"- row_key_definition_summary: {inputs.get('row_key_definition_summary', 'n/a')}",
        "",
        "## Alignment Overview",
        f"- candidate_a_total_rows: {alignment.get('candidate_a_total_rows', 0)}",
        f"- candidate_c_total_rows: {alignment.get('candidate_c_total_rows', 0)}",
        f"- shared_rows: {alignment.get('shared_rows', 0)}",
        f"- candidate_a_only_rows: {alignment.get('candidate_a_only_rows', 0)}",
        f"- candidate_c_only_rows: {alignment.get('candidate_c_only_rows', 0)}",
        f"- shared_row_ratio_from_a: {_format_pct(alignment.get('shared_row_ratio_from_a'))}",
        f"- shared_row_ratio_from_c: {_format_pct(alignment.get('shared_row_ratio_from_c'))}",
        "",
        "## C2-Exclusive Row Profile",
        f"- total_exclusive_rows: {profile.get('total_exclusive_rows', 0)}",
    ]

    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(by_horizon.get(horizon))
        counts = _safe_dict(payload.get("directional_counts"))
        ratios = _safe_dict(payload.get("directional_ratios"))
        lines.append(
            f"- {horizon}: "
            f"labeled={payload.get('labeled_rows', 0)}, "
            f"up={counts.get('up', 0)} ({_format_pct(ratios.get('up'))}), "
            f"down={counts.get('down', 0)} ({_format_pct(ratios.get('down'))}), "
            f"flat={counts.get('flat', 0)} ({_format_pct(ratios.get('flat'))}), "
            f"directional_total={payload.get('directional_total', 0)}, "
            f"directional_ratio={_format_pct(payload.get('directional_ratio'))}, "
            f"directional_ratio_vs_total_rows={_format_pct(payload.get('directional_ratio_vs_total_rows'))}, "
            f"unlabeled={payload.get('unlabeled_rows', 0)}, "
            f"unlabeled_ratio_vs_total_rows={_format_pct(payload.get('unlabeled_ratio_vs_total_rows'))}"
        )

    lines.extend(["", "## Distribution Breakdown"])
    for section_key, label in (
        ("symbol_distribution", "symbols"),
        ("selected_strategy_distribution", "strategies"),
        ("symbol_strategy_distribution", "symbol x strategy"),
    ):
        payload = _safe_dict(distributions.get(section_key))
        top_rows = _safe_list(payload.get("top"))
        if not top_rows:
            lines.append(f"- {label}: none")
            continue
        preview = "; ".join(
            f"{_safe_dict(row).get('name', 'unknown')}={_safe_dict(row).get('count', 0)} ({_format_pct(_safe_dict(row).get('ratio'))})"
            for row in top_rows[:5]
        )
        lines.append(
            f"- {label}: unique={payload.get('unique_count', 0)}; "
            f"top_ratio={_format_pct(payload.get('top_ratio'))}; top={preview}"
        )

    optional_fields = _safe_dict(distributions.get("optional_field_distributions"))
    if optional_fields:
        for field_name, payload in sorted(optional_fields.items()):
            field_payload = _safe_dict(payload)
            preview = "; ".join(
                f"{_safe_dict(row).get('name', 'unknown')}={_safe_dict(row).get('count', 0)} ({_format_pct(_safe_dict(row).get('ratio'))})"
                for row in _safe_list(field_payload.get("top"))[:5]
            )
            lines.append(
                f"- {field_name}: unique={field_payload.get('unique_count', 0)}; "
                f"top_ratio={_format_pct(field_payload.get('top_ratio'))}; top={preview}"
            )

    lines.extend(
        [
            "",
            "## Seed Supply Interpretation",
            "- This report asks whether C2-only rows plausibly improve downstream candidate availability, not whether they are directly profitable.",
            f"- directional_support_horizons: {', '.join(str(item) for item in _safe_list(interpretation.get('directional_support_horizons'))) or 'none'}",
            f"- flat_heavy_horizons: {', '.join(str(item) for item in _safe_list(interpretation.get('flat_heavy_horizons'))) or 'none'}",
            f"- unlabeled_heavy_horizons: {', '.join(str(item) for item in _safe_list(interpretation.get('unlabeled_heavy_horizons'))) or 'none'}",
            f"- unique_symbols: {breadth.get('unique_symbols', 0)}",
            f"- unique_strategies: {breadth.get('unique_strategies', 0)}",
            f"- top_symbol_ratio: {_format_pct(breadth.get('top_symbol_ratio'))}",
            f"- top_strategy_ratio: {_format_pct(breadth.get('top_strategy_ratio'))}",
        ]
    )
    for note in interpretation.get("notes", []):
        lines.append(f"- note: {note}")

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

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_c_exclusive_rows_diagnosis(
    candidate_a_path: Path = CANDIDATE_A_DEFAULT_PATH,
    candidate_c_path: Path = DEFAULT_CANDIDATE_C_DATASET,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
) -> dict[str, Any]:
    candidate_a_records, candidate_a_instrumentation = load_jsonl_records(candidate_a_path)
    candidate_c_records, candidate_c_instrumentation = load_jsonl_records(candidate_c_path)

    summary = build_experimental_candidate_c_exclusive_rows_diagnosis(
        candidate_a_records,
        candidate_c_records,
        candidate_a_path=candidate_a_path,
        candidate_c_path=candidate_c_path,
        candidate_a_instrumentation=candidate_a_instrumentation,
        candidate_c_instrumentation=candidate_c_instrumentation,
        candidate_c_variant=candidate_c_variant,
    )
    markdown = build_experimental_candidate_c_exclusive_rows_markdown(summary)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Candidate C2 exclusive rows for seed-supply usefulness"
    )
    parser.add_argument("--candidate-a-path", type=Path, default=CANDIDATE_A_DEFAULT_PATH)
    parser.add_argument("--candidate-c-path", type=Path, default=DEFAULT_CANDIDATE_C_DATASET)
    parser.add_argument("--candidate-c-variant", default=DEFAULT_CANDIDATE_C_VARIANT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_c_exclusive_rows_diagnosis(
        candidate_a_path=args.candidate_a_path,
        candidate_c_path=args.candidate_c_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
        candidate_c_variant=args.candidate_c_variant,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()