from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPORT_TYPE = "analyzer_mapper_bottleneck_diagnosis_report"
REPORT_TITLE = "Analyzer Mapper Bottleneck Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_PRIMARY_INPUT = Path("logs/trade_analysis.jsonl")
DEFAULT_FALLBACK_INPUT = Path("logs/trade_analysis_cumulative.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

HORIZONS = ("15m", "1h", "4h")
PREVIEW_SLOT_KEYS = ("top_strategy", "top_symbol", "top_alignment_state")
INVALID_VISIBLE_GROUP_VALUES = {
    "insufficient_data",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
}

EXPLICIT_ANALYZER_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "latest": (
        ("latest_summary",),
        ("research_reports", "latest_summary"),
        ("research_reports", "latest", "summary"),
        ("research", "latest_summary"),
        ("research", "latest", "summary"),
        ("latest", "summary"),
    ),
    "cumulative": (
        ("cumulative_summary",),
        ("research_reports", "cumulative_summary"),
        ("research_reports", "cumulative", "summary"),
        ("research", "cumulative_summary"),
        ("research", "cumulative", "summary"),
        ("cumulative", "summary"),
    ),
    "generic": (
        ("analyzer_output",),
        ("research", "analyzer_output"),
        ("summary",),
    ),
}
EXPLICIT_MAPPER_PATHS = (
    ("edge_selection_mapper_payload",),
    ("edge_selection_input",),
    ("mapped_edge_selection_input",),
    ("historical_edge_selection_input",),
    ("research", "edge_selection_mapper_payload"),
    ("research", "edge_selection_input"),
)
EXPLICIT_ENGINE_PATHS = (
    ("edge_selection_output",),
    ("shadow_selection",),
    ("engine_output",),
    ("research", "edge_selection_output"),
    ("research", "shadow_selection"),
)
ANALYZER_LIKE_KEYS = frozenset(
    {
        "edge_candidates_preview",
        "edge_candidate_rows",
        "edge_stability_preview",
        "top_highlights",
        "strategy_lab",
        "schema_validation",
    }
)
MAPPER_LIKE_KEYS = frozenset(
    {
        "candidates",
        "candidate_seed_count",
        "candidate_seed_diagnostics",
    }
)
ENGINE_LIKE_KEYS = frozenset(
    {
        "selection_status",
        "reason_codes",
        "abstain_diagnosis",
        "ranking",
    }
)
JOINED_ROWS_PRESENT_BUT_EMPTY = "JOINED_EDGE_CANDIDATE_ROWS_PRESENT_BUT_EMPTY"


@dataclass(frozen=True)
class LoadedRow:
    line_number: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class ParsedRow:
    line_number: int
    analyzer_summary: dict[str, Any] | None
    analyzer_source: str | None
    analyzer_available_sources: tuple[str, ...]
    mapper_payload: dict[str, Any] | None
    mapper_path: str | None
    candidate_seed_diagnostics: dict[str, Any] | None
    engine_output: dict[str, Any] | None
    engine_path: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether the next selection bottleneck is analyzer-side, "
            "analyzer-to-mapper handoff, or engine-side using trade-analysis JSONL rows."
        )
    )
    parser.add_argument(
        "--trade-analysis",
        type=Path,
        default=None,
        help=(
            "Optional trade-analysis JSONL path. When omitted, the command checks "
            "logs/trade_analysis.jsonl and then logs/trade_analysis_cumulative.jsonl."
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
        help="Directory for optional JSON and Markdown output files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path, resolution = resolve_trade_analysis_path(args.trade_analysis)
    report = build_report(input_path=input_path, input_resolution=resolution)

    written_paths: dict[str, str] = {}
    if args.write_latest_copy:
        written_paths = write_report_files(report, args.output_dir)

    summary = {
        "report_type": REPORT_TYPE,
        "trade_analysis_path": report["inputs"]["trade_analysis_path"],
        "trade_analysis_resolution": report["inputs"]["trade_analysis_resolution"],
        "parsed_row_count": report["data_quality"]["parsed_row_count"],
        "rows_with_analyzer_output": report["data_quality"]["rows_with_analyzer_output"],
        "rows_with_mapper_payload": report["data_quality"]["rows_with_mapper_payload"],
        "rows_with_engine_output": report["data_quality"]["rows_with_engine_output"],
        "analyzer_coverage_ratio": report["data_quality"]["analyzer_coverage_ratio"],
        "analyzer_layer_diagnosis_reliable": report["data_quality"][
            "analyzer_layer_diagnosis_reliable"
        ],
        "primary_bottleneck_layer": report["final_verdict"]["primary_bottleneck_layer"],
        "final_verdict_coverage_limited": report["final_verdict"]["coverage_limited"],
        "written_paths": written_paths,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def resolve_trade_analysis_path(explicit_path: Path | None) -> tuple[Path, str]:
    if explicit_path is not None:
        return _resolve_existing_file(explicit_path, label="Trade-analysis input"), "explicit"

    primary = DEFAULT_PRIMARY_INPUT
    if primary.exists():
        return _resolve_existing_file(primary, label="Trade-analysis input"), "default_primary"

    fallback = DEFAULT_FALLBACK_INPUT
    if fallback.exists():
        return _resolve_existing_file(fallback, label="Trade-analysis input"), "default_fallback"

    raise FileNotFoundError(
        "Could not find a trade-analysis JSONL input path. Checked: "
        f"{DEFAULT_PRIMARY_INPUT} and {DEFAULT_FALLBACK_INPUT}"
    )


def build_report(
    *,
    input_path: Path,
    input_resolution: str,
) -> dict[str, Any]:
    loaded = load_trade_analysis_rows(input_path)
    parsed_rows = [_parse_row(item) for item in loaded["rows"]]

    analyzer_section = _build_analyzer_section(parsed_rows)
    joined_section = _build_joined_row_section(parsed_rows)
    mapper_section = _build_mapper_section(parsed_rows)
    engine_section = _build_engine_section(parsed_rows)
    final_verdict = _build_final_verdict(
        rows=parsed_rows,
        analyzer_section=analyzer_section,
        joined_section=joined_section,
        mapper_section=mapper_section,
        engine_section=engine_section,
        row_count=len(parsed_rows),
    )

    data_quality = {
        **loaded["diagnostics"],
        "rows_with_analyzer_output": sum(
            1 for row in parsed_rows if row.analyzer_summary is not None
        ),
        "rows_without_analyzer_output": sum(
            1 for row in parsed_rows if row.analyzer_summary is None
        ),
        "rows_with_latest_analyzer_output": sum(
            1 for row in parsed_rows if "latest" in row.analyzer_available_sources
        ),
        "rows_with_cumulative_analyzer_output": sum(
            1 for row in parsed_rows if "cumulative" in row.analyzer_available_sources
        ),
        "rows_with_generic_analyzer_output": sum(
            1 for row in parsed_rows if "generic" in row.analyzer_available_sources
        ),
        "rows_with_mapper_payload": sum(
            1 for row in parsed_rows if row.mapper_payload is not None
        ),
        "rows_without_mapper_payload": sum(
            1 for row in parsed_rows if row.mapper_payload is None
        ),
        "rows_with_candidate_seed_diagnostics": sum(
            1 for row in parsed_rows if row.candidate_seed_diagnostics is not None
        ),
        "rows_without_candidate_seed_diagnostics": sum(
            1 for row in parsed_rows if row.candidate_seed_diagnostics is None
        ),
        "rows_with_engine_output": sum(
            1 for row in parsed_rows if row.engine_output is not None
        ),
        "rows_without_engine_output": sum(
            1 for row in parsed_rows if row.engine_output is None
        ),
        "rows_with_abstain_diagnosis": sum(
            1
            for row in parsed_rows
            if _safe_dict(_safe_dict(row.engine_output).get("abstain_diagnosis"))
        ),
        "rows_with_edge_candidate_rows_block": analyzer_section[
            "rows_with_edge_candidate_rows_block"
        ],
        "rows_without_edge_candidate_rows_block": analyzer_section[
            "rows_missing_edge_candidate_rows_block"
        ],
        "preferred_analyzer_source_counts": analyzer_section[
            "preferred_analyzer_source_counts"
        ],
        "analyzer_coverage_ratio": _safe_float(
            _safe_dict(final_verdict.get("coverage_assessment")).get(
                "analyzer_coverage_ratio"
            )
        )
        or 0.0,
        "rows_with_downstream_diagnostics": _safe_int(
            _safe_dict(final_verdict.get("coverage_assessment")).get(
                "rows_with_downstream_diagnostics"
            )
        )
        or 0,
        "rows_with_downstream_diagnostics_without_analyzer_output": _safe_int(
            _safe_dict(final_verdict.get("coverage_assessment")).get(
                "rows_with_downstream_diagnostics_without_analyzer_output"
            )
        )
        or 0,
        "rows_with_mapper_payload_without_analyzer_output": _safe_int(
            _safe_dict(final_verdict.get("coverage_assessment")).get(
                "rows_with_mapper_payload_without_analyzer_output"
            )
        )
        or 0,
        "rows_with_engine_output_without_analyzer_output": _safe_int(
            _safe_dict(final_verdict.get("coverage_assessment")).get(
                "rows_with_engine_output_without_analyzer_output"
            )
        )
        or 0,
        "analyzer_layer_diagnosis_reliable": bool(
            _safe_dict(final_verdict.get("coverage_assessment")).get(
                "analyzer_layer_diagnosis_reliable"
            )
        ),
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "inputs": {
            "trade_analysis_path": str(input_path),
            "trade_analysis_resolution": input_resolution,
        },
        "data_quality": data_quality,
        "analyzer_preview": analyzer_section,
        "joined_row_formation": joined_section,
        "mapper_seed_handoff": mapper_section,
        "engine_outcome": engine_section,
        "final_verdict": final_verdict,
        "assumptions": [
            "Analyzer snapshots are chosen per row with latest preferred, then generic embedded analyzer output, then cumulative.",
            "Rows without embedded analyzer snapshots are counted explicitly as missing analyzer coverage rather than inferred from mapper or engine fields.",
            "Analyzer preview-state counts are horizon-occurrence based and are not identity-deduplicated.",
            "Engine eligible/penalized/blocked totals prefer abstain_diagnosis counts when present and otherwise fall back to ranking candidate_status counts.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    inputs = _safe_dict(report.get("inputs"))
    data_quality = _safe_dict(report.get("data_quality"))
    analyzer = _safe_dict(report.get("analyzer_preview"))
    joined = _safe_dict(report.get("joined_row_formation"))
    mapper = _safe_dict(report.get("mapper_seed_handoff"))
    engine = _safe_dict(report.get("engine_outcome"))
    verdict = _safe_dict(report.get("final_verdict"))
    coverage = _safe_dict(verdict.get("coverage_assessment"))

    lines = [f"# {REPORT_TITLE}", ""]

    if coverage.get("analyzer_layer_diagnosis_reliable") is False:
        analyzer_coverage_count = _safe_int(coverage.get("analyzer_coverage_count")) or 0
        parsed_row_count = data_quality.get("parsed_row_count", 0)
        analyzer_coverage_ratio = _safe_float(coverage.get("analyzer_coverage_ratio")) or 0.0
        limitation_note = _text(coverage.get("coverage_limitation_note"))

        lines.extend(
            [
                "> **Coverage limitation warning**",
                (
                    "> Embedded analyzer snapshots were available on "
                    f"{analyzer_coverage_count}/{parsed_row_count} parsed rows "
                    f"({_format_ratio(analyzer_coverage_ratio)})."
                ),
                (
                    "> Analyzer-layer diagnosis is not reliable for this input, "
                    "so analyzer-to-mapper-to-engine conclusions are coverage-limited."
                ),
                (
                    "> Final verdict coverage-limited: "
                    f"{_format_bool(verdict.get('coverage_limited'))}."
                ),
            ]
        )
        if limitation_note is not None:
            lines.append(f"> {limitation_note}")
        lines.append("")

    lines.extend(
        [
            "## Executive Summary",
            "",
            f"- input_path: {inputs.get('trade_analysis_path')}",
            f"- parsed_rows: {data_quality.get('parsed_row_count', 0)}",
            (
                "- coverage: "
                f"analyzer={data_quality.get('rows_with_analyzer_output', 0)}, "
                f"mapper={data_quality.get('rows_with_mapper_payload', 0)}, "
                f"engine={data_quality.get('rows_with_engine_output', 0)}"
            ),
            (
                "- analyzer_coverage: "
                f"{coverage.get('analyzer_coverage_count', 0)}/"
                f"{data_quality.get('parsed_row_count', 0)} "
                f"({_format_ratio(_safe_float(coverage.get('analyzer_coverage_ratio')) or 0.0)})"
            ),
            (
                "- analyzer_layer_diagnosis_reliable: "
                f"{_format_bool(coverage.get('analyzer_layer_diagnosis_reliable'))}"
            ),
            (
                "- final_verdict_coverage_limited: "
                f"{_format_bool(verdict.get('coverage_limited'))}"
            ),
            (
                "- pre_coverage_primary_bottleneck_layer: "
                f"{verdict.get('pre_coverage_primary_bottleneck_layer', 'inconclusive')}"
            ),
            f"- primary_bottleneck_layer: {verdict.get('primary_bottleneck_layer', 'inconclusive')}",
            "",
            "## Analyzer Findings",
            "",
            (
                "- analyzer_specific_conclusions_available: "
                f"{_format_bool(coverage.get('analyzer_specific_conclusions_available'))}"
            ),
            (
                "- analyzer_embedded_joined_row_block_rows: "
                f"available={joined.get('rows_with_analyzer_embedded_joined_row_block', 0)}, "
                f"missing={joined.get('rows_missing_analyzer_embedded_joined_row_block', 0)}"
            ),
            (
                "- preview_state_counts: "
                f"failed_absolute_minimum={analyzer.get('failed_absolute_minimum_visibility_count', 0)}, "
                f"weak_or_borderline={analyzer.get('survived_sample_gate_but_weak_or_borderline_count', 0)}, "
                f"passed_quality_gate={analyzer.get('passed_quality_gate_count', 0)}"
            ),
            (
                "- visible_group_slots: "
                f"top_strategy={_safe_dict(analyzer.get('visible_group_slot_counts')).get('top_strategy', 0)}, "
                f"top_symbol={_safe_dict(analyzer.get('visible_group_slot_counts')).get('top_symbol', 0)}, "
                f"top_alignment_state={_safe_dict(analyzer.get('visible_group_slot_counts')).get('top_alignment_state', 0)}"
            ),
            "",
            "## Joined-Row / Mapper Findings",
            "",
            (
                "- analyzer_embedded_joined_rows: "
                f"total={joined.get('analyzer_embedded_total_joined_row_count', 0)}, "
                f"diagnostic={joined.get('analyzer_embedded_diagnostic_row_count', 0)}, "
                f"dropped={joined.get('analyzer_embedded_dropped_row_count', 0)}"
            ),
            (
                "- analyzer_embedded_empty_state_categories: "
                f"{_format_count_dict(joined.get('empty_state_category_counts'))}"
            ),
            (
                "- mapper_diagnostic_joined_rows: "
                f"joined_candidate_rows_total={mapper.get('mapper_diagnostic_joined_candidate_row_count_total', 0)}, "
                f"dropped_candidate_rows_total={mapper.get('mapper_diagnostic_dropped_candidate_row_count_total', 0)}"
            ),
            (
                "- mapper_seed_handoff: "
                f"total={mapper.get('candidate_seed_count_total', 0)}, "
                f"fallback_blocked={mapper.get('fallback_blocked_count', 0)}, "
                f"joined_rows_present_but_empty={mapper.get('joined_rows_present_but_empty_count', 0)}"
            ),
            (
                "- dropped_candidate_row_reasons: "
                f"{_format_count_dict(mapper.get('dropped_candidate_row_reasons'))}"
            ),
            (
                "- note: analyzer_embedded_* metrics come from embedded analyzer snapshots; "
                "mapper_diagnostic_* metrics come from mapper candidate_seed_diagnostics and may still be present when analyzer snapshots are missing."
            ),
            "",
            "## Engine Findings",
            "",
            (
                "- selection_status_counts: "
                f"{_format_count_dict(engine.get('selection_status_counts'))}"
            ),
            (
                "- abstain_category_counts: "
                f"{_format_count_dict(engine.get('abstain_category_counts'))}"
            ),
            (
                "- aggregate_candidate_status_counts: "
                f"{_format_count_dict(engine.get('aggregate_candidate_status_counts'))}"
            ),
            "",
            "## Final Bottleneck Judgment",
            "",
            (
                "- pre_coverage_primary_bottleneck_layer: "
                f"{verdict.get('pre_coverage_primary_bottleneck_layer', 'inconclusive')}"
            ),
            f"- primary_bottleneck_layer: {verdict.get('primary_bottleneck_layer', 'inconclusive')}",
        ]
    )

    for bullet in _safe_list(verdict.get("evidence_bullets")):
        lines.append(f"- {bullet}")

    lines.append("")
    return "\n".join(lines)


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    resolved_output_dir = output_dir.expanduser()
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = Path.cwd() / resolved_output_dir
    resolved_output_dir = resolved_output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_output_dir / REPORT_JSON_NAME
    md_path = resolved_output_dir / REPORT_MD_NAME

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def load_trade_analysis_rows(path: Path) -> dict[str, Any]:
    rows: list[LoadedRow] = []
    diagnostics = {
        "total_lines": 0,
        "blank_line_count": 0,
        "malformed_line_count": 0,
        "non_object_line_count": 0,
        "parsed_row_count": 0,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            diagnostics["total_lines"] += 1
            line = raw_line.strip()
            if not line:
                diagnostics["blank_line_count"] += 1
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                diagnostics["malformed_line_count"] += 1
                continue

            if not isinstance(payload, dict):
                diagnostics["non_object_line_count"] += 1
                continue

            rows.append(LoadedRow(line_number=line_number, payload=payload))

    diagnostics["parsed_row_count"] = len(rows)
    return {
        "rows": rows,
        "diagnostics": diagnostics,
    }


def _parse_row(loaded: LoadedRow) -> ParsedRow:
    analyzer_snapshots = _extract_analyzer_snapshots(loaded.payload)
    analyzer_summary, analyzer_source = _choose_preferred_analyzer_summary(analyzer_snapshots)
    mapper_payload, mapper_path = _extract_mapper_payload(loaded.payload)
    engine_output, engine_path = _extract_engine_output(loaded.payload)
    candidate_seed_diagnostics = _extract_candidate_seed_diagnostics(
        loaded.payload,
        mapper_payload,
        engine_output,
    )

    return ParsedRow(
        line_number=loaded.line_number,
        analyzer_summary=analyzer_summary,
        analyzer_source=analyzer_source,
        analyzer_available_sources=tuple(sorted(analyzer_snapshots)),
        mapper_payload=mapper_payload,
        mapper_path=mapper_path,
        candidate_seed_diagnostics=candidate_seed_diagnostics,
        engine_output=engine_output,
        engine_path=engine_path,
    )


def _build_analyzer_section(rows: list[ParsedRow]) -> dict[str, Any]:
    preferred_source_counts: Counter[str] = Counter()
    visible_group_slot_counts: Counter[str] = Counter()
    visible_group_value_counts = {key: Counter() for key in PREVIEW_SLOT_KEYS}
    horizon_sections: dict[str, dict[str, Counter[str]]] = {
        horizon: {
            "sample_gate_counts": Counter(),
            "quality_gate_counts": Counter(),
            "candidate_strength_counts": Counter(),
            "visibility_reason_counts": Counter(),
            "classification_counts": Counter(),
        }
        for horizon in HORIZONS
    }
    missing_preview_horizon_counts: Counter[str] = Counter()
    rows_with_preview_block = 0
    rows_missing_preview_block = 0
    rows_with_edge_candidate_rows_block = 0
    rows_missing_edge_candidate_rows_block = 0

    failed_absolute_minimum_visibility_count = 0
    survived_sample_gate_but_weak_or_borderline_count = 0
    passed_quality_gate_count = 0

    for row in rows:
        if row.analyzer_source is not None:
            preferred_source_counts[row.analyzer_source] += 1

        summary = row.analyzer_summary
        if summary is None:
            rows_missing_preview_block += 1
            rows_missing_edge_candidate_rows_block += 1
            continue

        preview_block = _safe_dict(summary.get("edge_candidates_preview"))
        by_horizon = _safe_dict(preview_block.get("by_horizon"))
        if by_horizon:
            rows_with_preview_block += 1
        else:
            rows_missing_preview_block += 1

        if _extract_edge_candidate_rows_block(summary) is not None:
            rows_with_edge_candidate_rows_block += 1
        else:
            rows_missing_edge_candidate_rows_block += 1

        for horizon in HORIZONS:
            horizon_payload = _safe_dict(by_horizon.get(horizon))
            if not horizon_payload:
                missing_preview_horizon_counts[horizon] += 1
                continue

            sample_gate = _text(horizon_payload.get("sample_gate")) or "missing"
            quality_gate = _text(horizon_payload.get("quality_gate")) or "missing"
            candidate_strength = _text(horizon_payload.get("candidate_strength")) or "missing"
            visibility_reason = _text(horizon_payload.get("visibility_reason")) or "missing"

            horizon_sections[horizon]["sample_gate_counts"][sample_gate] += 1
            horizon_sections[horizon]["quality_gate_counts"][quality_gate] += 1
            horizon_sections[horizon]["candidate_strength_counts"][candidate_strength] += 1
            horizon_sections[horizon]["visibility_reason_counts"][visibility_reason] += 1

            classification = _preview_classification(
                sample_gate=sample_gate,
                quality_gate=quality_gate,
                visibility_reason=visibility_reason,
            )
            horizon_sections[horizon]["classification_counts"][classification] += 1

            if classification == "failed_absolute_minimum_visibility":
                failed_absolute_minimum_visibility_count += 1
            elif classification == "survived_sample_gate_but_weak_or_borderline":
                survived_sample_gate_but_weak_or_borderline_count += 1
            elif classification == "passed_quality_gate":
                passed_quality_gate_count += 1

            for slot_key in PREVIEW_SLOT_KEYS:
                candidate = _safe_dict(horizon_payload.get(slot_key))
                group = _text(candidate.get("group"))
                if not _is_visible_group(candidate, group):
                    continue
                visible_group_slot_counts[slot_key] += 1
                visible_group_value_counts[slot_key][group] += 1

    return {
        "rows_with_analyzer_output": sum(1 for row in rows if row.analyzer_summary is not None),
        "rows_missing_analyzer_output": sum(1 for row in rows if row.analyzer_summary is None),
        "preferred_analyzer_source_counts": _sorted_counter_dict(preferred_source_counts),
        "rows_with_edge_candidates_preview": rows_with_preview_block,
        "rows_missing_edge_candidates_preview": rows_missing_preview_block,
        "rows_with_edge_candidate_rows_block": rows_with_edge_candidate_rows_block,
        "rows_missing_edge_candidate_rows_block": rows_missing_edge_candidate_rows_block,
        "missing_preview_horizon_counts": _sorted_counter_dict(missing_preview_horizon_counts),
        "by_horizon": {
            horizon: {
                key: _sorted_counter_dict(counter)
                for key, counter in horizon_sections[horizon].items()
            }
            for horizon in HORIZONS
        },
        "visible_group_slot_counts": {
            key: visible_group_slot_counts.get(key, 0) for key in PREVIEW_SLOT_KEYS
        },
        "visible_group_value_counts": {
            key: _sorted_counter_dict(counter)
            for key, counter in visible_group_value_counts.items()
        },
        "failed_absolute_minimum_visibility_count": failed_absolute_minimum_visibility_count,
        "survived_sample_gate_but_weak_or_borderline_count": (
            survived_sample_gate_but_weak_or_borderline_count
        ),
        "passed_quality_gate_count": passed_quality_gate_count,
    }


def _build_joined_row_section(rows: list[ParsedRow]) -> dict[str, Any]:
    empty_state_category_counts: Counter[str] = Counter()
    diagnostic_rejection_reason_counts: Counter[str] = Counter()
    diagnostic_category_counts: Counter[str] = Counter()

    total_joined_row_count = 0
    diagnostic_row_count = 0
    dropped_row_count = 0
    identities_blocked_only_by_incompatibility_count = 0
    strategies_without_analyzer_compatible_horizons_count = 0
    has_only_incompatibility_rejections_count = 0
    has_only_weak_or_insufficient_candidates_count = 0
    rows_with_analyzer_embedded_joined_row_block = 0
    rows_missing_analyzer_embedded_joined_row_block = 0
    rows_missing_empty_reason_summary = 0

    for row in rows:
        summary = row.analyzer_summary
        if summary is None:
            rows_missing_analyzer_embedded_joined_row_block += 1
            rows_missing_empty_reason_summary += 1
            continue

        block = _extract_edge_candidate_rows_block(summary)
        if block is None:
            rows_missing_analyzer_embedded_joined_row_block += 1
            rows_missing_empty_reason_summary += 1
            continue
        rows_with_analyzer_embedded_joined_row_block += 1

        total_joined_row_count += _safe_int(block.get("row_count")) or len(
            _safe_list(block.get("rows"))
        )
        diagnostic_row_count += _safe_int(block.get("diagnostic_row_count")) or len(
            _safe_list(block.get("diagnostic_rows"))
        )
        dropped_row_count += _safe_int(block.get("dropped_row_count")) or len(
            _safe_list(block.get("dropped_rows"))
        )

        empty_reason_summary = _safe_dict(block.get("empty_reason_summary"))
        if not empty_reason_summary:
            rows_missing_empty_reason_summary += 1
            continue

        empty_state = _text(empty_reason_summary.get("empty_state_category")) or "missing"
        empty_state_category_counts[empty_state] += 1

        _update_counter_from_mapping(
            diagnostic_rejection_reason_counts,
            empty_reason_summary.get("diagnostic_rejection_reason_counts"),
        )
        _update_counter_from_mapping(
            diagnostic_category_counts,
            empty_reason_summary.get("diagnostic_category_counts"),
        )

        identities_blocked_only_by_incompatibility_count += len(
            _safe_list(empty_reason_summary.get("identities_blocked_only_by_incompatibility"))
        )
        strategies_without_analyzer_compatible_horizons_count += len(
            _safe_list(empty_reason_summary.get("strategies_without_analyzer_compatible_horizons"))
        )
        if empty_reason_summary.get("has_only_incompatibility_rejections") is True:
            has_only_incompatibility_rejections_count += 1
        if empty_reason_summary.get("has_only_weak_or_insufficient_candidates") is True:
            has_only_weak_or_insufficient_candidates_count += 1

    return {
        "evidence_source": "analyzer_embedded_edge_candidate_rows",
        "rows_with_analyzer_embedded_joined_row_block": (
            rows_with_analyzer_embedded_joined_row_block
        ),
        "rows_missing_analyzer_embedded_joined_row_block": (
            rows_missing_analyzer_embedded_joined_row_block
        ),
        "analyzer_embedded_total_joined_row_count": total_joined_row_count,
        "analyzer_embedded_diagnostic_row_count": diagnostic_row_count,
        "analyzer_embedded_dropped_row_count": dropped_row_count,
        "total_joined_row_count": total_joined_row_count,
        "diagnostic_row_count": diagnostic_row_count,
        "dropped_row_count": dropped_row_count,
        "empty_state_category_counts": _sorted_counter_dict(empty_state_category_counts),
        "diagnostic_rejection_reason_counts": _sorted_counter_dict(
            diagnostic_rejection_reason_counts
        ),
        "diagnostic_category_counts": _sorted_counter_dict(diagnostic_category_counts),
        "identities_blocked_only_by_incompatibility_count": (
            identities_blocked_only_by_incompatibility_count
        ),
        "strategies_without_analyzer_compatible_horizons_count": (
            strategies_without_analyzer_compatible_horizons_count
        ),
        "has_only_incompatibility_rejections_count": has_only_incompatibility_rejections_count,
        "has_only_weak_or_insufficient_candidates_count": (
            has_only_weak_or_insufficient_candidates_count
        ),
        "rows_missing_empty_reason_summary": rows_missing_empty_reason_summary,
    }


def _build_mapper_section(rows: list[ParsedRow]) -> dict[str, Any]:
    seed_source_counts: Counter[str] = Counter()
    candidate_seed_count_distribution: Counter[str] = Counter()
    candidate_seed_count_by_horizon = {horizon: 0 for horizon in HORIZONS}
    dropped_candidate_row_reasons: Counter[str] = Counter()
    fallback_block_reason_counts: Counter[str] = Counter()

    candidate_seed_count_total = 0
    joined_candidate_row_count_total = 0
    dropped_candidate_row_count_total = 0
    fallback_blocked_count = 0

    for row in rows:
        diagnostics = row.candidate_seed_diagnostics
        if diagnostics is None:
            continue

        seed_source = _text(diagnostics.get("seed_source")) or "missing"
        seed_source_counts[seed_source] += 1

        candidate_seed_count = _extract_candidate_seed_count(
            diagnostics,
            row.mapper_payload,
            row.engine_output,
        )
        candidate_seed_count_total += candidate_seed_count
        candidate_seed_count_distribution[str(candidate_seed_count)] += 1

        joined_candidate_row_count_total += _safe_int(
            diagnostics.get("joined_candidate_row_count")
        ) or 0
        dropped_candidate_row_count_total += _safe_int(
            diagnostics.get("dropped_candidate_row_count")
        ) or 0

        _update_counter_from_mapping(
            dropped_candidate_row_reasons,
            diagnostics.get("dropped_candidate_row_reasons"),
        )

        if diagnostics.get("fallback_blocked") is True:
            fallback_blocked_count += 1

        fallback_block_reason = _text(diagnostics.get("fallback_block_reason"))
        if fallback_block_reason is not None:
            fallback_block_reason_counts[fallback_block_reason] += 1

        for horizon, count in _extract_seed_count_by_horizon(diagnostics).items():
            candidate_seed_count_by_horizon[horizon] += count

    return {
        "evidence_source": "mapper_candidate_seed_diagnostics",
        "seed_source_counts": _sorted_counter_dict(seed_source_counts),
        "candidate_seed_count_total": candidate_seed_count_total,
        "candidate_seed_count_distribution": _sorted_counter_dict(
            candidate_seed_count_distribution
        ),
        "candidate_seed_count_by_horizon": candidate_seed_count_by_horizon,
        "mapper_diagnostic_joined_candidate_row_count_total": joined_candidate_row_count_total,
        "mapper_diagnostic_dropped_candidate_row_count_total": (
            dropped_candidate_row_count_total
        ),
        "joined_candidate_row_count_total": joined_candidate_row_count_total,
        "dropped_candidate_row_count_total": dropped_candidate_row_count_total,
        "dropped_candidate_row_reasons": _sorted_counter_dict(dropped_candidate_row_reasons),
        "fallback_blocked_count": fallback_blocked_count,
        "fallback_block_reason_counts": _sorted_counter_dict(fallback_block_reason_counts),
        "joined_rows_present_but_empty_count": fallback_block_reason_counts.get(
            JOINED_ROWS_PRESENT_BUT_EMPTY,
            0,
        ),
    }


def _build_engine_section(rows: list[ParsedRow]) -> dict[str, Any]:
    selection_status_counts: Counter[str] = Counter()
    reason_code_counts: Counter[str] = Counter()
    abstain_category_counts: Counter[str] = Counter()
    aggregate_candidate_status_counts: Counter[str] = Counter()

    rows_with_ranking = 0

    for row in rows:
        output = row.engine_output
        if output is None:
            continue

        selection_status = _text(output.get("selection_status")) or "missing"
        selection_status_counts[selection_status] += 1

        for reason_code in _normalize_string_list(output.get("reason_codes")):
            reason_code_counts[reason_code] += 1

        abstain = _safe_dict(output.get("abstain_diagnosis"))
        abstain_category = _text(abstain.get("category"))
        if abstain_category is not None:
            abstain_category_counts[abstain_category] += 1

        ranking = [item for item in _safe_list(output.get("ranking")) if isinstance(item, dict)]
        if ranking:
            rows_with_ranking += 1

        candidate_status_counts = _extract_engine_candidate_status_counts(output)
        for key in ("eligible", "penalized", "blocked"):
            aggregate_candidate_status_counts[key] += candidate_status_counts.get(key, 0)

    return {
        "selection_status_counts": _sorted_counter_dict(selection_status_counts),
        "reason_code_counts": _sorted_counter_dict(reason_code_counts),
        "abstain_category_counts": _sorted_counter_dict(abstain_category_counts),
        "aggregate_candidate_status_counts": {
            key: aggregate_candidate_status_counts.get(key, 0)
            for key in ("eligible", "penalized", "blocked")
        },
        "rows_with_ranking": rows_with_ranking,
    }


def _build_final_verdict(
    *,
    rows: list[ParsedRow],
    analyzer_section: dict[str, Any],
    joined_section: dict[str, Any],
    mapper_section: dict[str, Any],
    engine_section: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    row_bottleneck_counts: Counter[str] = Counter(
        _classify_row_bottleneck(row) for row in rows
    )

    rows_with_analyzer_output = _safe_int(analyzer_section.get("rows_with_analyzer_output")) or 0
    rows_missing_analyzer_output = _safe_int(
        analyzer_section.get("rows_missing_analyzer_output")
    ) or 0
    coverage_assessment = _build_analyzer_coverage_assessment(
        rows=rows,
        row_count=row_count,
        rows_with_analyzer_output=rows_with_analyzer_output,
        rows_missing_analyzer_output=rows_missing_analyzer_output,
    )
    rows_with_engine_output = sum(
        int(value)
        for value in _safe_dict(engine_section.get("selection_status_counts")).values()
        if isinstance(value, int)
    )

    preview_failed = _safe_int(
        analyzer_section.get("failed_absolute_minimum_visibility_count")
    ) or 0
    preview_weak = _safe_int(
        analyzer_section.get("survived_sample_gate_but_weak_or_borderline_count")
    ) or 0
    preview_passed = _safe_int(analyzer_section.get("passed_quality_gate_count")) or 0
    joined_total = _safe_int(joined_section.get("total_joined_row_count")) or 0
    seed_total = _safe_int(mapper_section.get("candidate_seed_count_total")) or 0
    dropped_reason_total = sum(
        int(value)
        for value in _safe_dict(mapper_section.get("dropped_candidate_row_reasons")).values()
        if isinstance(value, int)
    )
    fallback_blocked = _safe_int(mapper_section.get("fallback_blocked_count")) or 0
    joined_present_but_empty = _safe_int(
        mapper_section.get("joined_rows_present_but_empty_count")
    ) or 0
    selection_status_counts = _safe_dict(engine_section.get("selection_status_counts"))
    engine_abstain = _safe_int(selection_status_counts.get("abstain")) or 0
    engine_blocked = _safe_int(selection_status_counts.get("blocked")) or 0
    engine_selected = _safe_int(selection_status_counts.get("selected")) or 0
    engine_candidate_status_counts = _safe_dict(
        engine_section.get("aggregate_candidate_status_counts")
    )

    layer_signal_counts = {
        "analyzer": row_bottleneck_counts.get("analyzer", 0),
        "mapper": row_bottleneck_counts.get("mapper", 0),
        "engine": row_bottleneck_counts.get("engine", 0),
        "inconclusive": row_bottleneck_counts.get("inconclusive", 0),
        "none": row_bottleneck_counts.get("none", 0),
    }

    pre_coverage_primary_bottleneck_layer = _select_primary_bottleneck(
        {
            "analyzer": layer_signal_counts["analyzer"],
            "mapper": layer_signal_counts["mapper"],
            "engine": layer_signal_counts["engine"],
        }
    )
    primary_bottleneck_layer = pre_coverage_primary_bottleneck_layer
    coverage_limited = False

    analyzer_reliable = coverage_assessment["analyzer_layer_diagnosis_reliable"] is True
    has_downstream_diagnostics = coverage_assessment["rows_with_downstream_diagnostics"] > 0

    if (
        analyzer_reliable is False
        and has_downstream_diagnostics
        and primary_bottleneck_layer in {"mapper", "engine"}
    ):
        primary_bottleneck_layer = "inconclusive"
        coverage_limited = True
    elif analyzer_reliable is False and row_count > 0:
        coverage_limited = True

    evidence_bullets: list[str] = []
    evidence_bullets.append(
        "Analyzer output was embedded on "
        f"{rows_with_analyzer_output}/{row_count} parsed rows "
        f"({_format_ratio(coverage_assessment['analyzer_coverage_ratio'])}); "
        f"{rows_missing_analyzer_output} rows had no embedded analyzer snapshot."
    )
    evidence_bullets.append(
        "Analyzer-layer diagnosis reliability was "
        f"{_format_bool(coverage_assessment['analyzer_layer_diagnosis_reliable'])}; "
        f"pre-coverage primary_bottleneck_layer={pre_coverage_primary_bottleneck_layer}; "
        f"final primary_bottleneck_layer={primary_bottleneck_layer}; "
        f"final verdict coverage-limited={_format_bool(coverage_limited)}."
    )
    if coverage_assessment["coverage_limitation_note"] is not None:
        evidence_bullets.append(coverage_assessment["coverage_limitation_note"])
    evidence_bullets.append(
        "Analyzer preview states across all horizon snapshots were "
        f"failed_absolute_minimum={preview_failed}, weak_or_borderline={preview_weak}, passed_quality_gate={preview_passed}."
    )
    evidence_bullets.append(
        "Analyzer-embedded joined-row formation produced "
        f"{joined_total} engine-facing joined rows, "
        f"{joined_section.get('analyzer_embedded_diagnostic_row_count', 0)} diagnostic rows, "
        f"and {joined_section.get('analyzer_embedded_dropped_row_count', 0)} dropped rows."
    )
    evidence_bullets.append(
        "Mapper diagnostics recorded "
        f"joined_candidate_rows_total={mapper_section.get('mapper_diagnostic_joined_candidate_row_count_total', 0)}, "
        f"dropped_candidate_rows_total={mapper_section.get('mapper_diagnostic_dropped_candidate_row_count_total', 0)}, "
        f"and {seed_total} total seeds with fallback_blocked={fallback_blocked}; "
        f"{JOINED_ROWS_PRESENT_BUT_EMPTY} appeared {joined_present_but_empty} times."
    )
    if dropped_reason_total > 0:
        evidence_bullets.append(
            "Mapper dropped candidate rows for "
            f"{dropped_reason_total} counted reasons: {_format_count_dict(mapper_section.get('dropped_candidate_row_reasons'))}."
        )
    evidence_bullets.append(
        "Engine outcomes were "
        f"selected={engine_selected}, abstain={engine_abstain}, blocked={engine_blocked} "
        f"across {rows_with_engine_output} rows with engine output."
    )
    evidence_bullets.append(
        "Engine candidate-status aggregates were "
        f"eligible={engine_candidate_status_counts.get('eligible', 0)}, "
        f"penalized={engine_candidate_status_counts.get('penalized', 0)}, "
        f"blocked={engine_candidate_status_counts.get('blocked', 0)}."
    )
    evidence_bullets.append(
        "Layer signal counts were "
        f"analyzer={layer_signal_counts['analyzer']}, mapper={layer_signal_counts['mapper']}, "
        f"engine={layer_signal_counts['engine']}, none={layer_signal_counts['none']}, "
        f"inconclusive={layer_signal_counts['inconclusive']}."
    )

    return {
        "pre_coverage_primary_bottleneck_layer": pre_coverage_primary_bottleneck_layer,
        "primary_bottleneck_layer": primary_bottleneck_layer,
        "coverage_limited": coverage_limited,
        "coverage_limitation_reason": coverage_assessment["coverage_limitation_reason"],
        "coverage_assessment": coverage_assessment,
        "layer_signal_counts": layer_signal_counts,
        "evidence_bullets": evidence_bullets,
    }


def _select_primary_bottleneck(layer_signal_counts: dict[str, int]) -> str:
    ordered = sorted(
        layer_signal_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    top_layer, top_count = ordered[0]
    if top_count <= 0:
        return "inconclusive"
    if len(ordered) > 1 and ordered[1][1] == top_count:
        return "inconclusive"
    return top_layer


def _classify_row_bottleneck(row: ParsedRow) -> str:
    seed_diagnostics = row.candidate_seed_diagnostics or {}
    seed_count = _extract_candidate_seed_count(
        seed_diagnostics,
        row.mapper_payload,
        row.engine_output,
    )
    selection_status = _text(_safe_dict(row.engine_output).get("selection_status"))

    if seed_count > 0:
        if selection_status in {"abstain", "blocked"} and row.analyzer_summary is not None:
            return "engine"
        if selection_status in {"abstain", "blocked"} and _has_explicit_engine_only_evidence(
            row
        ):
            return "engine"
        if selection_status == "selected":
            return "none"
        return "inconclusive"

    analyzer_summary = row.analyzer_summary
    if analyzer_summary is None:
        return "inconclusive"

    block = _extract_edge_candidate_rows_block(analyzer_summary)
    joined_row_count = _safe_int(_safe_dict(block).get("row_count")) or len(
        _safe_list(_safe_dict(block).get("rows"))
    )
    empty_reason_summary = _safe_dict(_safe_dict(block).get("empty_reason_summary"))
    empty_state = _text(empty_reason_summary.get("empty_state_category"))
    fallback_block_reason = _text(seed_diagnostics.get("fallback_block_reason"))
    dropped_candidate_row_count = _safe_int(
        seed_diagnostics.get("dropped_candidate_row_count")
    ) or 0

    preview_state_counts = _row_preview_state_counts(analyzer_summary)

    if dropped_candidate_row_count > 0:
        return "mapper"

    if joined_row_count > 0:
        return "mapper"

    if fallback_block_reason == JOINED_ROWS_PRESENT_BUT_EMPTY:
        return "analyzer"

    if block is not None and joined_row_count <= 0:
        return "analyzer"

    if empty_state in {
        "only_incompatibility_rejections",
        "only_weak_or_insufficient_candidates",
        "mixed_rejections_without_eligible_rows",
        "no_joined_candidates_evaluated",
    }:
        return "analyzer"

    if preview_state_counts["passed_quality_gate"] > 0:
        return "mapper"

    if (
        preview_state_counts["failed_absolute_minimum_visibility"] > 0
        or preview_state_counts["survived_sample_gate_but_weak_or_borderline"] > 0
    ):
        return "analyzer"

    return "inconclusive"


def _build_analyzer_coverage_assessment(
    *,
    rows: list[ParsedRow],
    row_count: int,
    rows_with_analyzer_output: int,
    rows_missing_analyzer_output: int,
) -> dict[str, Any]:
    rows_with_downstream_diagnostics = 0
    rows_with_downstream_diagnostics_without_analyzer_output = 0
    rows_with_mapper_payload_without_analyzer_output = 0
    rows_with_engine_output_without_analyzer_output = 0

    for row in rows:
        has_downstream_diagnostics = any(
            (
                row.mapper_payload is not None,
                row.candidate_seed_diagnostics is not None,
                row.engine_output is not None,
            )
        )
        if has_downstream_diagnostics:
            rows_with_downstream_diagnostics += 1
            if row.analyzer_summary is None:
                rows_with_downstream_diagnostics_without_analyzer_output += 1

        if row.mapper_payload is not None and row.analyzer_summary is None:
            rows_with_mapper_payload_without_analyzer_output += 1

        if row.engine_output is not None and row.analyzer_summary is None:
            rows_with_engine_output_without_analyzer_output += 1

    analyzer_coverage_ratio = _compute_ratio(rows_with_analyzer_output, row_count)
    analyzer_specific_conclusions_available = rows_with_analyzer_output > 0
    analyzer_layer_diagnosis_reliable = (
        analyzer_specific_conclusions_available
        and rows_with_downstream_diagnostics_without_analyzer_output == 0
    )

    coverage_limitation_reason: str | None = None
    coverage_limitation_note: str | None = None

    if rows_with_analyzer_output <= 0:
        coverage_limitation_reason = "ANALYZER_SNAPSHOTS_MISSING"
        coverage_limitation_note = (
            "Analyzer-specific conclusions are unavailable for this input because "
            "embedded analyzer snapshots were missing on every parsed row."
        )
    elif rows_with_downstream_diagnostics_without_analyzer_output > 0:
        coverage_limitation_reason = "DOWNSTREAM_ROWS_MISSING_ANALYZER_SNAPSHOTS"
        coverage_limitation_note = (
            "Some rows reached mapper or engine diagnostics without an embedded analyzer "
            "snapshot, so downstream bottleneck claims must be treated as inconclusive."
        )

    return {
        "analyzer_coverage_count": rows_with_analyzer_output,
        "analyzer_missing_count": rows_missing_analyzer_output,
        "analyzer_coverage_ratio": analyzer_coverage_ratio,
        "rows_with_downstream_diagnostics": rows_with_downstream_diagnostics,
        "rows_with_downstream_diagnostics_without_analyzer_output": (
            rows_with_downstream_diagnostics_without_analyzer_output
        ),
        "rows_with_mapper_payload_without_analyzer_output": (
            rows_with_mapper_payload_without_analyzer_output
        ),
        "rows_with_engine_output_without_analyzer_output": (
            rows_with_engine_output_without_analyzer_output
        ),
        "analyzer_specific_conclusions_available": analyzer_specific_conclusions_available,
        "analyzer_layer_diagnosis_reliable": analyzer_layer_diagnosis_reliable,
        "coverage_limitation_reason": coverage_limitation_reason,
        "coverage_limitation_note": coverage_limitation_note,
    }


def _has_explicit_engine_only_evidence(row: ParsedRow) -> bool:
    output = _safe_dict(row.engine_output)
    if not output:
        return False

    selection_status = _text(output.get("selection_status"))
    if selection_status not in {"abstain", "blocked"}:
        return False

    abstain = _safe_dict(output.get("abstain_diagnosis"))
    abstain_category = _text(abstain.get("category"))
    if abstain_category == "tied_top_candidates":
        return True

    candidate_status_counts = _extract_engine_candidate_status_counts(output)
    return candidate_status_counts.get("eligible", 0) > 0


def _row_preview_state_counts(analyzer_summary: dict[str, Any]) -> dict[str, int]:
    counts = {
        "failed_absolute_minimum_visibility": 0,
        "survived_sample_gate_but_weak_or_borderline": 0,
        "passed_quality_gate": 0,
    }
    by_horizon = _safe_dict(
        _safe_dict(analyzer_summary.get("edge_candidates_preview")).get("by_horizon")
    )
    for horizon in HORIZONS:
        horizon_payload = _safe_dict(by_horizon.get(horizon))
        if not horizon_payload:
            continue
        classification = _preview_classification(
            sample_gate=_text(horizon_payload.get("sample_gate")) or "missing",
            quality_gate=_text(horizon_payload.get("quality_gate")) or "missing",
            visibility_reason=_text(horizon_payload.get("visibility_reason")) or "missing",
        )
        if classification in counts:
            counts[classification] += 1
    return counts


def _extract_analyzer_snapshots(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}

    for source, path_list in EXPLICIT_ANALYZER_PATHS.items():
        for path in path_list:
            value = _get_nested_value(payload, *path)
            if _looks_like_analyzer_summary(value):
                snapshots[source] = value
                break

    if _looks_like_analyzer_summary(payload) and "generic" not in snapshots:
        snapshots["generic"] = payload

    for path_parts, value in _recursive_find_matching_dicts(
        payload,
        predicate=_looks_like_analyzer_summary,
        max_depth=6,
    ):
        source = _infer_analyzer_source(path_parts)
        if source not in snapshots:
            snapshots[source] = value

    return snapshots


def _choose_preferred_analyzer_summary(
    snapshots: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    for source in ("latest", "generic", "cumulative"):
        summary = snapshots.get(source)
        if isinstance(summary, dict) and summary:
            return summary, source
    return None, None


def _extract_mapper_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    for path in EXPLICIT_MAPPER_PATHS:
        value = _get_nested_value(payload, *path)
        if _looks_like_explicit_mapper_payload(value):
            return value, ".".join(path)

    if _looks_like_mapper_payload(payload):
        return payload, "root"

    found = _recursive_find_first_dict(payload, predicate=_looks_like_mapper_payload, max_depth=5)
    if found is None:
        return None, None
    path_parts, value = found
    return value, ".".join(path_parts)


def _extract_engine_output(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    for path in EXPLICIT_ENGINE_PATHS:
        value = _get_nested_value(payload, *path)
        if _looks_like_engine_output(value):
            return value, ".".join(path)

    if _looks_like_engine_output(payload):
        return payload, "root"

    found = _recursive_find_first_dict(payload, predicate=_looks_like_engine_output, max_depth=5)
    if found is None:
        return None, None
    path_parts, value = found
    return value, ".".join(path_parts)


def _extract_candidate_seed_diagnostics(
    row: dict[str, Any],
    mapper_payload: dict[str, Any] | None,
    engine_output: dict[str, Any] | None,
) -> dict[str, Any] | None:
    for candidate in (
        _safe_dict(_safe_dict(mapper_payload).get("candidate_seed_diagnostics")),
        _safe_dict(row.get("candidate_seed_diagnostics")),
        _safe_dict(_safe_dict(engine_output).get("candidate_seed_diagnostics")),
        _safe_dict(
            _safe_dict(_safe_dict(engine_output).get("abstain_diagnosis")).get(
                "candidate_seed_diagnostics"
            )
        ),
    ):
        if candidate:
            return candidate
    return None


def _extract_edge_candidate_rows_block(summary: dict[str, Any]) -> dict[str, Any] | None:
    direct = _safe_dict(summary.get("edge_candidate_rows"))
    if direct:
        return direct

    found = _recursive_find_first_dict_for_key(summary, key="edge_candidate_rows", max_depth=5)
    if found is None:
        return None
    _, value = found
    return value


def _extract_candidate_seed_count(
    diagnostics: dict[str, Any],
    mapper_payload: dict[str, Any] | None,
    engine_output: dict[str, Any] | None,
) -> int:
    candidates = (
        diagnostics.get("candidate_seed_count"),
        _safe_dict(mapper_payload).get("candidate_seed_count"),
        _safe_dict(engine_output).get("candidate_seed_count"),
        _safe_dict(_safe_dict(engine_output).get("abstain_diagnosis")).get(
            "candidate_seed_count"
        ),
    )
    for value in candidates:
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return 0


def _extract_seed_count_by_horizon(diagnostics: dict[str, Any]) -> dict[str, int]:
    counts = {horizon: 0 for horizon in HORIZONS}
    for item in _iter_horizon_diagnostics(diagnostics):
        horizon = _normalize_horizon(item.get("horizon"))
        if horizon is None:
            continue
        count = _safe_int(item.get("seed_generated_count"))
        if count is None:
            count = 1 if item.get("seed_generated") is True else 0
        counts[horizon] += count
    return counts


def _extract_engine_candidate_status_counts(output: dict[str, Any]) -> dict[str, int]:
    abstain = _safe_dict(output.get("abstain_diagnosis"))
    explicit_counts = {
        "eligible": _safe_int(abstain.get("eligible_candidate_count")),
        "penalized": _safe_int(abstain.get("penalized_candidate_count")),
        "blocked": _safe_int(abstain.get("blocked_candidate_count")),
    }
    if all(value is not None for value in explicit_counts.values()):
        return {
            key: int(value or 0) for key, value in explicit_counts.items()
        }

    counts: Counter[str] = Counter()
    for item in _safe_list(output.get("ranking")):
        if not isinstance(item, dict):
            continue
        candidate_status = _text(item.get("candidate_status"))
        if candidate_status in {"eligible", "penalized", "blocked"}:
            counts[candidate_status] += 1
    return {
        key: counts.get(key, 0) for key in ("eligible", "penalized", "blocked")
    }


def _iter_horizon_diagnostics(diagnostics: dict[str, Any]) -> Iterable[dict[str, Any]]:
    raw = diagnostics.get("horizon_diagnostics")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(raw, dict):
        for horizon, item in raw.items():
            if not isinstance(item, dict):
                continue
            if _text(item.get("horizon")) is None:
                yield {"horizon": horizon, **item}
            else:
                yield item


def _preview_classification(
    *,
    sample_gate: str,
    quality_gate: str,
    visibility_reason: str,
) -> str:
    if visibility_reason == "failed_absolute_minimum_gate" or sample_gate != "passed":
        return "failed_absolute_minimum_visibility"
    if quality_gate == "passed":
        return "passed_quality_gate"
    if sample_gate == "passed":
        return "survived_sample_gate_but_weak_or_borderline"
    return "unknown"


def _is_visible_group(candidate: dict[str, Any], group: str | None) -> bool:
    if not candidate:
        return False
    if candidate.get("sample_gate") != "passed":
        return False
    if _text(candidate.get("candidate_strength")) == "insufficient_data":
        return False
    if group is None:
        return False
    return group.strip().lower() not in INVALID_VISIBLE_GROUP_VALUES


def _looks_like_analyzer_summary(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in ANALYZER_LIKE_KEYS)


def _looks_like_explicit_mapper_payload(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in MAPPER_LIKE_KEYS)


def _looks_like_mapper_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if _looks_like_engine_output(value):
        return False
    if "candidates" in value:
        return True
    return (
        "candidate_seed_count" in value
        and "candidate_seed_diagnostics" in value
        and ("errors" in value or "warnings" in value or "ok" in value)
    )


def _looks_like_engine_output(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in ENGINE_LIKE_KEYS)


def _infer_analyzer_source(path_parts: tuple[str, ...]) -> str:
    lowered = [part.lower() for part in path_parts]
    if any("latest" in part for part in lowered):
        return "latest"
    if any("cumulative" in part for part in lowered):
        return "cumulative"
    return "generic"


def _get_nested_value(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _recursive_find_first_dict(
    payload: Any,
    *,
    predicate: Callable[[Any], bool],
    max_depth: int,
    current_path: tuple[str, ...] = (),
    current_depth: int = 0,
) -> tuple[tuple[str, ...], dict[str, Any]] | None:
    matches = _recursive_find_matching_dicts(
        payload,
        predicate=predicate,
        max_depth=max_depth,
        current_path=current_path,
        current_depth=current_depth,
    )
    for match in matches:
        return match
    return None


def _recursive_find_matching_dicts(
    payload: Any,
    *,
    predicate: Callable[[Any], bool],
    max_depth: int,
    current_path: tuple[str, ...] = (),
    current_depth: int = 0,
) -> Iterable[tuple[tuple[str, ...], dict[str, Any]]]:
    if current_depth > max_depth:
        return

    if isinstance(payload, dict):
        if current_path and predicate(payload):
            yield current_path, payload
        for key, value in payload.items():
            yield from _recursive_find_matching_dicts(
                value,
                predicate=predicate,
                max_depth=max_depth,
                current_path=current_path + (str(key),),
                current_depth=current_depth + 1,
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _recursive_find_matching_dicts(
                value,
                predicate=predicate,
                max_depth=max_depth,
                current_path=current_path + (f"[{index}]",),
                current_depth=current_depth + 1,
            )


def _recursive_find_first_dict_for_key(
    payload: Any,
    *,
    key: str,
    max_depth: int,
    current_path: tuple[str, ...] = (),
    current_depth: int = 0,
) -> tuple[tuple[str, ...], dict[str, Any]] | None:
    if current_depth > max_depth:
        return None

    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, dict):
            return current_path + (key,), value
        for child_key, child_value in payload.items():
            found = _recursive_find_first_dict_for_key(
                child_value,
                key=key,
                max_depth=max_depth,
                current_path=current_path + (str(child_key),),
                current_depth=current_depth + 1,
            )
            if found is not None:
                return found
    elif isinstance(payload, list):
        for index, child_value in enumerate(payload):
            found = _recursive_find_first_dict_for_key(
                child_value,
                key=key,
                max_depth=max_depth,
                current_path=current_path + (f"[{index}]",),
                current_depth=current_depth + 1,
            )
            if found is not None:
                return found
    return None


def _update_counter_from_mapping(counter: Counter[str], mapping: Any) -> None:
    if not isinstance(mapping, dict):
        return
    for key, value in mapping.items():
        text = _text(key)
        count = _safe_int(value)
        if text is None or count is None:
            continue
        counter[text] += count


def _resolve_existing_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} path must be a file: {resolved}")
    return resolved


def _normalize_horizon(value: Any) -> str | None:
    text = _text(value)
    return text if text in HORIZONS else None


def _normalize_string_list(value: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(value, list):
        return result
    for item in value:
        text = _text(item)
        if text is not None:
            result.append(text)
    return result


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        key: counter[key]
        for key in sorted(counter, key=lambda item: str(item))
    }


def _format_count_dict(value: Any) -> str:
    mapping = _safe_dict(value)
    if not mapping:
        return "none"
    items = [f"{key}={mapping[key]}" for key in sorted(mapping)]
    return ", ".join(items)


def _compute_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _format_ratio(value: float) -> str:
    return f"{value:.1%}"


def _format_bool(value: Any) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
