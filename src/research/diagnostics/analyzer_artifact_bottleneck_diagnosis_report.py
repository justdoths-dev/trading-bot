from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


REPORT_TYPE = "analyzer_artifact_bottleneck_diagnosis_report"
REPORT_TITLE = "Analyzer Artifact Bottleneck Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_SUMMARY_PATH = Path("logs/research_reports/latest/summary.json")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
KNOWN_HORIZON_ORDER = ("1m", "5m", "15m", "1h", "4h", "1d")
_KNOWN_HORIZON_INDEX = {
    horizon: index for index, horizon in enumerate(KNOWN_HORIZON_ORDER)
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose analyzer-side preview/joined-row bottlenecks from a single "
            "analyzer summary.json artifact."
        )
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help=(
            "Analyzer summary.json path. Defaults to "
            "logs/research_reports/latest/summary.json."
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
    summary_path, resolution = resolve_summary_path(args.summary_path)
    report = build_report(
        summary_path=summary_path,
        summary_path_resolution=resolution,
    )

    written_paths: dict[str, str] = {}
    if args.write_latest_copy:
        written_paths = write_report_files(report, args.output_dir)

    summary = {
        "report_type": REPORT_TYPE,
        "summary_path": report["inputs"]["summary_path"],
        "summary_path_resolution": report["inputs"]["summary_path_resolution"],
        "summary_file_exists": report["artifact_presence"]["summary_file_exists"],
        "summary_payload_is_object": report["artifact_presence"][
            "summary_payload_is_object"
        ],
        "load_error": report["artifact_presence"]["load_error"],
        "preview_block_exists": report["analyzer_preview"]["preview_block_exists"],
        "joined_row_block_exists": report["joined_row_artifact"][
            "joined_row_block_exists"
        ],
        "compatibility_diagnostics_present": report["compatibility_diagnostics"][
            "compatibility_diagnostics_present"
        ],
        "artifact_sufficient_for_current_snapshot_diagnosis": report["final_verdict"][
            "artifact_sufficient_for_current_snapshot_diagnosis"
        ],
        "verdict_category": report["final_verdict"]["verdict_category"],
        "written_paths": written_paths,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def resolve_summary_path(explicit_path: Path | None) -> tuple[Path, str]:
    candidate = explicit_path if explicit_path is not None else DEFAULT_SUMMARY_PATH
    resolution = "explicit" if explicit_path is not None else "default"

    resolved = candidate.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve(), resolution


def build_report(
    *,
    summary_path: Path,
    summary_path_resolution: str = "explicit",
) -> dict[str, Any]:
    loaded = load_summary_artifact(summary_path)
    payload = loaded["payload"]

    preview = _build_preview_section(payload)
    joined = _build_joined_row_section(payload)
    compatibility = _build_compatibility_section(
        joined_block=_safe_dict(payload.get("edge_candidate_rows")),
        empty_reason_summary=_safe_dict(joined.get("empty_reason_summary")),
    )
    final_verdict = _build_final_verdict(
        loaded=loaded,
        preview=preview,
        joined=joined,
        compatibility=compatibility,
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "inputs": {
            "summary_path": str(summary_path),
            "summary_path_resolution": summary_path_resolution,
        },
        "artifact_presence": {
            "summary_file_exists": loaded["exists"],
            "summary_path_is_file": loaded["is_file"],
            "summary_json_loaded": loaded["json_loaded"],
            "summary_payload_is_object": loaded["payload_is_object"],
            "load_error": loaded["load_error"],
            "top_level_keys": loaded["top_level_keys"],
        },
        "analyzer_preview": preview,
        "joined_row_artifact": joined,
        "compatibility_diagnostics": compatibility,
        "final_verdict": final_verdict,
        "assumptions": [
            "This report reads a single analyzer summary artifact only.",
            "It does not inspect trade-analysis JSONL, mapper payloads, or engine output.",
            "Compatibility summaries are derived only from persisted analyzer artifact fields when present.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    inputs = _safe_dict(report.get("inputs"))
    artifact = _safe_dict(report.get("artifact_presence"))
    preview = _safe_dict(report.get("analyzer_preview"))
    joined = _safe_dict(report.get("joined_row_artifact"))
    compatibility = _safe_dict(report.get("compatibility_diagnostics"))
    verdict = _safe_dict(report.get("final_verdict"))

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Executive Summary",
        "",
        f"- summary_path: {inputs.get('summary_path')}",
        f"- summary_path_resolution: {inputs.get('summary_path_resolution', 'unknown')}",
        f"- summary_file_exists: {_format_bool(artifact.get('summary_file_exists'))}",
        f"- summary_payload_is_object: {_format_bool(artifact.get('summary_payload_is_object'))}",
        f"- preview_block_exists: {_format_bool(preview.get('preview_block_exists'))}",
        f"- joined_row_block_exists: {_format_bool(joined.get('joined_row_block_exists'))}",
        (
            "- artifact_sufficient_for_current_snapshot_diagnosis: "
            f"{_format_bool(verdict.get('artifact_sufficient_for_current_snapshot_diagnosis'))}"
        ),
        f"- analyzer_only_verdict: {verdict.get('verdict_category', 'inconclusive')}",
        "",
        "## Artifact Presence",
        "",
        f"- summary_path_is_file: {_format_bool(artifact.get('summary_path_is_file'))}",
        f"- summary_json_loaded: {_format_bool(artifact.get('summary_json_loaded'))}",
        f"- load_error: {artifact.get('load_error') or 'none'}",
        f"- top_level_keys: {_format_string_list(artifact.get('top_level_keys'))}",
        "",
        "## Analyzer Preview",
        "",
        f"- preview_by_horizon_present: {_format_bool(preview.get('preview_by_horizon_present'))}",
        f"- preview_horizons_present: {_format_string_list(preview.get('preview_horizons_present'))}",
        f"- preview_data_present: {_format_bool(preview.get('preview_data_present'))}",
        (
            "- expected_missing_sub_blocks: "
            f"{_format_string_list(preview.get('expected_missing_sub_blocks'))}"
        ),
        "",
        "## Joined-Row Artifact",
        "",
        f"- row_count: {_format_optional_int(joined.get('row_count'))}",
        f"- diagnostic_row_count: {_format_optional_int(joined.get('diagnostic_row_count'))}",
        f"- dropped_row_count: {_format_optional_int(joined.get('dropped_row_count'))}",
        f"- rows_list_count: {_format_optional_int(joined.get('rows_list_count'))}",
        (
            "- diagnostic_rows_list_count: "
            f"{_format_optional_int(joined.get('diagnostic_rows_list_count'))}"
        ),
        (
            "- identity_horizon_evaluation_count: "
            f"{_format_optional_int(joined.get('identity_horizon_evaluation_count'))}"
        ),
        (
            "- empty_reason_summary_present: "
            f"{_format_bool(joined.get('empty_reason_summary_present'))}"
        ),
        (
            "- empty_state_category: "
            f"{_safe_dict(joined.get('empty_reason_summary')).get('empty_state_category', 'n/a')}"
        ),
        (
            "- dominant_rejection_reason: "
            f"{_safe_dict(joined.get('empty_reason_summary')).get('dominant_rejection_reason', 'n/a')}"
        ),
        (
            "- expected_missing_sub_blocks: "
            f"{_format_string_list(joined.get('expected_missing_sub_blocks'))}"
        ),
        "",
        "## Compatibility Signals",
        "",
        (
            "- compatibility_diagnostics_present: "
            f"{_format_bool(compatibility.get('compatibility_diagnostics_present'))}"
        ),
        (
            "- identities_with_raw_preview_visibility_count: "
            f"{compatibility.get('identities_with_raw_preview_visibility_count', 0)}"
        ),
        (
            "- identities_with_compatibility_filtered_visibility_count: "
            f"{compatibility.get('identities_with_compatibility_filtered_visibility_count', 0)}"
        ),
        (
            "- raw_preview_visible_but_compatibility_filtered_invisible_count: "
            f"{compatibility.get('raw_preview_visible_but_compatibility_filtered_invisible_count', 0)}"
        ),
        (
            "- identities_with_no_analyzer_compatible_horizons_count: "
            f"{compatibility.get('identities_with_no_analyzer_compatible_horizons_count', 0)}"
        ),
        (
            "- identities_blocked_only_by_incompatibility_count: "
            f"{compatibility.get('identities_blocked_only_by_incompatibility_count', 0)}"
        ),
        (
            "- strategies_without_analyzer_compatible_horizons_count: "
            f"{compatibility.get('strategies_without_analyzer_compatible_horizons_count', 0)}"
        ),
        (
            "- conservative_signals: "
            f"{_format_string_list(compatibility.get('conservative_signals'))}"
        ),
        "",
        "## Final Verdict",
        "",
        f"- scope: {verdict.get('scope', 'analyzer_artifact_only')}",
        (
            "- expected_missing_blocks: "
            f"{_format_string_list(verdict.get('expected_missing_blocks'))}"
        ),
        (
            "- expected_missing_sub_blocks: "
            f"{_format_string_list(verdict.get('expected_missing_sub_blocks'))}"
        ),
    ]

    for bullet in _safe_list(verdict.get("facts")):
        lines.append(f"- fact: {bullet}")

    for bullet in _safe_list(verdict.get("inference_notes")):
        lines.append(f"- inference: {bullet}")

    for bullet in _safe_list(verdict.get("uncertainty_notes")):
        lines.append(f"- uncertainty: {bullet}")

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


def load_summary_artifact(path: Path) -> dict[str, Any]:
    result = {
        "exists": path.exists(),
        "is_file": path.is_file(),
        "json_loaded": False,
        "payload_is_object": False,
        "load_error": None,
        "payload": {},
        "top_level_keys": [],
    }

    if not result["exists"]:
        result["load_error"] = "summary_path_missing"
        return result

    if not result["is_file"]:
        result["load_error"] = "summary_path_not_file"
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        result["load_error"] = "summary_path_unreadable"
        return result
    except json.JSONDecodeError:
        result["load_error"] = "summary_json_invalid"
        return result

    result["json_loaded"] = True
    if not isinstance(payload, dict):
        result["load_error"] = "summary_payload_not_object"
        return result

    result["payload_is_object"] = True
    result["payload"] = payload
    result["top_level_keys"] = sorted(str(key) for key in payload)
    return result


def _build_preview_section(summary: dict[str, Any]) -> dict[str, Any]:
    preview_value = summary.get("edge_candidates_preview")
    preview_block = _safe_dict(preview_value)
    by_horizon_value = preview_block.get("by_horizon")

    preview_horizons_present: list[str] = []
    if isinstance(by_horizon_value, dict):
        preview_horizons_present = _sort_horizon_strings(
            _normalize_string_list(list(by_horizon_value.keys()))
        )

    expected_missing_sub_blocks: list[str] = []
    if isinstance(preview_value, dict) and not isinstance(by_horizon_value, dict):
        expected_missing_sub_blocks.append("edge_candidates_preview.by_horizon")

    return {
        "preview_block_exists": isinstance(preview_value, dict),
        "preview_by_horizon_present": isinstance(by_horizon_value, dict),
        "preview_horizons_present": preview_horizons_present,
        "preview_data_present": bool(preview_horizons_present),
        "expected_missing_sub_blocks": expected_missing_sub_blocks,
    }


def _build_joined_row_section(summary: dict[str, Any]) -> dict[str, Any]:
    joined_value = summary.get("edge_candidate_rows")
    joined_block = _safe_dict(joined_value)
    empty_reason_value = joined_block.get("empty_reason_summary")
    rows_value = joined_block.get("rows")
    diagnostic_rows_value = joined_block.get("diagnostic_rows")
    identity_evaluations_value = joined_block.get("identity_horizon_evaluations")

    expected_missing_sub_blocks: list[str] = []
    if isinstance(joined_value, dict):
        if not isinstance(rows_value, list):
            expected_missing_sub_blocks.append("edge_candidate_rows.rows")
        if not isinstance(diagnostic_rows_value, list):
            expected_missing_sub_blocks.append("edge_candidate_rows.diagnostic_rows")
        if not isinstance(identity_evaluations_value, list):
            expected_missing_sub_blocks.append(
                "edge_candidate_rows.identity_horizon_evaluations"
            )
        if not isinstance(empty_reason_value, dict):
            expected_missing_sub_blocks.append(
                "edge_candidate_rows.empty_reason_summary"
            )

    return {
        "joined_row_block_exists": isinstance(joined_value, dict),
        "row_count": _safe_int(joined_block.get("row_count")),
        "diagnostic_row_count": _safe_int(joined_block.get("diagnostic_row_count")),
        "dropped_row_count": _safe_int(joined_block.get("dropped_row_count")),
        "rows_list_count": _safe_list_count(rows_value),
        "diagnostic_rows_list_count": _safe_list_count(diagnostic_rows_value),
        "identity_horizon_evaluations_present": isinstance(
            identity_evaluations_value, list
        ),
        "identity_horizon_evaluation_count": _safe_list_count(
            identity_evaluations_value
        ),
        "empty_reason_summary_present": isinstance(empty_reason_value, dict),
        "empty_reason_summary": (
            _safe_dict(empty_reason_value)
            if isinstance(empty_reason_value, dict)
            else None
        ),
        "expected_missing_sub_blocks": expected_missing_sub_blocks,
    }


def _build_compatibility_section(
    *,
    joined_block: dict[str, Any],
    empty_reason_summary: dict[str, Any],
) -> dict[str, Any]:
    identity_evaluations = _safe_dict_list(
        joined_block.get("identity_horizon_evaluations")
    )

    raw_preview_visible_count = 0
    compatibility_visible_count = 0
    raw_but_filtered_invisible_count = 0
    identities_with_no_compatible_horizons_count = 0

    for entry in identity_evaluations:
        raw_visibility = _safe_dict(entry.get("raw_preview_visibility"))
        compatibility_visibility = _safe_dict(
            entry.get("compatibility_filtered_preview_visibility")
        )

        raw_union_horizons = _extract_visibility_horizons(
            raw_visibility,
            primary_key="raw_category_union_horizons",
            fallback_key="raw_category_overlap_horizons",
        )
        compatibility_union_horizons = _extract_visibility_horizons(
            compatibility_visibility,
            primary_key="compatibility_filtered_category_union_horizons",
            fallback_key="compatibility_filtered_category_overlap_horizons",
        )

        strategy_compatible_horizons = _normalize_horizon_list(
            entry.get("strategy_compatible_horizons")
        )
        if strategy_compatible_horizons is None:
            strategy_compatible_horizons = _normalize_horizon_list(
                compatibility_visibility.get("strategy_compatible_horizons")
            )

        if raw_union_horizons:
            raw_preview_visible_count += 1
        if compatibility_union_horizons:
            compatibility_visible_count += 1
        if raw_union_horizons and not compatibility_union_horizons:
            raw_but_filtered_invisible_count += 1
        if strategy_compatible_horizons is not None and not strategy_compatible_horizons:
            identities_with_no_compatible_horizons_count += 1

    identities_blocked_only_by_incompatibility = _normalize_string_list(
        empty_reason_summary.get("identities_blocked_only_by_incompatibility")
    )
    strategies_without_analyzer_compatible_horizons = _normalize_string_list(
        empty_reason_summary.get("strategies_without_analyzer_compatible_horizons")
    )

    conservative_signals: list[str] = []
    if raw_but_filtered_invisible_count > 0:
        conservative_signals.append(
            "raw_preview_visible_but_compatibility_filtered_invisible"
        )

    all_identities_incompatible = (
        len(identity_evaluations) > 0
        and identities_with_no_compatible_horizons_count == len(identity_evaluations)
    )
    if all_identities_incompatible:
        conservative_signals.append("all_horizons_incompatible")
    elif compatibility_visible_count > 0:
        conservative_signals.append("compatibility_filtered_visibility_present")

    compatibility_diagnostics_present = bool(identity_evaluations) or bool(
        identities_blocked_only_by_incompatibility
        or strategies_without_analyzer_compatible_horizons
    )

    return {
        "compatibility_diagnostics_present": compatibility_diagnostics_present,
        "identity_horizon_evaluation_count": len(identity_evaluations),
        "identities_with_raw_preview_visibility_count": raw_preview_visible_count,
        "identities_with_compatibility_filtered_visibility_count": (
            compatibility_visible_count
        ),
        "raw_preview_visible_but_compatibility_filtered_invisible_count": (
            raw_but_filtered_invisible_count
        ),
        "identities_with_no_analyzer_compatible_horizons_count": (
            identities_with_no_compatible_horizons_count
        ),
        "all_identities_incompatible": all_identities_incompatible,
        "identities_blocked_only_by_incompatibility_count": (
            len(identities_blocked_only_by_incompatibility)
        ),
        "strategies_without_analyzer_compatible_horizons_count": (
            len(strategies_without_analyzer_compatible_horizons)
        ),
        "conservative_signals": conservative_signals,
    }


def _build_final_verdict(
    *,
    loaded: dict[str, Any],
    preview: dict[str, Any],
    joined: dict[str, Any],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    expected_missing_blocks: list[str] = []
    if not preview.get("preview_block_exists"):
        expected_missing_blocks.append("edge_candidates_preview")
    if not joined.get("joined_row_block_exists"):
        expected_missing_blocks.append("edge_candidate_rows")

    expected_missing_sub_blocks = _merge_string_lists(
        preview.get("expected_missing_sub_blocks"),
        joined.get("expected_missing_sub_blocks"),
    )

    joined_effective_row_count = joined.get("row_count")
    if joined_effective_row_count is None:
        joined_effective_row_count = joined.get("rows_list_count")

    joined_effective_diagnostic_count = joined.get("diagnostic_row_count")
    if joined_effective_diagnostic_count is None:
        joined_effective_diagnostic_count = joined.get("diagnostic_rows_list_count")

    empty_reason_summary = _safe_dict(joined.get("empty_reason_summary"))
    empty_state_category = _text(empty_reason_summary.get("empty_state_category"))
    dominant_rejection_reason = _text(
        empty_reason_summary.get("dominant_rejection_reason")
    )
    has_eligible_rows = _safe_bool(empty_reason_summary.get("has_eligible_rows"))

    artifact_sufficient = bool(
        loaded["payload_is_object"]
        and preview.get("preview_block_exists")
        and joined.get("joined_row_block_exists")
        and (
            joined.get("row_count") is not None
            or joined.get("diagnostic_row_count") is not None
            or joined.get("dropped_row_count") is not None
            or joined.get("empty_reason_summary_present")
            or joined.get("identity_horizon_evaluations_present")
        )
    )

    facts: list[str] = []
    inference_notes: list[str] = []
    uncertainty_notes = [
        "This report stays analyzer-only and does not assess downstream mapper or engine bottlenecks.",
        "It does not inspect trade-analysis JSONL or comparison summaries.",
    ]

    if not loaded["exists"]:
        facts.append("Analyzer summary artifact is missing.")
        uncertainty_notes.append(
            "No analyzer snapshot was available at the requested summary path."
        )
        return {
            "scope": "analyzer_artifact_only",
            "verdict_category": "artifact_missing",
            "artifact_sufficient_for_current_snapshot_diagnosis": False,
            "expected_missing_blocks": expected_missing_blocks,
            "expected_missing_sub_blocks": expected_missing_sub_blocks,
            "facts": facts,
            "inference_notes": inference_notes,
            "uncertainty_notes": uncertainty_notes,
        }

    if not loaded["is_file"]:
        facts.append("Summary path exists but is not a file.")
        uncertainty_notes.append("Analyzer artifact could not be read as summary.json.")
        return {
            "scope": "analyzer_artifact_only",
            "verdict_category": "artifact_unreadable_or_invalid",
            "artifact_sufficient_for_current_snapshot_diagnosis": False,
            "expected_missing_blocks": expected_missing_blocks,
            "expected_missing_sub_blocks": expected_missing_sub_blocks,
            "facts": facts,
            "inference_notes": inference_notes,
            "uncertainty_notes": uncertainty_notes,
        }

    if not loaded["json_loaded"]:
        facts.append(f"Summary artifact could not be parsed: {loaded['load_error']}.")
        uncertainty_notes.append(
            "Analyzer artifact exists, but JSON loading failed before block-level diagnosis."
        )
        return {
            "scope": "analyzer_artifact_only",
            "verdict_category": "artifact_unreadable_or_invalid",
            "artifact_sufficient_for_current_snapshot_diagnosis": False,
            "expected_missing_blocks": expected_missing_blocks,
            "expected_missing_sub_blocks": expected_missing_sub_blocks,
            "facts": facts,
            "inference_notes": inference_notes,
            "uncertainty_notes": uncertainty_notes,
        }

    if not loaded["payload_is_object"]:
        facts.append("Summary JSON loaded, but the top-level payload is not an object.")
        uncertainty_notes.append(
            "Analyzer artifact shape is invalid for block-level diagnosis."
        )
        return {
            "scope": "analyzer_artifact_only",
            "verdict_category": "artifact_unreadable_or_invalid",
            "artifact_sufficient_for_current_snapshot_diagnosis": False,
            "expected_missing_blocks": expected_missing_blocks,
            "expected_missing_sub_blocks": expected_missing_sub_blocks,
            "facts": facts,
            "inference_notes": inference_notes,
            "uncertainty_notes": uncertainty_notes,
        }

    facts.append(
        f"Preview block present={_format_bool(preview.get('preview_block_exists'))}; "
        f"joined-row block present={_format_bool(joined.get('joined_row_block_exists'))}."
    )
    facts.append(
        "Joined-row counts: "
        f"row_count={_format_optional_int(joined.get('row_count'))}, "
        f"diagnostic_row_count={_format_optional_int(joined.get('diagnostic_row_count'))}, "
        f"dropped_row_count={_format_optional_int(joined.get('dropped_row_count'))}."
    )
    if empty_state_category is not None:
        facts.append(f"empty_state_category={empty_state_category}.")
    if dominant_rejection_reason is not None:
        facts.append(f"dominant_rejection_reason={dominant_rejection_reason}.")
    if compatibility.get("conservative_signals"):
        facts.append(
            "Compatibility signals="
            + ", ".join(
                _normalize_string_list(compatibility.get("conservative_signals"))
            )
            + "."
        )

    if expected_missing_blocks:
        if preview.get("preview_block_exists") and not joined.get(
            "joined_row_block_exists"
        ):
            verdict_category = "preview_present_but_no_joined_row_block"
        else:
            verdict_category = "artifact_present_but_expected_blocks_missing"
        uncertainty_notes.append(
            "Missing expected analyzer blocks limit current-snapshot diagnosis coverage."
        )
        return {
            "scope": "analyzer_artifact_only",
            "verdict_category": verdict_category,
            "artifact_sufficient_for_current_snapshot_diagnosis": artifact_sufficient,
            "expected_missing_blocks": expected_missing_blocks,
            "expected_missing_sub_blocks": expected_missing_sub_blocks,
            "facts": facts,
            "inference_notes": inference_notes,
            "uncertainty_notes": uncertainty_notes,
        }

    if expected_missing_sub_blocks:
        uncertainty_notes.append(
            "Some expected analyzer sub-blocks were absent, so compatibility or rejection detail may be partially reduced."
        )

    if isinstance(joined_effective_row_count, int) and joined_effective_row_count > 0:
        verdict_category = "joined_rows_present_with_compatible_candidates"
        inference_notes.append(
            "The analyzer artifact currently shows at least one eligible joined candidate row."
        )
        uncertainty_notes.append(
            "Joined-row presence here does not imply anything about downstream selection outcomes."
        )
    elif (
        joined.get("joined_row_block_exists")
        and (
            has_eligible_rows is False
            or (
                isinstance(joined_effective_row_count, int)
                and joined_effective_row_count == 0
            )
            or (
                isinstance(joined_effective_diagnostic_count, int)
                and joined_effective_diagnostic_count > 0
            )
            or joined.get("empty_reason_summary_present")
        )
    ):
        verdict_category = "joined_row_block_present_but_no_eligible_rows"
        if compatibility.get(
            "raw_preview_visible_but_compatibility_filtered_invisible_count",
            0,
        ) > 0:
            inference_notes.append(
                "Current snapshot evidence is consistent with raw preview visibility being narrowed away by analyzer-side compatibility filtering."
            )
        elif empty_state_category is not None:
            inference_notes.append(
                f"Current snapshot evidence is consistent with analyzer-side joined-row rejection in category '{empty_state_category}'."
            )
    else:
        verdict_category = "inconclusive"
        uncertainty_notes.append(
            "The artifact is present, but persisted fields were not decisive enough for a narrow analyzer verdict."
        )

    return {
        "scope": "analyzer_artifact_only",
        "verdict_category": verdict_category,
        "artifact_sufficient_for_current_snapshot_diagnosis": artifact_sufficient,
        "expected_missing_blocks": expected_missing_blocks,
        "expected_missing_sub_blocks": expected_missing_sub_blocks,
        "facts": facts,
        "inference_notes": inference_notes,
        "uncertainty_notes": uncertainty_notes,
    }


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_list_count(value: Any) -> int | None:
    if not isinstance(value, list):
        return None
    return len(value)


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        return int(numeric) if numeric.is_integer() else None
    return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_horizon_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return _sort_horizon_strings(_normalize_string_list(value))


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _text(item)
        if text is not None and text not in result:
            result.append(text)
    return result


def _sort_horizon_strings(values: Iterable[str]) -> list[str]:
    items = _normalize_string_list(list(values))
    indexed_items = list(enumerate(items))

    def sort_key(item: tuple[int, str]) -> tuple[int, int, str]:
        original_index, horizon = item
        known_index = _KNOWN_HORIZON_INDEX.get(horizon)
        if known_index is not None:
            return (0, known_index, horizon)
        return (1, original_index, horizon)

    return [horizon for _, horizon in sorted(indexed_items, key=sort_key)]


def _merge_string_lists(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        for item in _normalize_string_list(value):
            if item not in merged:
                merged.append(item)
    return merged


def _extract_visibility_horizons(
    visibility: dict[str, Any],
    *,
    primary_key: str,
    fallback_key: str,
) -> list[str] | None:
    primary = _normalize_horizon_list(visibility.get(primary_key))
    if primary is not None:
        return primary
    return _normalize_horizon_list(visibility.get(fallback_key))


def _format_bool(value: Any) -> str:
    return "yes" if value else "no"


def _format_optional_int(value: Any) -> str:
    parsed = _safe_int(value)
    return str(parsed) if parsed is not None else "n/a"


def _format_string_list(value: Any) -> str:
    items = _normalize_string_list(value)
    return ", ".join(items) if items else "none"


if __name__ == "__main__":
    main()
