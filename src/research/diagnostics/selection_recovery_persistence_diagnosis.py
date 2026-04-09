from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "research_reports" / "latest"
DEFAULT_PRIMARY_INPUT = REPO_ROOT / "logs" / "trade_analysis.jsonl"
DEFAULT_FALLBACK_INPUT = REPO_ROOT / "logs" / "trade_analysis_cumulative.jsonl"

RELAXED_REASON_CODE = "CANDIDATE_STABILITY_SINGLE_HORIZON_RELAXED"

SELECTED_STATUSES = {"selected", "winner", "chosen", "accepted"}
NON_SELECTED_STATUSES = {
    "abstain",
    "blocked",
    "hold",
    "rejected",
    "none",
    "no_selection",
    "no_candidate",
    "filtered",
}

TIMESTAMP_KEYS = (
    "generated_at",
    "timestamp",
    "logged_at",
    "created_at",
    "run_at",
    "completed_at",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose selection recovery persistence, concentration, and relaxed "
            "single-horizon dependence from trade analysis logs."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional explicit JSONL input path.",
    )
    parser.add_argument(
        "--recent-rows",
        type=int,
        default=None,
        help="Keep only the most recent N valid edge-selection rows.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown copies into the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for optional JSON/Markdown output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_selection_recovery_persistence_diagnosis(
        input_path=args.input_path,
        recent_rows=args.recent_rows,
        write_latest_copy=args.write_latest_copy,
        output_dir=resolve_output_dir(args.output_dir),
    )
    print(render_markdown_report(report), end="")


def run_selection_recovery_persistence_diagnosis(
    *,
    input_path: Path | None = None,
    recent_rows: int | None = None,
    write_latest_copy: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_input = resolve_input_path(input_path)
    loaded = read_trade_analysis_rows(resolved_input)
    analyzed_rows = apply_recent_limit(loaded["edge_rows"], recent_rows)
    report = build_report(
        input_path=resolved_input,
        total_rows_read=loaded["total_rows_read"],
        source_malformed_row_count=loaded["source_malformed_row_count"],
        edge_rows_available=len(loaded["edge_rows"]),
        analyzed_rows=analyzed_rows,
        recent_rows=recent_rows,
    )

    if write_latest_copy:
        write_report_files(report, output_dir)

    return report


def resolve_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def resolve_input_path(explicit_input_path: Path | None) -> Path:
    if explicit_input_path is not None:
        resolved = explicit_input_path.expanduser()
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input path does not exist: {resolved}")
        return resolved

    if DEFAULT_PRIMARY_INPUT.exists():
        return DEFAULT_PRIMARY_INPUT
    if DEFAULT_FALLBACK_INPUT.exists():
        return DEFAULT_FALLBACK_INPUT

    raise FileNotFoundError(
        "No default trade analysis input found. Expected one of: "
        f"{DEFAULT_PRIMARY_INPUT} or {DEFAULT_FALLBACK_INPUT}"
    )


def read_trade_analysis_rows(input_path: Path) -> dict[str, Any]:
    total_rows_read = 0
    source_malformed_row_count = 0
    edge_rows: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_rows_read += 1
            line = raw_line.strip()
            if not line:
                source_malformed_row_count += 1
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                source_malformed_row_count += 1
                continue

            if not isinstance(payload, dict):
                source_malformed_row_count += 1
                continue

            selection_output = payload.get("edge_selection_output")
            if not isinstance(selection_output, dict):
                source_malformed_row_count += 1
                continue

            edge_rows.append(
                {
                    "line_number": line_number,
                    "selection_output": selection_output,
                    "timestamp": extract_timestamp(payload, selection_output),
                }
            )

    edge_rows.sort(
        key=lambda row: (
            row.get("timestamp") is None,
            row.get("timestamp") or "",
            int(row.get("line_number", 0)),
        )
    )
    return {
        "total_rows_read": total_rows_read,
        "source_malformed_row_count": source_malformed_row_count,
        "edge_rows": edge_rows,
    }


def apply_recent_limit(rows: list[dict[str, Any]], recent_rows: int | None) -> list[dict[str, Any]]:
    if recent_rows is None:
        return list(rows)
    if recent_rows <= 0:
        return []
    return list(rows[-recent_rows:])


def build_report(
    *,
    input_path: Path,
    total_rows_read: int,
    source_malformed_row_count: int,
    edge_rows_available: int,
    analyzed_rows: list[dict[str, Any]],
    recent_rows: int | None,
) -> dict[str, Any]:
    selected_entries: list[dict[str, Any]] = []
    malformed_row_count = 0
    non_selected_rows_count = 0
    selection_status_counts: Counter[str] = Counter()

    for row in analyzed_rows:
        analyzed = analyze_selection_row(row)
        classification = analyzed["classification"]
        if classification == "malformed":
            malformed_row_count += 1
            continue

        selection_status_counts[analyzed["selection_status"]] += 1
        if classification == "selected":
            selected_entries.append(analyzed)
        else:
            non_selected_rows_count += 1

    usable_rows_count = len(selected_entries) + non_selected_rows_count
    selected_rows_count = len(selected_entries)
    selected_ratio = safe_ratio(selected_rows_count, usable_rows_count)
    abstain_ratio = safe_ratio(non_selected_rows_count, usable_rows_count)

    selected_identity_counts = Counter(entry["identity"] for entry in selected_entries)
    selected_symbol_counts = Counter(entry["symbol"] for entry in selected_entries if entry["symbol"])
    selected_strategy_counts = Counter(
        entry["strategy"] for entry in selected_entries if entry["strategy"]
    )
    selected_horizon_counts = Counter(entry["horizon"] for entry in selected_entries if entry["horizon"])

    top_candidate_identity, top_candidate_count = most_common_identity(selected_identity_counts)
    unique_selected_candidate_count = len(selected_identity_counts)
    top_candidate_share = safe_ratio(top_candidate_count, selected_rows_count)

    chronological_streaks = build_streak_details(selected_entries, sort_for_output=False)
    streak_details = build_streak_details(selected_entries, sort_for_output=True)
    longest_repeat_streak = max((item["length"] for item in chronological_streaks), default=0)
    current_repeat_streak = chronological_streaks[-1]["length"] if chronological_streaks else 0
    candidate_transition_count = sum(
        1
        for left, right in zip(selected_entries, selected_entries[1:])
        if left["identity"] != right["identity"]
    )

    selected_scores = [
        float(entry["selection_score"])
        for entry in selected_entries
        if entry["selection_score"] is not None
    ]
    selected_confidences = [
        float(entry["selection_confidence"])
        for entry in selected_entries
        if entry["selection_confidence"] is not None
    ]

    stability_counts = Counter(entry["selected_stability_label"] for entry in selected_entries)
    relaxed_single_horizon_selected_count = sum(
        1 for entry in selected_entries if entry["relaxed_single_horizon"] is True
    )
    advisory_relaxed_reason_count = sum(
        1 for entry in selected_entries if RELAXED_REASON_CODE in entry["advisory_reason_codes"]
    )
    visible_len_1_count = sum(
        1 for entry in selected_entries if entry["visible_horizons_count"] == 1
    )
    visible_len_ge_2_count = sum(
        1 for entry in selected_entries if entry["visible_horizons_count"] >= 2
    )

    relaxed_single_horizon_selected_share = safe_ratio(
        relaxed_single_horizon_selected_count,
        selected_rows_count,
    )

    interpretation_notes = build_interpretation_notes(
        selected_rows_count=selected_rows_count,
        top_candidate_identity=top_candidate_identity,
        top_candidate_share=top_candidate_share,
        unique_selected_candidate_count=unique_selected_candidate_count,
        current_repeat_streak=current_repeat_streak,
        candidate_transition_count=candidate_transition_count,
        relaxed_single_horizon_selected_share=relaxed_single_horizon_selected_share,
        multi_horizon_confirmed_selected_count=stability_counts.get(
            "multi_horizon_confirmed",
            0,
        ),
        single_horizon_only_selected_count=stability_counts.get("single_horizon_only", 0),
    )

    verdict_label = determine_verdict(
        selected_rows_count=selected_rows_count,
        top_candidate_share=top_candidate_share,
        relaxed_single_horizon_selected_share=relaxed_single_horizon_selected_share,
        candidate_transition_count=candidate_transition_count,
        longest_repeat_streak=longest_repeat_streak,
        unique_selected_candidate_count=unique_selected_candidate_count,
        multi_horizon_confirmed_selected_count=stability_counts.get(
            "multi_horizon_confirmed",
            0,
        ),
        single_horizon_only_selected_count=stability_counts.get("single_horizon_only", 0),
    )

    return {
        "input_path_used": str(input_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "recent_rows_limit": recent_rows,
        "total_rows_read": total_rows_read,
        "source_malformed_row_count": source_malformed_row_count,
        "rows_with_edge_selection_output_count": edge_rows_available,
        "malformed_row_count": malformed_row_count,
        "usable_rows_count": usable_rows_count,
        "selected_rows_count": selected_rows_count,
        "non_selected_rows_count": non_selected_rows_count,
        "selected_ratio": round(selected_ratio, 6),
        "abstain_ratio": round(abstain_ratio, 6),
        "unique_selected_candidate_count": unique_selected_candidate_count,
        "top_candidate_identity": top_candidate_identity,
        "top_candidate_count": top_candidate_count,
        "top_candidate_share": round(top_candidate_share, 6),
        "longest_repeat_streak": longest_repeat_streak,
        "current_repeat_streak": current_repeat_streak,
        "candidate_transition_count": candidate_transition_count,
        "selected_identity_counts": sort_counter_dict(selected_identity_counts),
        "selected_symbol_counts": sort_counter_dict(selected_symbol_counts),
        "selected_strategy_counts": sort_counter_dict(selected_strategy_counts),
        "selected_horizon_counts": sort_counter_dict(selected_horizon_counts),
        "selected_score_count": len(selected_scores),
        "selected_score_mean": round(mean(selected_scores), 6) if selected_scores else None,
        "selected_score_median": round(median(selected_scores), 6) if selected_scores else None,
        "selected_score_min": round(min(selected_scores), 6) if selected_scores else None,
        "selected_score_max": round(max(selected_scores), 6) if selected_scores else None,
        "recent_selected_scores_tail": [round(value, 6) for value in selected_scores[-10:]],
        "selected_confidence_count": len(selected_confidences),
        "selected_confidence_mean": round(mean(selected_confidences), 6)
        if selected_confidences
        else None,
        "selected_confidence_median": round(median(selected_confidences), 6)
        if selected_confidences
        else None,
        "selected_confidence_min": round(min(selected_confidences), 6)
        if selected_confidences
        else None,
        "selected_confidence_max": round(max(selected_confidences), 6)
        if selected_confidences
        else None,
        "recent_selected_confidences_tail": [
            round(value, 6) for value in selected_confidences[-10:]
        ],
        "single_horizon_only_selected_count": stability_counts.get("single_horizon_only", 0),
        "multi_horizon_confirmed_selected_count": stability_counts.get(
            "multi_horizon_confirmed",
            0,
        ),
        "other_stability_label_count": sum(
            count
            for label, count in stability_counts.items()
            if label not in {"single_horizon_only", "multi_horizon_confirmed"}
        ),
        "relaxed_single_horizon_selected_count": relaxed_single_horizon_selected_count,
        "relaxed_single_horizon_selected_share": round(
            relaxed_single_horizon_selected_share,
            6,
        ),
        "advisory_relaxed_reason_count": advisory_relaxed_reason_count,
        "selected_with_visible_horizons_len_1_count": visible_len_1_count,
        "selected_with_visible_horizons_len_ge_2_count": visible_len_ge_2_count,
        "selection_status_counts": sort_counter_dict(selection_status_counts),
        "streak_details": streak_details[:10],
        "recent_selected_identity_tail": [entry["identity"] for entry in selected_entries[-10:]],
        "interpretation_notes": interpretation_notes,
        "verdict": verdict_label,
    }


def analyze_selection_row(row: dict[str, Any]) -> dict[str, Any]:
    selection_output = as_dict(row.get("selection_output"))
    if not selection_output:
        return {"classification": "malformed", "selection_status": "malformed"}

    selection_status = normalize_text(selection_output.get("selection_status"), default="unknown")
    top_entry = first_ranking_entry(selection_output)

    symbol = first_non_empty_text(
        selection_output.get("selected_symbol"),
        top_entry.get("symbol"),
    )
    strategy = first_non_empty_text(
        selection_output.get("selected_strategy"),
        top_entry.get("strategy"),
    )
    horizon = first_non_empty_text(
        selection_output.get("selected_horizon"),
        top_entry.get("horizon"),
    )
    identity = build_identity(symbol, strategy, horizon)

    is_selected = classify_selected(selection_status, identity)
    if selection_status == "unknown" and identity is None and not top_entry:
        return {"classification": "malformed", "selection_status": selection_status}

    if is_selected is None:
        return {"classification": "malformed", "selection_status": selection_status}

    selected_stability_label = first_non_empty_text(
        top_entry.get("selected_stability_label"),
        selection_output.get("selected_stability_label"),
    )
    visible_horizons = extract_visible_horizons(top_entry, selection_output)
    advisory_reason_codes = extract_advisory_reason_codes(top_entry, selection_output)
    relaxed_single_horizon = extract_relaxed_single_horizon(top_entry, selection_output)

    selection_score = coerce_float(
        first_non_none(
            top_entry.get("selection_score"),
            selection_output.get("selection_score"),
        )
    )
    selection_confidence = coerce_float(
        first_non_none(
            top_entry.get("selection_confidence"),
            selection_output.get("selection_confidence"),
        )
    )

    return {
        "classification": "selected" if is_selected else "non_selected",
        "selection_status": selection_status,
        "identity": identity,
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "selection_score": selection_score,
        "selection_confidence": selection_confidence,
        "selected_stability_label": selected_stability_label or "unknown",
        "visible_horizons": visible_horizons,
        "visible_horizons_count": len(visible_horizons),
        "advisory_reason_codes": advisory_reason_codes,
        "relaxed_single_horizon": relaxed_single_horizon,
        "line_number": int(row.get("line_number", 0)),
        "timestamp": row.get("timestamp"),
    }


def classify_selected(selection_status: str, identity: str | None) -> bool | None:
    if selection_status in SELECTED_STATUSES:
        return identity is not None
    if selection_status in NON_SELECTED_STATUSES:
        return False
    if identity is not None:
        return True
    return False if selection_status != "unknown" else None


def first_ranking_entry(selection_output: dict[str, Any]) -> dict[str, Any]:
    ranking = selection_output.get("ranking")
    if not isinstance(ranking, list):
        return {}
    for item in ranking:
        if isinstance(item, dict):
            return item
    return {}


def extract_visible_horizons(*objects: dict[str, Any]) -> list[str]:
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        candidate = obj.get("selected_visible_horizons")
        normalized = normalize_horizon_list(candidate)
        if normalized:
            return normalized
    return []


def extract_advisory_reason_codes(*objects: dict[str, Any]) -> list[str]:
    codes: list[str] = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        codes.extend(normalize_reason_codes(obj.get("advisory_reason_codes")))

        gate_diagnostics = as_dict(obj.get("gate_diagnostics"))
        if gate_diagnostics:
            advisory = as_dict(gate_diagnostics.get("advisory"))
            if advisory:
                codes.extend(normalize_reason_codes(advisory.get("reason_codes")))

    return dedupe_preserve_order(codes)


def extract_relaxed_single_horizon(*objects: dict[str, Any]) -> bool | None:
    for obj in objects:
        if not isinstance(obj, dict):
            continue

        direct = coerce_bool(obj.get("relaxed_single_horizon"))
        if direct is not None:
            return direct

        gate_diagnostics = as_dict(obj.get("gate_diagnostics"))
        if gate_diagnostics:
            nested = find_bool_for_key(gate_diagnostics, "relaxed_single_horizon")
            if nested is not None:
                return nested

    for obj in objects:
        nested = find_bool_for_key(obj, "relaxed_single_horizon")
        if nested is not None:
            return nested
    return None


def build_streak_details(
    selected_entries: list[dict[str, Any]],
    *,
    sort_for_output: bool,
) -> list[dict[str, Any]]:
    if not selected_entries:
        return []

    streaks: list[dict[str, Any]] = []
    start_index = 0

    for index in range(1, len(selected_entries) + 1):
        is_boundary = index == len(selected_entries)
        if (
            not is_boundary
            and selected_entries[index]["identity"] == selected_entries[start_index]["identity"]
        ):
            continue

        streak_slice = selected_entries[start_index:index]
        streaks.append(
            {
                "identity": streak_slice[0]["identity"],
                "length": len(streak_slice),
                "start_selected_index": start_index,
                "end_selected_index": index - 1,
                "start_line_number": streak_slice[0]["line_number"],
                "end_line_number": streak_slice[-1]["line_number"],
                "start_timestamp": streak_slice[0]["timestamp"],
                "end_timestamp": streak_slice[-1]["timestamp"],
            }
        )
        start_index = index

    if sort_for_output:
        streaks.sort(
            key=lambda item: (
                -int(item["length"]),
                -int(item["end_selected_index"]),
                str(item["identity"]),
            )
        )
    return streaks


def build_interpretation_notes(
    *,
    selected_rows_count: int,
    top_candidate_identity: str | None,
    top_candidate_share: float,
    unique_selected_candidate_count: int,
    current_repeat_streak: int,
    candidate_transition_count: int,
    relaxed_single_horizon_selected_share: float,
    multi_horizon_confirmed_selected_count: int,
    single_horizon_only_selected_count: int,
) -> list[str]:
    notes: list[str] = []

    if selected_rows_count == 0:
        return ["No selected rows were found in the analyzed window."]

    if top_candidate_identity:
        notes.append(
            "Top selected identity is "
            f"{top_candidate_identity} with share={round(top_candidate_share, 4)}."
        )

    notes.append(
        "Selected identities changed "
        f"{candidate_transition_count} times across {selected_rows_count} selected rows."
    )
    notes.append(
        "Current repeat streak is "
        f"{current_repeat_streak}, with {unique_selected_candidate_count} unique selected identities."
    )
    notes.append(
        "Relaxed single-horizon share among selected rows is "
        f"{round(relaxed_single_horizon_selected_share, 4)}."
    )
    notes.append(
        "Multi-horizon confirmed selections="
        f"{multi_horizon_confirmed_selected_count}, "
        "single-horizon-only selections="
        f"{single_horizon_only_selected_count}."
    )
    return notes


def determine_verdict(
    *,
    selected_rows_count: int,
    top_candidate_share: float,
    relaxed_single_horizon_selected_share: float,
    candidate_transition_count: int,
    longest_repeat_streak: int,
    unique_selected_candidate_count: int,
    multi_horizon_confirmed_selected_count: int,
    single_horizon_only_selected_count: int,
) -> str:
    if selected_rows_count < 5:
        return "insufficient_recent_data"

    if (
        top_candidate_share >= 0.7
        and relaxed_single_horizon_selected_share >= 0.5
        and single_horizon_only_selected_count >= multi_horizon_confirmed_selected_count
    ):
        return "relaxation_dependent_concentration"

    if top_candidate_share >= 0.75 and longest_repeat_streak >= max(4, selected_rows_count // 3):
        return "dominance_risk"

    if top_candidate_share >= 0.55:
        return "concentrated_but_monitor"

    if (
        unique_selected_candidate_count >= max(4, selected_rows_count // 2)
        and candidate_transition_count >= max(3, selected_rows_count - 2)
        and longest_repeat_streak <= 2
    ):
        return "unstable_recovery"

    if (
        relaxed_single_horizon_selected_share <= 0.35
        and multi_horizon_confirmed_selected_count >= single_horizon_only_selected_count
        and top_candidate_share <= 0.5
    ):
        return "healthy_recovery_persistence"

    return "mixed_recovery_state"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Selection Recovery Persistence Diagnosis",
        "",
        "## Run",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- input_path: {report.get('input_path_used')}",
        f"- total_rows_read: {report.get('total_rows_read')}",
        f"- rows_with_edge_selection_output_count: {report.get('rows_with_edge_selection_output_count')}",
        f"- usable_rows_count: {report.get('usable_rows_count')}",
        f"- malformed_rows: {report.get('malformed_row_count')}",
        f"- source_malformed_rows: {report.get('source_malformed_row_count')}",
        f"- recent_rows_limit: {report.get('recent_rows_limit')}",
        "",
        "## Selection Volume",
        "",
        f"- selected_rows_count: {report.get('selected_rows_count')}",
        f"- non_selected_rows_count: {report.get('non_selected_rows_count')}",
        f"- selected_ratio: {report.get('selected_ratio')}",
        f"- abstain_ratio: {report.get('abstain_ratio')}",
        "",
        "## Concentration",
        "",
        f"- unique_selected_candidate_count: {report.get('unique_selected_candidate_count')}",
        f"- top_candidate_identity: {report.get('top_candidate_identity')}",
        f"- top_candidate_count: {report.get('top_candidate_count')}",
        f"- top_candidate_share: {report.get('top_candidate_share')}",
        f"- longest_repeat_streak: {report.get('longest_repeat_streak')}",
        f"- current_repeat_streak: {report.get('current_repeat_streak')}",
        f"- candidate_transition_count: {report.get('candidate_transition_count')}",
        "",
        "## Score Trend",
        "",
        f"- selected_score_count: {report.get('selected_score_count')}",
        f"- selected_score_mean: {report.get('selected_score_mean')}",
        f"- selected_score_median: {report.get('selected_score_median')}",
        f"- selected_score_min: {report.get('selected_score_min')}",
        f"- selected_score_max: {report.get('selected_score_max')}",
        f"- recent_selected_scores_tail: {report.get('recent_selected_scores_tail')}",
        "",
        "## Confidence Trend",
        "",
        f"- selected_confidence_count: {report.get('selected_confidence_count')}",
        f"- selected_confidence_mean: {report.get('selected_confidence_mean')}",
        f"- selected_confidence_median: {report.get('selected_confidence_median')}",
        f"- selected_confidence_min: {report.get('selected_confidence_min')}",
        f"- selected_confidence_max: {report.get('selected_confidence_max')}",
        f"- recent_selected_confidences_tail: {report.get('recent_selected_confidences_tail')}",
        "",
        "## Stability / Relaxation Dependence",
        "",
        f"- single_horizon_only_selected_count: {report.get('single_horizon_only_selected_count')}",
        f"- multi_horizon_confirmed_selected_count: {report.get('multi_horizon_confirmed_selected_count')}",
        f"- other_stability_label_count: {report.get('other_stability_label_count')}",
        f"- relaxed_single_horizon_selected_count: {report.get('relaxed_single_horizon_selected_count')}",
        f"- relaxed_single_horizon_selected_share: {report.get('relaxed_single_horizon_selected_share')}",
        f"- advisory_relaxed_reason_count: {report.get('advisory_relaxed_reason_count')}",
        f"- visible_horizons_len_1_count: {report.get('selected_with_visible_horizons_len_1_count')}",
        f"- visible_horizons_len_ge_2_count: {report.get('selected_with_visible_horizons_len_ge_2_count')}",
        "",
        "## Identity Breakdown",
        "",
    ]

    lines.extend(render_mapping_block("selected_identity_counts", report.get("selected_identity_counts")))
    lines.extend(render_mapping_block("selected_symbol_counts", report.get("selected_symbol_counts")))
    lines.extend(render_mapping_block("selected_strategy_counts", report.get("selected_strategy_counts")))
    lines.extend(render_mapping_block("selected_horizon_counts", report.get("selected_horizon_counts")))

    lines.extend(
        [
            "## Recent Selected Tail",
            "",
            f"- recent_selected_identity_tail: {report.get('recent_selected_identity_tail')}",
            "",
            "## Interpretation Notes",
            "",
        ]
    )

    interpretation_notes = report.get("interpretation_notes") or []
    if interpretation_notes:
        for note in interpretation_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none")

    lines.extend(["", "## Verdict", "", f"- {report.get('verdict')}", ""])
    return "\n".join(lines)


def render_mapping_block(title: str, values: dict[str, int] | None) -> list[str]:
    lines = [f"- {title}:"]
    if not values:
        lines.append("  - none")
        lines.append("")
        return lines

    for key, count in values.items():
        lines.append(f"  - {key}: {count}")
    lines.append("")
    return lines


def write_report_files(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "selection_recovery_persistence_diagnosis.json"
    md_path = output_dir / "selection_recovery_persistence_diagnosis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")


def extract_timestamp(*objects: dict[str, Any]) -> str | None:
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for key in TIMESTAMP_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def find_bool_for_key(value: Any, key: str, depth: int = 0, max_depth: int = 6) -> bool | None:
    if depth > max_depth:
        return None

    if isinstance(value, dict):
        direct = coerce_bool(value.get(key))
        if direct is not None:
            return direct
        for nested in value.values():
            found = find_bool_for_key(nested, key, depth + 1, max_depth)
            if found is not None:
                return found

    elif isinstance(value, list):
        for item in value:
            found = find_bool_for_key(item, key, depth + 1, max_depth)
            if found is not None:
                return found

    return None


def build_identity(symbol: str | None, strategy: str | None, horizon: str | None) -> str | None:
    if not symbol or not strategy or not horizon:
        return None
    return f"{symbol}/{strategy}/{horizon}"


def most_common_identity(counter: Counter[str]) -> tuple[str | None, int]:
    if not counter:
        return None, 0
    identity, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return identity, count


def sort_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: count for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}


def normalize_horizon_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    horizons: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                horizons.append(text)
    return dedupe_preserve_order(horizons)


def normalize_reason_codes(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
    return dedupe_preserve_order(normalized)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_text(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return default


def first_non_empty_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()

