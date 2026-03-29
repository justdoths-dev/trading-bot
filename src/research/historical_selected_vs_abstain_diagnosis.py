from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_TARGET_WINDOWS = [3550, 3575, 3600, 3625, 3650]
DEFAULT_RUNS_ROOT = Path("logs/research_reports/historical_direct_edge_selection")
DEFAULT_LATEST_DIR = Path("logs/research_reports/latest")

LOAD_BEARING_FIELDS = [
    "candidate_seed_count",
    "horizons_with_seed",
    "horizons_without_seed",
    "selected_visible_horizons",
    "selected_stability_label",
    "candidate_status",
    "reason_codes",
    "gate_diagnostics",
    "aggregate_score",
    "sample_count",
    "positive_rate_pct",
    "robustness_signal_pct",
]


@dataclass
class StepSnapshot:
    step_index: Optional[int]
    end_record_index_inclusive: Optional[int]
    selection_status: Optional[str]
    selection_reason: Optional[str]
    selected_symbol: Optional[str]
    selected_strategy: Optional[str]
    selected_horizon: Optional[str]
    candidate_seed_count: Optional[int]
    horizons_with_seed: List[str]
    horizons_without_seed: List[str]
    selected_visible_horizons: List[str]
    selected_stability_label: Optional[str]
    candidate_status: Optional[str]
    reason_codes: List[str]
    gate_diagnostics: Dict[str, Any]
    aggregate_score: Optional[float]
    sample_count: Optional[int]
    positive_rate_pct: Optional[float]
    robustness_signal_pct: Optional[float]
    raw_output_path: Optional[str]
    source_summary: Dict[str, str]
    raw_step_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "end_record_index_inclusive": self.end_record_index_inclusive,
            "selection_status": self.selection_status,
            "selection_reason": self.selection_reason,
            "selected_symbol": self.selected_symbol,
            "selected_strategy": self.selected_strategy,
            "selected_horizon": self.selected_horizon,
            "candidate_seed_count": self.candidate_seed_count,
            "horizons_with_seed": self.horizons_with_seed,
            "horizons_without_seed": self.horizons_without_seed,
            "selected_visible_horizons": self.selected_visible_horizons,
            "selected_stability_label": self.selected_stability_label,
            "candidate_status": self.candidate_status,
            "reason_codes": self.reason_codes,
            "gate_diagnostics": self.gate_diagnostics,
            "aggregate_score": self.aggregate_score,
            "sample_count": self.sample_count,
            "positive_rate_pct": self.positive_rate_pct,
            "robustness_signal_pct": self.robustness_signal_pct,
            "raw_output_path": self.raw_output_path,
            "source_summary": self.source_summary,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected vs abstain windows from historical direct edge selection "
            "and extract only load-bearing diagnostic fields."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific historical_direct_edge_selection run directory. If omitted, latest run is used.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Root directory containing historical direct edge selection runs.",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        type=int,
        default=DEFAULT_TARGET_WINDOWS,
        help="Target end_record_index_inclusive windows to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files. Defaults to the selected run directory.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Also write copies into logs/research_reports/latest/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = resolve_run_dir(args.run_dir, args.runs_root)
    step_results_path = run_dir / "step_results.jsonl"
    if not step_results_path.exists():
        raise FileNotFoundError(f"Missing step_results.jsonl: {step_results_path}")

    rows = load_jsonl(step_results_path)
    indexed_rows = index_rows_by_window(rows)

    targets = []
    missing_windows = []
    for window in args.windows:
        row = indexed_rows.get(window)
        if row is None:
            missing_windows.append(window)
            continue
        snapshot = build_step_snapshot(row)
        targets.append(snapshot)

    comparison_groups = build_comparison_groups(targets)
    explanations = build_explanations(comparison_groups)

    report = {
        "metadata": {
            "run_dir": str(run_dir),
            "step_results_path": str(step_results_path),
            "target_windows": args.windows,
            "resolved_windows": [s.end_record_index_inclusive for s in targets],
            "missing_windows": missing_windows,
        },
        "snapshots": [s.to_dict() for s in targets],
        "comparison_groups": comparison_groups,
        "explanations": explanations,
    }

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "selected_vs_abstain_diagnosis.json"
    md_path = output_dir / "selected_vs_abstain_diagnosis.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "resolved_windows": [s.end_record_index_inclusive for s in targets],
            "missing_windows": missing_windows,
        },
        indent=2,
        ensure_ascii=False,
    ))

    if args.write_latest_copy:
        DEFAULT_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        latest_json = DEFAULT_LATEST_DIR / "selected_vs_abstain_diagnosis.json"
        latest_md = DEFAULT_LATEST_DIR / "selected_vs_abstain_diagnosis.md"
        latest_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        latest_md.write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps(
            {
                "latest_json_report": str(latest_json),
                "latest_markdown_report": str(latest_md),
            },
            indent=2,
            ensure_ascii=False,
        ))


def resolve_run_dir(explicit_run_dir: Optional[Path], runs_root: Path) -> Path:
    if explicit_run_dir is not None:
        return explicit_run_dir

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {runs_root}")

    run_dirs.sort(key=lambda p: p.name, reverse=True)
    return run_dirs[0]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def index_rows_by_window(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        window = row.get("end_record_index_inclusive")
        if isinstance(window, int):
            indexed[window] = row
    return indexed


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def recursive_find_first(obj: Any, target_key: str) -> Any:
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for value in obj.values():
            result = recursive_find_first(value, target_key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = recursive_find_first(item, target_key)
            if result is not None:
                return result
    return None


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if value == "":
            continue
        return value
    return None


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_string_list(value: Any) -> List[str]:
    result: List[str] = []
    for item in ensure_list(value):
        if item is None:
            continue
        result.append(str(item))
    return result


def normalize_reason_codes(value: Any) -> List[str]:
    items = ensure_list(value)
    normalized: List[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            code = item.get("code") or item.get("reason_code") or item.get("reason")
            if code is not None:
                normalized.append(str(code))
        else:
            normalized.append(str(item))
    return normalized


def normalize_gate_diagnostics(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def extract_ranking_candidates(raw_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranking = coalesce(
        raw_payload.get("ranking"),
        raw_payload.get("candidate_rankings"),
        recursive_find_first(raw_payload, "ranking"),
        recursive_find_first(raw_payload, "candidate_rankings"),
    )
    if isinstance(ranking, list):
        return [item for item in ranking if isinstance(item, dict)]
    return []


def extract_selected_candidate(raw_payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    explicit_selected = coalesce(
        raw_payload.get("selected_candidate"),
        recursive_find_first(raw_payload, "selected_candidate"),
    )
    if isinstance(explicit_selected, dict):
        return explicit_selected

    ranking = extract_ranking_candidates(raw_payload)
    if not ranking:
        return {}

    row_symbol = row.get("selected_symbol")
    row_strategy = row.get("selected_strategy")
    row_horizon = row.get("selected_horizon")

    for candidate in ranking:
        if (
            candidate.get("symbol") == row_symbol
            and candidate.get("strategy") == row_strategy
            and candidate.get("horizon") == row_horizon
        ):
            return candidate

    return ranking[0]


def build_step_snapshot(row: Dict[str, Any]) -> StepSnapshot:
    raw_output_path_value = row.get("raw_output_path")
    raw_payload: Dict[str, Any] = {}
    if isinstance(raw_output_path_value, str) and raw_output_path_value:
        raw_payload = load_json_file(Path(raw_output_path_value))

    selected_candidate = extract_selected_candidate(raw_payload, row)

    candidate_seed_count = to_int(coalesce(
        row.get("candidate_seed_count"),
        recursive_find_first(raw_payload, "candidate_seed_count"),
    ))

    horizons_with_seed = normalize_string_list(coalesce(
        row.get("horizons_with_seed"),
        recursive_find_first(raw_payload, "horizons_with_seed"),
    ))

    horizons_without_seed = normalize_string_list(coalesce(
        row.get("horizons_without_seed"),
        recursive_find_first(raw_payload, "horizons_without_seed"),
    ))

    selected_visible_horizons = normalize_string_list(coalesce(
        selected_candidate.get("selected_visible_horizons"),
        selected_candidate.get("visible_horizons"),
        recursive_find_first(selected_candidate, "selected_visible_horizons"),
        recursive_find_first(selected_candidate, "visible_horizons"),
        recursive_find_first(raw_payload, "selected_visible_horizons"),
    ))

    selected_stability_label = coalesce(
        selected_candidate.get("selected_stability_label"),
        selected_candidate.get("stability_label"),
        row.get("selected_stability_label"),
        recursive_find_first(selected_candidate, "selected_stability_label"),
        recursive_find_first(selected_candidate, "stability_label"),
        recursive_find_first(raw_payload, "selected_stability_label"),
        recursive_find_first(raw_payload, "stability_label"),
    )

    candidate_status = coalesce(
        selected_candidate.get("candidate_status"),
        selected_candidate.get("status"),
        row.get("candidate_status"),
        recursive_find_first(selected_candidate, "candidate_status"),
        recursive_find_first(raw_payload, "candidate_status"),
    )

    reason_codes = normalize_reason_codes(coalesce(
        selected_candidate.get("reason_codes"),
        row.get("reason_codes"),
        recursive_find_first(selected_candidate, "reason_codes"),
        recursive_find_first(raw_payload, "reason_codes"),
    ))

    if not reason_codes:
        reason_codes = normalize_reason_codes([
            row.get("reason"),
            selected_candidate.get("reason"),
        ])

    gate_diagnostics = normalize_gate_diagnostics(coalesce(
        selected_candidate.get("gate_diagnostics"),
        row.get("gate_diagnostics"),
        recursive_find_first(selected_candidate, "gate_diagnostics"),
        recursive_find_first(raw_payload, "gate_diagnostics"),
    ))

    aggregate_score = to_float(coalesce(
        selected_candidate.get("aggregate_score"),
        row.get("aggregate_score"),
        recursive_find_first(selected_candidate, "aggregate_score"),
        recursive_find_first(raw_payload, "aggregate_score"),
    ))

    sample_count = to_int(coalesce(
        selected_candidate.get("sample_count"),
        row.get("sample_count"),
        recursive_find_first(selected_candidate, "sample_count"),
        recursive_find_first(raw_payload, "sample_count"),
    ))

    positive_rate_pct = to_float(coalesce(
        selected_candidate.get("positive_rate_pct"),
        row.get("positive_rate_pct"),
        recursive_find_first(selected_candidate, "positive_rate_pct"),
        recursive_find_first(raw_payload, "positive_rate_pct"),
    ))

    robustness_signal_pct = to_float(coalesce(
        selected_candidate.get("robustness_signal_pct"),
        row.get("robustness_signal_pct"),
        recursive_find_first(selected_candidate, "robustness_signal_pct"),
        recursive_find_first(raw_payload, "robustness_signal_pct"),
    ))

    source_summary = {
        "candidate_seed_count": detect_source(
            row=row,
            raw_payload=raw_payload,
            selected_candidate=selected_candidate,
            keys=["candidate_seed_count"],
        ),
        "selected_stability_label": detect_source(
            row=row,
            raw_payload=raw_payload,
            selected_candidate=selected_candidate,
            keys=["selected_stability_label", "stability_label"],
        ),
        "candidate_status": detect_source(
            row=row,
            raw_payload=raw_payload,
            selected_candidate=selected_candidate,
            keys=["candidate_status", "status"],
        ),
        "aggregate_score": detect_source(
            row=row,
            raw_payload=raw_payload,
            selected_candidate=selected_candidate,
            keys=["aggregate_score"],
        ),
        "sample_count": detect_source(
            row=row,
            raw_payload=raw_payload,
            selected_candidate=selected_candidate,
            keys=["sample_count"],
        ),
        "positive_rate_pct": detect_source(
            row=row,
            raw_payload=raw_payload,
            selected_candidate=selected_candidate,
            keys=["positive_rate_pct"],
        ),
        "robustness_signal_pct": detect_source(
            row=row,
            raw_payload=raw_payload,
            selected_candidate=selected_candidate,
            keys=["robustness_signal_pct"],
        ),
    }

    return StepSnapshot(
        step_index=to_int(row.get("step_index")),
        end_record_index_inclusive=to_int(row.get("end_record_index_inclusive")),
        selection_status=coalesce(row.get("selection_status"), recursive_find_first(raw_payload, "selection_status")),
        selection_reason=coalesce(row.get("reason"), recursive_find_first(raw_payload, "reason")),
        selected_symbol=coalesce(row.get("selected_symbol"), recursive_find_first(selected_candidate, "symbol")),
        selected_strategy=coalesce(row.get("selected_strategy"), recursive_find_first(selected_candidate, "strategy")),
        selected_horizon=coalesce(row.get("selected_horizon"), recursive_find_first(selected_candidate, "horizon")),
        candidate_seed_count=candidate_seed_count,
        horizons_with_seed=horizons_with_seed,
        horizons_without_seed=horizons_without_seed,
        selected_visible_horizons=selected_visible_horizons,
        selected_stability_label=selected_stability_label,
        candidate_status=candidate_status,
        reason_codes=reason_codes,
        gate_diagnostics=gate_diagnostics,
        aggregate_score=aggregate_score,
        sample_count=sample_count,
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        raw_output_path=raw_output_path_value if isinstance(raw_output_path_value, str) else None,
        source_summary=source_summary,
        raw_step_payload=raw_payload,
    )


def detect_source(
    row: Dict[str, Any],
    raw_payload: Dict[str, Any],
    selected_candidate: Dict[str, Any],
    keys: List[str],
) -> str:
    for key in keys:
        if key in selected_candidate:
            return "selected_candidate"
    for key in keys:
        if key in row:
            return "step_result_row"
    for key in keys:
        value = recursive_find_first(raw_payload, key)
        if value is not None:
            return "raw_payload_nested"
    return "not_found"


def build_comparison_groups(snapshots: List[StepSnapshot]) -> List[Dict[str, Any]]:
    snapshots_sorted = sorted(
        [s for s in snapshots if s.end_record_index_inclusive is not None],
        key=lambda s: s.end_record_index_inclusive or 0,
    )

    selected = [s for s in snapshots_sorted if s.selection_status == "selected"]
    abstain = [s for s in snapshots_sorted if s.selection_status != "selected"]

    comparison_groups: List[Dict[str, Any]] = []

    for abstain_snapshot in abstain:
        nearest_selected = find_nearest_selected(abstain_snapshot, selected)
        if nearest_selected is None:
            continue
        field_diffs = compare_fields(nearest_selected, abstain_snapshot)
        comparison_groups.append(
            {
                "selected_window": nearest_selected.end_record_index_inclusive,
                "abstain_window": abstain_snapshot.end_record_index_inclusive,
                "selected_identity": format_identity(nearest_selected),
                "abstain_identity": format_identity(abstain_snapshot),
                "field_differences": field_diffs,
            }
        )

    return comparison_groups


def find_nearest_selected(target: StepSnapshot, selected_snapshots: List[StepSnapshot]) -> Optional[StepSnapshot]:
    if target.end_record_index_inclusive is None or not selected_snapshots:
        return None
    return min(
        selected_snapshots,
        key=lambda s: abs((s.end_record_index_inclusive or 0) - target.end_record_index_inclusive),
    )


def compare_fields(selected_snapshot: StepSnapshot, abstain_snapshot: StepSnapshot) -> Dict[str, Dict[str, Any]]:
    selected_dict = selected_snapshot.to_dict()
    abstain_dict = abstain_snapshot.to_dict()

    diffs: Dict[str, Dict[str, Any]] = {}
    for field in LOAD_BEARING_FIELDS:
        selected_value = selected_dict.get(field)
        abstain_value = abstain_dict.get(field)
        if selected_value != abstain_value:
            diffs[field] = {
                "selected": selected_value,
                "abstain": abstain_value,
            }
    return diffs


def build_explanations(comparison_groups: List[Dict[str, Any]]) -> List[str]:
    explanations: List[str] = []

    for group in comparison_groups:
        selected_window = group["selected_window"]
        abstain_window = group["abstain_window"]
        diffs = group["field_differences"]

        lines: List[str] = []
        lines.append(
            f"Window {selected_window} -> {abstain_window}: "
            f"the system moves from selected to abstain because the load-bearing state weakens."
        )

        if "horizons_with_seed" in diffs:
            lines.append(
                f"- horizons_with_seed changes from {format_value(diffs['horizons_with_seed']['selected'])} "
                f"to {format_value(diffs['horizons_with_seed']['abstain'])}."
            )
        if "selected_visible_horizons" in diffs:
            lines.append(
                f"- selected_visible_horizons changes from {format_value(diffs['selected_visible_horizons']['selected'])} "
                f"to {format_value(diffs['selected_visible_horizons']['abstain'])}."
            )
        if "selected_stability_label" in diffs:
            lines.append(
                f"- selected_stability_label downgrades from "
                f"{format_value(diffs['selected_stability_label']['selected'])} to "
                f"{format_value(diffs['selected_stability_label']['abstain'])}."
            )
        if "candidate_status" in diffs:
            lines.append(
                f"- candidate_status changes from {format_value(diffs['candidate_status']['selected'])} "
                f"to {format_value(diffs['candidate_status']['abstain'])}."
            )
        if "reason_codes" in diffs:
            lines.append(
                f"- reason_codes changes from {format_value(diffs['reason_codes']['selected'])} "
                f"to {format_value(diffs['reason_codes']['abstain'])}."
            )
        if "gate_diagnostics" in diffs:
            lines.append(
                f"- gate_diagnostics differs: selected={format_value(diffs['gate_diagnostics']['selected'])} "
                f"vs abstain={format_value(diffs['gate_diagnostics']['abstain'])}."
            )

        numeric_shift_lines = []
        for numeric_field in [
            "aggregate_score",
            "sample_count",
            "positive_rate_pct",
            "robustness_signal_pct",
        ]:
            if numeric_field in diffs:
                numeric_shift_lines.append(
                    f"{numeric_field}: {format_value(diffs[numeric_field]['selected'])} -> "
                    f"{format_value(diffs[numeric_field]['abstain'])}"
                )
        if numeric_shift_lines:
            lines.append("- numeric shifts: " + "; ".join(numeric_shift_lines) + ".")

        if not diffs:
            lines.append("- no differences found across the load-bearing fields.")

        explanations.append("\n".join(lines))

    return explanations


def format_identity(snapshot: StepSnapshot) -> str:
    return " / ".join([
        snapshot.selected_symbol or "n/a",
        snapshot.selected_strategy or "n/a",
        snapshot.selected_horizon or "n/a",
    ])


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return json.dumps(value, ensure_ascii=False)


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    metadata = report["metadata"]
    snapshots = report["snapshots"]
    comparison_groups = report["comparison_groups"]
    explanations = report["explanations"]

    lines.append("# Selected vs Abstain Diagnosis")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- run_dir: `{metadata['run_dir']}`")
    lines.append(f"- step_results_path: `{metadata['step_results_path']}`")
    lines.append(f"- target_windows: `{metadata['target_windows']}`")
    lines.append(f"- resolved_windows: `{metadata['resolved_windows']}`")
    lines.append(f"- missing_windows: `{metadata['missing_windows']}`")
    lines.append("")

    lines.append("## Step Snapshots")
    lines.append("")
    header = [
        "window",
        "status",
        "reason",
        "identity",
        "candidate_seed_count",
        "horizons_with_seed",
        "selected_visible_horizons",
        "selected_stability_label",
        "candidate_status",
        "aggregate_score",
        "sample_count",
        "positive_rate_pct",
        "robustness_signal_pct",
        "reason_codes",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for item in sorted(snapshots, key=lambda x: x.get("end_record_index_inclusive") or 0):
        row = [
            str(item.get("end_record_index_inclusive")),
            str(item.get("selection_status")),
            str(item.get("selection_reason")),
            " / ".join([
                str(item.get("selected_symbol") or "n/a"),
                str(item.get("selected_strategy") or "n/a"),
                str(item.get("selected_horizon") or "n/a"),
            ]),
            str(item.get("candidate_seed_count")),
            json.dumps(item.get("horizons_with_seed"), ensure_ascii=False),
            json.dumps(item.get("selected_visible_horizons"), ensure_ascii=False),
            str(item.get("selected_stability_label")),
            str(item.get("candidate_status")),
            safe_cell(item.get("aggregate_score")),
            safe_cell(item.get("sample_count")),
            safe_cell(item.get("positive_rate_pct")),
            safe_cell(item.get("robustness_signal_pct")),
            json.dumps(item.get("reason_codes"), ensure_ascii=False),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Pairwise Selected vs Abstain Comparisons")
    lines.append("")

    if not comparison_groups:
        lines.append("- No comparison groups generated.")
    else:
        for group in comparison_groups:
            lines.append(
                f"### selected {group['selected_window']} vs abstain {group['abstain_window']}"
            )
            lines.append(f"- selected_identity: `{group['selected_identity']}`")
            lines.append(f"- abstain_identity: `{group['abstain_identity']}`")
            lines.append("")
            lines.append("| field | selected | abstain |")
            lines.append("|---|---|---|")
            diffs = group["field_differences"]
            if diffs:
                for field_name in LOAD_BEARING_FIELDS:
                    if field_name not in diffs:
                        continue
                    lines.append(
                        "| "
                        + " | ".join([
                            field_name,
                            safe_cell(diffs[field_name]["selected"]),
                            safe_cell(diffs[field_name]["abstain"]),
                        ])
                        + " |"
                    )
            else:
                lines.append("| none | identical across load-bearing fields | identical across load-bearing fields |")
            lines.append("")

    lines.append("## Explanations")
    lines.append("")
    if explanations:
        for explanation in explanations:
            lines.append(explanation)
            lines.append("")
    else:
        lines.append("- No explanations generated.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def safe_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


if __name__ == "__main__":
    main()