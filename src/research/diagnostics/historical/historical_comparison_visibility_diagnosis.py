from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RUNS_ROOT = Path("logs/research_reports/historical_direct_edge_selection")
DEFAULT_LATEST_DIR = Path("logs/research_reports/latest")
DEFAULT_SYMBOL = "ETHUSDT"
DEFAULT_STRATEGY = "swing"
DEFAULT_HORIZON = "1h"


@dataclass
class VisibilityRow:
    step_index: Optional[int]
    end_record_index_inclusive: Optional[int]
    selection_status: Optional[str]
    selection_reason: Optional[str]

    tracked_symbol: str
    tracked_strategy: str
    tracked_horizon: str

    comparison_latest_top_symbol: Optional[str]
    comparison_latest_top_strategy: Optional[str]
    comparison_cumulative_top_symbol: Optional[str]
    comparison_cumulative_top_strategy: Optional[str]

    comparison_symbol_latest_visible_horizons: List[str]
    comparison_symbol_cumulative_visible_horizons: List[str]
    comparison_strategy_latest_visible_horizons: List[str]
    comparison_strategy_cumulative_visible_horizons: List[str]

    comparison_symbol_has_tracked_horizon: bool
    comparison_strategy_has_tracked_horizon: bool

    mapper_candidate_present_for_tracked_triplet: bool
    mapper_candidate_visible_horizons: List[str]
    mapper_candidate_stability_label: Optional[str]
    mapper_candidate_symbol: Optional[str]
    mapper_candidate_strategy: Optional[str]
    mapper_candidate_horizon: Optional[str]

    engine_selected_symbol: Optional[str]
    engine_selected_strategy: Optional[str]
    engine_selected_horizon: Optional[str]

    raw_output_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "end_record_index_inclusive": self.end_record_index_inclusive,
            "selection_status": self.selection_status,
            "selection_reason": self.selection_reason,
            "tracked_symbol": self.tracked_symbol,
            "tracked_strategy": self.tracked_strategy,
            "tracked_horizon": self.tracked_horizon,
            "comparison_latest_top_symbol": self.comparison_latest_top_symbol,
            "comparison_latest_top_strategy": self.comparison_latest_top_strategy,
            "comparison_cumulative_top_symbol": self.comparison_cumulative_top_symbol,
            "comparison_cumulative_top_strategy": self.comparison_cumulative_top_strategy,
            "comparison_symbol_latest_visible_horizons": self.comparison_symbol_latest_visible_horizons,
            "comparison_symbol_cumulative_visible_horizons": self.comparison_symbol_cumulative_visible_horizons,
            "comparison_strategy_latest_visible_horizons": self.comparison_strategy_latest_visible_horizons,
            "comparison_strategy_cumulative_visible_horizons": self.comparison_strategy_cumulative_visible_horizons,
            "comparison_symbol_has_tracked_horizon": self.comparison_symbol_has_tracked_horizon,
            "comparison_strategy_has_tracked_horizon": self.comparison_strategy_has_tracked_horizon,
            "mapper_candidate_present_for_tracked_triplet": self.mapper_candidate_present_for_tracked_triplet,
            "mapper_candidate_visible_horizons": self.mapper_candidate_visible_horizons,
            "mapper_candidate_stability_label": self.mapper_candidate_stability_label,
            "mapper_candidate_symbol": self.mapper_candidate_symbol,
            "mapper_candidate_strategy": self.mapper_candidate_strategy,
            "mapper_candidate_horizon": self.mapper_candidate_horizon,
            "engine_selected_symbol": self.engine_selected_symbol,
            "engine_selected_strategy": self.engine_selected_strategy,
            "engine_selected_horizon": self.engine_selected_horizon,
            "raw_output_path": self.raw_output_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether tracked horizon visibility disappears first in the "
            "comparison layer or only later in mapper output."
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
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help="Tracked symbol identity.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        help="Tracked strategy identity.",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default=DEFAULT_HORIZON,
        help="Tracked horizon to diagnose.",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=None,
        help="Optional inclusive lower bound for end_record_index_inclusive.",
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=None,
        help="Optional inclusive upper bound for end_record_index_inclusive.",
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
    filtered_rows = filter_rows_by_window(rows, args.start_window, args.end_window)

    visibility_rows = [
        build_visibility_row(
            row=row,
            tracked_symbol=args.symbol,
            tracked_strategy=args.strategy,
            tracked_horizon=args.horizon,
        )
        for row in filtered_rows
    ]

    summary = build_summary(
        rows=visibility_rows,
        tracked_symbol=args.symbol,
        tracked_strategy=args.strategy,
        tracked_horizon=args.horizon,
    )
    transitions = build_transitions(visibility_rows)
    explanations = build_explanations(summary, transitions)

    report = {
        "metadata": {
            "run_dir": str(run_dir),
            "step_results_path": str(step_results_path),
            "tracked_symbol": args.symbol,
            "tracked_strategy": args.strategy,
            "tracked_horizon": args.horizon,
            "start_window": args.start_window,
            "end_window": args.end_window,
            "row_count": len(visibility_rows),
        },
        "summary": summary,
        "transitions": transitions,
        "rows": [row.to_dict() for row in visibility_rows],
        "explanations": explanations,
    }

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_symbol = args.symbol.lower()
    safe_strategy = args.strategy.lower()
    safe_horizon = args.horizon.lower().replace("/", "_")

    json_path = output_dir / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_comparison_visibility_diagnosis.json"
    md_path = output_dir / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_comparison_visibility_diagnosis.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "row_count": len(visibility_rows),
            "summary": {
                "comparison_symbol_has_horizon_count": summary["comparison_symbol_has_horizon_count"],
                "comparison_strategy_has_horizon_count": summary["comparison_strategy_has_horizon_count"],
                "mapper_triplet_present_count": summary["mapper_triplet_present_count"],
                "first_comparison_loss_window": summary["first_comparison_loss_window"],
                "first_mapper_loss_window": summary["first_mapper_loss_window"],
                "first_comparison_to_mapper_gap_window": summary["first_comparison_to_mapper_gap_window"],
            },
        },
        indent=2,
        ensure_ascii=False,
    ))

    if args.write_latest_copy:
        DEFAULT_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        latest_json = DEFAULT_LATEST_DIR / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_comparison_visibility_diagnosis.json"
        latest_md = DEFAULT_LATEST_DIR / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_comparison_visibility_diagnosis.md"
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


def filter_rows_by_window(
    rows: Iterable[Dict[str, Any]],
    start_window: Optional[int],
    end_window: Optional[int],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        window = to_int(row.get("end_record_index_inclusive"))
        if window is None:
            continue
        if start_window is not None and window < start_window:
            continue
        if end_window is not None and window > end_window:
            continue
        filtered.append(row)
    filtered.sort(key=lambda item: to_int(item.get("end_record_index_inclusive")) or 0)
    return filtered


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


def extract_comparison_report(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    comparison_pipeline_output = raw_payload.get("comparison_pipeline_output")
    if not isinstance(comparison_pipeline_output, dict):
        return {}
    comparison_report = comparison_pipeline_output.get("comparison_report")
    if isinstance(comparison_report, dict):
        return comparison_report
    return {}


def extract_mapper_candidates(raw_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    mapper_payload = raw_payload.get("mapper_payload")
    if not isinstance(mapper_payload, dict):
        return []
    candidates = mapper_payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def extract_engine_output(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    engine_output = raw_payload.get("engine_output")
    if isinstance(engine_output, dict):
        return engine_output
    return {}


def candidate_symbol(candidate: Dict[str, Any]) -> Optional[str]:
    return coalesce(candidate.get("symbol"), recursive_find_first(candidate, "symbol"))


def candidate_strategy(candidate: Dict[str, Any]) -> Optional[str]:
    return coalesce(candidate.get("strategy"), recursive_find_first(candidate, "strategy"))


def candidate_horizon(candidate: Dict[str, Any]) -> Optional[str]:
    value = coalesce(candidate.get("horizon"), recursive_find_first(candidate, "horizon"))
    return str(value) if value is not None else None


def find_mapper_triplet_candidate(
    mapper_candidates: List[Dict[str, Any]],
    tracked_symbol: str,
    tracked_strategy: str,
    tracked_horizon: str,
) -> Dict[str, Any]:
    for candidate in mapper_candidates:
        if (
            candidate_symbol(candidate) == tracked_symbol
            and candidate_strategy(candidate) == tracked_strategy
            and candidate_horizon(candidate) == tracked_horizon
        ):
            return candidate
    return {}


def get_nested_dict(source: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    if isinstance(current, dict):
        return current
    return {}


def get_top_highlights_comparison_horizon_block(comparison_report: Dict[str, Any], tracked_horizon: str) -> Dict[str, Any]:
    return get_nested_dict(comparison_report, "top_highlights_comparison", tracked_horizon)


def get_edge_stability_comparison_block(comparison_report: Dict[str, Any], category: str) -> Dict[str, Any]:
    return get_nested_dict(comparison_report, "edge_stability_comparison", category)


def build_visibility_row(
    row: Dict[str, Any],
    tracked_symbol: str,
    tracked_strategy: str,
    tracked_horizon: str,
) -> VisibilityRow:
    raw_output_path_value = row.get("raw_output_path")
    raw_payload: Dict[str, Any] = {}
    if isinstance(raw_output_path_value, str) and raw_output_path_value:
        raw_payload = load_json_file(Path(raw_output_path_value))

    comparison_report = extract_comparison_report(raw_payload)
    mapper_candidates = extract_mapper_candidates(raw_payload)
    engine_output = extract_engine_output(raw_payload)

    horizon_block = get_top_highlights_comparison_horizon_block(comparison_report, tracked_horizon)
    symbol_stability_block = get_edge_stability_comparison_block(comparison_report, "symbol")
    strategy_stability_block = get_edge_stability_comparison_block(comparison_report, "strategy")

    mapper_triplet_candidate = find_mapper_triplet_candidate(
        mapper_candidates=mapper_candidates,
        tracked_symbol=tracked_symbol,
        tracked_strategy=tracked_strategy,
        tracked_horizon=tracked_horizon,
    )

    comparison_latest_top_symbol = horizon_block.get("latest_top_symbol")
    comparison_latest_top_strategy = horizon_block.get("latest_top_strategy")
    comparison_cumulative_top_symbol = horizon_block.get("cumulative_top_symbol")
    comparison_cumulative_top_strategy = horizon_block.get("cumulative_top_strategy")

    comparison_symbol_latest_visible_horizons = normalize_string_list(
        symbol_stability_block.get("latest_visible_horizons")
    )
    comparison_symbol_cumulative_visible_horizons = normalize_string_list(
        symbol_stability_block.get("cumulative_visible_horizons")
    )
    comparison_strategy_latest_visible_horizons = normalize_string_list(
        strategy_stability_block.get("latest_visible_horizons")
    )
    comparison_strategy_cumulative_visible_horizons = normalize_string_list(
        strategy_stability_block.get("cumulative_visible_horizons")
    )

    comparison_symbol_has_tracked_horizon = tracked_horizon in comparison_symbol_latest_visible_horizons
    comparison_strategy_has_tracked_horizon = tracked_horizon in comparison_strategy_latest_visible_horizons

    mapper_candidate_visible_horizons = normalize_string_list(coalesce(
        mapper_triplet_candidate.get("selected_visible_horizons"),
        mapper_triplet_candidate.get("visible_horizons"),
        recursive_find_first(mapper_triplet_candidate, "selected_visible_horizons"),
        recursive_find_first(mapper_triplet_candidate, "visible_horizons"),
    ))

    mapper_candidate_stability_label = coalesce(
        mapper_triplet_candidate.get("selected_stability_label"),
        mapper_triplet_candidate.get("stability_label"),
        recursive_find_first(mapper_triplet_candidate, "selected_stability_label"),
        recursive_find_first(mapper_triplet_candidate, "stability_label"),
    )

    return VisibilityRow(
        step_index=to_int(row.get("step_index")),
        end_record_index_inclusive=to_int(row.get("end_record_index_inclusive")),
        selection_status=coalesce(row.get("selection_status"), recursive_find_first(raw_payload, "selection_status")),
        selection_reason=coalesce(row.get("reason"), recursive_find_first(raw_payload, "reason")),
        tracked_symbol=tracked_symbol,
        tracked_strategy=tracked_strategy,
        tracked_horizon=tracked_horizon,
        comparison_latest_top_symbol=comparison_latest_top_symbol,
        comparison_latest_top_strategy=comparison_latest_top_strategy,
        comparison_cumulative_top_symbol=comparison_cumulative_top_symbol,
        comparison_cumulative_top_strategy=comparison_cumulative_top_strategy,
        comparison_symbol_latest_visible_horizons=comparison_symbol_latest_visible_horizons,
        comparison_symbol_cumulative_visible_horizons=comparison_symbol_cumulative_visible_horizons,
        comparison_strategy_latest_visible_horizons=comparison_strategy_latest_visible_horizons,
        comparison_strategy_cumulative_visible_horizons=comparison_strategy_cumulative_visible_horizons,
        comparison_symbol_has_tracked_horizon=comparison_symbol_has_tracked_horizon,
        comparison_strategy_has_tracked_horizon=comparison_strategy_has_tracked_horizon,
        mapper_candidate_present_for_tracked_triplet=bool(mapper_triplet_candidate),
        mapper_candidate_visible_horizons=mapper_candidate_visible_horizons,
        mapper_candidate_stability_label=mapper_candidate_stability_label,
        mapper_candidate_symbol=candidate_symbol(mapper_triplet_candidate),
        mapper_candidate_strategy=candidate_strategy(mapper_triplet_candidate),
        mapper_candidate_horizon=candidate_horizon(mapper_triplet_candidate),
        engine_selected_symbol=engine_output.get("selected_symbol"),
        engine_selected_strategy=engine_output.get("selected_strategy"),
        engine_selected_horizon=engine_output.get("selected_horizon"),
        raw_output_path=raw_output_path_value if isinstance(raw_output_path_value, str) else None,
    )


def build_summary(
    rows: List[VisibilityRow],
    tracked_symbol: str,
    tracked_strategy: str,
    tracked_horizon: str,
) -> Dict[str, Any]:
    comparison_symbol_has_rows = [row for row in rows if row.comparison_symbol_has_tracked_horizon]
    comparison_strategy_has_rows = [row for row in rows if row.comparison_strategy_has_tracked_horizon]
    mapper_triplet_rows = [row for row in rows if row.mapper_candidate_present_for_tracked_triplet]

    first_comparison_loss_window = None
    first_mapper_loss_window = None
    first_comparison_to_mapper_gap_window = None

    prev_comparison_state: Optional[bool] = None
    prev_mapper_state: Optional[bool] = None

    for row in rows:
        current_comparison_state = row.comparison_symbol_has_tracked_horizon and row.comparison_strategy_has_tracked_horizon
        current_mapper_state = row.mapper_candidate_present_for_tracked_triplet

        if prev_comparison_state is True and current_comparison_state is False and first_comparison_loss_window is None:
            first_comparison_loss_window = row.end_record_index_inclusive

        if prev_mapper_state is True and current_mapper_state is False and first_mapper_loss_window is None:
            first_mapper_loss_window = row.end_record_index_inclusive

        if (
            current_comparison_state is True
            and current_mapper_state is False
            and first_comparison_to_mapper_gap_window is None
        ):
            first_comparison_to_mapper_gap_window = row.end_record_index_inclusive

        prev_comparison_state = current_comparison_state
        prev_mapper_state = current_mapper_state

    return {
        "tracked_symbol": tracked_symbol,
        "tracked_strategy": tracked_strategy,
        "tracked_horizon": tracked_horizon,
        "row_count": len(rows),
        "comparison_symbol_has_horizon_count": len(comparison_symbol_has_rows),
        "comparison_strategy_has_horizon_count": len(comparison_strategy_has_rows),
        "comparison_joint_has_horizon_count": len([
            row for row in rows
            if row.comparison_symbol_has_tracked_horizon and row.comparison_strategy_has_tracked_horizon
        ]),
        "mapper_triplet_present_count": len(mapper_triplet_rows),
        "first_window": rows[0].end_record_index_inclusive if rows else None,
        "last_window": rows[-1].end_record_index_inclusive if rows else None,
        "first_comparison_loss_window": first_comparison_loss_window,
        "first_mapper_loss_window": first_mapper_loss_window,
        "first_comparison_to_mapper_gap_window": first_comparison_to_mapper_gap_window,
        "windows_comparison_symbol_has_horizon": [
            row.end_record_index_inclusive for row in comparison_symbol_has_rows
        ],
        "windows_comparison_strategy_has_horizon": [
            row.end_record_index_inclusive for row in comparison_strategy_has_rows
        ],
        "windows_mapper_triplet_present": [
            row.end_record_index_inclusive for row in mapper_triplet_rows
        ],
    }


def visibility_state(row: VisibilityRow) -> str:
    comparison_joint = row.comparison_symbol_has_tracked_horizon and row.comparison_strategy_has_tracked_horizon
    mapper_present = row.mapper_candidate_present_for_tracked_triplet

    if comparison_joint and mapper_present:
        return "comparison_and_mapper_present"
    if comparison_joint and not mapper_present:
        return "comparison_present_mapper_missing"
    if (not comparison_joint) and mapper_present:
        return "comparison_missing_mapper_present"
    return "comparison_and_mapper_missing"


def build_transitions(rows: List[VisibilityRow]) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        prev_state = visibility_state(previous)
        curr_state = visibility_state(current)
        if prev_state != curr_state:
            transitions.append(
                {
                    "from_window": previous.end_record_index_inclusive,
                    "to_window": current.end_record_index_inclusive,
                    "from_state": prev_state,
                    "to_state": curr_state,
                    "comparison_change": {
                        "from_symbol_has_horizon": previous.comparison_symbol_has_tracked_horizon,
                        "to_symbol_has_horizon": current.comparison_symbol_has_tracked_horizon,
                        "from_strategy_has_horizon": previous.comparison_strategy_has_tracked_horizon,
                        "to_strategy_has_horizon": current.comparison_strategy_has_tracked_horizon,
                        "from_latest_top_symbol": previous.comparison_latest_top_symbol,
                        "to_latest_top_symbol": current.comparison_latest_top_symbol,
                        "from_latest_top_strategy": previous.comparison_latest_top_strategy,
                        "to_latest_top_strategy": current.comparison_latest_top_strategy,
                    },
                    "mapper_change": {
                        "from_triplet_present": previous.mapper_candidate_present_for_tracked_triplet,
                        "to_triplet_present": current.mapper_candidate_present_for_tracked_triplet,
                        "from_mapper_horizon": previous.mapper_candidate_horizon,
                        "to_mapper_horizon": current.mapper_candidate_horizon,
                        "from_mapper_stability_label": previous.mapper_candidate_stability_label,
                        "to_mapper_stability_label": current.mapper_candidate_stability_label,
                    },
                }
            )
    return transitions


def build_explanations(summary: Dict[str, Any], transitions: List[Dict[str, Any]]) -> List[str]:
    explanations: List[str] = []

    explanations.append(
        "This report checks whether the tracked horizon is still visible in the comparison layer before mapper candidate generation."
    )

    if summary["first_comparison_loss_window"] is not None:
        explanations.append(
            f"The first comparison-layer loss of the tracked horizon occurs at window {summary['first_comparison_loss_window']}."
        )

    if summary["first_mapper_loss_window"] is not None:
        explanations.append(
            f"The first mapper-level loss of the tracked symbol/strategy/horizon triplet occurs at window {summary['first_mapper_loss_window']}."
        )

    if summary["first_comparison_to_mapper_gap_window"] is not None:
        explanations.append(
            f"At window {summary['first_comparison_to_mapper_gap_window']}, comparison still supports the tracked horizon while mapper no longer emits the full triplet."
        )
    else:
        explanations.append(
            "No comparison-to-mapper gap was detected in this range; mapper disappearance appears aligned with or downstream of comparison disappearance."
        )

    if transitions:
        first_transition = transitions[0]
        explanations.append(
            f"The earliest visibility state change is {first_transition['from_state']} -> {first_transition['to_state']} "
            f"between windows {first_transition['from_window']} and {first_transition['to_window']}."
        )

    explanations.append(
        "If comparison loses the tracked horizon at the same window mapper loses the tracked triplet, the bottleneck likely begins in comparison/top-group visibility rather than mapper-only filtering."
    )

    return explanations


def render_markdown(report: Dict[str, Any]) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    transitions = report["transitions"]
    rows = report["rows"]
    explanations = report["explanations"]

    lines: List[str] = []
    lines.append("# Comparison Visibility Diagnosis")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- run_dir: `{metadata['run_dir']}`")
    lines.append(f"- step_results_path: `{metadata['step_results_path']}`")
    lines.append(f"- tracked_symbol: `{metadata['tracked_symbol']}`")
    lines.append(f"- tracked_strategy: `{metadata['tracked_strategy']}`")
    lines.append(f"- tracked_horizon: `{metadata['tracked_horizon']}`")
    lines.append(f"- start_window: `{metadata['start_window']}`")
    lines.append(f"- end_window: `{metadata['end_window']}`")
    lines.append(f"- row_count: `{metadata['row_count']}`")
    lines.append("")

    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## Visibility Rows")
    lines.append("")
    header = [
        "window",
        "selection_status",
        "comparison_latest_top_symbol",
        "comparison_latest_top_strategy",
        "comparison_symbol_has_tracked_horizon",
        "comparison_strategy_has_tracked_horizon",
        "mapper_candidate_present_for_tracked_triplet",
        "mapper_candidate_symbol",
        "mapper_candidate_strategy",
        "mapper_candidate_horizon",
        "mapper_candidate_stability_label",
        "engine_selected_symbol",
        "engine_selected_strategy",
        "engine_selected_horizon",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in rows:
        lines.append(
            "| " + " | ".join([
                safe_cell(row.get("end_record_index_inclusive")),
                safe_cell(row.get("selection_status")),
                safe_cell(row.get("comparison_latest_top_symbol")),
                safe_cell(row.get("comparison_latest_top_strategy")),
                safe_cell(row.get("comparison_symbol_has_tracked_horizon")),
                safe_cell(row.get("comparison_strategy_has_tracked_horizon")),
                safe_cell(row.get("mapper_candidate_present_for_tracked_triplet")),
                safe_cell(row.get("mapper_candidate_symbol")),
                safe_cell(row.get("mapper_candidate_strategy")),
                safe_cell(row.get("mapper_candidate_horizon")),
                safe_cell(row.get("mapper_candidate_stability_label")),
                safe_cell(row.get("engine_selected_symbol")),
                safe_cell(row.get("engine_selected_strategy")),
                safe_cell(row.get("engine_selected_horizon")),
            ]) + " |"
        )

    lines.append("")
    lines.append("## State Transitions")
    lines.append("")
    if not transitions:
        lines.append("- No visibility state transitions detected.")
    else:
        for item in transitions:
            lines.append(f"### {item['from_window']} -> {item['to_window']}")
            lines.append(f"- from_state: `{item['from_state']}`")
            lines.append(f"- to_state: `{item['to_state']}`")
            lines.append(f"- comparison_change: `{item['comparison_change']}`")
            lines.append(f"- mapper_change: `{item['mapper_change']}`")
            lines.append("")

    lines.append("## Explanations")
    lines.append("")
    for explanation in explanations:
        lines.append(f"- {explanation}")

    return "\n".join(lines).strip() + "\n"


def safe_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


if __name__ == "__main__":
    main()