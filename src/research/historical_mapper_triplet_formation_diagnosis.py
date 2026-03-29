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


LOAD_BEARING_FIELDS = [
    "comparison_latest_top_symbol",
    "comparison_latest_top_strategy",
    "comparison_cumulative_top_symbol",
    "comparison_cumulative_top_strategy",
    "comparison_symbol_latest_visible_horizons",
    "comparison_strategy_latest_visible_horizons",
    "comparison_symbol_has_tracked_horizon",
    "comparison_strategy_has_tracked_horizon",
    "triplet_present_in_mapper",
    "triplet_visible_horizons",
    "triplet_stability_label",
    "triplet_candidate_strength",
    "triplet_sample_count",
    "triplet_positive_rate_pct",
    "triplet_robustness_signal_pct",
    "candidate_seed_count",
    "horizons_with_seed",
    "horizons_without_seed",
    "engine_selected_symbol",
    "engine_selected_strategy",
    "engine_selected_horizon",
    "engine_abstain_top_symbol",
    "engine_abstain_top_strategy",
    "engine_abstain_top_horizon",
]


@dataclass
class MapperTripletRow:
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
    comparison_strategy_latest_visible_horizons: List[str]
    comparison_symbol_has_tracked_horizon: bool
    comparison_strategy_has_tracked_horizon: bool

    triplet_present_in_mapper: bool
    triplet_visible_horizons: List[str]
    triplet_stability_label: Optional[str]
    triplet_candidate_strength: Optional[str]
    triplet_sample_count: Optional[int]
    triplet_positive_rate_pct: Optional[float]
    triplet_robustness_signal_pct: Optional[float]

    candidate_seed_count: Optional[int]
    horizons_with_seed: List[str]
    horizons_without_seed: List[str]

    engine_selected_symbol: Optional[str]
    engine_selected_strategy: Optional[str]
    engine_selected_horizon: Optional[str]

    engine_abstain_top_symbol: Optional[str]
    engine_abstain_top_strategy: Optional[str]
    engine_abstain_top_horizon: Optional[str]

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
            "comparison_strategy_latest_visible_horizons": self.comparison_strategy_latest_visible_horizons,
            "comparison_symbol_has_tracked_horizon": self.comparison_symbol_has_tracked_horizon,
            "comparison_strategy_has_tracked_horizon": self.comparison_strategy_has_tracked_horizon,
            "triplet_present_in_mapper": self.triplet_present_in_mapper,
            "triplet_visible_horizons": self.triplet_visible_horizons,
            "triplet_stability_label": self.triplet_stability_label,
            "triplet_candidate_strength": self.triplet_candidate_strength,
            "triplet_sample_count": self.triplet_sample_count,
            "triplet_positive_rate_pct": self.triplet_positive_rate_pct,
            "triplet_robustness_signal_pct": self.triplet_robustness_signal_pct,
            "candidate_seed_count": self.candidate_seed_count,
            "horizons_with_seed": self.horizons_with_seed,
            "horizons_without_seed": self.horizons_without_seed,
            "engine_selected_symbol": self.engine_selected_symbol,
            "engine_selected_strategy": self.engine_selected_strategy,
            "engine_selected_horizon": self.engine_selected_horizon,
            "engine_abstain_top_symbol": self.engine_abstain_top_symbol,
            "engine_abstain_top_strategy": self.engine_abstain_top_strategy,
            "engine_abstain_top_horizon": self.engine_abstain_top_horizon,
            "raw_output_path": self.raw_output_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mapper triplet formation for a tracked symbol/strategy/horizon "
            "across present vs missing windows."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    parser.add_argument("--horizon", type=str, default=DEFAULT_HORIZON)
    parser.add_argument("--start-window", type=int, default=None)
    parser.add_argument("--end-window", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-latest-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = resolve_run_dir(args.run_dir, args.runs_root)
    step_results_path = run_dir / "step_results.jsonl"
    if not step_results_path.exists():
        raise FileNotFoundError(f"Missing step_results.jsonl: {step_results_path}")

    rows = load_jsonl(step_results_path)
    filtered_rows = filter_rows_by_window(rows, args.start_window, args.end_window)

    mapped_rows = [
        build_mapper_triplet_row(
            row=row,
            tracked_symbol=args.symbol,
            tracked_strategy=args.strategy,
            tracked_horizon=args.horizon,
        )
        for row in filtered_rows
    ]

    summary = build_summary(mapped_rows, args.symbol, args.strategy, args.horizon)
    pairwise_comparisons = build_pairwise_comparisons(mapped_rows)
    explanations = build_explanations(summary, pairwise_comparisons)

    report = {
        "metadata": {
            "run_dir": str(run_dir),
            "step_results_path": str(step_results_path),
            "tracked_symbol": args.symbol,
            "tracked_strategy": args.strategy,
            "tracked_horizon": args.horizon,
            "start_window": args.start_window,
            "end_window": args.end_window,
            "row_count": len(mapped_rows),
        },
        "summary": summary,
        "rows": [row.to_dict() for row in mapped_rows],
        "pairwise_comparisons": pairwise_comparisons,
        "explanations": explanations,
    }

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_symbol = args.symbol.lower()
    safe_strategy = args.strategy.lower()
    safe_horizon = args.horizon.lower().replace("/", "_")

    json_path = output_dir / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_mapper_triplet_formation_diagnosis.json"
    md_path = output_dir / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_mapper_triplet_formation_diagnosis.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "row_count": len(mapped_rows),
            "summary": {
                "triplet_present_count": summary["triplet_present_count"],
                "triplet_missing_count": summary["triplet_missing_count"],
                "present_windows": summary["present_windows"],
                "missing_windows": summary["missing_windows"],
            },
        },
        indent=2,
        ensure_ascii=False,
    ))

    if args.write_latest_copy:
        DEFAULT_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        latest_json = DEFAULT_LATEST_DIR / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_mapper_triplet_formation_diagnosis.json"
        latest_md = DEFAULT_LATEST_DIR / f"{safe_symbol}_{safe_strategy}_{safe_horizon}_mapper_triplet_formation_diagnosis.md"
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


def get_nested_dict(source: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    if isinstance(current, dict):
        return current
    return {}


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


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def extract_comparison_report(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    return get_nested_dict(raw_payload, "comparison_pipeline_output", "comparison_report")


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


def extract_analyzer_edge_candidate_rows(raw_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = get_nested_dict(raw_payload, "analyzer_output", "edge_candidate_rows").get("rows")
    if isinstance(rows, list):
        return [item for item in rows if isinstance(item, dict)]
    return []


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


def find_analyzer_triplet_row(
    analyzer_rows: List[Dict[str, Any]],
    tracked_symbol: str,
    tracked_strategy: str,
    tracked_horizon: str,
) -> Dict[str, Any]:
    for row in analyzer_rows:
        if (
            coalesce(row.get("symbol"), recursive_find_first(row, "symbol")) == tracked_symbol
            and coalesce(row.get("strategy"), recursive_find_first(row, "strategy")) == tracked_strategy
            and str(coalesce(row.get("horizon"), recursive_find_first(row, "horizon"))) == tracked_horizon
        ):
            return row
    return {}


def build_mapper_triplet_row(
    row: Dict[str, Any],
    tracked_symbol: str,
    tracked_strategy: str,
    tracked_horizon: str,
) -> MapperTripletRow:
    raw_output_path_value = row.get("raw_output_path")
    raw_payload: Dict[str, Any] = {}
    if isinstance(raw_output_path_value, str) and raw_output_path_value:
        raw_payload = load_json_file(Path(raw_output_path_value))

    comparison_report = extract_comparison_report(raw_payload)
    mapper_candidates = extract_mapper_candidates(raw_payload)
    engine_output = extract_engine_output(raw_payload)
    analyzer_rows = extract_analyzer_edge_candidate_rows(raw_payload)

    horizon_block = get_nested_dict(comparison_report, "top_highlights_comparison", tracked_horizon)
    symbol_stability_block = get_nested_dict(comparison_report, "edge_stability_comparison", "symbol")
    strategy_stability_block = get_nested_dict(comparison_report, "edge_stability_comparison", "strategy")

    mapper_triplet_candidate = find_mapper_triplet_candidate(
        mapper_candidates=mapper_candidates,
        tracked_symbol=tracked_symbol,
        tracked_strategy=tracked_strategy,
        tracked_horizon=tracked_horizon,
    )
    analyzer_triplet_row = find_analyzer_triplet_row(
        analyzer_rows=analyzer_rows,
        tracked_symbol=tracked_symbol,
        tracked_strategy=tracked_strategy,
        tracked_horizon=tracked_horizon,
    )

    comparison_latest_top_symbol = horizon_block.get("latest_top_symbol")
    comparison_latest_top_strategy = horizon_block.get("latest_top_strategy")
    comparison_cumulative_top_symbol = horizon_block.get("cumulative_top_symbol")
    comparison_cumulative_top_strategy = horizon_block.get("cumulative_top_strategy")

    comparison_symbol_latest_visible_horizons = normalize_string_list(symbol_stability_block.get("latest_visible_horizons"))
    comparison_strategy_latest_visible_horizons = normalize_string_list(strategy_stability_block.get("latest_visible_horizons"))

    comparison_symbol_has_tracked_horizon = tracked_horizon in comparison_symbol_latest_visible_horizons
    comparison_strategy_has_tracked_horizon = tracked_horizon in comparison_strategy_latest_visible_horizons

    triplet_visible_horizons = normalize_string_list(coalesce(
        mapper_triplet_candidate.get("selected_visible_horizons"),
        mapper_triplet_candidate.get("visible_horizons"),
        recursive_find_first(mapper_triplet_candidate, "selected_visible_horizons"),
        recursive_find_first(mapper_triplet_candidate, "visible_horizons"),
        analyzer_triplet_row.get("selected_visible_horizons"),
        recursive_find_first(analyzer_triplet_row, "selected_visible_horizons"),
    ))

    triplet_stability_label = coalesce(
        mapper_triplet_candidate.get("selected_stability_label"),
        mapper_triplet_candidate.get("stability_label"),
        recursive_find_first(mapper_triplet_candidate, "selected_stability_label"),
        recursive_find_first(mapper_triplet_candidate, "stability_label"),
        analyzer_triplet_row.get("selected_stability_label"),
        recursive_find_first(analyzer_triplet_row, "selected_stability_label"),
    )

    triplet_candidate_strength = coalesce(
        mapper_triplet_candidate.get("selected_candidate_strength"),
        mapper_triplet_candidate.get("candidate_strength"),
        recursive_find_first(mapper_triplet_candidate, "selected_candidate_strength"),
        recursive_find_first(mapper_triplet_candidate, "candidate_strength"),
        analyzer_triplet_row.get("selected_candidate_strength"),
        recursive_find_first(analyzer_triplet_row, "selected_candidate_strength"),
    )

    triplet_sample_count = to_int(coalesce(
        mapper_triplet_candidate.get("sample_count"),
        recursive_find_first(mapper_triplet_candidate, "sample_count"),
        analyzer_triplet_row.get("sample_count"),
        recursive_find_first(analyzer_triplet_row, "sample_count"),
    ))

    triplet_positive_rate_pct = to_float(coalesce(
        mapper_triplet_candidate.get("positive_rate_pct"),
        recursive_find_first(mapper_triplet_candidate, "positive_rate_pct"),
        analyzer_triplet_row.get("positive_rate_pct"),
        recursive_find_first(analyzer_triplet_row, "positive_rate_pct"),
    ))

    triplet_robustness_signal_pct = to_float(coalesce(
        mapper_triplet_candidate.get("robustness_signal_pct"),
        recursive_find_first(mapper_triplet_candidate, "robustness_signal_pct"),
        analyzer_triplet_row.get("robustness_signal_pct"),
        recursive_find_first(analyzer_triplet_row, "robustness_signal_pct"),
    ))

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

    abstain_top_candidate = get_nested_dict(engine_output, "abstain_diagnosis", "top_candidate")

    return MapperTripletRow(
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
        comparison_strategy_latest_visible_horizons=comparison_strategy_latest_visible_horizons,
        comparison_symbol_has_tracked_horizon=comparison_symbol_has_tracked_horizon,
        comparison_strategy_has_tracked_horizon=comparison_strategy_has_tracked_horizon,
        triplet_present_in_mapper=bool(mapper_triplet_candidate),
        triplet_visible_horizons=triplet_visible_horizons,
        triplet_stability_label=triplet_stability_label,
        triplet_candidate_strength=str(triplet_candidate_strength) if triplet_candidate_strength is not None else None,
        triplet_sample_count=triplet_sample_count,
        triplet_positive_rate_pct=triplet_positive_rate_pct,
        triplet_robustness_signal_pct=triplet_robustness_signal_pct,
        candidate_seed_count=candidate_seed_count,
        horizons_with_seed=horizons_with_seed,
        horizons_without_seed=horizons_without_seed,
        engine_selected_symbol=engine_output.get("selected_symbol"),
        engine_selected_strategy=engine_output.get("selected_strategy"),
        engine_selected_horizon=engine_output.get("selected_horizon"),
        engine_abstain_top_symbol=abstain_top_candidate.get("symbol") if isinstance(abstain_top_candidate, dict) else None,
        engine_abstain_top_strategy=abstain_top_candidate.get("strategy") if isinstance(abstain_top_candidate, dict) else None,
        engine_abstain_top_horizon=abstain_top_candidate.get("horizon") if isinstance(abstain_top_candidate, dict) else None,
        raw_output_path=raw_output_path_value if isinstance(raw_output_path_value, str) else None,
    )


def build_summary(
    rows: List[MapperTripletRow],
    tracked_symbol: str,
    tracked_strategy: str,
    tracked_horizon: str,
) -> Dict[str, Any]:
    present_rows = [row for row in rows if row.triplet_present_in_mapper]
    missing_rows = [row for row in rows if not row.triplet_present_in_mapper]

    return {
        "tracked_symbol": tracked_symbol,
        "tracked_strategy": tracked_strategy,
        "tracked_horizon": tracked_horizon,
        "row_count": len(rows),
        "triplet_present_count": len(present_rows),
        "triplet_missing_count": len(missing_rows),
        "present_windows": [row.end_record_index_inclusive for row in present_rows],
        "missing_windows": [row.end_record_index_inclusive for row in missing_rows],
        "comparison_joint_horizon_support_windows": [
            row.end_record_index_inclusive
            for row in rows
            if row.comparison_symbol_has_tracked_horizon and row.comparison_strategy_has_tracked_horizon
        ],
    }


def row_to_field_dict(row: MapperTripletRow) -> Dict[str, Any]:
    return row.to_dict()


def build_pairwise_comparisons(rows: List[MapperTripletRow]) -> List[Dict[str, Any]]:
    present_rows = [row for row in rows if row.triplet_present_in_mapper]
    missing_rows = [row for row in rows if not row.triplet_present_in_mapper]

    comparisons: List[Dict[str, Any]] = []
    for missing_row in missing_rows:
        if not present_rows:
            continue
        nearest_present = min(
            present_rows,
            key=lambda r: abs((r.end_record_index_inclusive or 0) - (missing_row.end_record_index_inclusive or 0)),
        )

        present_dict = row_to_field_dict(nearest_present)
        missing_dict = row_to_field_dict(missing_row)
        diffs: Dict[str, Dict[str, Any]] = {}

        for field in LOAD_BEARING_FIELDS:
            if present_dict.get(field) != missing_dict.get(field):
                diffs[field] = {
                    "present": present_dict.get(field),
                    "missing": missing_dict.get(field),
                }

        comparisons.append(
            {
                "present_window": nearest_present.end_record_index_inclusive,
                "missing_window": missing_row.end_record_index_inclusive,
                "field_differences": diffs,
            }
        )

    return comparisons


def build_explanations(summary: Dict[str, Any], pairwise_comparisons: List[Dict[str, Any]]) -> List[str]:
    explanations: List[str] = []

    explanations.append(
        "This report compares windows where the tracked mapper triplet is emitted versus windows where it is missing."
    )

    if summary["triplet_missing_count"] > 0:
        explanations.append(
            f"The tracked triplet is missing in {summary['triplet_missing_count']} windows even though comparison support remains available."
        )

    if pairwise_comparisons:
        first = pairwise_comparisons[0]
        explanations.append(
            f"The earliest nearest comparison is present window {first['present_window']} versus missing window {first['missing_window']}."
        )

    explanations.append(
        "If comparison support fields remain unchanged while mapper triplet presence flips, the decisive bottleneck is mapper triplet emission logic rather than comparison visibility."
    )

    return explanations


def render_markdown(report: Dict[str, Any]) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    rows = report["rows"]
    pairwise_comparisons = report["pairwise_comparisons"]
    explanations = report["explanations"]

    lines: List[str] = []
    lines.append("# Mapper Triplet Formation Diagnosis")
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

    lines.append("## Triplet Rows")
    lines.append("")
    header = [
        "window",
        "selection_status",
        "comparison_latest_top_symbol",
        "comparison_latest_top_strategy",
        "comparison_symbol_has_tracked_horizon",
        "comparison_strategy_has_tracked_horizon",
        "triplet_present_in_mapper",
        "triplet_visible_horizons",
        "triplet_stability_label",
        "triplet_candidate_strength",
        "triplet_sample_count",
        "triplet_positive_rate_pct",
        "triplet_robustness_signal_pct",
        "candidate_seed_count",
        "horizons_with_seed",
        "engine_selected_horizon",
        "engine_abstain_top_horizon",
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
                safe_cell(row.get("triplet_present_in_mapper")),
                safe_cell(row.get("triplet_visible_horizons")),
                safe_cell(row.get("triplet_stability_label")),
                safe_cell(row.get("triplet_candidate_strength")),
                safe_cell(row.get("triplet_sample_count")),
                safe_cell(row.get("triplet_positive_rate_pct")),
                safe_cell(row.get("triplet_robustness_signal_pct")),
                safe_cell(row.get("candidate_seed_count")),
                safe_cell(row.get("horizons_with_seed")),
                safe_cell(row.get("engine_selected_horizon")),
                safe_cell(row.get("engine_abstain_top_horizon")),
            ]) + " |"
        )

    lines.append("")
    lines.append("## Pairwise Present vs Missing Comparisons")
    lines.append("")
    if not pairwise_comparisons:
        lines.append("- No pairwise comparisons generated.")
    else:
        for comp in pairwise_comparisons:
            lines.append(f"### present {comp['present_window']} vs missing {comp['missing_window']}")
            lines.append("| field | present | missing |")
            lines.append("|---|---|---|")
            diffs = comp["field_differences"]
            if diffs:
                for field in LOAD_BEARING_FIELDS:
                    if field not in diffs:
                        continue
                    lines.append(
                        "| "
                        + " | ".join([
                            field,
                            safe_cell(diffs[field]["present"]),
                            safe_cell(diffs[field]["missing"]),
                        ])
                        + " |"
                    )
            else:
                lines.append("| none | identical | identical |")
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