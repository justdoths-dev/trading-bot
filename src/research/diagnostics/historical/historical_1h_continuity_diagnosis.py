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


@dataclass
class ContinuityRow:
    step_index: Optional[int]
    end_record_index_inclusive: Optional[int]
    selection_status: Optional[str]
    selection_reason: Optional[str]

    tracked_symbol: str
    tracked_strategy: str

    has_1h_seed: bool
    has_4h_seed: bool
    horizons_present_for_identity: List[str]

    selected_symbol: Optional[str]
    selected_strategy: Optional[str]
    selected_horizon: Optional[str]

    tracked_candidate_present: bool
    tracked_candidate_horizon: Optional[str]
    tracked_candidate_status: Optional[str]
    tracked_candidate_stability_label: Optional[str]
    tracked_candidate_visible_horizons: List[str]
    tracked_candidate_reason_codes: List[str]
    tracked_candidate_gate_diagnostics: Dict[str, Any]
    tracked_candidate_aggregate_score: Optional[float]
    tracked_candidate_sample_count: Optional[int]
    tracked_candidate_positive_rate_pct: Optional[float]
    tracked_candidate_robustness_signal_pct: Optional[float]

    candidate_seed_count: Optional[int]
    horizons_with_seed: List[str]
    horizons_without_seed: List[str]

    raw_output_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "end_record_index_inclusive": self.end_record_index_inclusive,
            "selection_status": self.selection_status,
            "selection_reason": self.selection_reason,
            "tracked_symbol": self.tracked_symbol,
            "tracked_strategy": self.tracked_strategy,
            "has_1h_seed": self.has_1h_seed,
            "has_4h_seed": self.has_4h_seed,
            "horizons_present_for_identity": self.horizons_present_for_identity,
            "selected_symbol": self.selected_symbol,
            "selected_strategy": self.selected_strategy,
            "selected_horizon": self.selected_horizon,
            "tracked_candidate_present": self.tracked_candidate_present,
            "tracked_candidate_horizon": self.tracked_candidate_horizon,
            "tracked_candidate_status": self.tracked_candidate_status,
            "tracked_candidate_stability_label": self.tracked_candidate_stability_label,
            "tracked_candidate_visible_horizons": self.tracked_candidate_visible_horizons,
            "tracked_candidate_reason_codes": self.tracked_candidate_reason_codes,
            "tracked_candidate_gate_diagnostics": self.tracked_candidate_gate_diagnostics,
            "tracked_candidate_aggregate_score": self.tracked_candidate_aggregate_score,
            "tracked_candidate_sample_count": self.tracked_candidate_sample_count,
            "tracked_candidate_positive_rate_pct": self.tracked_candidate_positive_rate_pct,
            "tracked_candidate_robustness_signal_pct": self.tracked_candidate_robustness_signal_pct,
            "candidate_seed_count": self.candidate_seed_count,
            "horizons_with_seed": self.horizons_with_seed,
            "horizons_without_seed": self.horizons_without_seed,
            "raw_output_path": self.raw_output_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace 1h continuity for a specific symbol/strategy identity across "
            "historical direct edge selection windows."
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
    filtered_rows = filter_rows_by_window(
        rows,
        start_window=args.start_window,
        end_window=args.end_window,
    )

    continuity_rows = [
        build_continuity_row(
            row=row,
            tracked_symbol=args.symbol,
            tracked_strategy=args.strategy,
        )
        for row in filtered_rows
    ]

    summary = build_summary(continuity_rows, args.symbol, args.strategy)
    transitions = build_transitions(continuity_rows)
    explanations = build_explanations(summary, transitions)

    report = {
        "metadata": {
            "run_dir": str(run_dir),
            "step_results_path": str(step_results_path),
            "tracked_symbol": args.symbol,
            "tracked_strategy": args.strategy,
            "start_window": args.start_window,
            "end_window": args.end_window,
            "row_count": len(continuity_rows),
        },
        "summary": summary,
        "transitions": transitions,
        "rows": [row.to_dict() for row in continuity_rows],
        "explanations": explanations,
    }

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_symbol = args.symbol.lower()
    safe_strategy = args.strategy.lower()

    json_path = output_dir / f"{safe_symbol}_{safe_strategy}_1h_continuity_diagnosis.json"
    md_path = output_dir / f"{safe_symbol}_{safe_strategy}_1h_continuity_diagnosis.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "row_count": len(continuity_rows),
            "summary": {
                "identity_present_count": summary["identity_present_count"],
                "has_1h_seed_count": summary["has_1h_seed_count"],
                "has_4h_seed_count": summary["has_4h_seed_count"],
                "selected_count": summary["selected_count"],
                "abstain_count": summary["abstain_count"],
                "first_1h_loss_window": summary["first_1h_loss_window"],
                "first_abstain_after_1h_loss_window": summary["first_abstain_after_1h_loss_window"],
            },
        },
        indent=2,
        ensure_ascii=False,
    ))

    if args.write_latest_copy:
        DEFAULT_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        latest_json = DEFAULT_LATEST_DIR / f"{safe_symbol}_{safe_strategy}_1h_continuity_diagnosis.json"
        latest_md = DEFAULT_LATEST_DIR / f"{safe_symbol}_{safe_strategy}_1h_continuity_diagnosis.md"
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


def horizon_sort_key(value: str) -> int:
    order = {"15m": 0, "1h": 1, "4h": 2}
    return order.get(value, 999)


def candidate_symbol(candidate: Dict[str, Any]) -> Optional[str]:
    return coalesce(
        candidate.get("symbol"),
        recursive_find_first(candidate, "symbol"),
    )


def candidate_strategy(candidate: Dict[str, Any]) -> Optional[str]:
    return coalesce(
        candidate.get("strategy"),
        recursive_find_first(candidate, "strategy"),
    )


def candidate_horizon(candidate: Dict[str, Any]) -> Optional[str]:
    value = coalesce(
        candidate.get("horizon"),
        recursive_find_first(candidate, "horizon"),
    )
    return str(value) if value is not None else None


def extract_engine_ranking(raw_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    engine_output = raw_payload.get("engine_output")
    if not isinstance(engine_output, dict):
        return []

    ranking = engine_output.get("ranking")
    if isinstance(ranking, list):
        return [item for item in ranking if isinstance(item, dict)]
    return []


def extract_mapper_candidates(raw_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    mapper_payload = raw_payload.get("mapper_payload")
    if not isinstance(mapper_payload, dict):
        return []

    candidates = mapper_payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def extract_abstain_top_candidate(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    engine_output = raw_payload.get("engine_output")
    if not isinstance(engine_output, dict):
        return {}

    abstain_diagnosis = engine_output.get("abstain_diagnosis")
    if not isinstance(abstain_diagnosis, dict):
        return {}

    top_candidate = abstain_diagnosis.get("top_candidate")
    if isinstance(top_candidate, dict):
        return top_candidate
    return {}


def extract_selected_candidate(raw_payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    engine_output = raw_payload.get("engine_output")
    if isinstance(engine_output, dict):
        ranking = engine_output.get("ranking")
        if isinstance(ranking, list):
            row_symbol = row.get("selected_symbol") or engine_output.get("selected_symbol")
            row_strategy = row.get("selected_strategy") or engine_output.get("selected_strategy")
            row_horizon = row.get("selected_horizon") or engine_output.get("selected_horizon")

            for candidate in ranking:
                if not isinstance(candidate, dict):
                    continue
                if (
                    candidate_symbol(candidate) == row_symbol
                    and candidate_strategy(candidate) == row_strategy
                    and candidate_horizon(candidate) == row_horizon
                ):
                    return candidate

            if ranking and isinstance(ranking[0], dict):
                return ranking[0]

    return {}


def find_candidates_for_identity(
    candidates: List[Dict[str, Any]],
    symbol: str,
    strategy: str,
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate_symbol(candidate) == symbol and candidate_strategy(candidate) == strategy:
            matched.append(candidate)
    return matched


def choose_primary_engine_candidate(
    engine_candidates: List[Dict[str, Any]],
    abstain_top_candidate: Dict[str, Any],
    selected_candidate: Dict[str, Any],
    tracked_symbol: str,
    tracked_strategy: str,
) -> Dict[str, Any]:
    if selected_candidate:
        if candidate_symbol(selected_candidate) == tracked_symbol and candidate_strategy(selected_candidate) == tracked_strategy:
            return selected_candidate

    matched_engine = find_candidates_for_identity(engine_candidates, tracked_symbol, tracked_strategy)
    if matched_engine:
        for preferred_horizon in ["4h", "1h", "15m"]:
            for candidate in matched_engine:
                if candidate_horizon(candidate) == preferred_horizon:
                    return candidate
        return matched_engine[0]

    if abstain_top_candidate:
        if candidate_symbol(abstain_top_candidate) == tracked_symbol and candidate_strategy(abstain_top_candidate) == tracked_strategy:
            return abstain_top_candidate

    return {}


def choose_primary_mapper_candidate(
    mapper_candidates: List[Dict[str, Any]],
    tracked_symbol: str,
    tracked_strategy: str,
) -> Dict[str, Any]:
    matched = find_candidates_for_identity(mapper_candidates, tracked_symbol, tracked_strategy)
    if not matched:
        return {}

    for preferred_horizon in ["4h", "1h", "15m"]:
        for candidate in matched:
            if candidate_horizon(candidate) == preferred_horizon:
                return candidate
    return matched[0]


def build_continuity_row(
    row: Dict[str, Any],
    tracked_symbol: str,
    tracked_strategy: str,
) -> ContinuityRow:
    raw_output_path_value = row.get("raw_output_path")
    raw_payload: Dict[str, Any] = {}
    if isinstance(raw_output_path_value, str) and raw_output_path_value:
        raw_payload = load_json_file(Path(raw_output_path_value))

    mapper_candidates = extract_mapper_candidates(raw_payload)
    engine_ranking = extract_engine_ranking(raw_payload)
    abstain_top_candidate = extract_abstain_top_candidate(raw_payload)
    selected_candidate = extract_selected_candidate(raw_payload, row)

    matched_mapper_candidates = find_candidates_for_identity(
        mapper_candidates,
        symbol=tracked_symbol,
        strategy=tracked_strategy,
    )

    mapper_primary_candidate = choose_primary_mapper_candidate(
        mapper_candidates=mapper_candidates,
        tracked_symbol=tracked_symbol,
        tracked_strategy=tracked_strategy,
    )

    primary_engine_candidate = choose_primary_engine_candidate(
        engine_candidates=engine_ranking,
        abstain_top_candidate=abstain_top_candidate,
        selected_candidate=selected_candidate,
        tracked_symbol=tracked_symbol,
        tracked_strategy=tracked_strategy,
    )

    horizons_present_for_identity = sorted(
        {
            candidate_horizon(candidate)
            for candidate in matched_mapper_candidates
            if candidate_horizon(candidate) is not None
        },
        key=lambda value: horizon_sort_key(value),
    )

    has_1h_seed = "1h" in horizons_present_for_identity
    has_4h_seed = "4h" in horizons_present_for_identity

    tracked_candidate_present = bool(matched_mapper_candidates or primary_engine_candidate)

    tracked_candidate_horizon = candidate_horizon(primary_engine_candidate) or candidate_horizon(mapper_primary_candidate)

    tracked_candidate_status = coalesce(
        primary_engine_candidate.get("candidate_status"),
        primary_engine_candidate.get("status"),
        recursive_find_first(primary_engine_candidate, "candidate_status"),
    )

    tracked_candidate_stability_label = coalesce(
        primary_engine_candidate.get("selected_stability_label"),
        primary_engine_candidate.get("stability_label"),
        recursive_find_first(primary_engine_candidate, "selected_stability_label"),
        recursive_find_first(primary_engine_candidate, "stability_label"),
        mapper_primary_candidate.get("selected_stability_label"),
        mapper_primary_candidate.get("stability_label"),
        recursive_find_first(mapper_primary_candidate, "selected_stability_label"),
        recursive_find_first(mapper_primary_candidate, "stability_label"),
    )

    tracked_candidate_visible_horizons = normalize_string_list(coalesce(
        primary_engine_candidate.get("selected_visible_horizons"),
        primary_engine_candidate.get("visible_horizons"),
        recursive_find_first(primary_engine_candidate, "selected_visible_horizons"),
        recursive_find_first(primary_engine_candidate, "visible_horizons"),
        mapper_primary_candidate.get("selected_visible_horizons"),
        mapper_primary_candidate.get("visible_horizons"),
        recursive_find_first(mapper_primary_candidate, "selected_visible_horizons"),
        recursive_find_first(mapper_primary_candidate, "visible_horizons"),
    ))

    tracked_candidate_reason_codes = normalize_reason_codes(coalesce(
        primary_engine_candidate.get("reason_codes"),
        recursive_find_first(primary_engine_candidate, "reason_codes"),
        abstain_top_candidate.get("reason_codes"),
        recursive_find_first(abstain_top_candidate, "reason_codes"),
    ))

    tracked_candidate_gate_diagnostics = normalize_gate_diagnostics(coalesce(
        primary_engine_candidate.get("gate_diagnostics"),
        recursive_find_first(primary_engine_candidate, "gate_diagnostics"),
        abstain_top_candidate.get("gate_diagnostics"),
        recursive_find_first(abstain_top_candidate, "gate_diagnostics"),
    ))

    tracked_candidate_aggregate_score = to_float(coalesce(
        primary_engine_candidate.get("aggregate_score"),
        recursive_find_first(primary_engine_candidate, "aggregate_score"),
        abstain_top_candidate.get("aggregate_score"),
        recursive_find_first(abstain_top_candidate, "aggregate_score"),
    ))

    tracked_candidate_sample_count = to_int(coalesce(
        primary_engine_candidate.get("sample_count"),
        recursive_find_first(primary_engine_candidate, "sample_count"),
        abstain_top_candidate.get("sample_count"),
        recursive_find_first(abstain_top_candidate, "sample_count"),
    ))

    tracked_candidate_positive_rate_pct = to_float(coalesce(
        primary_engine_candidate.get("positive_rate_pct"),
        recursive_find_first(primary_engine_candidate, "positive_rate_pct"),
        abstain_top_candidate.get("positive_rate_pct"),
        recursive_find_first(abstain_top_candidate, "positive_rate_pct"),
    ))

    tracked_candidate_robustness_signal_pct = to_float(coalesce(
        primary_engine_candidate.get("robustness_signal_pct"),
        recursive_find_first(primary_engine_candidate, "robustness_signal_pct"),
        abstain_top_candidate.get("robustness_signal_pct"),
        recursive_find_first(abstain_top_candidate, "robustness_signal_pct"),
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

    engine_output = raw_payload.get("engine_output")
    selected_symbol = coalesce(
        row.get("selected_symbol"),
        engine_output.get("selected_symbol") if isinstance(engine_output, dict) else None,
        candidate_symbol(selected_candidate),
    )
    selected_strategy = coalesce(
        row.get("selected_strategy"),
        engine_output.get("selected_strategy") if isinstance(engine_output, dict) else None,
        candidate_strategy(selected_candidate),
    )
    selected_horizon = coalesce(
        row.get("selected_horizon"),
        engine_output.get("selected_horizon") if isinstance(engine_output, dict) else None,
        candidate_horizon(selected_candidate),
    )

    return ContinuityRow(
        step_index=to_int(row.get("step_index")),
        end_record_index_inclusive=to_int(row.get("end_record_index_inclusive")),
        selection_status=coalesce(
            row.get("selection_status"),
            recursive_find_first(raw_payload, "selection_status"),
        ),
        selection_reason=coalesce(
            row.get("reason"),
            recursive_find_first(raw_payload, "reason"),
        ),
        tracked_symbol=tracked_symbol,
        tracked_strategy=tracked_strategy,
        has_1h_seed=has_1h_seed,
        has_4h_seed=has_4h_seed,
        horizons_present_for_identity=horizons_present_for_identity,
        selected_symbol=selected_symbol,
        selected_strategy=selected_strategy,
        selected_horizon=selected_horizon,
        tracked_candidate_present=tracked_candidate_present,
        tracked_candidate_horizon=tracked_candidate_horizon,
        tracked_candidate_status=tracked_candidate_status,
        tracked_candidate_stability_label=tracked_candidate_stability_label,
        tracked_candidate_visible_horizons=tracked_candidate_visible_horizons,
        tracked_candidate_reason_codes=tracked_candidate_reason_codes,
        tracked_candidate_gate_diagnostics=tracked_candidate_gate_diagnostics,
        tracked_candidate_aggregate_score=tracked_candidate_aggregate_score,
        tracked_candidate_sample_count=tracked_candidate_sample_count,
        tracked_candidate_positive_rate_pct=tracked_candidate_positive_rate_pct,
        tracked_candidate_robustness_signal_pct=tracked_candidate_robustness_signal_pct,
        candidate_seed_count=candidate_seed_count,
        horizons_with_seed=horizons_with_seed,
        horizons_without_seed=horizons_without_seed,
        raw_output_path=raw_output_path_value if isinstance(raw_output_path_value, str) else None,
    )


def build_summary(
    rows: List[ContinuityRow],
    tracked_symbol: str,
    tracked_strategy: str,
) -> Dict[str, Any]:
    identity_present_rows = [row for row in rows if row.tracked_candidate_present]
    has_1h_rows = [row for row in rows if row.has_1h_seed]
    no_1h_rows = [row for row in rows if not row.has_1h_seed and row.has_4h_seed]
    selected_rows = [row for row in rows if row.selection_status == "selected"]
    abstain_rows = [row for row in rows if row.selection_status != "selected"]

    first_1h_loss_window = None
    first_abstain_after_1h_loss_window = None

    prev_has_1h: Optional[bool] = None
    loss_detected = False

    for row in rows:
        if prev_has_1h is True and row.has_1h_seed is False and row.has_4h_seed is True:
            first_1h_loss_window = row.end_record_index_inclusive
            loss_detected = True
            break
        prev_has_1h = row.has_1h_seed

    if loss_detected:
        for row in rows:
            if (
                first_1h_loss_window is not None
                and row.end_record_index_inclusive is not None
                and row.end_record_index_inclusive >= first_1h_loss_window
                and row.selection_status != "selected"
            ):
                first_abstain_after_1h_loss_window = row.end_record_index_inclusive
                break

    return {
        "tracked_symbol": tracked_symbol,
        "tracked_strategy": tracked_strategy,
        "row_count": len(rows),
        "identity_present_count": len(identity_present_rows),
        "has_1h_seed_count": len(has_1h_rows),
        "has_4h_seed_count": len([row for row in rows if row.has_4h_seed]),
        "has_4h_without_1h_count": len(no_1h_rows),
        "selected_count": len(selected_rows),
        "abstain_count": len(abstain_rows),
        "first_window": rows[0].end_record_index_inclusive if rows else None,
        "last_window": rows[-1].end_record_index_inclusive if rows else None,
        "first_1h_loss_window": first_1h_loss_window,
        "first_abstain_after_1h_loss_window": first_abstain_after_1h_loss_window,
        "selected_windows": [row.end_record_index_inclusive for row in selected_rows],
        "abstain_windows": [row.end_record_index_inclusive for row in abstain_rows],
        "windows_with_1h_seed": [row.end_record_index_inclusive for row in has_1h_rows],
        "windows_with_4h_without_1h": [row.end_record_index_inclusive for row in no_1h_rows],
    }


def build_transitions(rows: List[ContinuityRow]) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        previous_state = continuity_state(previous)
        current_state = continuity_state(current)
        if previous_state != current_state:
            transitions.append(
                {
                    "from_window": previous.end_record_index_inclusive,
                    "to_window": current.end_record_index_inclusive,
                    "from_state": previous_state,
                    "to_state": current_state,
                    "selection_change": {
                        "from_status": previous.selection_status,
                        "to_status": current.selection_status,
                        "from_reason": previous.selection_reason,
                        "to_reason": current.selection_reason,
                    },
                    "tracked_candidate_change": {
                        "from_status": previous.tracked_candidate_status,
                        "to_status": current.tracked_candidate_status,
                        "from_stability_label": previous.tracked_candidate_stability_label,
                        "to_stability_label": current.tracked_candidate_stability_label,
                        "from_visible_horizons": previous.tracked_candidate_visible_horizons,
                        "to_visible_horizons": current.tracked_candidate_visible_horizons,
                        "from_reason_codes": previous.tracked_candidate_reason_codes,
                        "to_reason_codes": current.tracked_candidate_reason_codes,
                    },
                }
            )
    return transitions


def continuity_state(row: ContinuityRow) -> str:
    if row.has_1h_seed and row.has_4h_seed:
        return "1h_and_4h_present"
    if (not row.has_1h_seed) and row.has_4h_seed:
        return "4h_only"
    if row.has_1h_seed and (not row.has_4h_seed):
        return "1h_only"
    return "identity_not_present"


def build_explanations(summary: Dict[str, Any], transitions: List[Dict[str, Any]]) -> List[str]:
    explanations: List[str] = []

    explanations.append(
        "This report tracks whether the target identity keeps both 1h and 4h support across consecutive windows."
    )

    if summary["has_4h_without_1h_count"] > 0:
        explanations.append(
            f"The tracked identity falls into 4h-only mode {summary['has_4h_without_1h_count']} times. "
            "These are the key windows where multi-horizon confirmation can be lost."
        )

    if summary["first_1h_loss_window"] is not None:
        explanations.append(
            f"The first detected 1h continuity loss occurs at window {summary['first_1h_loss_window']}."
        )

    if summary["first_abstain_after_1h_loss_window"] is not None:
        explanations.append(
            f"The first abstain observed after 1h loss occurs at window {summary['first_abstain_after_1h_loss_window']}."
        )

    if transitions:
        first_transition = transitions[0]
        explanations.append(
            "The earliest continuity state change is "
            f"{first_transition['from_state']} -> {first_transition['to_state']} "
            f"between windows {first_transition['from_window']} and {first_transition['to_window']}."
        )

    explanations.append(
        "If abstain begins only after the state changes from 1h_and_4h_present to 4h_only, "
        "then the load-bearing failure remains 1h continuity loss rather than score weakness."
    )

    return explanations


def render_markdown(report: Dict[str, Any]) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    transitions = report["transitions"]
    rows = report["rows"]
    explanations = report["explanations"]

    lines: List[str] = []
    lines.append("# 1h Continuity Diagnosis")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- run_dir: `{metadata['run_dir']}`")
    lines.append(f"- step_results_path: `{metadata['step_results_path']}`")
    lines.append(f"- tracked_symbol: `{metadata['tracked_symbol']}`")
    lines.append(f"- tracked_strategy: `{metadata['tracked_strategy']}`")
    lines.append(f"- start_window: `{metadata['start_window']}`")
    lines.append(f"- end_window: `{metadata['end_window']}`")
    lines.append(f"- row_count: `{metadata['row_count']}`")
    lines.append("")

    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## Continuity Rows")
    lines.append("")
    header = [
        "window",
        "selection_status",
        "selection_reason",
        "has_1h_seed",
        "has_4h_seed",
        "horizons_present_for_identity",
        "tracked_candidate_present",
        "tracked_candidate_horizon",
        "tracked_candidate_status",
        "tracked_candidate_stability_label",
        "tracked_candidate_visible_horizons",
        "tracked_candidate_aggregate_score",
        "tracked_candidate_sample_count",
        "tracked_candidate_positive_rate_pct",
        "tracked_candidate_robustness_signal_pct",
        "tracked_candidate_reason_codes",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in rows:
        lines.append(
            "| " + " | ".join([
                safe_cell(row.get("end_record_index_inclusive")),
                safe_cell(row.get("selection_status")),
                safe_cell(row.get("selection_reason")),
                safe_cell(row.get("has_1h_seed")),
                safe_cell(row.get("has_4h_seed")),
                safe_cell(row.get("horizons_present_for_identity")),
                safe_cell(row.get("tracked_candidate_present")),
                safe_cell(row.get("tracked_candidate_horizon")),
                safe_cell(row.get("tracked_candidate_status")),
                safe_cell(row.get("tracked_candidate_stability_label")),
                safe_cell(row.get("tracked_candidate_visible_horizons")),
                safe_cell(row.get("tracked_candidate_aggregate_score")),
                safe_cell(row.get("tracked_candidate_sample_count")),
                safe_cell(row.get("tracked_candidate_positive_rate_pct")),
                safe_cell(row.get("tracked_candidate_robustness_signal_pct")),
                safe_cell(row.get("tracked_candidate_reason_codes")),
            ]) + " |"
        )

    lines.append("")
    lines.append("## State Transitions")
    lines.append("")
    if not transitions:
        lines.append("- No continuity state transitions detected.")
    else:
        for item in transitions:
            lines.append(f"### {item['from_window']} -> {item['to_window']}")
            lines.append(f"- from_state: `{item['from_state']}`")
            lines.append(f"- to_state: `{item['to_state']}`")
            lines.append(f"- selection_change: `{item['selection_change']}`")
            lines.append(f"- tracked_candidate_change: `{item['tracked_candidate_change']}`")
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