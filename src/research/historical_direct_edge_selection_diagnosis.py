from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


RUNNER_VERSION = "historical_direct_edge_selection_diagnosis_v4"


@dataclass
class StepResult:
    step_index: int
    window_record_count: int
    end_record_index_inclusive: int
    selection_status: str
    reason: str
    selected_symbol: Optional[str]
    selected_strategy: Optional[str]
    selected_horizon: Optional[str]
    candidate_seed_count: Optional[int]
    horizons_with_seed: List[str]
    horizons_without_seed: List[str]
    ranking_count: Optional[int]
    raw_output_path: str
    error: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Non-object JSON at line {line_no} in {path}")
            records.append(obj)
    return records


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "src").exists() and (candidate / "logs").exists():
            return candidate
    raise RuntimeError("Could not detect repository root containing both 'src/' and 'logs/'.")


@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def import_first_available(candidates: List[Tuple[str, str]]) -> Tuple[str, str, Callable[..., Any]]:
    errors: List[str] = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, attr_name)
            if callable(func):
                return module_name, attr_name, func
            errors.append(f"{module_name}.{attr_name} exists but is not callable")
        except Exception as exc:
            errors.append(f"{module_name}.{attr_name} -> {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to import any candidate callable.\n{joined}")


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    sig = inspect.signature(func)
    accepted = {}
    for name in sig.parameters.keys():
        if name in kwargs:
            accepted[name] = kwargs[name]
    return func(**accepted)


def resolve_mapper_callable() -> Tuple[str, str, Callable[..., Any]]:
    candidates = [
        ("src.research.edge_selection_input_mapper", "map_edge_selection_input"),
        ("src.research.edge_selection.input_mapper", "map_edge_selection_input"),
        ("src.research.edge_selection_mapper", "map_edge_selection_input"),
    ]
    return import_first_available(candidates)


def resolve_engine_callable() -> Tuple[str, str, Callable[..., Any]]:
    candidates = [
        ("src.research.edge_selection_engine", "run_edge_selection_engine"),
        ("src.research.edge_selection_engine", "run_edge_selection"),
        ("src.research.edge_selection.engine", "run_edge_selection_engine"),
        ("src.research.edge_selection.engine", "run_edge_selection"),
    ]
    return import_first_available(candidates)


def resolve_comparison_pipeline_callable() -> Tuple[str, str, Callable[..., Any]]:
    candidates = [
        ("src.research.run_comparison_pipeline", "run_comparison_pipeline"),
    ]
    return import_first_available(candidates)


def run_research_analyzer_cli(
    repo_root: Path,
    workspace_root: Path,
    python_executable: str,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"

    cmd = [python_executable, "-m", "src.research.research_analyzer"]
    return subprocess.run(
        cmd,
        cwd=str(workspace_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_comparison_pipeline_step(
    comparison_pipeline_func: Callable[..., Any],
    workspace_root: Path,
) -> Any:
    logs_dir = workspace_root / "logs"
    reports_dir = logs_dir / "research_reports"

    latest_summary = reports_dir / "latest" / "summary.json"
    cumulative_output = logs_dir / "trade_analysis_cumulative.jsonl"
    cumulative_output_dir = reports_dir / "cumulative"
    comparison_output_dir = reports_dir / "comparison"
    edge_scores_output_dir = reports_dir / "edge_scores"
    edge_score_history_output = reports_dir / "edge_scores_history.jsonl"
    score_drift_output_dir = reports_dir / "score_drift"

    return comparison_pipeline_func(
        logs_dir=logs_dir,
        latest_summary=latest_summary,
        cumulative_output=cumulative_output,
        cumulative_output_dir=cumulative_output_dir,
        comparison_output_dir=comparison_output_dir,
        edge_scores_output_dir=edge_scores_output_dir,
        edge_score_history_output=edge_score_history_output,
        score_drift_output_dir=score_drift_output_dir,
    )


def run_mapper(
    mapper_func: Callable[..., Any],
    workspace_root: Path,
) -> Any:
    reports_base_dir = workspace_root / "logs" / "research_reports"

    attempts: List[Tuple[str, Callable[[], Any]]] = [
        ("kwargs_reports_root_path", lambda: call_with_supported_kwargs(mapper_func, base_dir=reports_base_dir)),
        ("kwargs_reports_root_str", lambda: call_with_supported_kwargs(mapper_func, base_dir=str(reports_base_dir))),
        ("positional_reports_root_path", lambda: mapper_func(reports_base_dir)),
        ("positional_reports_root_str", lambda: mapper_func(str(reports_base_dir))),
    ]

    errors: List[str] = []
    for label, runner in attempts:
        try:
            return runner()
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise RuntimeError(f"Mapper execution failed after all attempts: {' | '.join(errors)}")


def run_engine(
    engine_func: Callable[..., Any],
    mapper_payload: Any,
) -> Any:
    attempts = [
        {"mapper_payload": mapper_payload},
        {"edge_selection_input": mapper_payload},
        {"selection_input": mapper_payload},
        {"input_payload": mapper_payload},
        {"payload": mapper_payload},
        {"candidate_payload": mapper_payload},
    ]

    errors: List[str] = []

    for kwargs in attempts:
        try:
            return call_with_supported_kwargs(engine_func, **kwargs)
        except Exception as exc:
            errors.append(f"kwargs={list(kwargs.keys())}: {exc}")

    try:
        return engine_func(mapper_payload)
    except Exception as exc:
        errors.append(f"positional_payload: {exc}")

    raise RuntimeError(f"Engine execution failed after all attempts: {' | '.join(errors)}")


def to_plain_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_data(v) for v in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return to_plain_data(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return to_plain_data(value.dict())
    if hasattr(value, "__dict__"):
        try:
            return to_plain_data(vars(value))
        except Exception:
            pass
    return value


def extract_candidate_seed_info(mapper_payload: Any) -> Tuple[Optional[int], List[str], List[str]]:
    payload = to_plain_data(mapper_payload)
    if not isinstance(payload, dict):
        return None, [], []

    candidate_seed_count = safe_int(payload.get("candidate_seed_count"))

    diagnostics = payload.get("candidate_seed_diagnostics") or {}
    horizons_with_seed = normalize_list(
        payload.get("horizons_with_seed")
        or payload.get("seeded_horizons")
        or payload.get("visible_horizons")
        or diagnostics.get("horizons_with_seed")
    )
    horizons_without_seed = normalize_list(
        payload.get("horizons_without_seed")
        or payload.get("missing_seed_horizons")
        or diagnostics.get("horizons_without_seed")
    )

    return candidate_seed_count, horizons_with_seed, horizons_without_seed


def extract_selection_fields(engine_output: Any) -> Dict[str, Any]:
    data = to_plain_data(engine_output)
    if not isinstance(data, dict):
        return {
            "selection_status": "unknown",
            "reason": "non_dict_output",
            "selected_symbol": None,
            "selected_strategy": None,
            "selected_horizon": None,
            "ranking_count": None,
        }

    selection_status = (
        data.get("selection_status")
        or data.get("status")
        or data.get("result")
        or "unknown"
    )

    reason_codes = data.get("reason_codes")
    selection_explanation = data.get("selection_explanation")
    reason = (
        data.get("reason")
        or data.get("abstain_reason")
        or data.get("selection_reason")
        or (reason_codes[0] if isinstance(reason_codes, list) and reason_codes else None)
        or selection_explanation
        or "unknown"
    )

    selected_symbol = data.get("selected_symbol")
    selected_strategy = data.get("selected_strategy")
    selected_horizon = data.get("selected_horizon")

    selected_block = data.get("selected_candidate") or data.get("selected") or data.get("winner")
    if isinstance(selected_block, dict):
        selected_symbol = selected_symbol or selected_block.get("symbol")
        selected_strategy = selected_strategy or selected_block.get("strategy")
        selected_horizon = selected_horizon or selected_block.get("horizon") or selected_block.get("timeframe")

    if selected_symbol is None:
        ranking = data.get("ranking") or data.get("ranked_candidates") or []
        if isinstance(ranking, list) and ranking:
            top = ranking[0]
            if isinstance(top, dict) and selection_status == "selected":
                selected_symbol = top.get("symbol")
                selected_strategy = top.get("strategy")
                selected_horizon = top.get("horizon") or top.get("timeframe")

    ranking_count = None
    ranking = data.get("ranking") or data.get("ranked_candidates")
    if isinstance(ranking, list):
        ranking_count = len(ranking)

    return {
        "selection_status": str(selection_status),
        "reason": str(reason),
        "selected_symbol": selected_symbol,
        "selected_strategy": selected_strategy,
        "selected_horizon": selected_horizon,
        "ranking_count": ranking_count,
    }


def reset_workspace(workspace_root: Path) -> None:
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    ensure_dir(workspace_root / "logs")
    ensure_dir(workspace_root / "logs" / "research_reports" / "latest")


def seed_workspace_input(workspace_root: Path, window_rows: List[dict]) -> Path:
    input_path = workspace_root / "logs" / "trade_analysis.jsonl"
    write_jsonl(input_path, window_rows)
    return input_path


def build_markdown_summary(
    run_id: str,
    source_path: Path,
    output_dir: Path,
    total_records: int,
    total_steps: int,
    selected_count: int,
    abstain_count: int,
    blocked_count: int,
    error_count: int,
    selection_rate: float,
    status_counts: Dict[str, int],
    top_selected_candidates: List[Tuple[str, int]],
    step_results: List[StepResult],
) -> str:
    lines: List[str] = []
    lines.append("# Historical Direct Edge Selection Diagnosis")
    lines.append("")
    lines.append("## Run")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- generated_at: {utc_now_iso()}")
    lines.append(f"- runner_version: `{RUNNER_VERSION}`")
    lines.append(f"- source_path: `{source_path}`")
    lines.append(f"- output_dir: `{output_dir}`")
    lines.append("")
    lines.append("## Totals")
    lines.append(f"- total_records: {total_records}")
    lines.append(f"- total_steps: {total_steps}")
    lines.append(f"- selected_count: {selected_count}")
    lines.append(f"- abstain_count: {abstain_count}")
    lines.append(f"- blocked_count: {blocked_count}")
    lines.append(f"- error_count: {error_count}")
    lines.append(f"- selection_rate: {selection_rate:.4f}")
    lines.append("")
    lines.append("## Status Counts")
    for key, value in sorted(status_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Top Selected Candidates")
    if top_selected_candidates:
        for name, count in top_selected_candidates[:10]:
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Recent Steps")
    for item in step_results[-10:]:
        candidate = " / ".join([x for x in [item.selected_symbol, item.selected_strategy, item.selected_horizon] if x]) or "n/a"
        lines.append(
            f"- step={item.step_index}, window={item.window_record_count}, "
            f"status={item.selection_status}, reason={item.reason}, selected={candidate}, "
            f"seeds={item.candidate_seed_count}, with_seed={item.horizons_with_seed}, "
            f"without_seed={item.horizons_without_seed}, error={item.error or 'none'}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct historical edge selection diagnosis without replay-ready logs.")
    parser.add_argument(
        "--input",
        default="logs/research_reports/latest/resolved_historical_input.jsonl",
        help="Historical JSONL input path (default: logs/research_reports/latest/resolved_historical_input.jsonl)",
    )
    parser.add_argument(
        "--warmup-records",
        type=int,
        default=400,
        help="Minimum cumulative record count before first evaluation step.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=50,
        help="Evaluate every N additional records.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional cap on step count. 0 means no cap.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to invoke the research analyzer CLI.",
    )
    args = parser.parse_args()

    repo_root = detect_repo_root()
    source_path = (repo_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    all_rows = read_jsonl(source_path)
    total_records = len(all_rows)
    if total_records == 0:
        raise RuntimeError(f"No records found in input: {source_path}")

    if args.warmup_records >= total_records:
        raise RuntimeError(
            f"--warmup-records ({args.warmup_records}) must be smaller than total record count ({total_records})."
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = repo_root / "logs" / "research_reports" / "historical_direct_edge_selection" / run_id
    raw_steps_dir = output_dir / "raw_steps"
    workspace_root = output_dir / "_workspace"

    ensure_dir(output_dir)
    ensure_dir(raw_steps_dir)

    mapper_module, mapper_attr, mapper_func = resolve_mapper_callable()
    engine_module, engine_attr, engine_func = resolve_engine_callable()
    comparison_module, comparison_attr, comparison_pipeline_func = resolve_comparison_pipeline_callable()

    step_results: List[StepResult] = []
    selected_counter: Dict[str, int] = {}
    selected_count = 0
    abstain_count = 0
    blocked_count = 0
    error_count = 0
    status_counts: Dict[str, int] = {}

    planned_indices = list(range(args.warmup_records, total_records + 1, args.step_size))
    if planned_indices[-1] != total_records:
        planned_indices.append(total_records)

    if args.max_steps > 0:
        planned_indices = planned_indices[:args.max_steps]

    metadata = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "runner_version": RUNNER_VERSION,
        "repo_root": str(repo_root),
        "input_path": str(source_path),
        "warmup_records": args.warmup_records,
        "step_size": args.step_size,
        "max_steps": args.max_steps,
        "python_executable": args.python_executable,
        "imports": {
            "mapper": f"{mapper_module}.{mapper_attr}",
            "engine": f"{engine_module}.{engine_attr}",
            "comparison_pipeline": f"{comparison_module}.{comparison_attr}",
        },
        "planned_steps": len(planned_indices),
        "total_records": total_records,
    }
    write_json(output_dir / "run_metadata.json", metadata)

    for step_index, end_exclusive in enumerate(planned_indices, start=1):
        window_rows = all_rows[:end_exclusive]
        window_count = len(window_rows)

        reset_workspace(workspace_root)
        seed_workspace_input(workspace_root, window_rows)

        analyzer_proc = run_research_analyzer_cli(
            repo_root=repo_root,
            workspace_root=workspace_root,
            python_executable=args.python_executable,
        )

        raw_output_path = raw_steps_dir / f"step_{step_index:05d}.json"

        if analyzer_proc.returncode != 0:
            error_count += 1
            status_counts["error"] = status_counts.get("error", 0) + 1
            result = StepResult(
                step_index=step_index,
                window_record_count=window_count,
                end_record_index_inclusive=end_exclusive - 1,
                selection_status="error",
                reason="research_analyzer_failed",
                selected_symbol=None,
                selected_strategy=None,
                selected_horizon=None,
                candidate_seed_count=None,
                horizons_with_seed=[],
                horizons_without_seed=[],
                ranking_count=None,
                raw_output_path=str(raw_output_path),
                error=analyzer_proc.stderr[-4000:] if analyzer_proc.stderr else "research_analyzer_failed",
            )
            write_json(
                raw_output_path,
                {
                    "step_index": step_index,
                    "window_record_count": window_count,
                    "research_analyzer": {
                        "returncode": analyzer_proc.returncode,
                        "stdout_tail": analyzer_proc.stdout[-4000:] if analyzer_proc.stdout else "",
                        "stderr_tail": analyzer_proc.stderr[-4000:] if analyzer_proc.stderr else "",
                    },
                    "result": asdict(result),
                },
            )
            step_results.append(result)
            continue

        try:
            with pushd(workspace_root):
                comparison_pipeline_output = run_comparison_pipeline_step(
                    comparison_pipeline_func=comparison_pipeline_func,
                    workspace_root=workspace_root,
                )
                mapper_payload = run_mapper(
                    mapper_func=mapper_func,
                    workspace_root=workspace_root,
                )
                engine_output = run_engine(engine_func, mapper_payload)

            candidate_seed_count, horizons_with_seed, horizons_without_seed = extract_candidate_seed_info(mapper_payload)
            selection_fields = extract_selection_fields(engine_output)

            status = selection_fields["selection_status"]
            reason = selection_fields["reason"]

            status_counts[status] = status_counts.get(status, 0) + 1

            if status == "selected":
                selected_count += 1
                candidate_name = " / ".join(
                    [x for x in [
                        selection_fields["selected_symbol"],
                        selection_fields["selected_strategy"],
                        selection_fields["selected_horizon"],
                    ] if x]
                ) or "unknown"
                selected_counter[candidate_name] = selected_counter.get(candidate_name, 0) + 1
            elif status == "abstain":
                abstain_count += 1
            elif status == "blocked":
                blocked_count += 1

            result = StepResult(
                step_index=step_index,
                window_record_count=window_count,
                end_record_index_inclusive=end_exclusive - 1,
                selection_status=status,
                reason=reason,
                selected_symbol=selection_fields["selected_symbol"],
                selected_strategy=selection_fields["selected_strategy"],
                selected_horizon=selection_fields["selected_horizon"],
                candidate_seed_count=candidate_seed_count,
                horizons_with_seed=horizons_with_seed,
                horizons_without_seed=horizons_without_seed,
                ranking_count=selection_fields["ranking_count"],
                raw_output_path=str(raw_output_path),
                error=None,
            )

            write_json(
                raw_output_path,
                {
                    "step_index": step_index,
                    "window_record_count": window_count,
                    "comparison_pipeline_output": to_plain_data(comparison_pipeline_output),
                    "mapper_payload": to_plain_data(mapper_payload),
                    "engine_output": to_plain_data(engine_output),
                    "result": asdict(result),
                },
            )
            step_results.append(result)

        except Exception as exc:
            error_count += 1
            status_counts["error"] = status_counts.get("error", 0) + 1
            tb = traceback.format_exc()
            result = StepResult(
                step_index=step_index,
                window_record_count=window_count,
                end_record_index_inclusive=end_exclusive - 1,
                selection_status="error",
                reason="comparison_mapper_or_engine_failed",
                selected_symbol=None,
                selected_strategy=None,
                selected_horizon=None,
                candidate_seed_count=None,
                horizons_with_seed=[],
                horizons_without_seed=[],
                ranking_count=None,
                raw_output_path=str(raw_output_path),
                error=str(exc),
            )
            write_json(
                raw_output_path,
                {
                    "step_index": step_index,
                    "window_record_count": window_count,
                    "error": str(exc),
                    "traceback": tb,
                    "result": asdict(result),
                },
            )
            step_results.append(result)

    total_steps = len(step_results)
    selection_rate = (selected_count / total_steps) if total_steps else 0.0
    top_selected_candidates = sorted(selected_counter.items(), key=lambda x: (-x[1], x[0]))

    summary = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "runner_version": RUNNER_VERSION,
        "input_path": str(source_path),
        "total_records": total_records,
        "total_steps": total_steps,
        "selected_count": selected_count,
        "abstain_count": abstain_count,
        "blocked_count": blocked_count,
        "error_count": error_count,
        "selection_rate": selection_rate,
        "status_counts": status_counts,
        "top_selected_candidates": [
            {"candidate": name, "count": count} for name, count in top_selected_candidates[:20]
        ],
        "imports": {
            "mapper": f"{mapper_module}.{mapper_attr}",
            "engine": f"{engine_module}.{engine_attr}",
            "comparison_pipeline": f"{comparison_module}.{comparison_attr}",
        },
    }

    write_jsonl(output_dir / "step_results.jsonl", [asdict(item) for item in step_results])
    write_json(output_dir / "summary.json", summary)

    summary_md = build_markdown_summary(
        run_id=run_id,
        source_path=source_path,
        output_dir=output_dir,
        total_records=total_records,
        total_steps=total_steps,
        selected_count=selected_count,
        abstain_count=abstain_count,
        blocked_count=blocked_count,
        error_count=error_count,
        selection_rate=selection_rate,
        status_counts=status_counts,
        top_selected_candidates=top_selected_candidates,
        step_results=step_results,
    )
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()