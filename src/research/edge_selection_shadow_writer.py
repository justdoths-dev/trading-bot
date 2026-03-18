from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.research.edge_selection_schema_validator import validate_shadow_output

logger = logging.getLogger(__name__)

DEFAULT_SHADOW_OUTPUT_PATH = Path(
    "logs/edge_selection_shadow/edge_selection_shadow.jsonl"
)


def write_edge_selection_shadow_output(
    payload: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Validate and append a shadow selection payload to a JSONL audit log."""
    final_path = Path(output_path) if output_path is not None else DEFAULT_SHADOW_OUTPUT_PATH

    validation_result = validate_shadow_output(payload)
    if not validation_result.is_valid:
        joined_errors = "; ".join(validation_result.errors)
        raise ValueError(f"Invalid shadow output payload: {joined_errors}")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    with final_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    logger.debug(
        "Shadow selection payload appended: %s",
        json.dumps(
            _build_shadow_write_log_context(payload),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return final_path


def read_edge_selection_shadow_outputs(path: Path) -> list[dict[str, Any]]:
    """Read JSONL shadow outputs from disk, skipping blank lines."""
    records: list[dict[str, Any]] = []
    final_path = Path(path)

    if not final_path.exists():
        return records

    if not final_path.is_file():
        raise ValueError(f"Shadow output path is not a file: {final_path}")

    with final_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue

            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Shadow output JSONL line {line_number} is not valid JSON: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Shadow output JSONL line {line_number} must contain a JSON object."
                )

            records.append(payload)

    return records


def _build_shadow_write_log_context(payload: dict[str, Any]) -> dict[str, Any]:
    ranking = payload.get("ranking")
    ranking_items = ranking if isinstance(ranking, list) else []
    ranking_dicts = [item for item in ranking_items if isinstance(item, dict)]

    candidate_status_counts = {
        "eligible": sum(
            1 for item in ranking_dicts if item.get("candidate_status") == "eligible"
        ),
        "penalized": sum(
            1 for item in ranking_dicts if item.get("candidate_status") == "penalized"
        ),
        "blocked": sum(
            1 for item in ranking_dicts if item.get("candidate_status") == "blocked"
        ),
    }

    context = {
        "selection_status": payload.get("selection_status"),
        "reason_codes": list(payload.get("reason_codes") or []),
        "selection_explanation": payload.get("selection_explanation"),
        "candidates_considered": payload.get("candidates_considered"),
        "latest_window_record_count": payload.get("latest_window_record_count"),
        "cumulative_record_count": payload.get("cumulative_record_count"),
        "candidate_status_counts": candidate_status_counts,
        "selected_candidate": _build_selected_candidate_snapshot(payload),
        "top_ranked_candidate": _build_top_ranked_candidate_snapshot(ranking_dicts),
    }

    abstain_diagnosis = payload.get("abstain_diagnosis")
    if isinstance(abstain_diagnosis, dict):
        context["abstain_diagnosis"] = abstain_diagnosis

    return context


def _build_selected_candidate_snapshot(payload: dict[str, Any]) -> dict[str, Any] | None:
    selected_symbol = payload.get("selected_symbol")
    selected_strategy = payload.get("selected_strategy")
    selected_horizon = payload.get("selected_horizon")
    selection_score = payload.get("selection_score")
    selection_confidence = payload.get("selection_confidence")

    if (
        selected_symbol is None
        and selected_strategy is None
        and selected_horizon is None
        and selection_score is None
        and selection_confidence is None
    ):
        return None

    return {
        "symbol": selected_symbol,
        "strategy": selected_strategy,
        "horizon": selected_horizon,
        "selection_score": selection_score,
        "selection_confidence": selection_confidence,
    }


def _build_top_ranked_candidate_snapshot(
    ranking: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not ranking:
        return None

    top_candidate = ranking[0]
    return {
        "rank": top_candidate.get("rank"),
        "symbol": top_candidate.get("symbol"),
        "strategy": top_candidate.get("strategy"),
        "horizon": top_candidate.get("horizon"),
        "candidate_status": top_candidate.get("candidate_status"),
        "selection_score": top_candidate.get("selection_score"),
        "selection_confidence": top_candidate.get("selection_confidence"),
        "reason_codes": list(top_candidate.get("reason_codes") or []),
        "advisory_reason_codes": list(top_candidate.get("advisory_reason_codes") or []),
        "selected_candidate_strength": top_candidate.get("selected_candidate_strength"),
        "selected_stability_label": top_candidate.get("selected_stability_label"),
        "drift_direction": top_candidate.get("drift_direction"),
        "edge_stability_score": top_candidate.get("edge_stability_score"),
        "latest_sample_size": top_candidate.get("latest_sample_size"),
        "cumulative_sample_size": top_candidate.get("cumulative_sample_size"),
        "symbol_cumulative_support": top_candidate.get("symbol_cumulative_support"),
        "strategy_cumulative_support": top_candidate.get("strategy_cumulative_support"),
        "gate_diagnostics": top_candidate.get("gate_diagnostics") or {},
    }