from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

VALID_SELECTION_STATUSES = {"selected", "abstain", "blocked"}
VALID_HORIZONS = {"15m", "1h", "4h"}
VALID_CANDIDATE_STATUSES = {"eligible", "penalized", "blocked"}
VALID_CANDIDATE_STRENGTHS = {"insufficient_data", "weak", "moderate", "strong"}
VALID_STABILITY_LABELS = {
    "insufficient_data",
    "unstable",
    "single_horizon_only",
    "multi_horizon_confirmed",
}
VALID_DRIFT_DIRECTIONS = {"increase", "decrease", "flat", "insufficient_history"}
VALID_SOURCE_PREFERENCES = {"latest", "cumulative", "n/a"}
VALID_GATE_DIAGNOSTICS_KEYS = {
    "score_gate",
    "stability_gate",
    "drift_gate",
    "eligibility_gate",
    "advisory",
}
VALID_BOOLEAN_GATE_KEYS = {
    "score_gate",
    "stability_gate",
    "drift_gate",
    "eligibility_gate",
}

REQUIRED_REPORT_PATHS = (
    Path("latest") / "summary.json",
    Path("comparison") / "summary.json",
    Path("edge_scores") / "summary.json",
    Path("score_drift") / "summary.json",
)
OPTIONAL_HISTORY_PATH = Path("edge_scores_history.jsonl")


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_shadow_output(payload: dict[str, Any]) -> ValidationResult:
    result = ValidationResult()

    if not isinstance(payload, dict):
        result.add_error("Shadow output payload must be a dict.")
        return result

    generated_at = payload.get("generated_at")
    if generated_at is not None:
        _validate_optional_iso_datetime(
            generated_at,
            field_name="generated_at",
            result=result,
        )

    mode = payload.get("mode")
    if mode != "shadow":
        result.add_error("mode must be 'shadow'.")

    selection_status = payload.get("selection_status")
    if selection_status not in VALID_SELECTION_STATUSES:
        result.add_error(
            "selection_status must be one of: selected, abstain, blocked."
        )

    _validate_optional_non_empty_string(
        payload.get("selected_symbol"),
        field_name="selected_symbol",
        result=result,
    )
    _validate_optional_non_empty_string(
        payload.get("selected_strategy"),
        field_name="selected_strategy",
        result=result,
    )

    selected_horizon = payload.get("selected_horizon")
    if selected_horizon is not None and selected_horizon not in VALID_HORIZONS:
        result.add_error("selected_horizon must be None or one of: 15m, 1h, 4h.")

    _validate_optional_number(
        payload.get("selection_score"),
        field_name="selection_score",
        result=result,
    )
    _validate_optional_probability(
        payload.get("selection_confidence"),
        field_name="selection_confidence",
        result=result,
    )

    _validate_reason_codes(
        payload.get("reason_codes"),
        field_name="reason_codes",
        result=result,
    )

    _validate_optional_non_negative_int(
        payload.get("candidates_considered"),
        field_name="candidates_considered",
        result=result,
    )
    _validate_optional_non_negative_int(
        payload.get("latest_window_record_count"),
        field_name="latest_window_record_count",
        result=result,
    )
    _validate_optional_non_negative_int(
        payload.get("cumulative_record_count"),
        field_name="cumulative_record_count",
        result=result,
    )
    _validate_optional_non_empty_string(
        payload.get("selection_explanation"),
        field_name="selection_explanation",
        result=result,
    )

    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        result.add_error("ranking must be a list.")
        ranking_items: list[dict[str, Any]] = []
    else:
        ranking_items = [item for item in ranking if isinstance(item, dict)]
        if len(ranking_items) != len(ranking):
            result.add_error("ranking must contain only dict items.")

    seen_ranks: set[int] = set()
    for index, item in enumerate(ranking_items, start=1):
        _validate_ranking_item(
            item=item,
            index=index,
            seen_ranks=seen_ranks,
            result=result,
        )

    candidates_considered = payload.get("candidates_considered")
    if isinstance(candidates_considered, int) and not isinstance(candidates_considered, bool):
        if candidates_considered < len(ranking_items):
            result.add_error(
                "candidates_considered must be greater than or equal to the number of ranking items."
            )

    abstain_diagnosis = payload.get("abstain_diagnosis")
    if abstain_diagnosis is not None:
        _validate_abstain_diagnosis(
            abstain_diagnosis,
            ranking_items=ranking_items,
            result=result,
        )

    if selection_status in {"abstain", "blocked"}:
        for field_name in (
            "selected_symbol",
            "selected_strategy",
            "selected_horizon",
            "selection_score",
            "selection_confidence",
        ):
            if payload.get(field_name) is not None:
                result.add_error(
                    f"{field_name} must be None when selection_status is '{selection_status}'."
                )

    if selection_status == "selected":
        for field_name in ("selected_symbol", "selected_strategy", "selected_horizon"):
            value = payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                result.add_error(
                    f"{field_name} must be present when selection_status is 'selected'."
                )

        if not ranking_items:
            result.add_error(
                "ranking must contain at least one item when selection_status is 'selected'."
            )

        selected_symbol = payload.get("selected_symbol")
        selected_strategy = payload.get("selected_strategy")
        selected_horizon = payload.get("selected_horizon")
        matched_selected = any(
            item.get("symbol") == selected_symbol
            and item.get("strategy") == selected_strategy
            and item.get("horizon") == selected_horizon
            for item in ranking_items
        )
        if ranking_items and not matched_selected:
            result.add_error(
                "selected_symbol/selected_strategy/selected_horizon must match a ranking item."
            )

        if abstain_diagnosis is not None:
            result.add_error(
                "abstain_diagnosis must be omitted when selection_status is 'selected'."
            )

    if selection_status == "blocked" and abstain_diagnosis is not None:
        result.add_error(
            "abstain_diagnosis must be omitted when selection_status is 'blocked'."
        )

    return result


def validate_upstream_reports(
    base_dir: Path,
    max_age_minutes: int = 90,
) -> ValidationResult:
    result = ValidationResult()

    if isinstance(max_age_minutes, bool) or not isinstance(max_age_minutes, int):
        result.add_error("max_age_minutes must be a positive integer.")
        return result
    if max_age_minutes <= 0:
        result.add_error("max_age_minutes must be greater than 0.")
        return result

    now = datetime.now(UTC)
    stale_cutoff = now - timedelta(minutes=max_age_minutes)

    for relative_path in REQUIRED_REPORT_PATHS:
        report_path = base_dir / relative_path
        _validate_required_json_report(
            path=report_path,
            stale_cutoff=stale_cutoff,
            result=result,
        )

    optional_history_path = base_dir / OPTIONAL_HISTORY_PATH
    if not optional_history_path.exists():
        result.add_warning(
            f"Optional upstream report is missing: {optional_history_path}"
        )
    elif not optional_history_path.is_file():
        result.add_warning(
            f"Optional upstream history path is not a file: {optional_history_path}"
        )
    else:
        _validate_optional_history_file(
            path=optional_history_path,
            stale_cutoff=stale_cutoff,
            result=result,
        )

    return result


def _validate_ranking_item(
    *,
    item: dict[str, Any],
    index: int,
    seen_ranks: set[int],
    result: ValidationResult,
) -> None:
    prefix = f"ranking[{index}]"

    rank = item.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        result.add_error(f"{prefix}.rank must be a positive integer.")
    elif rank in seen_ranks:
        result.add_error(f"{prefix}.rank must be unique across ranking items.")
    else:
        seen_ranks.add(rank)

    _validate_optional_non_empty_string(
        item.get("symbol"),
        field_name=f"{prefix}.symbol",
        result=result,
    )
    _validate_optional_non_empty_string(
        item.get("strategy"),
        field_name=f"{prefix}.strategy",
        result=result,
    )

    horizon = item.get("horizon")
    if horizon is not None and horizon not in VALID_HORIZONS:
        result.add_error(f"{prefix}.horizon must be one of: 15m, 1h, 4h.")

    candidate_status = item.get("candidate_status")
    if candidate_status not in VALID_CANDIDATE_STATUSES:
        result.add_error(
            f"{prefix}.candidate_status must be one of: eligible, penalized, blocked."
        )

    _validate_optional_number(
        item.get("selection_score"),
        field_name=f"{prefix}.selection_score",
        result=result,
    )
    _validate_optional_probability(
        item.get("selection_confidence"),
        field_name=f"{prefix}.selection_confidence",
        result=result,
    )

    selected_candidate_strength = item.get("selected_candidate_strength")
    if selected_candidate_strength not in VALID_CANDIDATE_STRENGTHS:
        result.add_error(
            f"{prefix}.selected_candidate_strength must be one of: insufficient_data, weak, moderate, strong."
        )

    selected_stability_label = item.get("selected_stability_label")
    if selected_stability_label not in VALID_STABILITY_LABELS:
        result.add_error(
            f"{prefix}.selected_stability_label must be one of: insufficient_data, unstable, single_horizon_only, multi_horizon_confirmed."
        )

    drift_direction = item.get("drift_direction")
    if drift_direction not in VALID_DRIFT_DIRECTIONS:
        result.add_error(
            f"{prefix}.drift_direction must be one of: increase, decrease, flat, insufficient_history."
        )

    reason_codes = item.get("reason_codes")
    _validate_reason_codes(
        reason_codes,
        field_name=f"{prefix}.reason_codes",
        result=result,
    )

    advisory_reason_codes = item.get("advisory_reason_codes")
    if advisory_reason_codes is not None:
        _validate_reason_codes(
            advisory_reason_codes,
            field_name=f"{prefix}.advisory_reason_codes",
            result=result,
        )

    if candidate_status == "blocked":
        if not isinstance(reason_codes, list) or len(reason_codes) == 0:
            result.add_error(
                f"{prefix}.reason_codes must contain at least one entry when candidate_status is 'blocked'."
            )
        if item.get("selection_score") is not None:
            result.add_error(
                f"{prefix}.selection_score must be None when candidate_status is 'blocked'."
            )
        if item.get("selection_confidence") is not None:
            result.add_error(
                f"{prefix}.selection_confidence must be None when candidate_status is 'blocked'."
            )

    if candidate_status in {"eligible", "penalized"}:
        if item.get("selection_score") is None:
            result.add_error(
                f"{prefix}.selection_score must be present when candidate_status is '{candidate_status}'."
            )
        if item.get("selection_confidence") is None:
            result.add_error(
                f"{prefix}.selection_confidence must be present when candidate_status is '{candidate_status}'."
            )

    selected_visible_horizons = item.get("selected_visible_horizons")
    if selected_visible_horizons is not None:
        if not isinstance(selected_visible_horizons, list):
            result.add_error(f"{prefix}.selected_visible_horizons must be a list.")
        else:
            normalized_horizons: list[str] = []
            for horizon_value in selected_visible_horizons:
                if horizon_value not in VALID_HORIZONS:
                    result.add_error(
                        f"{prefix}.selected_visible_horizons must contain only: 15m, 1h, 4h."
                    )
                    continue
                normalized_horizons.append(horizon_value)
            if len(set(normalized_horizons)) != len(normalized_horizons):
                result.add_error(
                    f"{prefix}.selected_visible_horizons must not contain duplicates."
                )

    source_preference = item.get("source_preference")
    if source_preference is not None and source_preference not in VALID_SOURCE_PREFERENCES:
        result.add_error(
            f"{prefix}.source_preference must be one of: latest, cumulative, n/a."
        )

    _validate_optional_number(
        item.get("edge_stability_score"),
        field_name=f"{prefix}.edge_stability_score",
        result=result,
    )

    stability_gate_pass = item.get("stability_gate_pass")
    if stability_gate_pass is not None and not isinstance(stability_gate_pass, bool):
        result.add_error(f"{prefix}.stability_gate_pass must be a boolean or None.")

    _validate_optional_non_negative_int(
        item.get("latest_sample_size"),
        field_name=f"{prefix}.latest_sample_size",
        result=result,
    )
    _validate_optional_non_negative_int(
        item.get("cumulative_sample_size"),
        field_name=f"{prefix}.cumulative_sample_size",
        result=result,
    )
    _validate_optional_non_negative_int(
        item.get("symbol_cumulative_support"),
        field_name=f"{prefix}.symbol_cumulative_support",
        result=result,
    )
    _validate_optional_non_negative_int(
        item.get("strategy_cumulative_support"),
        field_name=f"{prefix}.strategy_cumulative_support",
        result=result,
    )
    _validate_optional_non_negative_int(
        item.get("consecutive_visible_cycles"),
        field_name=f"{prefix}.consecutive_visible_cycles",
        result=result,
    )
    _validate_optional_non_negative_int(
        item.get("consecutive_stable_cycles"),
        field_name=f"{prefix}.consecutive_stable_cycles",
        result=result,
    )

    _validate_optional_number(
        item.get("score_delta"),
        field_name=f"{prefix}.score_delta",
        result=result,
    )

    drift_blocked = item.get("drift_blocked")
    if drift_blocked is not None and not isinstance(drift_blocked, bool):
        result.add_error(f"{prefix}.drift_blocked must be a boolean or None.")

    gate_diagnostics = item.get("gate_diagnostics")
    if gate_diagnostics is not None:
        _validate_gate_diagnostics(
            gate_diagnostics,
            field_name=f"{prefix}.gate_diagnostics",
            result=result,
        )


def _validate_abstain_diagnosis(
    value: Any,
    *,
    ranking_items: list[dict[str, Any]],
    result: ValidationResult,
) -> None:
    if not isinstance(value, dict):
        result.add_error("abstain_diagnosis must be a dict.")
        return

    _validate_required_non_empty_string(
        value.get("category"),
        field_name="abstain_diagnosis.category",
        result=result,
    )
    _validate_required_non_empty_string(
        value.get("summary"),
        field_name="abstain_diagnosis.summary",
        result=result,
    )
    _validate_required_non_negative_int(
        value.get("eligible_candidate_count"),
        field_name="abstain_diagnosis.eligible_candidate_count",
        result=result,
    )
    _validate_required_non_negative_int(
        value.get("penalized_candidate_count"),
        field_name="abstain_diagnosis.penalized_candidate_count",
        result=result,
    )
    _validate_required_non_negative_int(
        value.get("blocked_candidate_count"),
        field_name="abstain_diagnosis.blocked_candidate_count",
        result=result,
    )

    eligible_count = value.get("eligible_candidate_count")
    penalized_count = value.get("penalized_candidate_count")
    blocked_count = value.get("blocked_candidate_count")
    if all(
        isinstance(v, int) and not isinstance(v, bool)
        for v in (eligible_count, penalized_count, blocked_count)
    ):
        total_count = eligible_count + penalized_count + blocked_count
        if total_count > len(ranking_items):
            result.add_error(
                "abstain_diagnosis candidate counts must not exceed the number of ranking items."
            )

    _validate_optional_candidate_snapshot(
        value.get("top_candidate"),
        field_name="abstain_diagnosis.top_candidate",
        result=result,
    )
    _validate_optional_candidate_snapshot(
        value.get("compared_candidate"),
        field_name="abstain_diagnosis.compared_candidate",
        result=result,
    )


def _validate_optional_candidate_snapshot(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        result.add_error(f"{field_name} must be a dict or None.")
        return

    _validate_optional_non_empty_string(
        value.get("symbol"),
        field_name=f"{field_name}.symbol",
        result=result,
    )
    _validate_optional_non_empty_string(
        value.get("strategy"),
        field_name=f"{field_name}.strategy",
        result=result,
    )

    horizon = value.get("horizon")
    if horizon is not None and horizon not in VALID_HORIZONS:
        result.add_error(f"{field_name}.horizon must be one of: 15m, 1h, 4h.")

    candidate_status = value.get("candidate_status")
    if candidate_status is not None and candidate_status not in VALID_CANDIDATE_STATUSES:
        result.add_error(
            f"{field_name}.candidate_status must be one of: eligible, penalized, blocked."
        )

    _validate_optional_number(
        value.get("selection_score"),
        field_name=f"{field_name}.selection_score",
        result=result,
    )
    _validate_optional_probability(
        value.get("selection_confidence"),
        field_name=f"{field_name}.selection_confidence",
        result=result,
    )
    _validate_reason_codes(
        value.get("reason_codes"),
        field_name=f"{field_name}.reason_codes",
        result=result,
    )

    advisory_reason_codes = value.get("advisory_reason_codes")
    if advisory_reason_codes is not None:
        _validate_reason_codes(
            advisory_reason_codes,
            field_name=f"{field_name}.advisory_reason_codes",
            result=result,
        )

    gate_diagnostics = value.get("gate_diagnostics")
    if gate_diagnostics is not None:
        _validate_gate_diagnostics(
            gate_diagnostics,
            field_name=f"{field_name}.gate_diagnostics",
            result=result,
        )


def _validate_gate_diagnostics(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if not isinstance(value, dict):
        result.add_error(f"{field_name} must be a dict.")
        return

    for key, gate_value in value.items():
        if key not in VALID_GATE_DIAGNOSTICS_KEYS:
            result.add_error(
                f"{field_name} contains unsupported key '{key}'."
            )
            continue

        if not isinstance(gate_value, dict):
            result.add_error(f"{field_name}.{key} must be a dict.")
            continue

        if key in VALID_BOOLEAN_GATE_KEYS:
            passed = gate_value.get("passed")
            if not isinstance(passed, bool):
                result.add_error(f"{field_name}.{key}.passed must be a boolean.")

        _validate_reason_codes(
            gate_value.get("reason_codes"),
            field_name=f"{field_name}.{key}.reason_codes",
            result=result,
        )


def _validate_reason_codes(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if not isinstance(value, list):
        result.add_error(f"{field_name} must be a list.")
        return

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            result.add_error(f"{field_name} must contain only non-empty strings.")
            continue
        normalized.append(item.strip())

    if len(set(normalized)) != len(normalized):
        result.add_error(f"{field_name} must not contain duplicates.")


def _validate_required_json_report(
    *,
    path: Path,
    stale_cutoff: datetime,
    result: ValidationResult,
) -> None:
    if not path.exists():
        result.add_error(f"Missing required upstream report: {path}")
        return

    if not path.is_file():
        result.add_error(f"Required upstream report is not a file: {path}")
        return

    if _is_stale(path, stale_cutoff):
        result.add_error(f"Required upstream report is stale: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        result.add_error(f"Failed to read required upstream report {path}: {exc}")
        return
    except json.JSONDecodeError as exc:
        result.add_error(f"Required upstream report is not valid JSON {path}: {exc}")
        return

    if not isinstance(payload, dict):
        result.add_error(f"Required upstream report must contain a JSON object: {path}")


def _validate_optional_history_file(
    *,
    path: Path,
    stale_cutoff: datetime,
    result: ValidationResult,
) -> None:
    if _is_stale(path, stale_cutoff):
        result.add_warning(f"Optional upstream history file is stale: {path}")

    non_empty_line_count = 0
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                content = line.strip()
                if not content:
                    continue
                non_empty_line_count += 1
                json.loads(content)
    except OSError as exc:
        result.add_warning(f"Failed to read optional upstream history file {path}: {exc}")
        return
    except json.JSONDecodeError as exc:
        result.add_warning(
            f"Optional upstream history file contains invalid JSONL at line {line_number}: {exc}"
        )
        return

    if non_empty_line_count == 0:
        result.add_warning(f"Optional upstream history file is empty: {path}")


def _is_stale(path: Path, stale_cutoff: datetime) -> bool:
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return modified_at < stale_cutoff


def _validate_optional_iso_datetime(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if not isinstance(value, str) or not value.strip():
        result.add_error(f"{field_name} must be a non-empty ISO datetime string.")
        return

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        result.add_error(f"{field_name} must be a valid ISO datetime string.")
        return

    if parsed.tzinfo is None:
        result.add_error(f"{field_name} must include timezone information.")


def _validate_required_non_empty_string(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if not isinstance(value, str) or not value.strip():
        result.add_error(f"{field_name} must be a non-empty string.")


def _validate_optional_non_empty_string(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        result.add_error(f"{field_name} must be a non-empty string or None.")


def _validate_optional_number(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        result.add_error(f"{field_name} must be a number or None.")


def _validate_optional_probability(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        result.add_error(f"{field_name} must be None or a number between 0 and 1.")
        return
    if not 0.0 <= float(value) <= 1.0:
        result.add_error(f"{field_name} must be between 0 and 1.")


def _validate_required_non_negative_int(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        result.add_error(f"{field_name} must be a non-negative integer.")


def _validate_optional_non_negative_int(
    value: Any,
    *,
    field_name: str,
    result: ValidationResult,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        result.add_error(f"{field_name} must be a non-negative integer or None.")