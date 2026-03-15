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

    if candidate_status == "blocked":
        if not isinstance(reason_codes, list) or len(reason_codes) == 0:
            result.add_error(
                f"{prefix}.reason_codes must contain at least one entry when candidate_status is 'blocked'."
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
