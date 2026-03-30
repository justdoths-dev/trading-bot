from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.research.edge_selection_schema_validator import validate_shadow_output

logger = logging.getLogger(__name__)

BLOCKED_REASON_MAP = {
    "identity_incomplete": "CANDIDATE_IDENTITY_INCOMPLETE",
    "insufficient_data_strength": "CANDIDATE_STRENGTH_INSUFFICIENT_DATA",
    "insufficient_data_stability": "CANDIDATE_STABILITY_INSUFFICIENT_DATA",
    "unstable_stability": "CANDIDATE_STABILITY_UNSTABLE",
    "latest_sample_too_low": "CANDIDATE_LATEST_SAMPLE_TOO_LOW",
    "cumulative_sample_too_low": "CANDIDATE_CUMULATIVE_SAMPLE_TOO_LOW",
    "symbol_support_too_low": "CANDIDATE_SYMBOL_SUPPORT_TOO_LOW",
    "strategy_support_too_low": "CANDIDATE_STRATEGY_SUPPORT_TOO_LOW",
}
PENALTY_REASON_MAP = {
    "weak_strength": "CANDIDATE_STRENGTH_WEAK",
    "single_horizon_stability": "CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY",
    "decreasing_drift": "CANDIDATE_DRIFT_DECREASING",
    "low_edge_score": "CANDIDATE_EDGE_STABILITY_SCORE_LOW",
}
ADVISORY_REASON_MAP = {
    "preferred_latest_sample": "CANDIDATE_LATEST_SAMPLE_PREFERRED_RANGE",
    "preferred_cumulative_sample": "CANDIDATE_CUMULATIVE_SAMPLE_PREFERRED_RANGE",
    "preferred_symbol_support": "CANDIDATE_SYMBOL_SUPPORT_PREFERRED_RANGE",
    "preferred_strategy_support": "CANDIDATE_STRATEGY_SUPPORT_PREFERRED_RANGE",
    "single_horizon_relaxed": "CANDIDATE_STABILITY_SINGLE_HORIZON_RELAXED",
}
STATUS_PRIORITY = {
    "eligible": 2,
    "penalized": 1,
    "blocked": 0,
}
STRENGTH_PRIORITY = {
    "insufficient_data": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}
STABILITY_PRIORITY = {
    "insufficient_data": 0,
    "unstable": 1,
    "single_horizon_only": 2,
    "multi_horizon_confirmed": 3,
}
DRIFT_PRIORITY = {
    "decrease": 0,
    "insufficient_history": 1,
    "flat": 2,
    "increase": 3,
}
TOP_LEVEL_REASON_CODES = {
    "upstream_invalid": "UPSTREAM_INPUT_INVALID",
    "no_candidates": "NO_CANDIDATES_AVAILABLE",
    "all_blocked": "ALL_CANDIDATES_BLOCKED",
    "no_eligible": "NO_ELIGIBLE_CANDIDATES",
    "tied": "TOP_CANDIDATES_TIED",
    "selected": "CLEAR_TOP_CANDIDATE",
}

VALID_CANDIDATE_STATUSES = tuple(STATUS_PRIORITY.keys())
VALID_HORIZONS = {"15m", "1h", "4h"}
VALID_SOURCE_PREFERENCES = {"latest", "cumulative", "n/a"}
ELIGIBLE_CONSERVATIVE_PASS = "ELIGIBLE_CONSERVATIVE_PASS"

SHADOW_MIN_EDGE_STABILITY_SCORE = 3.0
HARD_MIN_LATEST_SAMPLE = 20
HARD_MIN_CUMULATIVE_SAMPLE = 60
HARD_MIN_SYMBOL_SUPPORT = 150
HARD_MIN_STRATEGY_SUPPORT = 120

PREFERRED_LATEST_SAMPLE_MIN = 20
PREFERRED_LATEST_SAMPLE_MAX = 39
PREFERRED_CUMULATIVE_SAMPLE_MIN = 60
PREFERRED_CUMULATIVE_SAMPLE_MAX = 119
PREFERRED_SYMBOL_SUPPORT_MIN = 150
PREFERRED_SYMBOL_SUPPORT_MAX = 249
PREFERRED_STRATEGY_SUPPORT_MIN = 120
PREFERRED_STRATEGY_SUPPORT_MAX = 199


def run_edge_selection_engine(mapped_payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate mapped candidates conservatively and return validated shadow output."""

    generated_at = datetime.now(UTC).isoformat()

    if not isinstance(mapped_payload, dict):
        return _finalize_output(
            _build_shadow_output(
                generated_at=generated_at,
                selection_status="blocked",
                reason_codes=[TOP_LEVEL_REASON_CODES["upstream_invalid"]],
                ranking=[],
                candidates_considered=0,
                latest_window_record_count=None,
                cumulative_record_count=None,
                selection_explanation="Blocked because mapped payload is not a valid dict.",
            )
        )

    raw_candidates = mapped_payload.get("candidates")
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    upstream_errors = mapped_payload.get("errors")
    is_ok = mapped_payload.get("ok") is True
    latest_window_record_count = _coerce_non_negative_int(
        mapped_payload.get("latest_window_record_count")
    )
    cumulative_record_count = _coerce_non_negative_int(
        mapped_payload.get("cumulative_record_count")
    )
    candidate_seed_count = _coerce_non_negative_int(
        mapped_payload.get("candidate_seed_count")
    )
    candidate_seed_diagnostics = (
        mapped_payload.get("candidate_seed_diagnostics")
        if isinstance(mapped_payload.get("candidate_seed_diagnostics"), dict)
        else {}
    )

    if not is_ok or (isinstance(upstream_errors, list) and upstream_errors):
        return _finalize_output(
            _build_shadow_output(
                generated_at=generated_at,
                selection_status="blocked",
                reason_codes=[TOP_LEVEL_REASON_CODES["upstream_invalid"]],
                ranking=[],
                candidates_considered=len(candidates),
                latest_window_record_count=latest_window_record_count,
                cumulative_record_count=cumulative_record_count,
                selection_explanation="Blocked because mapped input reported upstream validation errors.",
                candidate_seed_count=candidate_seed_count,
                candidate_seed_diagnostics=candidate_seed_diagnostics,
            )
        )

    evaluated = [_evaluate_candidate(candidate) for candidate in candidates]
    evaluated.sort(key=_ranking_sort_key)

    ranking = [
        _build_ranking_item(item, rank=index)
        for index, item in enumerate(evaluated, start=1)
    ]

    if not ranking:
        explanation = "Abstained because no candidates were available from the mapper payload."
        return _finalize_output(
            _build_shadow_output(
                generated_at=generated_at,
                selection_status="abstain",
                reason_codes=[TOP_LEVEL_REASON_CODES["no_candidates"]],
                ranking=ranking,
                candidates_considered=0,
                latest_window_record_count=latest_window_record_count,
                cumulative_record_count=cumulative_record_count,
                selection_explanation=explanation,
                abstain_diagnosis=_build_abstain_diagnosis(
                    category="no_candidates_available",
                    summary=explanation,
                    ranking=ranking,
                    eligible_count=0,
                    penalized_count=0,
                    blocked_count=0,
                    candidate_seed_count=candidate_seed_count,
                    candidate_seed_diagnostics=candidate_seed_diagnostics,
                ),
                candidate_seed_count=candidate_seed_count,
                candidate_seed_diagnostics=candidate_seed_diagnostics,
            )
        )

    eligible = [item for item in evaluated if item["candidate_status"] == "eligible"]
    penalized = [item for item in evaluated if item["candidate_status"] == "penalized"]
    blocked = [item for item in evaluated if item["candidate_status"] == "blocked"]

    if not eligible:
        reason_code = (
            TOP_LEVEL_REASON_CODES["all_blocked"]
            if len(blocked) == len(evaluated)
            else TOP_LEVEL_REASON_CODES["no_eligible"]
        )
        explanation = (
            "Abstained because all candidates were blocked."
            if reason_code == TOP_LEVEL_REASON_CODES["all_blocked"]
            else "Abstained because no candidate passed the conservative eligibility threshold."
        )
        return _finalize_output(
            _build_shadow_output(
                generated_at=generated_at,
                selection_status="abstain",
                reason_codes=[reason_code],
                ranking=ranking,
                candidates_considered=len(ranking),
                latest_window_record_count=latest_window_record_count,
                cumulative_record_count=cumulative_record_count,
                selection_explanation=explanation,
                abstain_diagnosis=_build_abstain_diagnosis(
                    category=(
                        "all_candidates_blocked"
                        if reason_code == TOP_LEVEL_REASON_CODES["all_blocked"]
                        else "no_eligible_candidates"
                    ),
                    summary=explanation,
                    ranking=ranking,
                    eligible_count=len(eligible),
                    penalized_count=len(penalized),
                    blocked_count=len(blocked),
                    candidate_seed_count=candidate_seed_count,
                    candidate_seed_diagnostics=candidate_seed_diagnostics,
                ),
                candidate_seed_count=candidate_seed_count,
                candidate_seed_diagnostics=candidate_seed_diagnostics,
            )
        )

    top_candidate = eligible[0]
    if len(eligible) > 1 and _is_tied(top_candidate, eligible[1]):
        explanation = (
            "Abstained because the top eligible candidates were tied without sufficient distinction."
        )
        return _finalize_output(
            _build_shadow_output(
                generated_at=generated_at,
                selection_status="abstain",
                reason_codes=[TOP_LEVEL_REASON_CODES["tied"]],
                ranking=ranking,
                candidates_considered=len(ranking),
                latest_window_record_count=latest_window_record_count,
                cumulative_record_count=cumulative_record_count,
                selection_explanation=explanation,
                abstain_diagnosis=_build_abstain_diagnosis(
                    category="tied_top_candidates",
                    summary=explanation,
                    ranking=ranking,
                    eligible_count=len(eligible),
                    penalized_count=len(penalized),
                    blocked_count=len(blocked),
                    compared_candidate=eligible[1],
                    candidate_seed_count=candidate_seed_count,
                    candidate_seed_diagnostics=candidate_seed_diagnostics,
                ),
                candidate_seed_count=candidate_seed_count,
                candidate_seed_diagnostics=candidate_seed_diagnostics,
            )
        )

    return _finalize_output(
        _build_shadow_output(
            generated_at=generated_at,
            selection_status="selected",
            reason_codes=[TOP_LEVEL_REASON_CODES["selected"]],
            ranking=ranking,
            candidates_considered=len(ranking),
            latest_window_record_count=latest_window_record_count,
            cumulative_record_count=cumulative_record_count,
            selected=top_candidate,
            selection_explanation="Selected the clear top candidate under conservative shadow-only rules.",
            candidate_seed_count=candidate_seed_count,
            candidate_seed_diagnostics=candidate_seed_diagnostics,
        )
    )


def _evaluate_candidate(candidate: Any) -> dict[str, Any]:
    payload = candidate if isinstance(candidate, dict) else {}

    symbol = _normalize_text(payload.get("symbol"))
    strategy = _normalize_text(payload.get("strategy"))
    horizon = _normalize_horizon(payload.get("horizon"))

    strength = _normalize_strength(payload.get("selected_candidate_strength"))
    stability = _normalize_stability(payload.get("selected_stability_label"))
    drift_direction = _normalize_drift_direction(payload.get("drift_direction"))
    edge_stability_score = _coerce_number(payload.get("edge_stability_score"))
    score_delta = _coerce_number(payload.get("score_delta"))

    latest_sample_size = _coerce_non_negative_int(payload.get("latest_sample_size"))
    cumulative_sample_size = _coerce_non_negative_int(payload.get("cumulative_sample_size"))
    symbol_cumulative_support = _coerce_non_negative_int(
        payload.get("symbol_cumulative_support")
    )
    strategy_cumulative_support = _coerce_non_negative_int(
        payload.get("strategy_cumulative_support")
    )

    aggregate_score = _coerce_number(payload.get("aggregate_score"))
    sample_count = _coerce_non_negative_int(payload.get("sample_count"))
    labeled_count = _coerce_non_negative_int(payload.get("labeled_count"))
    coverage_pct = _coerce_number(payload.get("coverage_pct"))
    median_future_return_pct = _coerce_number(payload.get("median_future_return_pct"))
    avg_future_return_pct = _coerce_number(payload.get("avg_future_return_pct"))
    positive_rate_pct = _coerce_number(payload.get("positive_rate_pct"))
    robustness_signal_pct = _coerce_number(payload.get("robustness_signal_pct"))
    supporting_major_deficit_count = _coerce_non_negative_int(
        payload.get("supporting_major_deficit_count")
    )

    blocked_reasons = _blocked_reasons(
        symbol=symbol,
        strategy=strategy,
        horizon=horizon,
        strength=strength,
        stability=stability,
        latest_sample_size=latest_sample_size,
        cumulative_sample_size=cumulative_sample_size,
        symbol_cumulative_support=symbol_cumulative_support,
        strategy_cumulative_support=strategy_cumulative_support,
    )
    penalty_reasons = _penalty_reasons(
        strength=strength,
        stability=stability,
        drift_direction=drift_direction,
        edge_stability_score=edge_stability_score,
    )
    advisory_reason_codes = _advisory_reasons(
        latest_sample_size=latest_sample_size,
        cumulative_sample_size=cumulative_sample_size,
        symbol_cumulative_support=symbol_cumulative_support,
        strategy_cumulative_support=strategy_cumulative_support,
    )

    relaxed_single_horizon = _should_relax_single_horizon_penalty(
        strength=strength,
        stability=stability,
        drift_direction=drift_direction,
        edge_stability_score=edge_stability_score,
    )
    effective_penalty_reasons = _apply_relaxed_penalties(
        penalty_reasons=penalty_reasons,
        relaxed_single_horizon=relaxed_single_horizon,
    )
    if relaxed_single_horizon:
        advisory_reason_codes.append(ADVISORY_REASON_MAP["single_horizon_relaxed"])

    if blocked_reasons:
        candidate_status = "blocked"
        selection_score = None
        selection_confidence = None
        reason_codes = blocked_reasons
    else:
        candidate_status = "penalized" if effective_penalty_reasons else "eligible"
        selection_score = _calculate_selection_score(
            strength=strength,
            stability=stability,
            edge_stability_score=edge_stability_score,
            drift_direction=drift_direction,
            score_delta=score_delta,
            aggregate_score=aggregate_score,
            sample_count=sample_count,
            median_future_return_pct=median_future_return_pct,
            positive_rate_pct=positive_rate_pct,
            robustness_signal_pct=robustness_signal_pct,
            supporting_major_deficit_count=supporting_major_deficit_count,
            penalty_count=len(effective_penalty_reasons),
        )
        selection_confidence = _calculate_confidence(
            selection_score=selection_score,
            candidate_status=candidate_status,
        )
        reason_codes = effective_penalty_reasons or [ELIGIBLE_CONSERVATIVE_PASS]

    gate_diagnostics = _build_gate_diagnostics(
        candidate_status=candidate_status,
        strength=strength,
        stability=stability,
        drift_direction=drift_direction,
        edge_stability_score=edge_stability_score,
        blocked_reasons=blocked_reasons,
        penalty_reasons=effective_penalty_reasons,
        advisory_reason_codes=advisory_reason_codes,
        relaxed_single_horizon=relaxed_single_horizon,
    )

    ranking_signals = {
        "aggregate_score": aggregate_score,
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": coverage_pct,
        "median_future_return_pct": median_future_return_pct,
        "avg_future_return_pct": avg_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal_pct": robustness_signal_pct,
        "supporting_major_deficit_count": supporting_major_deficit_count,
    }
    ranking_signals = {
        key: value for key, value in ranking_signals.items() if value is not None
    }

    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "candidate_status": candidate_status,
        "selection_score": selection_score,
        "selection_confidence": selection_confidence,
        "reason_codes": reason_codes,
        "advisory_reason_codes": advisory_reason_codes,
        "selected_candidate_strength": strength,
        "selected_stability_label": stability,
        "drift_direction": drift_direction,
        "score_delta": score_delta,
        "source_preference": _normalize_source_preference(payload.get("source_preference")),
        "edge_stability_score": edge_stability_score,
        "selected_visible_horizons": _normalize_horizon_list(
            payload.get("selected_visible_horizons")
        ),
        "latest_sample_size": latest_sample_size,
        "cumulative_sample_size": cumulative_sample_size,
        "symbol_cumulative_support": symbol_cumulative_support,
        "strategy_cumulative_support": strategy_cumulative_support,
        "aggregate_score": aggregate_score,
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": coverage_pct,
        "median_future_return_pct": median_future_return_pct,
        "avg_future_return_pct": avg_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal_pct": robustness_signal_pct,
        "supporting_major_deficit_count": supporting_major_deficit_count,
        "ranking_signals": ranking_signals,
        "stability_gate_pass": candidate_status == "eligible",
        "drift_blocked": drift_direction == "decrease",
        "gate_diagnostics": gate_diagnostics,
        "relaxed_single_horizon": relaxed_single_horizon,
    }


def _blocked_reasons(
    *,
    symbol: str | None,
    strategy: str | None,
    horizon: str | None,
    strength: str,
    stability: str,
    latest_sample_size: int | None,
    cumulative_sample_size: int | None,
    symbol_cumulative_support: int | None,
    strategy_cumulative_support: int | None,
) -> list[str]:
    reasons: list[str] = []

    if symbol is None or strategy is None or horizon is None:
        reasons.append(BLOCKED_REASON_MAP["identity_incomplete"])

    if strength == "insufficient_data":
        reasons.append(BLOCKED_REASON_MAP["insufficient_data_strength"])
    if stability == "insufficient_data":
        reasons.append(BLOCKED_REASON_MAP["insufficient_data_stability"])
    if stability == "unstable":
        reasons.append(BLOCKED_REASON_MAP["unstable_stability"])

    if latest_sample_size is not None and latest_sample_size < HARD_MIN_LATEST_SAMPLE:
        reasons.append(BLOCKED_REASON_MAP["latest_sample_too_low"])
    if cumulative_sample_size is not None and cumulative_sample_size < HARD_MIN_CUMULATIVE_SAMPLE:
        reasons.append(BLOCKED_REASON_MAP["cumulative_sample_too_low"])
    if (
        symbol_cumulative_support is not None
        and symbol_cumulative_support < HARD_MIN_SYMBOL_SUPPORT
    ):
        reasons.append(BLOCKED_REASON_MAP["symbol_support_too_low"])
    if (
        strategy_cumulative_support is not None
        and strategy_cumulative_support < HARD_MIN_STRATEGY_SUPPORT
    ):
        reasons.append(BLOCKED_REASON_MAP["strategy_support_too_low"])

    return reasons


def _penalty_reasons(
    *,
    strength: str,
    stability: str,
    drift_direction: str,
    edge_stability_score: float | int | None,
) -> list[str]:
    reasons: list[str] = []

    if strength == "weak":
        reasons.append(PENALTY_REASON_MAP["weak_strength"])
    if stability == "single_horizon_only":
        reasons.append(PENALTY_REASON_MAP["single_horizon_stability"])
    if drift_direction == "decrease":
        reasons.append(PENALTY_REASON_MAP["decreasing_drift"])

    if (
        edge_stability_score is not None
        and float(edge_stability_score) < SHADOW_MIN_EDGE_STABILITY_SCORE
    ):
        reasons.append(PENALTY_REASON_MAP["low_edge_score"])

    return reasons


def _should_relax_single_horizon_penalty(
    *,
    strength: str,
    stability: str,
    drift_direction: str,
    edge_stability_score: float | int | None,
) -> bool:
    if stability != "single_horizon_only":
        return False
    if strength not in {"moderate", "strong"}:
        return False
    if drift_direction == "decrease":
        return False
    if (
        edge_stability_score is not None
        and float(edge_stability_score) < SHADOW_MIN_EDGE_STABILITY_SCORE
    ):
        return False
    return True


def _apply_relaxed_penalties(
    *,
    penalty_reasons: list[str],
    relaxed_single_horizon: bool,
) -> list[str]:
    if not relaxed_single_horizon:
        return list(penalty_reasons)

    return [
        reason
        for reason in penalty_reasons
        if reason != PENALTY_REASON_MAP["single_horizon_stability"]
    ]


def _advisory_reasons(
    *,
    latest_sample_size: int | None,
    cumulative_sample_size: int | None,
    symbol_cumulative_support: int | None,
    strategy_cumulative_support: int | None,
) -> list[str]:
    reasons: list[str] = []

    if (
        latest_sample_size is not None
        and PREFERRED_LATEST_SAMPLE_MIN <= latest_sample_size <= PREFERRED_LATEST_SAMPLE_MAX
    ):
        reasons.append(ADVISORY_REASON_MAP["preferred_latest_sample"])
    if (
        cumulative_sample_size is not None
        and PREFERRED_CUMULATIVE_SAMPLE_MIN
        <= cumulative_sample_size
        <= PREFERRED_CUMULATIVE_SAMPLE_MAX
    ):
        reasons.append(ADVISORY_REASON_MAP["preferred_cumulative_sample"])
    if (
        symbol_cumulative_support is not None
        and PREFERRED_SYMBOL_SUPPORT_MIN
        <= symbol_cumulative_support
        <= PREFERRED_SYMBOL_SUPPORT_MAX
    ):
        reasons.append(ADVISORY_REASON_MAP["preferred_symbol_support"])
    if (
        strategy_cumulative_support is not None
        and PREFERRED_STRATEGY_SUPPORT_MIN
        <= strategy_cumulative_support
        <= PREFERRED_STRATEGY_SUPPORT_MAX
    ):
        reasons.append(ADVISORY_REASON_MAP["preferred_strategy_support"])

    return reasons


def _calculate_selection_score(
    *,
    strength: str,
    stability: str,
    edge_stability_score: float | int | None,
    drift_direction: str,
    score_delta: float | int | None,
    aggregate_score: float | int | None,
    sample_count: int | None,
    median_future_return_pct: float | int | None,
    positive_rate_pct: float | int | None,
    robustness_signal_pct: float | int | None,
    supporting_major_deficit_count: int | None,
    penalty_count: int,
) -> float:
    strength_component = {
        "insufficient_data": 0.0,
        "weak": 1.0,
        "moderate": 2.5,
        "strong": 4.0,
    }[strength]
    stability_component = {
        "insufficient_data": 0.0,
        "unstable": 0.0,
        "single_horizon_only": 1.0,
        "multi_horizon_confirmed": 2.5,
    }[stability]
    drift_component = {
        "decrease": -0.75,
        "insufficient_history": -0.1,
        "flat": 0.2,
        "increase": 0.5,
    }[drift_direction]
    edge_component = float(edge_stability_score) if edge_stability_score is not None else 0.0

    delta_component = 0.0
    if score_delta is not None:
        delta_component = max(min(float(score_delta), 1.0), -1.0) * 0.2

    aggregate_component = 0.0
    if aggregate_score is not None:
        aggregate_component = (float(aggregate_score) - 60.0) / 10.0
        aggregate_component = max(min(aggregate_component, 4.0), 0.0)

    sample_component = 0.0
    if sample_count is not None:
        sample_component = min(max((float(sample_count) - 30.0) / 200.0, 0.0), 1.0)

    median_component = 0.0
    if median_future_return_pct is not None:
        median_component = min(max(float(median_future_return_pct) * 2.0, 0.0), 1.0)

    positive_component = 0.0
    if positive_rate_pct is not None:
        positive_component = min(max((float(positive_rate_pct) - 50.0) / 10.0, 0.0), 1.0)

    robustness_component = 0.0
    if robustness_signal_pct is not None:
        robustness_component = min(max((float(robustness_signal_pct) - 45.0) / 15.0, 0.0), 0.75)

    deficit_penalty = 0.0
    if supporting_major_deficit_count is not None:
        deficit_penalty = min(float(supporting_major_deficit_count) * 0.3, 1.2)

    score = (
        strength_component
        + stability_component
        + edge_component
        + drift_component
        + delta_component
        + aggregate_component
        + sample_component
        + median_component
        + positive_component
        + robustness_component
        - deficit_penalty
        - (penalty_count * 0.75)
    )
    return round(score, 6)


def _calculate_confidence(
    *,
    selection_score: float,
    candidate_status: str,
) -> float:
    baseline = 0.35 if candidate_status == "eligible" else 0.2
    confidence = baseline + (selection_score / 12.0)
    return round(min(max(confidence, 0.0), 0.95), 6)


def _ranking_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    selection_score = candidate.get("selection_score")
    numeric_score = (
        float(selection_score)
        if isinstance(selection_score, (int, float))
        else float("-inf")
    )
    aggregate_score = candidate.get("aggregate_score")
    numeric_aggregate_score = (
        float(aggregate_score)
        if isinstance(aggregate_score, (int, float))
        else float("-inf")
    )
    sample_count = candidate.get("sample_count")
    numeric_sample_count = int(sample_count) if isinstance(sample_count, int) else -1
    median_future_return_pct = candidate.get("median_future_return_pct")
    numeric_median_future_return_pct = (
        float(median_future_return_pct)
        if isinstance(median_future_return_pct, (int, float))
        else float("-inf")
    )
    positive_rate_pct = candidate.get("positive_rate_pct")
    numeric_positive_rate_pct = (
        float(positive_rate_pct)
        if isinstance(positive_rate_pct, (int, float))
        else float("-inf")
    )
    robustness_signal_pct = candidate.get("robustness_signal_pct")
    numeric_robustness_signal_pct = (
        float(robustness_signal_pct)
        if isinstance(robustness_signal_pct, (int, float))
        else float("-inf")
    )
    supporting_major_deficit_count = candidate.get("supporting_major_deficit_count")
    numeric_supporting_major_deficit_count = (
        int(supporting_major_deficit_count)
        if isinstance(supporting_major_deficit_count, int)
        else 10**9
    )
    score_delta = candidate.get("score_delta")
    numeric_delta = (
        float(score_delta) if isinstance(score_delta, (int, float)) else float("-inf")
    )

    return (
        -STATUS_PRIORITY[candidate["candidate_status"]],
        -numeric_score,
        -numeric_aggregate_score,
        numeric_supporting_major_deficit_count,
        -numeric_sample_count,
        -numeric_median_future_return_pct,
        -numeric_positive_rate_pct,
        -numeric_robustness_signal_pct,
        -STRENGTH_PRIORITY[candidate["selected_candidate_strength"]],
        -STABILITY_PRIORITY[candidate["selected_stability_label"]],
        -DRIFT_PRIORITY[candidate["drift_direction"]],
        -numeric_delta,
        str(candidate.get("symbol") or ""),
        str(candidate.get("strategy") or ""),
        str(candidate.get("horizon") or ""),
    )


def _is_tied(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return _tie_signature(first) == _tie_signature(second)


def _tie_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    selection_score = candidate.get("selection_score")
    aggregate_score = candidate.get("aggregate_score")
    median_future_return_pct = candidate.get("median_future_return_pct")
    positive_rate_pct = candidate.get("positive_rate_pct")
    robustness_signal_pct = candidate.get("robustness_signal_pct")
    edge_stability_score = candidate.get("edge_stability_score")

    return (
        candidate["candidate_status"],
        round(float(selection_score), 2)
        if isinstance(selection_score, (int, float))
        else None,
        round(float(aggregate_score), 2)
        if isinstance(aggregate_score, (int, float))
        else None,
        candidate.get("supporting_major_deficit_count"),
        candidate.get("sample_count"),
        round(float(median_future_return_pct), 2)
        if isinstance(median_future_return_pct, (int, float))
        else None,
        round(float(positive_rate_pct), 2)
        if isinstance(positive_rate_pct, (int, float))
        else None,
        round(float(robustness_signal_pct), 2)
        if isinstance(robustness_signal_pct, (int, float))
        else None,
        candidate["selected_candidate_strength"],
        candidate["selected_stability_label"],
        candidate["drift_direction"],
        round(float(edge_stability_score), 2)
        if isinstance(edge_stability_score, (int, float))
        else None,
    )


def _build_ranking_item(candidate: dict[str, Any], *, rank: int) -> dict[str, Any]:
    item = {
        "rank": rank,
        "symbol": candidate.get("symbol"),
        "strategy": candidate.get("strategy"),
        "horizon": candidate.get("horizon"),
        "candidate_status": candidate["candidate_status"],
        "selection_score": candidate.get("selection_score"),
        "selection_confidence": candidate.get("selection_confidence"),
        "reason_codes": list(candidate["reason_codes"]),
        "advisory_reason_codes": list(candidate.get("advisory_reason_codes") or []),
        "selected_candidate_strength": candidate["selected_candidate_strength"],
        "selected_stability_label": candidate["selected_stability_label"],
        "drift_direction": candidate["drift_direction"],
        "score_delta": candidate.get("score_delta"),
        "source_preference": candidate.get("source_preference"),
        "edge_stability_score": candidate.get("edge_stability_score"),
        "selected_visible_horizons": candidate.get("selected_visible_horizons"),
        "latest_sample_size": candidate.get("latest_sample_size"),
        "cumulative_sample_size": candidate.get("cumulative_sample_size"),
        "symbol_cumulative_support": candidate.get("symbol_cumulative_support"),
        "strategy_cumulative_support": candidate.get("strategy_cumulative_support"),
        "aggregate_score": candidate.get("aggregate_score"),
        "sample_count": candidate.get("sample_count"),
        "labeled_count": candidate.get("labeled_count"),
        "coverage_pct": candidate.get("coverage_pct"),
        "median_future_return_pct": candidate.get("median_future_return_pct"),
        "avg_future_return_pct": candidate.get("avg_future_return_pct"),
        "positive_rate_pct": candidate.get("positive_rate_pct"),
        "robustness_signal_pct": candidate.get("robustness_signal_pct"),
        "supporting_major_deficit_count": candidate.get("supporting_major_deficit_count"),
        "ranking_signals": candidate.get("ranking_signals"),
        "stability_gate_pass": candidate.get("stability_gate_pass"),
        "drift_blocked": candidate.get("drift_blocked"),
        "relaxed_single_horizon": candidate.get("relaxed_single_horizon"),
        "gate_diagnostics": candidate.get("gate_diagnostics"),
    }
    return {key: value for key, value in item.items() if value is not None}


def _build_shadow_output(
    *,
    generated_at: str,
    selection_status: str,
    reason_codes: list[str],
    ranking: list[dict[str, Any]],
    candidates_considered: int,
    latest_window_record_count: int | None,
    cumulative_record_count: int | None,
    selection_explanation: str,
    selected: dict[str, Any] | None = None,
    abstain_diagnosis: dict[str, Any] | None = None,
    candidate_seed_count: int | None = None,
    candidate_seed_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "generated_at": generated_at,
        "mode": "shadow",
        "selection_status": selection_status,
        "reason_codes": reason_codes,
        "candidates_considered": candidates_considered,
        "selected_symbol": selected.get("symbol") if selected is not None else None,
        "selected_strategy": selected.get("strategy") if selected is not None else None,
        "selected_horizon": selected.get("horizon") if selected is not None else None,
        "selection_score": selected.get("selection_score") if selected is not None else None,
        "selection_confidence": selected.get("selection_confidence")
        if selected is not None
        else None,
        "latest_window_record_count": latest_window_record_count,
        "cumulative_record_count": cumulative_record_count,
        "selection_explanation": selection_explanation,
        "ranking": ranking,
    }
    if abstain_diagnosis is not None:
        payload["abstain_diagnosis"] = abstain_diagnosis
    if candidate_seed_count is not None:
        payload["candidate_seed_count"] = candidate_seed_count
    if isinstance(candidate_seed_diagnostics, dict) and candidate_seed_diagnostics:
        payload["candidate_seed_diagnostics"] = candidate_seed_diagnostics
    return payload


def _build_gate_diagnostics(
    *,
    candidate_status: str,
    strength: str,
    stability: str,
    drift_direction: str,
    edge_stability_score: float | int | None,
    blocked_reasons: list[str],
    penalty_reasons: list[str],
    advisory_reason_codes: list[str],
    relaxed_single_horizon: bool,
) -> dict[str, dict[str, Any]]:
    score_gate_reasons: list[str] = []
    if (
        edge_stability_score is not None
        and float(edge_stability_score) < SHADOW_MIN_EDGE_STABILITY_SCORE
    ):
        score_gate_reasons.append(PENALTY_REASON_MAP["low_edge_score"])

    stability_gate_reasons: list[str] = []
    if strength == "insufficient_data":
        stability_gate_reasons.append(BLOCKED_REASON_MAP["insufficient_data_strength"])
    if stability == "insufficient_data":
        stability_gate_reasons.append(BLOCKED_REASON_MAP["insufficient_data_stability"])
    if stability == "unstable":
        stability_gate_reasons.append(BLOCKED_REASON_MAP["unstable_stability"])
    if stability == "single_horizon_only" and not relaxed_single_horizon:
        stability_gate_reasons.append(PENALTY_REASON_MAP["single_horizon_stability"])

    drift_gate_reasons: list[str] = []
    if drift_direction == "decrease":
        drift_gate_reasons.append(PENALTY_REASON_MAP["decreasing_drift"])

    eligibility_gate_reasons = list(blocked_reasons or penalty_reasons)

    return {
        "score_gate": {
            "passed": len(score_gate_reasons) == 0,
            "reason_codes": score_gate_reasons,
        },
        "stability_gate": {
            "passed": len(stability_gate_reasons) == 0,
            "reason_codes": stability_gate_reasons,
            "relaxed_single_horizon": relaxed_single_horizon,
        },
        "drift_gate": {
            "passed": len(drift_gate_reasons) == 0,
            "reason_codes": drift_gate_reasons,
        },
        "eligibility_gate": {
            "passed": candidate_status == "eligible",
            "reason_codes": eligibility_gate_reasons,
            "relaxed_single_horizon": relaxed_single_horizon,
        },
        "advisory": {
            "reason_codes": list(advisory_reason_codes),
        },
    }


def _build_abstain_diagnosis(
    *,
    category: str,
    summary: str,
    ranking: list[dict[str, Any]],
    eligible_count: int,
    penalized_count: int,
    blocked_count: int,
    compared_candidate: dict[str, Any] | None = None,
    candidate_seed_count: int | None = None,
    candidate_seed_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnosis = {
        "category": category,
        "summary": summary,
        "eligible_candidate_count": eligible_count,
        "penalized_candidate_count": penalized_count,
        "blocked_candidate_count": blocked_count,
        "top_candidate": _build_diagnostic_candidate_snapshot(ranking[0] if ranking else None),
    }
    if candidate_seed_count is not None:
        diagnosis["candidate_seed_count"] = candidate_seed_count
    if isinstance(candidate_seed_diagnostics, dict) and candidate_seed_diagnostics:
        diagnosis["candidate_seed_diagnostics"] = candidate_seed_diagnostics
    if compared_candidate is not None:
        diagnosis["compared_candidate"] = _build_diagnostic_candidate_snapshot(
            compared_candidate
        )
    return diagnosis


def _build_diagnostic_candidate_snapshot(
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None

    return {
        "symbol": candidate.get("symbol"),
        "strategy": candidate.get("strategy"),
        "horizon": candidate.get("horizon"),
        "candidate_status": candidate.get("candidate_status"),
        "selection_score": candidate.get("selection_score"),
        "selection_confidence": candidate.get("selection_confidence"),
        "reason_codes": list(candidate.get("reason_codes") or []),
        "advisory_reason_codes": list(candidate.get("advisory_reason_codes") or []),
        "aggregate_score": candidate.get("aggregate_score"),
        "sample_count": candidate.get("sample_count"),
        "median_future_return_pct": candidate.get("median_future_return_pct"),
        "positive_rate_pct": candidate.get("positive_rate_pct"),
        "robustness_signal_pct": candidate.get("robustness_signal_pct"),
        "supporting_major_deficit_count": candidate.get("supporting_major_deficit_count"),
        "ranking_signals": candidate.get("ranking_signals") or {},
        "gate_diagnostics": candidate.get("gate_diagnostics") or {},
    }


def _build_shadow_log_context(payload: dict[str, Any]) -> dict[str, Any]:
    ranking = payload.get("ranking") if isinstance(payload.get("ranking"), list) else []
    ranking_items = [item for item in ranking if isinstance(item, dict)]
    top_candidate = ranking_items[0] if ranking_items else None
    candidate_status_counts = {
        status: sum(
            1 for item in ranking_items if item.get("candidate_status") == status
        )
        for status in VALID_CANDIDATE_STATUSES
    }

    context = {
        "selection_status": payload.get("selection_status"),
        "reason_codes": list(payload.get("reason_codes") or []),
        "selection_explanation": payload.get("selection_explanation"),
        "candidates_considered": payload.get("candidates_considered"),
        "candidate_status_counts": candidate_status_counts,
        "top_candidate": _build_diagnostic_candidate_snapshot(top_candidate),
    }

    if payload.get("candidate_seed_count") is not None:
        context["candidate_seed_count"] = payload.get("candidate_seed_count")
    if isinstance(payload.get("candidate_seed_diagnostics"), dict):
        context["candidate_seed_diagnostics"] = payload.get("candidate_seed_diagnostics")

    abstain_diagnosis = payload.get("abstain_diagnosis")
    if isinstance(abstain_diagnosis, dict):
        context["abstain_diagnosis"] = abstain_diagnosis

    return context


def _finalize_output(payload: dict[str, Any]) -> dict[str, Any]:
    validation_result = validate_shadow_output(payload)
    if not validation_result.is_valid:
        joined = "; ".join(validation_result.errors)
        raise ValueError(f"Generated invalid shadow output: {joined}")

    logger.info(
        "Shadow selection decision: %s",
        json.dumps(
            _build_shadow_log_context(payload),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return payload


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_strength(value: Any) -> str:
    normalized = _normalize_text(value)
    if normalized in STRENGTH_PRIORITY:
        return normalized
    return "insufficient_data"


def _normalize_stability(value: Any) -> str:
    normalized = _normalize_text(value)
    if normalized in STABILITY_PRIORITY:
        return normalized
    return "insufficient_data"


def _normalize_drift_direction(value: Any) -> str:
    normalized = _normalize_text(value)
    if normalized in DRIFT_PRIORITY:
        return normalized
    return "insufficient_history"


def _normalize_source_preference(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized in VALID_SOURCE_PREFERENCES:
        return normalized
    return None


def _normalize_horizon(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized in VALID_HORIZONS:
        return normalized
    return None


def _normalize_horizon_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None

    normalized: list[str] = []
    for item in value:
        horizon = _normalize_horizon(item)
        if horizon is not None and horizon not in normalized:
            normalized.append(horizon)
    return normalized or None


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value