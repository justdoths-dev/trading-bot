import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research import sub_floor_candidate_validation as baseline

REPORT_TYPE = "historical_sub_floor_candidate_validation"
REPORT_VERSION = "v3"
DEFAULT_JSON_OUTPUT_NAME = "historical_sub_floor_candidate_validation_summary.json"
DEFAULT_MD_OUTPUT_NAME = "historical_sub_floor_candidate_validation_summary.md"
ELIGIBLE_CLASSIFICATION = "eligible"

MATURATION_OUTCOME_ORDER = {
    "matured_to_eligible": 0,
    "grew_but_remained_sub_floor": 1,
    "degraded_before_maturity": 2,
    "structurally_blocked_later": 3,
    "disappeared": 4,
}

SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE = (
    "prefer_edge_candidate_rows_rows_over_diagnostic_rows_for_same_snapshot_identity"
)

MIN_FOLLOWUP_IDENTITIES_FOR_SIGNAL = 3
MIN_MATURED_TO_ELIGIBLE_RATIO_FOR_SIGNAL = 0.5


def _classification(value: dict[str, Any]) -> str:
    return str(
        baseline._safe_dict(value.get("computed")).get("classification") or "unknown"
    )


def _identity_key(value: dict[str, Any]) -> str:
    return str(baseline._safe_dict(value.get("identity")).get("identity_key") or "")


def _sample_count(value: dict[str, Any]) -> int | None:
    return baseline._safe_int(
        baseline._safe_dict(value.get("facts")).get("sample_count")
    )


def _sample_count_band(value: dict[str, Any]) -> str:
    return str(
        baseline._safe_dict(value.get("computed")).get("sample_count_band") or "unknown"
    )


def _has_structural_blocker(value: dict[str, Any]) -> bool:
    return bool(
        baseline._safe_list(
            baseline._safe_dict(value.get("computed")).get(
                "structural_non_sample_blockers"
            )
        )
    )


def _is_eligible(value: dict[str, Any]) -> bool:
    return _classification(value) == ELIGIBLE_CLASSIFICATION


def _is_qualifying_sub_floor_observation(value: dict[str, Any]) -> bool:
    computed = baseline._safe_dict(value.get("computed"))
    return bool(computed.get("qualifying_sub_floor_candidate"))


def _history_observation_sort_key(value: dict[str, Any]) -> tuple[Any, ...]:
    facts = baseline._safe_dict(value.get("facts"))
    return (
        int(value.get("snapshot_index") or 0),
        baseline._window_sort_key(str(value.get("window_label") or "")),
        baseline._classification_sort_key(_classification(value)),
        -(_sample_count(value) or -1),
        str(facts.get("rejection_reason") or ""),
    )


def _ratio_or_none(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return baseline._ratio(count, total)


def _maturation_sort_key(name: str) -> tuple[int, str]:
    return (MATURATION_OUTCOME_ORDER.get(name, 999), name)


def _identity_summary_sort_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _maturation_sort_key(str(value.get("maturation_outcome") or "")),
        -int(value.get("total_appearances") or 0),
        -int(value.get("highest_sample_count") or 0),
        str(value.get("identity_key") or ""),
    )


def _attach_history_metadata(
    observation: dict[str, Any],
    *,
    snapshot_index: int,
    snapshot_kind: str,
) -> dict[str, Any]:
    cloned = dict(observation)
    computed = dict(baseline._safe_dict(cloned.get("computed")))
    classification = _classification(observation)

    qualifying_sub_floor_candidate = (
        snapshot_kind == "diagnostic"
        and bool(computed.get("is_sample_floor_blocked"))
        and not baseline._safe_list(computed.get("structural_non_sample_blockers"))
        and classification in baseline.NEAR_MISS_CLASSIFICATIONS
    )

    computed["qualifying_sub_floor_candidate"] = qualifying_sub_floor_candidate
    computed["eligible_at_snapshot"] = classification == ELIGIBLE_CLASSIFICATION
    cloned["computed"] = computed
    cloned["snapshot_index"] = snapshot_index
    cloned["snapshot_kind"] = snapshot_kind
    return cloned


def _normalize_selected_observation(
    row: dict[str, Any],
    *,
    source_path: Path,
    source_name: str,
    window_label: str,
    window_label_source: str,
) -> dict[str, Any]:
    symbol = baseline._normalize_symbol(row.get("symbol"))
    strategy = baseline._normalize_strategy(row.get("strategy"))
    horizon = baseline._normalize_horizon(row.get("horizon"))
    identity_key = f"{symbol}:{strategy}:{horizon}"

    sample_count = baseline._safe_int(row.get("sample_count"))
    labeled_count = baseline._safe_int(row.get("labeled_count"))
    coverage_pct = baseline._safe_float(row.get("coverage_pct"))
    median_future_return_pct = baseline._safe_float(row.get("median_future_return_pct"))
    positive_rate_pct = baseline._safe_float(row.get("positive_rate_pct"))
    robustness_signal = baseline._safe_text(row.get("robustness_signal"))
    robustness_signal_pct = baseline._safe_float(row.get("robustness_signal_pct"))
    aggregate_score = baseline._safe_float(row.get("aggregate_score"))
    candidate_strength = baseline._safe_text(
        row.get("selected_candidate_strength") or row.get("candidate_strength")
    )

    warnings = baseline._build_observation_warnings(
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        sample_count=sample_count,
        window_label_source=window_label_source,
    )
    if sample_count is not None and sample_count < 30:
        warnings.append("eligible_row_sample_count_below_30")

    return {
        "source_path": str(source_path),
        "source_name": source_name,
        "window_label": window_label,
        "identity": {
            "identity_key": identity_key,
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
        },
        "facts": {
            "status": "selected",
            "rejection_reason": None,
            "rejection_reasons": [],
            "diagnostic_category": None,
            "sample_gate": "passed",
            "quality_gate": "passed",
            "candidate_strength": candidate_strength,
            "classification_reason": None,
            "visibility_reason": baseline._safe_text(row.get("visibility_reason"))
            or "passed_sample_and_quality_gate",
            "sample_count": sample_count,
            "labeled_count": labeled_count,
            "coverage_pct": coverage_pct,
            "median_future_return_pct": median_future_return_pct,
            "positive_rate_pct": positive_rate_pct,
            "robustness_signal": robustness_signal,
            "robustness_signal_pct": robustness_signal_pct,
            "aggregate_score": aggregate_score,
        },
        "computed": {
            "sample_count_band": baseline._sample_count_band(sample_count),
            "is_sample_floor_blocked": False,
            "structural_non_sample_blockers": [],
            "directional_quality_flags": [],
            "classification": ELIGIBLE_CLASSIFICATION,
            "classification_reasons": [
                "row is present in edge_candidate_rows.rows at snapshot time"
            ],
        },
        "warnings": warnings,
    }


def _normalize_historical_horizon_evaluation_row(
    raw_row: dict[str, Any],
    *,
    default_symbol: str | None = None,
    default_strategy: str | None = None,
    default_horizon: str | None = None,
) -> dict[str, Any]:
    metrics = baseline._safe_dict(raw_row.get("metrics"))

    symbol = baseline._normalize_symbol(raw_row.get("symbol") or default_symbol)
    strategy = baseline._normalize_strategy(raw_row.get("strategy") or default_strategy)
    horizon = baseline._normalize_horizon(raw_row.get("horizon") or default_horizon)

    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": baseline._safe_text(raw_row.get("status")) or "rejected",
        "diagnostic_category": baseline._safe_text(raw_row.get("diagnostic_category")),
        "strategy_horizon_compatible": bool(
            raw_row.get("strategy_horizon_compatible", False)
        ),
        "rejection_reason": baseline._safe_text(raw_row.get("rejection_reason")),
        "rejection_reasons": baseline._safe_list(raw_row.get("rejection_reasons")),
        "sample_gate": baseline._safe_text(raw_row.get("sample_gate")),
        "quality_gate": baseline._safe_text(raw_row.get("quality_gate")),
        "candidate_strength": baseline._safe_text(raw_row.get("candidate_strength")),
        "classification_reason": baseline._safe_text(
            raw_row.get("classification_reason")
        ),
        "aggregate_score": baseline._safe_float(raw_row.get("aggregate_score")),
        "chosen_metric_summary": baseline._safe_text(
            raw_row.get("chosen_metric_summary")
        ),
        "visibility_reason": baseline._safe_text(raw_row.get("visibility_reason")),
        "sample_count": baseline._safe_int(metrics.get("sample_count")),
        "labeled_count": baseline._safe_int(metrics.get("labeled_count")),
        "coverage_pct": baseline._safe_float(metrics.get("coverage_pct")),
        "median_future_return_pct": baseline._safe_float(
            metrics.get("median_future_return_pct")
        ),
        "positive_rate_pct": baseline._safe_float(metrics.get("positive_rate_pct")),
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": baseline._safe_float(
            metrics.get("signal_match_rate_pct")
        ),
    }


def _extract_historical_diagnostic_rows_from_identity_horizon_evaluations(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    block = baseline._safe_dict(payload.get("edge_candidate_rows"))
    identity_evaluations = baseline._safe_list(block.get("identity_horizon_evaluations"))

    diagnostic_rows: list[dict[str, Any]] = []
    for identity_eval in identity_evaluations:
        identity_eval = baseline._safe_dict(identity_eval)
        symbol = baseline._safe_text(identity_eval.get("symbol"))
        strategy = baseline._safe_text(identity_eval.get("strategy"))
        horizon_evaluations = baseline._safe_dict(identity_eval.get("horizon_evaluations"))

        for horizon_name, raw_row in horizon_evaluations.items():
            if not isinstance(raw_row, dict):
                continue
            diagnostic_rows.append(
                _normalize_historical_horizon_evaluation_row(
                    raw_row,
                    default_symbol=symbol,
                    default_strategy=strategy,
                    default_horizon=str(horizon_name),
                )
            )

    return diagnostic_rows


def _prepare_payload_for_source_summary(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    block = baseline._safe_dict(payload.get("edge_candidate_rows"))
    existing_diagnostic_rows = baseline._safe_list(block.get("diagnostic_rows"))
    if existing_diagnostic_rows:
        return payload, []

    synthesized_diagnostic_rows = (
        _extract_historical_diagnostic_rows_from_identity_horizon_evaluations(payload)
    )
    if not synthesized_diagnostic_rows:
        return payload, []

    cloned_payload = dict(payload)
    cloned_block = dict(block)
    cloned_block["diagnostic_rows"] = synthesized_diagnostic_rows
    cloned_block["diagnostic_row_count"] = len(synthesized_diagnostic_rows)
    cloned_payload["edge_candidate_rows"] = cloned_block

    return cloned_payload, [
        "diagnostic_rows_synthesized_from_identity_horizon_evaluations"
    ]


def _same_snapshot_duplicate_precedence(value: dict[str, Any]) -> tuple[Any, ...]:
    classification = _classification(value)
    snapshot_kind = str(value.get("snapshot_kind") or "")
    return (
        0 if snapshot_kind == "eligible" else 1,
        0 if classification in baseline.NEAR_MISS_CLASSIFICATIONS else 1,
        -(_sample_count(value) or -1),
        baseline._classification_sort_key(classification),
        _history_observation_sort_key(value),
    )


def _deduplicate_same_snapshot_observations(
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_rows: list[dict[str, Any]] = []

    for observation in observations:
        identity_key = _identity_key(observation)
        if not identity_key:
            unique_rows.append(observation)
            continue
        grouped[identity_key].append(observation)

    duplicate_identity_count = 0
    cross_kind_duplicate_identity_count = 0
    dropped_observation_count = 0

    for rows in grouped.values():
        if len(rows) == 1:
            unique_rows.append(rows[0])
            continue

        duplicate_identity_count += 1
        if len({str(row.get("snapshot_kind") or "") for row in rows}) > 1:
            cross_kind_duplicate_identity_count += 1
        dropped_observation_count += len(rows) - 1
        chosen = min(rows, key=_same_snapshot_duplicate_precedence)
        unique_rows.append(chosen)

    unique_rows.sort(key=_history_observation_sort_key)
    return unique_rows, {
        "same_snapshot_duplicate_identity_count": duplicate_identity_count,
        "same_snapshot_cross_kind_duplicate_identity_count": (
            cross_kind_duplicate_identity_count
        ),
        "same_snapshot_dropped_observation_count": dropped_observation_count,
    }


def _build_source_snapshot(
    path: Path,
    payload: dict[str, Any],
    payload_warnings: list[str],
    *,
    snapshot_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared_payload, preparation_warnings = _prepare_payload_for_source_summary(payload)

    diagnostic_summary, diagnostic_observations = baseline._build_source_summary(
        path,
        prepared_payload,
        payload_warnings,
    )
    block, _ = baseline._extract_edge_candidate_rows_block(prepared_payload)
    selected_rows = [
        item for item in baseline._safe_list(baseline._safe_dict(block).get("rows"))
        if isinstance(item, dict)
    ]

    eligible_observations = [
        _normalize_selected_observation(
            row,
            source_path=path,
            source_name=str(diagnostic_summary.get("source_name") or path.name),
            window_label=str(diagnostic_summary.get("window_label") or path.stem),
            window_label_source=str(
                diagnostic_summary.get("window_label_source") or "unknown"
            ),
        )
        for row in selected_rows
    ]

    all_observations = [
        _attach_history_metadata(
            observation,
            snapshot_index=snapshot_index,
            snapshot_kind="diagnostic",
        )
        for observation in diagnostic_observations
    ]
    all_observations.extend(
        _attach_history_metadata(
            observation,
            snapshot_index=snapshot_index,
            snapshot_kind="eligible",
        )
        for observation in eligible_observations
    )
    all_observations, dedupe_details = _deduplicate_same_snapshot_observations(
        all_observations
    )

    source_warnings = list(
        baseline._normalize_string_list(diagnostic_summary.get("warnings"))
    )
    source_warnings.extend(preparation_warnings)
    if dedupe_details["same_snapshot_duplicate_identity_count"] > 0:
        source_warnings.append(
            "same_snapshot_duplicate_identities_resolved_with_precedence="
            f"{SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE}"
        )

    classification_counter = Counter(_classification(item) for item in all_observations)
    source_facts = dict(baseline._safe_dict(diagnostic_summary.get("source_facts")))
    source_facts["selected_row_count"] = len(selected_rows)
    source_facts.update(dedupe_details)
    source_facts["same_snapshot_duplicate_resolution_rule"] = (
        SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE
    )

    source_summary = {
        **diagnostic_summary,
        "snapshot_index": snapshot_index,
        "source_facts": source_facts,
        "diagnostic_observation_count": len(diagnostic_observations),
        "eligible_observation_count": len(eligible_observations),
        "observation_count": len(all_observations),
        "classification_distribution": baseline._counter_dict(classification_counter),
        "warnings": source_warnings,
    }
    return source_summary, all_observations


def _group_histories(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], set[str], list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualifying_rows: list[dict[str, Any]] = []

    for observation in observations:
        identity_key = _identity_key(observation)
        if identity_key:
            histories[identity_key].append(observation)
        if _is_qualifying_sub_floor_observation(observation):
            qualifying_rows.append(observation)

    for rows in histories.values():
        rows.sort(key=_history_observation_sort_key)

    qualifying_identity_keys = {
        _identity_key(item) for item in qualifying_rows if _identity_key(item)
    }
    return histories, qualifying_identity_keys, qualifying_rows


def _timeline_row(observation: dict[str, Any]) -> dict[str, Any]:
    facts = baseline._safe_dict(observation.get("facts"))
    computed = baseline._safe_dict(observation.get("computed"))
    return {
        "snapshot_index": int(observation.get("snapshot_index") or 0),
        "window_label": observation.get("window_label"),
        "snapshot_kind": observation.get("snapshot_kind"),
        "classification": computed.get("classification"),
        "sample_count_band": computed.get("sample_count_band"),
        "sample_count": facts.get("sample_count"),
        "median_future_return_pct": facts.get("median_future_return_pct"),
        "positive_rate_pct": facts.get("positive_rate_pct"),
        "robustness_signal_pct": facts.get("robustness_signal_pct"),
        "qualifying_sub_floor_candidate": bool(
            computed.get("qualifying_sub_floor_candidate")
        ),
        "eligible_at_snapshot": bool(computed.get("eligible_at_snapshot")),
        "structural_non_sample_blockers": computed.get(
            "structural_non_sample_blockers",
            [],
        ),
    }


def _classify_maturation_outcome(rows: list[dict[str, Any]]) -> tuple[str, bool]:
    qualifying_rows = [row for row in rows if _is_qualifying_sub_floor_observation(row)]
    if not qualifying_rows:
        return "disappeared", False

    first_anchor = qualifying_rows[0]
    anchor_snapshot_index = int(first_anchor.get("snapshot_index") or 0)
    anchor_sample_count = _sample_count(first_anchor) or 0
    later_rows = [
        row for row in rows if int(row.get("snapshot_index") or 0) > anchor_snapshot_index
    ]
    if any(_is_eligible(row) for row in later_rows):
        return "matured_to_eligible", True
    if any(_has_structural_blocker(row) for row in later_rows):
        return "structurally_blocked_later", True
    if not later_rows:
        return "disappeared", False

    latest_later_row = later_rows[-1]
    latest_classification = _classification(latest_later_row)
    latest_sample_count = _sample_count(latest_later_row) or 0
    if (
        latest_classification in baseline.NEAR_MISS_CLASSIFICATIONS
        and latest_sample_count < 30
        and latest_sample_count > anchor_sample_count
    ):
        return "grew_but_remained_sub_floor", True
    return "degraded_before_maturity", True


def _build_identity_summary(
    histories: dict[str, list[dict[str, Any]]],
    qualifying_identity_keys: set[str],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    summaries: list[dict[str, Any]] = []
    outcome_counter: Counter[str] = Counter()
    identities_with_followup_count = 0

    for identity_key in qualifying_identity_keys:
        rows = list(histories.get(identity_key, []))
        if not rows:
            continue

        first_identity = baseline._safe_dict(rows[0].get("identity"))
        appearances_by_band = Counter(_sample_count_band(row) for row in rows)
        classification_distribution = Counter(_classification(row) for row in rows)
        highest_sample_count = max((_sample_count(row) or 0) for row in rows)
        maturation_outcome, has_later_followup = _classify_maturation_outcome(rows)
        if has_later_followup:
            identities_with_followup_count += 1

        summary = {
            "identity_key": identity_key,
            "symbol": first_identity.get("symbol"),
            "strategy": first_identity.get("strategy"),
            "horizon": first_identity.get("horizon"),
            "total_appearances": len(rows),
            "appearances_by_band": baseline._counter_dict(appearances_by_band),
            "classification_distribution": baseline._counter_dict(
                classification_distribution
            ),
            "ever_reached_30_plus": any(_sample_count_band(row) == "30+" for row in rows),
            "ever_became_eligible": any(_is_eligible(row) for row in rows),
            "maturation_outcome": maturation_outcome,
            "highest_sample_count": highest_sample_count,
            "has_later_followup": has_later_followup,
            "observations": [_timeline_row(row) for row in rows],
        }
        summaries.append(summary)
        outcome_counter[maturation_outcome] += 1

    summaries.sort(key=_identity_summary_sort_key)
    return summaries, outcome_counter, identities_with_followup_count


def _later_matured_to_eligible_after_snapshot(
    rows: list[dict[str, Any]],
    *,
    snapshot_index: int,
) -> bool:
    return any(
        _is_eligible(row) and int(row.get("snapshot_index") or 0) > snapshot_index
        for row in rows
    )


def _all_band_labels(observations: list[dict[str, Any]]) -> list[str]:
    labels = [label for label, _, _ in baseline.SAMPLE_COUNT_BANDS]
    observed = {
        _sample_count_band(item)
        for item in observations
        if _sample_count_band(item) not in labels
    }
    return labels + sorted(observed, key=baseline._sample_band_sort_key)


def _build_sample_band_summary(
    histories: dict[str, list[dict[str, Any]]],
    qualifying_identity_keys: set[str],
) -> list[dict[str, Any]]:
    cohort_observations = [
        observation
        for identity_key in qualifying_identity_keys
        for observation in histories.get(identity_key, [])
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in cohort_observations:
        grouped[_sample_count_band(observation)].append(observation)

    summaries: list[dict[str, Any]] = []
    for band in _all_band_labels(cohort_observations):
        rows = sorted(grouped.get(band, []), key=_history_observation_sort_key)
        identities = {_identity_key(row) for row in rows if _identity_key(row)}
        median_values = [
            baseline._safe_float(
                baseline._safe_dict(row.get("facts")).get("median_future_return_pct")
            )
            for row in rows
        ]
        median_values = [value for value in median_values if value is not None]
        positive_median_count = sum(1 for value in median_values if value > 0)
        classification_distribution = Counter(_classification(row) for row in rows)

        anchor_identity_first_snapshot: dict[str, int] = {}
        for identity_key in qualifying_identity_keys:
            matching_rows = [
                row
                for row in histories.get(identity_key, [])
                if _sample_count_band(row) == band
                and _is_qualifying_sub_floor_observation(row)
            ]
            if matching_rows:
                anchor_identity_first_snapshot[identity_key] = min(
                    int(row.get("snapshot_index") or 0) for row in matching_rows
                )

        later_matured_to_eligible_count = sum(
            1
            for identity_key, anchor_snapshot_index in anchor_identity_first_snapshot.items()
            if _later_matured_to_eligible_after_snapshot(
                histories.get(identity_key, []),
                snapshot_index=anchor_snapshot_index,
            )
        )
        maturation_basis_count = len(anchor_identity_first_snapshot)

        observation_count = len(rows)
        summaries.append(
            {
                "sample_count_band": band,
                "observation_count": observation_count,
                "distinct_identity_count": len(identities),
                "positive_median_ratio": _ratio_or_none(
                    positive_median_count,
                    observation_count,
                ),
                "median_of_medians": baseline._median_or_none(median_values),
                "classification_distribution": baseline._counter_dict(
                    classification_distribution
                ),
                "later_matured_to_eligible_ratio": _ratio_or_none(
                    later_matured_to_eligible_count,
                    maturation_basis_count,
                ),
                "later_matured_to_eligible_identity_count": (
                    later_matured_to_eligible_count
                ),
                "later_maturation_basis_identity_count": maturation_basis_count,
            }
        )

    summaries.sort(
        key=lambda item: baseline._sample_band_sort_key(str(item["sample_count_band"]))
    )
    return summaries


def _build_maturation_summary(
    identity_summary: list[dict[str, Any]],
    outcome_counter: Counter[str],
    *,
    identities_with_followup_count: int,
) -> dict[str, Any]:
    matured_to_eligible_count = outcome_counter.get("matured_to_eligible", 0)
    return {
        "identity_count": len(identity_summary),
        "identities_with_followup_count": identities_with_followup_count,
        "matured_to_eligible_followup_ratio": _ratio_or_none(
            matured_to_eligible_count,
            identities_with_followup_count,
        ),
        "outcome_distribution": {
            outcome: outcome_counter.get(outcome, 0)
            for outcome in sorted(MATURATION_OUTCOME_ORDER, key=_maturation_sort_key)
        },
        "matured_to_eligible_identity_count": matured_to_eligible_count,
        "grew_but_remained_sub_floor_identity_count": outcome_counter.get(
            "grew_but_remained_sub_floor",
            0,
        ),
        "degraded_before_maturity_identity_count": outcome_counter.get(
            "degraded_before_maturity",
            0,
        ),
        "disappeared_identity_count": outcome_counter.get("disappeared", 0),
        "structurally_blocked_later_identity_count": outcome_counter.get(
            "structurally_blocked_later",
            0,
        ),
    }


def _build_conservative_conclusion(
    identity_summary: list[dict[str, Any]],
    outcome_counter: Counter[str],
    *,
    identities_with_followup_count: int,
) -> dict[str, Any]:
    matured_to_eligible_count = outcome_counter.get("matured_to_eligible", 0)
    degraded_before_maturity_count = outcome_counter.get(
        "degraded_before_maturity",
        0,
    )
    structurally_blocked_later_count = outcome_counter.get(
        "structurally_blocked_later",
        0,
    )
    matured_to_eligible_ratio = _ratio_or_none(
        matured_to_eligible_count,
        identities_with_followup_count,
    )
    adverse_followup_count = (
        degraded_before_maturity_count + structurally_blocked_later_count
    )

    if not identity_summary or identities_with_followup_count == 0:
        threshold_change_support = "insufficient_evidence"
        reason = (
            "No qualifying sub-floor cohort with later follow-up was available, so "
            "threshold changes are not evidence-supported."
        )
    elif matured_to_eligible_count == 0:
        threshold_change_support = "not_supported"
        reason = (
            "Follow-up snapshots did not show any qualifying sub-floor identity "
            "maturing into analyzer eligibility."
        )
    elif identities_with_followup_count < MIN_FOLLOWUP_IDENTITIES_FOR_SIGNAL:
        threshold_change_support = "insufficient_evidence"
        reason = (
            "Later follow-up exists, but the cohort is still too small to support "
            "anything beyond continued observation."
        )
    elif (
        matured_to_eligible_ratio is None
        or matured_to_eligible_ratio < MIN_MATURED_TO_ELIGIBLE_RATIO_FOR_SIGNAL
    ):
        if adverse_followup_count >= matured_to_eligible_count:
            threshold_change_support = "not_supported"
            reason = (
                "Later follow-up produced too few maturation outcomes relative to "
                "degradations and structural blockers."
            )
        else:
            threshold_change_support = "insufficient_evidence"
            reason = (
                "Later follow-up shows some maturation, but the maturation ratio "
                "is still too weak for a stronger conclusion."
            )
    elif adverse_followup_count >= matured_to_eligible_count:
        threshold_change_support = "not_supported"
        reason = (
            "Maturation exists, but degradations and structural blockers remain too "
            "frequent to treat the cohort as a reliable early edge."
        )
    else:
        threshold_change_support = "followup_only"
        reason = (
            "Some qualifying sub-floor identities later became eligible, but the "
            "evidence remains diagnostic-only and supports follow-up research rather "
            "than threshold changes."
        )

    return {
        "threshold_change_support": threshold_change_support,
        "cohort_identity_count": len(identity_summary),
        "identities_with_followup_count": identities_with_followup_count,
        "matured_to_eligible_identity_count": matured_to_eligible_count,
        "matured_to_eligible_followup_ratio": matured_to_eligible_ratio,
        "degraded_before_maturity_identity_count": degraded_before_maturity_count,
        "structurally_blocked_later_identity_count": (
            structurally_blocked_later_count
        ),
        "minimum_followup_identities_for_signal": MIN_FOLLOWUP_IDENTITIES_FOR_SIGNAL,
        "minimum_matured_to_eligible_ratio_for_signal": (
            MIN_MATURED_TO_ELIGIBLE_RATIO_FOR_SIGNAL
        ),
        "reason": reason,
        "note": (
            "Snapshot classifications reuse the existing conservative validator and "
            "later maturation is tracked separately using only subsequent snapshots."
        ),
    }


def build_historical_sub_floor_candidate_validation_report(
    summary_paths: list[Path],
) -> dict[str, Any]:
    source_summaries: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []
    warnings = [
        "Historical maturation tracking uses the provided summary_paths order; pass snapshots oldest-to-newest.",
        "Chronology cannot be independently verified from summary_paths alone; incorrect input order invalidates maturation interpretation.",
        "Snapshot classifications are computed from each snapshot row only; later maturation is tracked separately.",
        "This diagnostic is offline-only and does not recommend production threshold changes.",
    ]
    if len(summary_paths) < 2:
        warnings.append(
            "Fewer than two snapshots were provided; maturation tracking may be insufficient."
        )

    for snapshot_index, raw_path in enumerate(summary_paths):
        path = Path(raw_path)
        payload, payload_warnings = baseline.load_summary_json(path)
        source_summary, source_observations = _build_source_snapshot(
            path,
            payload,
            payload_warnings,
            snapshot_index=snapshot_index,
        )
        source_summaries.append(source_summary)
        all_observations.extend(source_observations)
        warnings.extend(
            f"{path.name}: {warning}" for warning in source_summary.get("warnings", [])
        )

    all_observations.sort(key=_history_observation_sort_key)
    histories, qualifying_identity_keys, qualifying_rows = _group_histories(
        all_observations
    )
    identity_summary, outcome_counter, identities_with_followup_count = (
        _build_identity_summary(histories, qualifying_identity_keys)
    )
    sample_band_summary = _build_sample_band_summary(
        histories,
        qualifying_identity_keys,
    )
    maturation_summary = _build_maturation_summary(
        identity_summary,
        outcome_counter,
        identities_with_followup_count=identities_with_followup_count,
    )
    conservative_conclusion = _build_conservative_conclusion(
        identity_summary,
        outcome_counter,
        identities_with_followup_count=identities_with_followup_count,
    )

    window_labels = [str(item.get("window_label") or "") for item in source_summaries]
    all_classification_distribution = Counter(
        _classification(item) for item in all_observations
    )
    cohort_observations = [
        observation
        for identity_key in qualifying_identity_keys
        for observation in histories.get(identity_key, [])
    ]
    cohort_classification_distribution = Counter(
        _classification(item) for item in cohort_observations
    )
    if not qualifying_identity_keys:
        warnings.append(
            "No qualifying sub-floor candidate identities were found after excluding structural blockers, false near-miss rows, and non-positive-or-flat rows."
        )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": REPORT_TYPE,
            "report_version": REPORT_VERSION,
            "classification_version": baseline.CLASSIFICATION_VERSION,
            "summary_input_count": len(summary_paths),
            "chronology_basis": "input_order_unverified",
            "chronology_verified": False,
            "sample_count_bands": [
                label for label, _, _ in baseline.SAMPLE_COUNT_BANDS
            ],
            "same_snapshot_duplicate_resolution_rule": (
                SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE
            ),
        },
        "source_summaries": source_summaries,
        "overall_summary": {
            "all_observation_count": len(all_observations),
            "qualifying_sub_floor_observation_count": len(qualifying_rows),
            "cohort_observation_count": len(cohort_observations),
            "cohort_identity_count": len(identity_summary),
            "window_labels": window_labels,
            "all_classification_distribution": baseline._counter_dict(
                all_classification_distribution
            ),
            "cohort_classification_distribution": baseline._counter_dict(
                cohort_classification_distribution
            ),
            "maturation_outcome_distribution": maturation_summary[
                "outcome_distribution"
            ],
        },
        "sample_band_summary": sample_band_summary,
        "identity_summary": identity_summary,
        "maturation_summary": maturation_summary,
        "conservative_conclusion": conservative_conclusion,
        "warnings": sorted(set(warnings)),
    }


def build_historical_sub_floor_candidate_validation_markdown(
    summary: dict[str, Any],
) -> str:
    metadata = baseline._safe_dict(summary.get("metadata"))
    overall = baseline._safe_dict(summary.get("overall_summary"))
    conclusion = baseline._safe_dict(summary.get("conservative_conclusion"))
    sample_band_summary = baseline._safe_list(summary.get("sample_band_summary"))
    identity_summary = baseline._safe_list(summary.get("identity_summary"))
    maturation_summary = baseline._safe_dict(summary.get("maturation_summary"))
    warning_rows = baseline._normalize_string_list(summary.get("warnings"))

    lines = [
        "# Historical Sub-Floor Candidate Validation",
        "",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- report_version: {metadata.get('report_version', 'n/a')}",
        f"- classification_version: {metadata.get('classification_version', 'n/a')}",
        f"- chronology_basis: {metadata.get('chronology_basis', 'n/a')}",
        f"- chronology_verified: {metadata.get('chronology_verified', False)}",
        f"- summary_input_count: {metadata.get('summary_input_count', 0)}",
        "",
        "## Conservative Conclusion",
        f"- threshold_change_support: {conclusion.get('threshold_change_support', 'n/a')}",
        f"- matured_to_eligible_identity_count: {conclusion.get('matured_to_eligible_identity_count', 0)}",
        f"- matured_to_eligible_followup_ratio: {baseline._format_number(baseline._safe_float(conclusion.get('matured_to_eligible_followup_ratio')))}",
        f"- identities_with_followup_count: {conclusion.get('identities_with_followup_count', 0)}",
        f"- degraded_before_maturity_identity_count: {conclusion.get('degraded_before_maturity_identity_count', 0)}",
        f"- structurally_blocked_later_identity_count: {conclusion.get('structurally_blocked_later_identity_count', 0)}",
        f"- minimum_followup_identities_for_signal: {conclusion.get('minimum_followup_identities_for_signal', 'n/a')}",
        f"- minimum_matured_to_eligible_ratio_for_signal: {baseline._format_number(baseline._safe_float(conclusion.get('minimum_matured_to_eligible_ratio_for_signal')))}",
        f"- reason: {conclusion.get('reason', 'n/a')}",
        f"- note: {conclusion.get('note', 'n/a')}",
        "",
        "## Overall Summary",
        f"- all_observation_count: {overall.get('all_observation_count', 0)}",
        f"- qualifying_sub_floor_observation_count: {overall.get('qualifying_sub_floor_observation_count', 0)}",
        f"- cohort_observation_count: {overall.get('cohort_observation_count', 0)}",
        f"- cohort_identity_count: {overall.get('cohort_identity_count', 0)}",
        f"- maturation_outcome_distribution: {maturation_summary.get('outcome_distribution', {})}",
        "",
        "## Sample Bands",
    ]

    for row in sample_band_summary:
        lines.append(
            f"- {row.get('sample_count_band', 'n/a')}: "
            f"observations={row.get('observation_count', 0)}, "
            f"distinct_identity_count={row.get('distinct_identity_count', 0)}, "
            f"positive_median_ratio={baseline._format_number(baseline._safe_float(row.get('positive_median_ratio')))}, "
            f"median_of_medians={baseline._format_number(baseline._safe_float(row.get('median_of_medians')))}, "
            f"classification_distribution={row.get('classification_distribution', {})}, "
            f"later_matured_to_eligible_ratio={baseline._format_number(baseline._safe_float(row.get('later_matured_to_eligible_ratio')))}"
        )

    lines.extend(["", "## Identity Summary"])
    if not identity_summary:
        lines.append("- none")
    else:
        for row in identity_summary:
            lines.append(
                f"- {row.get('identity_key', 'n/a')}: "
                f"total_appearances={row.get('total_appearances', 0)}, "
                f"appearances_by_band={row.get('appearances_by_band', {})}, "
                f"classification_distribution={row.get('classification_distribution', {})}, "
                f"ever_reached_30_plus={row.get('ever_reached_30_plus', False)}, "
                f"ever_became_eligible={row.get('ever_became_eligible', False)}, "
                f"maturation_outcome={row.get('maturation_outcome', 'n/a')}"
            )

    lines.extend(["", "## Warnings"])
    if not warning_rows:
        lines.append("- none")
    else:
        for warning in warning_rows:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def write_historical_sub_floor_candidate_validation_report(
    *,
    summary: dict[str, Any],
    markdown: str,
    json_output_path: Path,
    markdown_output_path: Path,
) -> dict[str, str]:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def run_historical_sub_floor_candidate_validation(
    *,
    summary_paths: list[Path],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    summary = build_historical_sub_floor_candidate_validation_report(summary_paths)
    markdown = build_historical_sub_floor_candidate_validation_markdown(summary)

    result: dict[str, Any] = {
        "summary": summary,
        "markdown": markdown,
    }

    if output_dir is not None:
        outputs = write_historical_sub_floor_candidate_validation_report(
            summary=summary,
            markdown=markdown,
            json_output_path=output_dir / DEFAULT_JSON_OUTPUT_NAME,
            markdown_output_path=output_dir / DEFAULT_MD_OUTPUT_NAME,
        )
        result.update(outputs)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track whether historically observed sub-floor candidates later mature "
            "into analyzer eligibility. Pass summary_paths oldest-to-newest."
        )
    )
    parser.add_argument(
        "summary_paths",
        type=Path,
        nargs="+",
        help="Analyzer summary JSON paths ordered from oldest snapshot to newest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where JSON and Markdown reports will be written",
    )
    parser.add_argument(
        "--stdout-format",
        choices=("json", "markdown"),
        default="json",
        help="Used only when --output-dir is omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_historical_sub_floor_candidate_validation(
        summary_paths=args.summary_paths,
        output_dir=args.output_dir,
    )

    if args.output_dir is not None:
        print(
            json.dumps(
                {
                    "summary_json": result["summary_json"],
                    "summary_md": result["summary_md"],
                    "overall_summary": result["summary"]["overall_summary"],
                    "conservative_conclusion": result["summary"][
                        "conservative_conclusion"
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.stdout_format == "markdown":
        print(result["markdown"])
        return

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))