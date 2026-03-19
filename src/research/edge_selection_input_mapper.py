from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.edge_selection_schema_validator import validate_upstream_reports

REQUIRED_REPORTS = {
    "latest": Path("latest") / "summary.json",
    "comparison": Path("comparison") / "summary.json",
    "edge_scores": Path("edge_scores") / "summary.json",
    "score_drift": Path("score_drift") / "summary.json",
}
OPTIONAL_HISTORY_REPORT = Path("edge_scores_history.jsonl")

SOURCE_FIELDS = {
    "latest": {
        "candidate_strength": "latest_candidate_strength",
        "stability_label": "latest_stability_label",
    },
    "cumulative": {
        "candidate_strength": "cumulative_candidate_strength",
        "stability_label": "cumulative_stability_label",
    },
}
VALID_SOURCE_PREFERENCES = {"latest", "cumulative", "n/a"}
VALID_HORIZONS = {"15m", "1h", "4h"}
SUPPORT_CATEGORY_ORDER = ("symbol", "strategy")
INVALID_IDENTIFIER_VALUES = {
    "insufficient_data",
    "unstable",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
}


def map_edge_selection_input(
    base_dir: Path,
    max_age_minutes: int = 90,
) -> dict[str, Any]:
    """Normalize research reports into a single payload for future edge selection."""

    reports_dir = Path(base_dir)
    generated_at = datetime.now(UTC).isoformat()

    validation_result = validate_upstream_reports(
        reports_dir,
        max_age_minutes=max_age_minutes,
    )
    if not validation_result.is_valid:
        return _build_result(
            ok=False,
            generated_at=generated_at,
            latest_window_record_count=None,
            cumulative_record_count=None,
            candidates=[],
            errors=list(validation_result.errors),
            warnings=list(validation_result.warnings),
            history_line_count=None,
            candidate_seed_count=0,
            candidate_seed_diagnostics={},
        )

    try:
        latest_summary = _load_json_report(reports_dir / REQUIRED_REPORTS["latest"])
        comparison_summary = _load_json_report(reports_dir / REQUIRED_REPORTS["comparison"])
        edge_scores_summary = _load_json_report(reports_dir / REQUIRED_REPORTS["edge_scores"])
        score_drift_summary = _load_json_report(reports_dir / REQUIRED_REPORTS["score_drift"])
        history_line_count = _load_optional_history_line_count(
            reports_dir / OPTIONAL_HISTORY_REPORT
        )
    except ValueError as exc:
        return _build_result(
            ok=False,
            generated_at=generated_at,
            latest_window_record_count=None,
            cumulative_record_count=None,
            candidates=[],
            errors=[str(exc)],
            warnings=list(validation_result.warnings),
            history_line_count=None,
            candidate_seed_count=0,
            candidate_seed_diagnostics={},
        )

    latest_window_record_count = _extract_latest_record_count(
        latest_summary,
        comparison_summary,
    )
    cumulative_record_count = _extract_cumulative_record_count(comparison_summary)

    score_lookup = _build_score_lookup(edge_scores_summary)
    drift_lookup = _build_drift_lookup(score_drift_summary)

    seeds = _build_candidate_seeds(comparison_summary)
    candidate_seed_diagnostics = _build_candidate_seed_diagnostics(comparison_summary, seeds)

    candidates = [
        _build_candidate(
            seed,
            score_lookup=score_lookup,
            drift_lookup=drift_lookup,
        )
        for seed in seeds
    ]
    candidates = [candidate for candidate in candidates if _is_valid_candidate(candidate)]
    candidates = _dedupe_candidates(candidates)
    candidates.sort(key=_candidate_sort_key)

    return _build_result(
        ok=True,
        generated_at=generated_at,
        latest_window_record_count=latest_window_record_count,
        cumulative_record_count=cumulative_record_count,
        candidates=candidates,
        errors=[],
        warnings=list(validation_result.warnings),
        history_line_count=history_line_count,
        candidate_seed_count=len(seeds),
        candidate_seed_diagnostics=candidate_seed_diagnostics,
    )


def _build_result(
    *,
    ok: bool,
    generated_at: str,
    latest_window_record_count: int | None,
    cumulative_record_count: int | None,
    candidates: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    history_line_count: int | None,
    candidate_seed_count: int,
    candidate_seed_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "generated_at": generated_at,
        "latest_window_record_count": latest_window_record_count,
        "cumulative_record_count": cumulative_record_count,
        "candidates": candidates,
        "errors": errors,
        "warnings": warnings,
        "history_line_count": history_line_count,
        "candidate_seed_count": candidate_seed_count,
        "candidate_seed_diagnostics": candidate_seed_diagnostics,
    }


def _load_json_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Failed to read upstream report {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Upstream report is not valid JSON {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Upstream report must contain a JSON object: {path}")

    return payload


def _load_optional_history_line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        raise ValueError(f"Failed to read optional history report {path}: {exc}") from exc


def _extract_latest_record_count(
    latest_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> int | None:
    latest_overview = _coerce_dict(latest_summary.get("dataset_overview"))
    value = _coerce_non_negative_int(latest_overview.get("total_records"))
    if value is not None:
        return value

    comparison_overview = _coerce_dict(
        comparison_summary.get("dataset_overview_comparison")
    )
    return _coerce_non_negative_int(comparison_overview.get("latest_total_records"))


def _extract_cumulative_record_count(comparison_summary: dict[str, Any]) -> int | None:
    comparison_overview = _coerce_dict(
        comparison_summary.get("dataset_overview_comparison")
    )
    return _coerce_non_negative_int(comparison_overview.get("cumulative_total_records"))


def _build_candidate_seeds(comparison_summary: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = _coerce_dict(comparison_summary.get("edge_candidates_comparison"))
    seeds: list[dict[str, Any]] = []

    for horizon in sorted(comparison):
        if horizon not in VALID_HORIZONS:
            continue

        horizon_payload = _coerce_dict(comparison.get(horizon))
        if not horizon_payload:
            continue

        symbol = _first_valid_identifier(
            horizon_payload.get("latest_top_symbol_group"),
            horizon_payload.get("cumulative_top_symbol_group"),
        )
        strategy = _first_valid_identifier(
            horizon_payload.get("latest_top_strategy_group"),
            horizon_payload.get("cumulative_top_strategy_group"),
        )

        if symbol is None and strategy is None:
            continue

        seeds.append(
            {
                "horizon": horizon,
                "symbol": symbol,
                "strategy": strategy,
                "latest_candidate_strength": _normalize_text(
                    horizon_payload.get("latest_candidate_strength")
                ),
                "cumulative_candidate_strength": _normalize_text(
                    horizon_payload.get("cumulative_candidate_strength")
                ),
            }
        )

    return seeds


def _build_candidate_seed_diagnostics(
    comparison_summary: dict[str, Any],
    seeds: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison = _coerce_dict(comparison_summary.get("edge_candidates_comparison"))
    horizon_diagnostics: list[dict[str, Any]] = []
    horizons_with_seed: list[str] = []
    horizons_without_seed: list[str] = []
    all_horizons_insufficient_data = True

    for horizon in sorted(comparison):
        if horizon not in VALID_HORIZONS:
            continue

        horizon_payload = _coerce_dict(comparison.get(horizon))
        if not horizon_payload:
            horizons_without_seed.append(horizon)
            horizon_diagnostics.append(
                {
                    "horizon": horizon,
                    "seed_generated": False,
                    "blocker_reasons": ["missing_horizon_payload"],
                }
            )
            all_horizons_insufficient_data = False
            continue

        latest_strength = _normalize_text(horizon_payload.get("latest_candidate_strength"))
        cumulative_strength = _normalize_text(
            horizon_payload.get("cumulative_candidate_strength")
        )

        latest_symbol = _normalize_text(horizon_payload.get("latest_top_symbol_group"))
        cumulative_symbol = _normalize_text(
            horizon_payload.get("cumulative_top_symbol_group")
        )
        latest_strategy = _normalize_text(horizon_payload.get("latest_top_strategy_group"))
        cumulative_strategy = _normalize_text(
            horizon_payload.get("cumulative_top_strategy_group")
        )
        latest_alignment_state = _normalize_text(
            horizon_payload.get("latest_top_alignment_state_group")
        )
        cumulative_alignment_state = _normalize_text(
            horizon_payload.get("cumulative_top_alignment_state_group")
        )

        valid_symbol = _first_valid_identifier(latest_symbol, cumulative_symbol)
        valid_strategy = _first_valid_identifier(latest_strategy, cumulative_strategy)

        seed_generated = any(
            seed.get("horizon") == horizon
            for seed in seeds
        )

        if seed_generated:
            horizons_with_seed.append(horizon)
        else:
            horizons_without_seed.append(horizon)

        latest_is_insufficient = latest_strength == "insufficient_data"
        cumulative_is_insufficient = cumulative_strength == "insufficient_data"
        if not (latest_is_insufficient and cumulative_is_insufficient):
            all_horizons_insufficient_data = False

        blocker_reasons: list[str] = []
        if latest_is_insufficient and cumulative_is_insufficient:
            blocker_reasons.append("candidate_strength_insufficient_data")
        if valid_symbol is None:
            blocker_reasons.append("no_valid_symbol_group")
        if valid_strategy is None:
            blocker_reasons.append("no_valid_strategy_group")
        if valid_symbol is None and valid_strategy is None:
            blocker_reasons.append("no_valid_symbol_or_strategy_group")

        horizon_diagnostics.append(
            {
                "horizon": horizon,
                "seed_generated": seed_generated,
                "latest_candidate_strength": latest_strength or "n/a",
                "cumulative_candidate_strength": cumulative_strength or "n/a",
                "latest_top_symbol_group": latest_symbol or "n/a",
                "cumulative_top_symbol_group": cumulative_symbol or "n/a",
                "latest_top_strategy_group": latest_strategy or "n/a",
                "cumulative_top_strategy_group": cumulative_strategy or "n/a",
                "latest_top_alignment_state_group": latest_alignment_state or "n/a",
                "cumulative_top_alignment_state_group": cumulative_alignment_state or "n/a",
                "has_valid_symbol_group": valid_symbol is not None,
                "has_valid_strategy_group": valid_strategy is not None,
                "blocker_reasons": blocker_reasons,
            }
        )

    return {
        "total_horizons_evaluated": len(horizon_diagnostics),
        "candidate_seed_count": len(seeds),
        "horizons_with_seed": horizons_with_seed,
        "horizons_without_seed": horizons_without_seed,
        "all_horizons_insufficient_data": all_horizons_insufficient_data,
        "horizon_diagnostics": horizon_diagnostics,
    }


def _build_score_lookup(
    edge_scores_summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    scores = _coerce_dict(edge_scores_summary.get("edge_stability_scores"))
    lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for category in SUPPORT_CATEGORY_ORDER:
        items = scores.get(category, [])
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            group = _normalize_identifier(item.get("group"))
            if group is None:
                continue

            lookup[(category, group)] = item

    return lookup


def _build_drift_lookup(
    score_drift_summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    items = score_drift_summary.get("score_drift", [])
    lookup: dict[tuple[str, str], dict[str, Any]] = {}

    if not isinstance(items, list):
        return lookup

    for item in items:
        if not isinstance(item, dict):
            continue

        category = _normalize_text(item.get("category"))
        group = _normalize_identifier(item.get("group"))
        if category is None or group is None:
            continue

        lookup[(category, group)] = item

    return lookup


def _build_candidate(
    seed: dict[str, Any],
    *,
    score_lookup: dict[tuple[str, str], dict[str, Any]],
    drift_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "symbol": _normalize_identifier(seed.get("symbol")),
        "strategy": _normalize_identifier(seed.get("strategy")),
        "horizon": _normalize_horizon(seed.get("horizon")),
    }
    candidate = {key: value for key, value in candidate.items() if value is not None}

    support = _select_supporting_score(seed, score_lookup)

    if support is None:
        fallback_strength = _first_non_empty(
            seed.get("latest_candidate_strength"),
            seed.get("cumulative_candidate_strength"),
        )
        if fallback_strength is not None:
            candidate["selected_candidate_strength"] = fallback_strength
        return candidate

    category, group, score_item = support
    source_preference = _normalize_source_preference(score_item.get("source_preference"))

    selected_strength = _resolve_strength(score_item, seed, source_preference)
    if selected_strength is not None:
        candidate["selected_candidate_strength"] = selected_strength

    selected_stability_label = _resolve_stability_label(score_item, source_preference)
    if selected_stability_label is not None:
        candidate["selected_stability_label"] = selected_stability_label

    if source_preference is not None:
        candidate["source_preference"] = source_preference

    edge_stability_score = _coerce_number(score_item.get("score"))
    if edge_stability_score is not None:
        candidate["edge_stability_score"] = edge_stability_score

    selected_visible_horizons = _normalize_horizon_list(
        score_item.get("selected_visible_horizons")
    )
    if selected_visible_horizons:
        candidate["selected_visible_horizons"] = selected_visible_horizons

    symbol_cumulative_support = _coerce_non_negative_int(
        score_item.get("symbol_cumulative_support")
    )
    if symbol_cumulative_support is not None:
        candidate["symbol_cumulative_support"] = symbol_cumulative_support

    strategy_cumulative_support = _coerce_non_negative_int(
        score_item.get("strategy_cumulative_support")
    )
    if strategy_cumulative_support is not None:
        candidate["strategy_cumulative_support"] = strategy_cumulative_support

    latest_sample_size = _coerce_non_negative_int(score_item.get("latest_sample_size"))
    if latest_sample_size is not None:
        candidate["latest_sample_size"] = latest_sample_size

    cumulative_sample_size = _coerce_non_negative_int(
        score_item.get("cumulative_sample_size")
    )
    if cumulative_sample_size is not None:
        candidate["cumulative_sample_size"] = cumulative_sample_size

    drift_item = drift_lookup.get((category, group))
    if drift_item is not None:
        drift_direction = _normalize_text(drift_item.get("drift_direction"))
        if drift_direction is not None:
            candidate["drift_direction"] = drift_direction

        score_delta = _coerce_number(drift_item.get("score_delta"))
        if score_delta is not None:
            candidate["score_delta"] = score_delta

    return candidate


def _select_supporting_score(
    seed: dict[str, Any],
    score_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str, dict[str, Any]] | None:
    matches: list[tuple[float, int, str, str, dict[str, Any]]] = []

    for category_index, category in enumerate(SUPPORT_CATEGORY_ORDER):
        group = _normalize_identifier(seed.get(category))
        if group is None:
            continue

        score_item = score_lookup.get((category, group))
        if score_item is None:
            continue

        score_value = _coerce_number(score_item.get("score"))
        matches.append(
            (
                float(score_value) if score_value is not None else float("-inf"),
                -category_index,
                category,
                group,
                score_item,
            )
        )

    if not matches:
        return None

    matches.sort(reverse=True)
    _, _, category, group, score_item = matches[0]
    return category, group, score_item


def _resolve_strength(
    score_item: dict[str, Any],
    seed: dict[str, Any],
    source_preference: str | None,
) -> str | None:
    if source_preference in {"latest", "cumulative"}:
        field_name = SOURCE_FIELDS[source_preference]["candidate_strength"]
        value = _normalize_text(score_item.get(field_name))
        if value is not None:
            return value

    if source_preference == "n/a":
        value = _first_non_empty(
            score_item.get(SOURCE_FIELDS["latest"]["candidate_strength"]),
            score_item.get(SOURCE_FIELDS["cumulative"]["candidate_strength"]),
        )
        if value is not None:
            return value

    return _first_non_empty(
        seed.get("latest_candidate_strength"),
        seed.get("cumulative_candidate_strength"),
    )


def _resolve_stability_label(
    score_item: dict[str, Any],
    source_preference: str | None,
) -> str | None:
    if source_preference in {"latest", "cumulative"}:
        field_name = SOURCE_FIELDS[source_preference]["stability_label"]
        value = _normalize_text(score_item.get(field_name))
        if value is not None:
            return value

    if source_preference == "n/a":
        return _first_non_empty(
            score_item.get(SOURCE_FIELDS["latest"]["stability_label"]),
            score_item.get(SOURCE_FIELDS["cumulative"]["stability_label"]),
        )

    return _first_non_empty(
        score_item.get(SOURCE_FIELDS["latest"]["stability_label"]),
        score_item.get(SOURCE_FIELDS["cumulative"]["stability_label"]),
    )


def _is_valid_candidate(candidate: dict[str, Any]) -> bool:
    symbol = _normalize_identifier(candidate.get("symbol"))
    strategy = _normalize_identifier(candidate.get("strategy"))
    horizon = _normalize_horizon(candidate.get("horizon"))
    return symbol is not None and strategy is not None and horizon is not None


def _normalize_source_preference(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized in VALID_SOURCE_PREFERENCES:
        return normalized
    return None


def _normalize_horizon_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None

    normalized: list[str] = []
    for item in value:
        item_text = _normalize_horizon(item)
        if item_text is not None and item_text not in normalized:
            normalized.append(item_text)

    return normalized or None


def _normalize_horizon(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized in VALID_HORIZONS:
        return normalized
    return None


def _normalize_identifier(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None
    if normalized.lower() in INVALID_IDENTIFIER_VALUES:
        return None
    return normalized


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            candidate.get("symbol"),
            candidate.get("strategy"),
            candidate.get("horizon"),
        )
        deduped[key] = candidate
    return list(deduped.values())


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate.get("horizon") or ""),
        str(candidate.get("symbol") or ""),
        str(candidate.get("strategy") or ""),
    )


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        normalized = _normalize_text(value)
        if normalized is not None:
            return normalized
    return None


def _first_valid_identifier(*values: Any) -> str | None:
    for value in values:
        normalized = _normalize_identifier(value)
        if normalized is not None:
            return normalized
    return None


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized or None


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value
