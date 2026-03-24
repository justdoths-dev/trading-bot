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
STRATEGY_HORIZON_COMPATIBILITY = {
    "scalping": {"1m", "5m"},
    "intraday": {"5m", "15m", "1h"},
    "swing": {"1h", "4h", "1d"},
}
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
STRENGTH_PRIORITY = {
    "insufficient_data": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}
RESEARCH_SIGNAL_INT_FIELDS = (
    "sample_count",
    "labeled_count",
    "supporting_major_deficit_count",
)
RESEARCH_SIGNAL_NUMBER_FIELDS = (
    "aggregate_score",
    "coverage_pct",
    "median_future_return_pct",
    "avg_future_return_pct",
    "positive_rate_pct",
    "robustness_signal_pct",
)


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

    seeds, candidate_seed_diagnostics = _build_candidate_seeds(
        latest_summary,
        comparison_summary=comparison_summary,
    )

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


def _build_candidate_seeds(
    latest_summary: dict[str, Any],
    *,
    comparison_summary: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edge_candidate_rows = _coerce_dict(latest_summary.get("edge_candidate_rows"))
    raw_rows = edge_candidate_rows.get("rows")
    rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []

    if not rows:
        return _build_legacy_candidate_seeds(comparison_summary)

    seeds: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    dropped_reason_counts: dict[str, int] = {}
    rows_by_horizon: dict[str, int] = {horizon: 0 for horizon in sorted(VALID_HORIZONS)}

    for index, row in enumerate(rows):
        seed, drop_reasons = _normalize_candidate_row_seed(row)
        horizon = _normalize_horizon(row.get("horizon"))
        if horizon is not None:
            rows_by_horizon[horizon] += 1

        if seed is None:
            dropped_rows.append(
                {
                    "row_index": index,
                    "symbol": row.get("symbol"),
                    "strategy": row.get("strategy"),
                    "horizon": row.get("horizon"),
                    "drop_reasons": drop_reasons,
                }
            )
            for reason in drop_reasons:
                dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1
            continue

        seeds.append(seed)

    horizon_diagnostics: list[dict[str, Any]] = []
    for horizon in sorted(VALID_HORIZONS):
        accepted = sum(1 for seed in seeds if seed.get("horizon") == horizon)
        dropped = sum(
            1
            for row in dropped_rows
            if _normalize_horizon(row.get("horizon")) == horizon
        )
        horizon_diagnostics.append(
            {
                "horizon": horizon,
                "joined_candidate_rows_seen": rows_by_horizon.get(horizon, 0),
                "seed_generated_count": accepted,
                "dropped_count": dropped,
                "seed_generated": accepted > 0,
            }
        )

    diagnostics = {
        "seed_source": "latest.edge_candidate_rows",
        "joined_candidate_row_count": len(rows),
        "candidate_seed_count": len(seeds),
        "dropped_candidate_row_count": len(dropped_rows),
        "dropped_candidate_row_reasons": dropped_reason_counts,
        "dropped_candidate_rows": dropped_rows,
        "horizons_with_seed": sorted({str(seed.get("horizon")) for seed in seeds if seed.get("horizon") is not None}),
        "horizons_without_seed": [
            horizon for horizon in sorted(VALID_HORIZONS)
            if not any(seed.get("horizon") == horizon for seed in seeds)
        ],
        "horizon_diagnostics": horizon_diagnostics,
    }
    return seeds, diagnostics


def _build_legacy_candidate_seeds(
    comparison_summary: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparison = _coerce_dict(_coerce_dict(comparison_summary).get("edge_candidates_comparison"))

    seeds: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    dropped_reason_counts: dict[str, int] = {}

    for horizon, item in comparison.items():
        if not isinstance(item, dict):
            continue

        seed, drop_reasons = _normalize_legacy_candidate_seed(horizon, item)
        if seed is None:
            dropped_rows.append(
                {
                    "horizon": horizon,
                    "symbol": item.get("latest_top_symbol_group"),
                    "strategy": item.get("latest_top_strategy_group"),
                    "drop_reasons": drop_reasons,
                }
            )
            for reason in drop_reasons:
                dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1
            continue

        seeds.append(seed)

    diagnostics = {
        "seed_source": "comparison.edge_candidates_comparison",
        "joined_candidate_row_count": 0,
        "candidate_seed_count": len(seeds),
        "dropped_candidate_row_count": len(dropped_rows),
        "dropped_candidate_row_reasons": dropped_reason_counts,
        "dropped_candidate_rows": dropped_rows,
        "horizons_with_seed": sorted(
            {str(seed.get("horizon")) for seed in seeds if seed.get("horizon") is not None}
        ),
        "horizons_without_seed": [
            horizon for horizon in sorted(VALID_HORIZONS)
            if not any(seed.get("horizon") == horizon for seed in seeds)
        ],
        "horizon_diagnostics": [
            {
                "horizon": horizon,
                "joined_candidate_rows_seen": 0,
                "seed_generated_count": sum(1 for seed in seeds if seed.get("horizon") == horizon),
                "dropped_count": sum(
                    1 for row in dropped_rows if _normalize_horizon(row.get("horizon")) == horizon
                ),
                "seed_generated": any(seed.get("horizon") == horizon for seed in seeds),
            }
            for horizon in sorted(VALID_HORIZONS)
        ],
    }
    return seeds, diagnostics


def _normalize_legacy_candidate_seed(
    horizon_value: Any,
    item: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    symbol = _first_valid_identifier(
        item.get("latest_top_symbol_group"),
        item.get("cumulative_top_symbol_group"),
    )
    strategy = _first_valid_identifier(
        item.get("latest_top_strategy_group"),
        item.get("cumulative_top_strategy_group"),
    )
    horizon = _normalize_horizon(horizon_value)

    reasons: list[str] = []
    if symbol is None:
        reasons.append("MISSING_SYMBOL")
    if strategy is None:
        reasons.append("MISSING_STRATEGY")
    if horizon is None:
        reasons.append("MISSING_OR_INVALID_HORIZON")

    if not reasons and not _is_strategy_horizon_compatible(strategy, horizon):
        reasons.append("INVALID_STRATEGY_HORIZON_COMBINATION")

    if reasons:
        return None, reasons

    seed = {
        "symbol": _normalize_group_for_category("symbol", symbol),
        "strategy": _normalize_group_for_category("strategy", strategy),
        "horizon": horizon,
        "selected_candidate_strength": _first_non_empty(
            item.get("latest_candidate_strength"),
            item.get("cumulative_candidate_strength"),
        ),
        "selected_stability_label": None,
        "source_preference": None,
        "edge_stability_score": None,
        "drift_direction": None,
        "score_delta": None,
        "selected_visible_horizons": None,
    }
    return seed, []


def _normalize_candidate_row_seed(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    symbol = _normalize_group_for_category("symbol", row.get("symbol"))
    strategy = _normalize_group_for_category("strategy", row.get("strategy"))
    horizon = _normalize_horizon(row.get("horizon"))

    reasons: list[str] = []
    if symbol is None:
        reasons.append("MISSING_SYMBOL")
    if strategy is None:
        reasons.append("MISSING_STRATEGY")
    if horizon is None:
        reasons.append("MISSING_OR_INVALID_HORIZON")

    if not reasons and not _is_strategy_horizon_compatible(strategy, horizon):
        reasons.append("INVALID_STRATEGY_HORIZON_COMBINATION")

    if reasons:
        return None, reasons

    seed = {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "selected_candidate_strength": _normalize_text(row.get("selected_candidate_strength")),
        "selected_stability_label": _normalize_text(row.get("selected_stability_label")),
        "source_preference": _normalize_source_preference(row.get("source_preference")),
        "edge_stability_score": _coerce_number(row.get("edge_stability_score")),
        "drift_direction": _normalize_text(row.get("drift_direction")),
        "score_delta": _coerce_number(row.get("score_delta")),
        "selected_visible_horizons": _normalize_horizon_list(row.get("selected_visible_horizons")),
    }
    _copy_research_signal_fields(source=row, target=seed)
    return seed, []


def _is_strategy_horizon_compatible(strategy: str | None, horizon: str | None) -> bool:
    if strategy is None or horizon is None:
        return False
    allowed_horizons = STRATEGY_HORIZON_COMPATIBILITY.get(strategy)
    if allowed_horizons is None:
        return False
    return horizon in allowed_horizons


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

            group = _normalize_group_for_category(category, item.get("group"))
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
        if category not in SUPPORT_CATEGORY_ORDER:
            continue

        group = _normalize_group_for_category(category, item.get("group"))
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
        "symbol": _normalize_group_for_category("symbol", seed.get("symbol")),
        "strategy": _normalize_group_for_category("strategy", seed.get("strategy")),
        "horizon": _normalize_horizon(seed.get("horizon")),
    }
    candidate = {key: value for key, value in candidate.items() if value is not None}

    for field_name in (
        "selected_candidate_strength",
        "selected_stability_label",
        "source_preference",
        "drift_direction",
    ):
        field_value = _normalize_text(seed.get(field_name))
        if field_value is not None:
            candidate[field_name] = field_value

    for field_name in ("edge_stability_score", "score_delta"):
        field_value = _coerce_number(seed.get(field_name))
        if field_value is not None:
            candidate[field_name] = field_value

    selected_visible_horizons = _normalize_horizon_list(seed.get("selected_visible_horizons"))
    if selected_visible_horizons:
        candidate["selected_visible_horizons"] = selected_visible_horizons

    _copy_research_signal_fields(source=seed, target=candidate)

    support = _select_supporting_score(seed, score_lookup)
    if support is not None:
        category, group, score_item = support
        source_preference = _normalize_source_preference(score_item.get("source_preference"))

        if candidate.get("selected_candidate_strength") is None:
            selected_strength = _resolve_score_item_selected_strength(score_item)
            if selected_strength is not None:
                candidate["selected_candidate_strength"] = selected_strength

        if candidate.get("selected_stability_label") is None:
            selected_stability_label = _resolve_stability_label(score_item, source_preference)
            if selected_stability_label is not None:
                candidate["selected_stability_label"] = selected_stability_label

        if candidate.get("source_preference") is None and source_preference is not None:
            candidate["source_preference"] = source_preference

        if candidate.get("edge_stability_score") is None:
            edge_stability_score = _coerce_number(score_item.get("score"))
            if edge_stability_score is not None:
                candidate["edge_stability_score"] = edge_stability_score

        if candidate.get("selected_visible_horizons") is None:
            selected_visible_horizons = _normalize_horizon_list(score_item.get("selected_visible_horizons"))
            if selected_visible_horizons:
                candidate["selected_visible_horizons"] = selected_visible_horizons

        symbol_cumulative_support = _coerce_non_negative_int(score_item.get("symbol_cumulative_support"))
        if symbol_cumulative_support is not None:
            candidate["symbol_cumulative_support"] = symbol_cumulative_support

        strategy_cumulative_support = _coerce_non_negative_int(score_item.get("strategy_cumulative_support"))
        if strategy_cumulative_support is not None:
            candidate["strategy_cumulative_support"] = strategy_cumulative_support

        latest_sample_size = _coerce_non_negative_int(score_item.get("latest_sample_size"))
        if latest_sample_size is not None:
            candidate["latest_sample_size"] = latest_sample_size

        cumulative_sample_size = _coerce_non_negative_int(score_item.get("cumulative_sample_size"))
        if cumulative_sample_size is not None:
            candidate["cumulative_sample_size"] = cumulative_sample_size

        drift_item = drift_lookup.get((category, group))
        if drift_item is not None:
            if candidate.get("drift_direction") is None:
                drift_direction = _normalize_text(drift_item.get("drift_direction"))
                if drift_direction is not None:
                    candidate["drift_direction"] = drift_direction

            if candidate.get("score_delta") is None:
                score_delta = _coerce_number(drift_item.get("score_delta"))
                if score_delta is not None:
                    candidate["score_delta"] = score_delta

    if candidate.get("source_preference") is None:
        candidate["source_preference"] = "n/a"

    return candidate


def _copy_research_signal_fields(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
) -> None:
    for field_name in RESEARCH_SIGNAL_INT_FIELDS:
        field_value = _coerce_non_negative_int(source.get(field_name))
        if field_value is not None:
            target[field_name] = field_value

    for field_name in RESEARCH_SIGNAL_NUMBER_FIELDS:
        field_value = _coerce_number(source.get(field_name))
        if field_value is not None:
            target[field_name] = field_value


def _select_supporting_score(
    seed: dict[str, Any],
    score_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str, dict[str, Any]] | None:
    matches: list[tuple[str, str, dict[str, Any]]] = []

    for category in SUPPORT_CATEGORY_ORDER:
        group = _normalize_group_for_category(category, seed.get(category))
        if group is None:
            continue

        score_item = score_lookup.get((category, group))
        if score_item is None:
            continue

        matches.append((category, group, score_item))

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    symbol_match = next((m for m in matches if m[0] == "symbol"), None)
    strategy_match = next((m for m in matches if m[0] == "strategy"), None)

    if symbol_match is not None and strategy_match is not None:
        symbol_strength = _resolve_score_item_selected_strength(symbol_match[2])
        strategy_strength = _resolve_score_item_selected_strength(strategy_match[2])

        if (
            _strength_rank(symbol_strength) >= _strength_rank("moderate")
            and _strength_rank(strategy_strength) <= _strength_rank("weak")
        ):
            return symbol_match

    scored_matches: list[tuple[float, int, str, str, dict[str, Any]]] = []
    for category, group, score_item in matches:
        score_value = _coerce_number(score_item.get("score"))
        category_index = SUPPORT_CATEGORY_ORDER.index(category)
        scored_matches.append(
            (
                float(score_value) if score_value is not None else float("-inf"),
                -category_index,
                category,
                group,
                score_item,
            )
        )

    scored_matches.sort(reverse=True)
    _, _, category, group, score_item = scored_matches[0]
    return category, group, score_item


def _resolve_score_item_selected_strength(score_item: dict[str, Any]) -> str:
    source_preference = _normalize_source_preference(score_item.get("source_preference"))
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

    value = _first_non_empty(
        score_item.get(SOURCE_FIELDS["latest"]["candidate_strength"]),
        score_item.get(SOURCE_FIELDS["cumulative"]["candidate_strength"]),
    )
    return value or "insufficient_data"


def _strength_rank(value: str | None) -> int:
    if value is None:
        return STRENGTH_PRIORITY["insufficient_data"]
    return STRENGTH_PRIORITY.get(value, STRENGTH_PRIORITY["insufficient_data"])


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


def _normalize_group_for_category(category: str, value: Any) -> str | None:
    normalized = _normalize_identifier(value)
    if normalized is None:
        return None
    if category == "symbol":
        return normalized.upper()
    return normalized


def _is_valid_candidate(candidate: dict[str, Any]) -> bool:
    symbol = _normalize_group_for_category("symbol", candidate.get("symbol"))
    strategy = _normalize_group_for_category("strategy", candidate.get("strategy"))
    horizon = _normalize_horizon(candidate.get("horizon"))
    return (
        symbol is not None
        and strategy is not None
        and horizon is not None
        and _is_strategy_horizon_compatible(strategy, horizon)
    )


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
