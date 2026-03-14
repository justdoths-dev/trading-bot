"""
edge_detector

Detects statistical edges between groups based on ranking metrics.

This module compares groups pairwise and evaluates whether a
meaningful performance edge exists between them.

Standard library only.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any


EDGE_STRONG_THRESHOLD = 0.15
EDGE_MEDIUM_THRESHOLD = 0.07
EDGE_WEAK_THRESHOLD = 0.03


def _to_float(value: Any) -> float | None:
    """Safely convert a value to float, preserving missing values as None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    """
    Return the first available numeric metric from the provided keys.

    Preserves missing/invalid values as None instead of coercing them to zero.
    """
    for key in keys:
        value = _to_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _metric_gap(winner_metrics: dict[str, Any], loser_metrics: dict[str, Any], *keys: str) -> float:
    """
    Compute a safe metric gap.

    Missing values are treated conservatively as 0.0 only for gap reporting,
    not for winner selection.
    """
    winner_value = _metric_value(winner_metrics, *keys)
    loser_value = _metric_value(loser_metrics, *keys)
    return (winner_value or 0.0) - (loser_value or 0.0)


def _edge_strength(gap: float) -> str:
    """Classify edge strength based on metric gap."""
    if gap >= EDGE_STRONG_THRESHOLD:
        return "strong_edge"
    if gap >= EDGE_MEDIUM_THRESHOLD:
        return "medium_edge"
    if gap >= EDGE_WEAK_THRESHOLD:
        return "weak_edge"
    return "no_edge"


def _compare_groups(group_a: dict[str, Any], group_b: dict[str, Any]) -> dict[str, Any] | None:
    """
    Compare two groups and determine which one has the edge.

    Comparison priority:
    1. avg_future_return_pct
    2. If both missing or equal, no edge
    """

    metrics_a = group_a.get("metrics", {}) or {}
    metrics_b = group_b.get("metrics", {}) or {}

    return_a = _metric_value(metrics_a, "avg_future_return_pct")
    return_b = _metric_value(metrics_b, "avg_future_return_pct")

    # If neither group has a usable return metric, do not fabricate an edge.
    if return_a is None and return_b is None:
        return None

    # If only one side has a usable return metric, prefer the side with actual data.
    if return_a is None and return_b is not None:
        winner = group_b
        loser = group_a
    elif return_b is None and return_a is not None:
        winner = group_a
        loser = group_b
    else:
        # Both are numeric here.
        if return_a == return_b:
            return None

        if return_a > return_b:
            winner = group_a
            loser = group_b
        else:
            winner = group_b
            loser = group_a

    winner_metrics = winner.get("metrics", {}) or {}
    loser_metrics = loser.get("metrics", {}) or {}

    return_gap = _metric_gap(
        winner_metrics,
        loser_metrics,
        "avg_future_return_pct",
    )
    signal_gap = _metric_gap(
        winner_metrics,
        loser_metrics,
        "signal_match_rate",
        "signal_match_rate_pct",
    )
    bias_gap = _metric_gap(
        winner_metrics,
        loser_metrics,
        "bias_match_rate",
        "bias_match_rate_pct",
    )

    strength = _edge_strength(return_gap)

    if strength == "no_edge":
        return None

    reasons: list[str] = []

    if return_gap > 0:
        reasons.append("higher avg_future_return_pct")

    if signal_gap > 0:
        reasons.append("higher signal_match_rate")

    if bias_gap > 0:
        reasons.append("higher bias_match_rate")

    return {
        "winner": winner.get("group"),
        "loser": loser.get("group"),
        "edge_strength": strength,
        "return_gap": round(return_gap, 4),
        "signal_gap": round(signal_gap, 4),
        "bias_gap": round(bias_gap, 4),
        "reasons": reasons,
    }


def _detect_edges(groups: list[dict[str, Any]], edge_type: str, horizon: str = "15m") -> dict[str, Any]:
    """
    Generic edge detection across group pairs.
    """

    findings = []

    pairs = list(combinations(groups, 2))

    for group_a, group_b in pairs:
        result = _compare_groups(group_a, group_b)

        if result:
            findings.append(result)

    return {
        "edge_type": edge_type,
        "horizon": horizon,
        "evaluated_pairs": len(pairs),
        "edge_findings": findings,
    }


def detect_symbol_edges(symbol_rankings: list[dict[str, Any]], horizon: str = "15m") -> dict[str, Any]:
    """
    Detect edges between symbols (BTC vs ETH etc).
    """
    return _detect_edges(symbol_rankings, "symbol", horizon)


def detect_strategy_edges(strategy_rankings: list[dict[str, Any]], horizon: str = "15m") -> dict[str, Any]:
    """
    Detect edges between strategies (scalping / intraday / swing).
    """
    return _detect_edges(strategy_rankings, "strategy", horizon)


def detect_alignment_state_edges(alignment_rankings: list[dict[str, Any]], horizon: str = "15m") -> dict[str, Any]:
    """
    Detect edges between alignment states.
    """
    return _detect_edges(alignment_rankings, "alignment_state", horizon)


def detect_ai_execution_state_edges(
    ai_execution_rankings: list[dict[str, Any]],
    horizon: str = "15m",
) -> dict[str, Any]:
    """
    Detect edges between AI execution states.
    """
    return _detect_edges(ai_execution_rankings, "ai_execution_state", horizon)