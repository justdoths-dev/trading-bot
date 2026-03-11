"""
edge_detector

Detects statistical edges between groups based on ranking metrics.

This module compares groups pairwise and evaluates whether a
meaningful performance edge exists between them.

Standard library only.
"""

from itertools import combinations


EDGE_STRONG_THRESHOLD = 0.15
EDGE_MEDIUM_THRESHOLD = 0.07
EDGE_WEAK_THRESHOLD = 0.03


def _edge_strength(gap):
    """Classify edge strength based on metric gap."""
    if gap >= EDGE_STRONG_THRESHOLD:
        return "strong_edge"
    if gap >= EDGE_MEDIUM_THRESHOLD:
        return "medium_edge"
    if gap >= EDGE_WEAK_THRESHOLD:
        return "weak_edge"
    return "no_edge"


def _compare_groups(group_a, group_b):
    """
    Compare two groups and determine which one has the edge.
    """

    metrics_a = group_a.get("metrics", {})
    metrics_b = group_b.get("metrics", {})

    return_a = metrics_a.get("avg_future_return_pct", 0.0)
    return_b = metrics_b.get("avg_future_return_pct", 0.0)

    if return_a == return_b:
        return None

    if return_a > return_b:
        winner = group_a
        loser = group_b
    else:
        winner = group_b
        loser = group_a

    winner_metrics = winner.get("metrics", {})
    loser_metrics = loser.get("metrics", {})

    return_gap = (
        winner_metrics.get("avg_future_return_pct", 0.0)
        - loser_metrics.get("avg_future_return_pct", 0.0)
    )

    signal_gap = (
        winner_metrics.get("signal_match_rate", 0.0)
        - loser_metrics.get("signal_match_rate", 0.0)
    )

    bias_gap = (
        winner_metrics.get("bias_match_rate", 0.0)
        - loser_metrics.get("bias_match_rate", 0.0)
    )

    strength = _edge_strength(return_gap)

    if strength == "no_edge":
        return None

    reasons = []

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


def _detect_edges(groups, edge_type, horizon="15m"):
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


def detect_symbol_edges(symbol_rankings, horizon="15m"):
    """
    Detect edges between symbols (BTC vs ETH etc).
    """
    return _detect_edges(symbol_rankings, "symbol", horizon)


def detect_strategy_edges(strategy_rankings, horizon="15m"):
    """
    Detect edges between strategies (scalping / intraday / swing).
    """
    return _detect_edges(strategy_rankings, "strategy", horizon)


def detect_alignment_state_edges(alignment_rankings, horizon="15m"):
    """
    Detect edges between alignment states.
    """
    return _detect_edges(alignment_rankings, "alignment_state", horizon)


def detect_ai_execution_state_edges(ai_execution_rankings, horizon="15m"):
    """
    Detect edges between AI execution states.
    """
    return _detect_edges(ai_execution_rankings, "ai_execution_state", horizon)