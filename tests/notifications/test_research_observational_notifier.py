from __future__ import annotations

from src.notifications.research_observational_notifier import (
    build_observational_message,
    build_shadow_selection_message,
)


def test_build_shadow_selection_message_surfaces_abstain_context() -> None:
    payload = {
        "generated_at": "2026-03-18T00:00:00+00:00",
        "selection_status": "abstain",
        "reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
        "selection_explanation": "Abstained because no candidate passed the conservative eligibility threshold.",
        "candidates_considered": 0,
        "abstain_diagnosis": {
            "category": "no_eligible_candidates",
            "summary": "Top candidate failed the eligibility gate after penalties.",
            "candidate_seed_diagnostics": {
                "horizon_diagnostics": [
                    {"horizon": "15m", "seed_generated": False},
                    {"horizon": "1h", "seed_generated": False},
                    {"horizon": "4h", "seed_generated": False},
                ]
            },
            "compared_candidate": {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "candidate_status": "penalized",
            },
        },
        "ranking": [
            {
                "symbol": "SOLUSDT",
                "strategy": "breakout",
                "horizon": "15m",
                "candidate_status": "penalized",
                "selection_score": 2.1,
                "selection_confidence": 0.37,
                "drift_direction": "decrease",
                "reason_codes": [
                    "CANDIDATE_STRENGTH_WEAK",
                    "CANDIDATE_EDGE_STABILITY_SCORE_LOW",
                ],
                "gate_diagnostics": {
                    "score_gate": {
                        "passed": False,
                        "reason_codes": ["CANDIDATE_EDGE_STABILITY_SCORE_LOW"],
                    },
                    "stability_gate": {
                        "passed": False,
                        "reason_codes": ["CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY"],
                    },
                    "drift_gate": {
                        "passed": False,
                        "reason_codes": ["CANDIDATE_DRIFT_DECREASING"],
                    },
                    "eligibility_gate": {
                        "passed": False,
                        "reason_codes": ["CANDIDATE_STRENGTH_WEAK"],
                    },
                },
            }
        ],
    }

    message = build_shadow_selection_message(payload)

    assert "Status: ABSTAIN" in message
    assert "Reason: NO_ELIGIBLE_CANDIDATES" in message
    assert "Candidates: 0 | Ranking: 1" in message
    assert "Blocked horizons: 15m, 1h, 4h" in message
    assert "Top: SOLUSDT / breakout / 15m" in message
    assert "Failed gates: score, stability, drift, eligibility" in message
    assert "Generated: 2026-03-18T00:00:00+00:00" in message


def test_build_observational_message_omits_empty_sections_and_formats_blocks() -> None:
    message = build_observational_message(
        score_drift_summary={
            "generated_at": "2026-03-18T00:00:00+00:00",
            "drift_summary": {
                "increase": 0,
                "decrease": 0,
                "flat": 3,
            },
            "score_drift": [],
        },
        edge_scores_summary=None,
        comparison_summary={
            "edge_candidates_preview": {
                "by_horizon": {
                    "15m": {
                        "top_strategy": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                        "top_symbol": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                        "top_alignment_state": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                    },
                    "1h": {
                        "top_strategy": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                        "top_symbol": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                        "top_alignment_state": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                    },
                    "4h": {
                        "top_strategy": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                        "top_symbol": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                        "top_alignment_state": {"candidate_strength": "insufficient_data", "visibility_reason": "failed_absolute_minimum_gate"},
                    },
                }
            },
            "edge_stability_preview": {
                "strategy": {"group": None, "visible_horizons": [], "stability_label": "insufficient_data"},
                "symbol": {"group": None, "visible_horizons": [], "stability_label": "insufficient_data"},
                "alignment_state": {"group": None, "visible_horizons": [], "stability_label": "insufficient_data"},
            },
        },
        shadow_selection={
            "generated_at": "2026-03-18T00:00:00+00:00",
            "selection_status": "abstain",
            "reason_codes": ["NO_CANDIDATES_AVAILABLE"],
            "candidates_considered": 0,
            "ranking": [],
        },
    )

    assert "Research Observation" in message
    assert "Status: ABSTAIN" in message
    assert "Reason: NO_CANDIDATES_AVAILABLE" in message
    assert "Why blocked" in message
    assert "15m: insufficient_data" in message
    assert "1h: insufficient_data" in message
    assert "4h: insufficient_data" in message
    assert "Drift" in message
    assert "increase 0 | decrease 0 | flat 3" in message
    assert "Current Snapshot" not in message
    assert "Changed Groups" not in message
