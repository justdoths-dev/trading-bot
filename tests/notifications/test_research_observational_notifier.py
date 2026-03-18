from __future__ import annotations

from src.notifications.research_observational_notifier import build_shadow_selection_message


def test_build_shadow_selection_message_surfaces_abstain_context() -> None:
    payload = {
        "generated_at": "2026-03-18T00:00:00+00:00",
        "selection_status": "abstain",
        "reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
        "selection_explanation": "Abstained because no candidate passed the conservative eligibility threshold.",
        "abstain_diagnosis": {
            "summary": "Top candidate failed the eligibility gate after penalties.",
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

    assert "Decision: abstain (NO_ELIGIBLE_CANDIDATES)" in message
    assert "Diagnosis: Top candidate failed the eligibility gate after penalties." in message
    assert "Top Candidate" in message
    assert "SOLUSDT / breakout / 15m" in message
    assert "main reasons: CANDIDATE_STRENGTH_WEAK, CANDIDATE_EDGE_STABILITY_SCORE_LOW" in message
    assert "score=fail" in message
    assert "eligibility=fail" in message
    assert "Tie Peer: ETHUSDT / trend / 1h [penalized]" in message
