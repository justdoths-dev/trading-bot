from __future__ import annotations

from src.research.edge_selection_diagnosis_report import (
    build_edge_selection_diagnosis_report,
)


def test_diagnosis_report_explains_filtered_selection_state() -> None:
    report = build_edge_selection_diagnosis_report(
        research_summary_data={
            "edge_candidates_preview": {
                "by_horizon": {
                    "15m": {
                        "top_strategy": {
                            "group": "swing",
                            "candidate_strength": "weak",
                            "quality_gate": "borderline",
                        },
                        "top_symbol": {
                            "group": "BTCUSDT",
                            "candidate_strength": "insufficient_data",
                            "quality_gate": "failed",
                        },
                        "top_alignment_state": {
                            "group": "aligned",
                            "candidate_strength": "insufficient_data",
                            "quality_gate": "failed",
                        },
                    },
                    "1h": {
                        "top_strategy": {
                            "group": "swing",
                            "candidate_strength": "weak",
                            "quality_gate": "borderline",
                        },
                        "top_symbol": {
                            "group": "BTCUSDT",
                            "candidate_strength": "insufficient_data",
                            "quality_gate": "failed",
                        },
                        "top_alignment_state": {
                            "group": "aligned",
                            "candidate_strength": "insufficient_data",
                            "quality_gate": "failed",
                        },
                    },
                }
            }
        },
        edge_candidates_preview={
            "by_horizon": {
                "15m": {
                    "top_strategy": {
                        "group": "swing",
                        "candidate_strength": "weak",
                        "quality_gate": "borderline",
                    },
                    "top_symbol": {
                        "group": "insufficient_data",
                        "candidate_strength": "insufficient_data",
                        "quality_gate": "failed",
                    },
                    "top_alignment_state": {
                        "group": "insufficient_data",
                        "candidate_strength": "insufficient_data",
                        "quality_gate": "failed",
                    },
                },
                "1h": {
                    "top_strategy": {
                        "group": "swing",
                        "candidate_strength": "weak",
                        "quality_gate": "borderline",
                    },
                    "top_symbol": {
                        "group": "insufficient_data",
                        "candidate_strength": "insufficient_data",
                        "quality_gate": "failed",
                    },
                    "top_alignment_state": {
                        "group": "insufficient_data",
                        "candidate_strength": "insufficient_data",
                        "quality_gate": "failed",
                    },
                },
            }
        },
        edge_stability_preview={
            "strategy": {
                "group": "swing",
                "visible_horizons": ["15m", "1h"],
                "stability_label": "multi_horizon_confirmed",
                "visibility_reason": "repeated_visible_candidate_across_horizons",
            },
            "symbol": {
                "group": None,
                "visible_horizons": [],
                "stability_label": "insufficient_data",
                "visibility_reason": "no_visible_candidates",
            },
            "alignment_state": {
                "group": None,
                "visible_horizons": [],
                "stability_label": "insufficient_data",
                "visibility_reason": "no_visible_candidates",
            },
        },
        shadow_selection={
            "selection_status": "abstain",
            "candidates_considered": 1,
            "ranking": [
                {
                    "symbol": "BTCUSDT",
                    "strategy": "swing",
                    "horizon": "15m",
                    "drift_direction": "decrease",
                    "gate_diagnostics": {
                        "drift_gate": {
                            "passed": False,
                            "reason_codes": ["CANDIDATE_DRIFT_DECREASING"],
                        },
                        "stability_gate": {
                            "passed": False,
                            "reason_codes": ["CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY"],
                        },
                    },
                }
            ],
            "abstain_diagnosis": {
                "top_candidate": {
                    "symbol": "BTCUSDT",
                    "strategy": "swing",
                    "horizon": "15m",
                    "drift_direction": "decrease",
                    "gate_diagnostics": {
                        "drift_gate": {
                            "passed": False,
                            "reason_codes": ["CANDIDATE_DRIFT_DECREASING"],
                        },
                        "stability_gate": {
                            "passed": False,
                            "reason_codes": ["CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY"],
                        },
                    },
                }
            },
        },
    )

    assert (
        report["candidate_generation_state"]
        == "candidates_generated_but_filtered_before_final_selection"
    )
    assert report["visible_groups"]["strategy"]["group"] == "swing"
    assert "symbol:no_visible_groups" in report["failed_layers"]
    assert "alignment:no_visible_groups" in report["failed_layers"]
    assert "candidate:CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY" in report["stability_issues"]
    assert "CANDIDATE_DRIFT_DECREASING" in report["drift_issues"]
    assert "status=abstain" in report["diagnosis_summary"]


def test_diagnosis_report_uses_conservative_preview_unavailable_label() -> None:
    report = build_edge_selection_diagnosis_report(
        research_summary_data={},
        edge_candidates_preview=None,
        edge_stability_preview=None,
        shadow_selection={
            "selection_status": "abstain",
            "candidates_considered": 0,
            "ranking": [],
        },
    )

    assert report["failed_layers"] == ["candidate_preview_unavailable"]
    assert report["candidate_generation_state"] == "no_ranked_candidates_visible"
