from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_direct_edge_selection_abstain_path_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_direct_edge_selection_abstain_path_diagnosis_report as report_module,
)


def _candidate(
    symbol: str,
    *,
    rank: int = 1,
    strategy: str = "intraday",
    horizon: str = "1h",
    status: str = "blocked",
    score: float | None = None,
    confidence: float | None = None,
    reason_codes: list[str] | None = None,
) -> dict:
    return {
        "rank": rank,
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "candidate_status": status,
        "selection_score": score,
        "selection_confidence": confidence,
        "reason_codes": reason_codes or [],
        "gate_diagnostics": {
            "eligibility_gate": {
                "passed": status == "eligible",
                "reason_codes": reason_codes or [],
            }
        },
    }


def _snapshot(
    *,
    name: str,
    status: str = "abstain",
    reason_codes: list[str] | None = None,
    abstain_category: str = "no_eligible_candidates",
    candidates: list[dict] | None = None,
    candidate_count: int | None = None,
) -> dict:
    ranking = list(candidates or [])
    return {
        "available": True,
        "snapshot_name": name,
        "snapshot_label": name,
        "selection_status": status,
        "reason": (reason_codes or ["unknown"])[0],
        "reason_codes": reason_codes or ["NO_ELIGIBLE_CANDIDATES"],
        "selection_explanation": "synthetic abstain",
        "candidate_count": len(ranking) if candidate_count is None else candidate_count,
        "ranking_count": len(ranking),
        "ranking": ranking,
        "abstain_diagnosis": {
            "category": abstain_category,
            "eligible_candidate_count": sum(
                1 for item in ranking if item.get("candidate_status") == "eligible"
            ),
            "penalized_candidate_count": sum(
                1 for item in ranking if item.get("candidate_status") == "penalized"
            ),
            "blocked_candidate_count": sum(
                1 for item in ranking if item.get("candidate_status") == "blocked"
            ),
            "top_candidate": ranking[0] if ranking else None,
        },
    }


def test_same_explicit_abstain_reason_is_reported_across_all_snapshots() -> None:
    top = _candidate(
        "BTCUSDT",
        score=None,
        confidence=None,
        reason_codes=["CANDIDATE_STABILITY_UNSTABLE"],
    )
    snapshots = {
        report_module._SNAPSHOT_BASELINE: _snapshot(
            name=report_module._SNAPSHOT_BASELINE,
            candidates=[top],
        ),
        report_module._SNAPSHOT_PATCH_CLASS_A: _snapshot(
            name=report_module._SNAPSHOT_PATCH_CLASS_A,
            candidates=[top],
        ),
        report_module._SNAPSHOT_PATCH_CLASS_B: _snapshot(
            name=report_module._SNAPSHOT_PATCH_CLASS_B,
            candidates=[top],
        ),
    }

    summary = report_module.build_abstain_path_summary(snapshots)

    assert summary["status"] == report_module._DIRECT_EDGE_SELECTION_AVAILABLE
    assert summary["same_top_candidate_across_snapshots"] is True
    assert summary["primary_abstain_path_classification"] == (
        "eligibility_or_acceptance_rejection"
    )
    assert summary["persistent_abstain_reason"]["persistent_reason_codes"] == [
        "NO_ELIGIBLE_CANDIDATES"
    ]
    assert summary["interpretation_status"] == (
        "direct_edge_selection_abstain_path_explained"
    )


def test_shadow_can_change_top_score_and_order_while_still_abstaining() -> None:
    baseline_top = _candidate(
        "BTCUSDT",
        rank=1,
        status="eligible",
        score=5.0,
        confidence=0.70,
        reason_codes=["ELIGIBLE_CONSERVATIVE_PASS"],
    )
    baseline_second = _candidate(
        "ETHUSDT",
        rank=2,
        status="eligible",
        score=4.8,
        confidence=0.68,
        reason_codes=["ELIGIBLE_CONSERVATIVE_PASS"],
    )
    shadow_top = {**baseline_second, "rank": 1, "selection_score": 5.2}
    shadow_second = {**baseline_top, "rank": 2, "selection_score": 5.0}
    snapshots = {
        report_module._SNAPSHOT_BASELINE: _snapshot(
            name=report_module._SNAPSHOT_BASELINE,
            reason_codes=["TOP_CANDIDATES_TIED"],
            abstain_category="tied_top_candidates",
            candidates=[baseline_top, baseline_second],
        ),
        report_module._SNAPSHOT_PATCH_CLASS_A: _snapshot(
            name=report_module._SNAPSHOT_PATCH_CLASS_A,
            reason_codes=["TOP_CANDIDATES_TIED"],
            abstain_category="tied_top_candidates",
            candidates=[shadow_top, shadow_second],
        ),
        report_module._SNAPSHOT_PATCH_CLASS_B: _snapshot(
            name=report_module._SNAPSHOT_PATCH_CLASS_B,
            reason_codes=["TOP_CANDIDATES_TIED"],
            abstain_category="tied_top_candidates",
            candidates=[shadow_top, shadow_second],
        ),
    }

    summary = report_module.build_abstain_path_summary(snapshots)
    delta = summary["comparisons"]["baseline_vs_patch_class_a"]

    assert summary["primary_abstain_path_classification"] == "selection_competition"
    assert delta["top_candidate_identity_changed"] is True
    assert delta["candidate_ordering_changed"] is True
    assert delta["top_candidate_score_delta"] == 0.2
    assert delta["selection_status_after"] == "abstain"


def test_candidate_count_can_change_while_selection_still_abstains() -> None:
    snapshots = {
        report_module._SNAPSHOT_BASELINE: _snapshot(
            name=report_module._SNAPSHOT_BASELINE,
            reason_codes=["NO_CANDIDATES_AVAILABLE"],
            abstain_category="no_candidates_available",
            candidates=[],
            candidate_count=0,
        ),
        report_module._SNAPSHOT_PATCH_CLASS_A: _snapshot(
            name=report_module._SNAPSHOT_PATCH_CLASS_A,
            reason_codes=["ALL_CANDIDATES_BLOCKED"],
            abstain_category="all_candidates_blocked",
            candidates=[
                _candidate(
                    "ETHUSDT",
                    reason_codes=["CANDIDATE_SYMBOL_SUPPORT_TOO_LOW"],
                )
            ],
        ),
        report_module._SNAPSHOT_PATCH_CLASS_B: _snapshot(
            name=report_module._SNAPSHOT_PATCH_CLASS_B,
            reason_codes=["ALL_CANDIDATES_BLOCKED"],
            abstain_category="all_candidates_blocked",
            candidates=[
                _candidate(
                    "ETHUSDT",
                    reason_codes=["CANDIDATE_SYMBOL_SUPPORT_TOO_LOW"],
                )
            ],
        ),
    }

    summary = report_module.build_abstain_path_summary(snapshots)
    delta = summary["comparisons"]["baseline_vs_patch_class_a"]

    assert delta["candidate_count_delta"] == 1
    assert delta["selection_status_after"] == "abstain"
    assert summary["primary_abstain_path_classification"] == "mixed_abstain_path"
    assert summary["minimum_question_answers"]["snapshot_answers"][
        report_module._SNAPSHOT_BASELINE
    ]["candidate_formation_scarcity"] is True


def test_wrapper_import_path_and_entrypoint_smoke(monkeypatch, capsys) -> None:
    def fake_runner(**_: object) -> dict:
        report = {
            "report_type": report_module.REPORT_TYPE,
            "input_path": "/tmp/input.jsonl",
            "output_dir": "/tmp/reports",
            "configurations_evaluated": [
                report_module.DiagnosisConfiguration(336, 10000).to_dict()
            ],
            "widest_configuration": (
                report_module.DiagnosisConfiguration(336, 10000).to_dict()
            ),
            "direct_edge_selection_abstain_path_summary": {
                "status": report_module._DIRECT_EDGE_SELECTION_AVAILABLE,
                "primary_abstain_path_classification": (
                    "eligibility_or_acceptance_rejection"
                ),
                "snapshots": {
                    report_module._SNAPSHOT_BASELINE: {
                        "selection_status": "abstain",
                        "candidate_count": 1,
                        "top_candidate_identity": "BTCUSDT / intraday / 1h",
                    },
                    report_module._SNAPSHOT_PATCH_CLASS_A: {
                        "selection_status": "abstain",
                        "candidate_count": 1,
                        "top_candidate_identity": "BTCUSDT / intraday / 1h",
                    },
                    report_module._SNAPSHOT_PATCH_CLASS_B: {
                        "selection_status": "abstain",
                        "candidate_count": 1,
                        "top_candidate_identity": "BTCUSDT / intraday / 1h",
                    },
                },
                "persistent_abstain_reason": {
                    "persistent_reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
                },
            },
            "interpretation_status": (
                "direct_edge_selection_abstain_path_explained"
            ),
        }
        return {"report": report, "written_paths": {}, "markdown": ""}

    monkeypatch.setattr(
        report_module,
        "run_selected_strategy_direct_edge_selection_abstain_path_diagnosis_report",
        fake_runner,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_direct_edge_selection_abstain_path_diagnosis_report",
            "--input",
            "/tmp/input.jsonl",
            "--output-dir",
            "/tmp/reports",
            "--config",
            "336/10000",
        ],
    )

    wrapper_path = (
        Path(__file__).parents[2]
        / "src"
        / "research"
        / "selected_strategy_direct_edge_selection_abstain_path_diagnosis_report.py"
    )
    runpy.run_path(str(wrapper_path), run_name="__main__")
    captured = json.loads(capsys.readouterr().out)

    assert captured["report_type"] == report_module.REPORT_TYPE
    assert captured["configuration_count"] == 1
    assert captured["primary_abstain_path_classification"] == (
        "eligibility_or_acceptance_rejection"
    )
    assert captured["baseline_selection_status"] == "abstain"
    assert wrapper_module.build_report is report_module.build_report
