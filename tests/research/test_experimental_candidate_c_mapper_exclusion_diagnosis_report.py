from __future__ import annotations

from pathlib import Path

from src.research.experimental_candidate_c_mapper_exclusion_diagnosis_report import (
    build_experimental_candidate_c_mapper_exclusion_diagnosis_summary,
    render_experimental_candidate_c_mapper_exclusion_diagnosis_markdown,
)


def _row(
    *,
    logged_at: str,
    symbol: str,
    strategy: str,
    return_15m: float,
    label_15m: str,
) -> dict:
    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "future_return_15m": return_15m,
        "future_return_1h": 0.0,
        "future_return_4h": 0.0,
        "future_label_15m": label_15m,
        "future_label_1h": "flat",
        "future_label_4h": "flat",
    }


def _base_inputs() -> tuple[list[dict], list[dict], Path]:
    baseline_rows = [
        _row(
            logged_at="2026-03-20T00:00:00+00:00",
            symbol="BTCUSDT",
            strategy="swing",
            return_15m=0.1,
            label_15m="up",
        )
    ]
    experiment_rows = baseline_rows + [
        _row(
            logged_at="2026-03-20T01:00:00+00:00",
            symbol="SOLUSDT",
            strategy="breakout",
            return_15m=0.4,
            label_15m="up",
        ),
        _row(
            logged_at="2026-03-20T02:00:00+00:00",
            symbol="SOLUSDT",
            strategy="breakout",
            return_15m=-0.2,
            label_15m="down",
        ),
    ]
    trace_dir = Path(
        "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate/edge_selection_trace"
    )
    return baseline_rows, experiment_rows, trace_dir


def test_identity_dropped_by_comparison_collapse() -> None:
    baseline_rows, experiment_rows, trace_dir = _base_inputs()
    stage_traces = {
        "15m": {
            "all_ranked_symbol_groups": ["SOLUSDT"],
            "all_ranked_strategy_groups": ["breakout"],
            "ranked_symbol_groups": [{"group": "SOLUSDT", "rank": 1}],
            "ranked_strategy_groups": [{"group": "breakout", "rank": 1}],
            "preview_selected_symbol_group": {"group": "SOLUSDT"},
            "preview_selected_strategy_group": {"group": "breakout"},
            "comparison_preserved_symbol_group": "BTCUSDT",
            "comparison_preserved_strategy_group": "swing",
            "mapper_seed_inputs_used": {"seed_generated": True},
            "mapper_seed_candidate_emitted": {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "15m",
            },
        },
        "1h": {},
        "4h": {},
    }

    summary = build_experimental_candidate_c_mapper_exclusion_diagnosis_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary={},
        comparison_summary={},
        mapped_payload={},
        shadow_output={"ranking": []},
        trace_dir=trace_dir,
        stage_traces=stage_traces,
    )

    assert summary["earliest_drop_stage_counts"][0]["value"] == "removed_by_comparison_collapse"
    assert (
        summary["root_bottleneck_assessment"]["dominant_bottleneck"]
        == "comparison_collapse_bottleneck"
    )
    assert summary["preview_constructed_count"] == 1
    assert summary["mapper_emitted_count"] == 0


def test_identity_dropped_by_mapper_seed_contract() -> None:
    baseline_rows, experiment_rows, trace_dir = _base_inputs()
    stage_traces = {
        "15m": {
            "all_ranked_symbol_groups": ["SOLUSDT"],
            "all_ranked_strategy_groups": ["breakout"],
            "ranked_symbol_groups": [{"group": "SOLUSDT", "rank": 1}],
            "ranked_strategy_groups": [{"group": "breakout", "rank": 1}],
            "preview_selected_symbol_group": {"group": "SOLUSDT"},
            "preview_selected_strategy_group": {"group": "breakout"},
            "comparison_preserved_symbol_group": "SOLUSDT",
            "comparison_preserved_strategy_group": "breakout",
            "mapper_seed_inputs_used": {"seed_generated": False},
            "mapper_seed_candidate_emitted": {},
        },
        "1h": {},
        "4h": {},
    }

    summary = build_experimental_candidate_c_mapper_exclusion_diagnosis_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary={},
        comparison_summary={},
        mapped_payload={},
        shadow_output={"ranking": []},
        trace_dir=trace_dir,
        stage_traces=stage_traces,
    )

    assert summary["earliest_drop_stage_counts"][0]["value"] == "removed_by_mapper_seed_contract"
    assert (
        summary["root_bottleneck_assessment"]["dominant_bottleneck"]
        == "mapper_seed_contract_bottleneck"
    )
    assert summary["preview_constructed_count"] == 1
    assert summary["mapper_emitted_count"] == 0


def test_identity_emitted_but_failing_eligibility() -> None:
    baseline_rows, experiment_rows, trace_dir = _base_inputs()
    stage_traces = {
        "15m": {
            "all_ranked_symbol_groups": ["SOLUSDT"],
            "all_ranked_strategy_groups": ["breakout"],
            "ranked_symbol_groups": [{"group": "SOLUSDT", "rank": 1}],
            "ranked_strategy_groups": [{"group": "breakout", "rank": 1}],
            "preview_selected_symbol_group": {"group": "SOLUSDT"},
            "preview_selected_strategy_group": {"group": "breakout"},
            "comparison_preserved_symbol_group": "SOLUSDT",
            "comparison_preserved_strategy_group": "breakout",
            "mapper_seed_inputs_used": {"seed_generated": True},
            "mapper_seed_candidate_emitted": {
                "symbol": "SOLUSDT",
                "strategy": "breakout",
                "horizon": "15m",
            },
        },
        "1h": {},
        "4h": {},
    }
    shadow_output = {
        "ranking": [
            {
                "symbol": "SOLUSDT",
                "strategy": "breakout",
                "horizon": "15m",
                "candidate_status": "blocked",
            }
        ]
    }

    summary = build_experimental_candidate_c_mapper_exclusion_diagnosis_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary={},
        comparison_summary={},
        mapped_payload={},
        shadow_output=shadow_output,
        trace_dir=trace_dir,
        stage_traces=stage_traces,
    )

    assert summary["earliest_drop_stage_counts"][0]["value"] == "emitted_but_failed_eligibility"
    assert summary["root_bottleneck_assessment"]["dominant_bottleneck"] == "eligibility_bottleneck"
    assert summary["preview_constructed_count"] == 1
    assert summary["mapper_emitted_count"] == 1
    assert summary["eligibility_passed_count"] == 0


def test_markdown_includes_new_summary_sections() -> None:
    summary = {
        "total_exclusive_rows": 4,
        "total_exclusive_identities": 2,
        "preview_constructed_count": 0,
        "mapper_emitted_count": 0,
        "eligibility_passed_count": 0,
        "final_selection_count": 0,
        "stage_trace_dir": (
            "logs/research_reports/experiments/candidate_c_asymmetric/"
            "c2_moderate/edge_selection_trace/stage_traces"
        ),
        "earliest_drop_stage_counts": [
            {"value": "removed_by_comparison_collapse", "count": 2}
        ],
        "per_horizon_stage_trace_summary": {
            "15m": {
                "ranked_symbol_group_count": 3,
                "ranked_strategy_group_count": 2,
                "preview_selected_symbol_group": "BTCUSDT",
                "preview_selected_strategy_group": "swing",
                "comparison_preserved_symbol_group": "BTCUSDT",
                "comparison_preserved_strategy_group": "swing",
                "mapper_seed_generated": True,
                "mapper_seed_candidate_emitted": {
                    "symbol": "BTCUSDT",
                    "strategy": "swing",
                },
            },
            "1h": {},
            "4h": {},
        },
        "root_bottleneck_assessment": {
            "comparison_collapse_bottleneck_count": 2,
            "mapper_seed_contract_bottleneck_count": 0,
            "eligibility_bottleneck_count": 0,
            "dominant_bottleneck": "comparison_collapse_bottleneck",
            "summary": "The dominant bottleneck is comparison collapse.",
        },
    }

    markdown = render_experimental_candidate_c_mapper_exclusion_diagnosis_markdown(summary)

    assert "Executive Summary" in markdown
    assert "Preview constructed count" in markdown
    assert "Earliest Drop Stage Counts" in markdown
    assert "Per-Horizon Stage Trace Summary" in markdown
    assert "Root Bottleneck Assessment" in markdown
    assert "comparison_collapse_bottleneck" in markdown
