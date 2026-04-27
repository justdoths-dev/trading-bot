from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report as report_module,
)


def _watchlist_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "paper_replay_watchlist_id": (
            "paper_replay_watchlist_v1:"
            "aggregate_metric_paper_replay_evaluation_v1:"
            "paper_replay_outcome_attachment_source_v1:"
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "aggregate_metric_evaluation_id": (
            "aggregate_metric_paper_replay_evaluation_v1:"
            "paper_replay_outcome_attachment_source_v1:"
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "source_outcome_attachment_source_id": (
            "paper_replay_outcome_attachment_source_v1:"
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "source_outcome_tracking_id": (
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "source_journal_entry_id": (
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "source_paper_replay_candidate_id": (
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "symbol": "BTCUSDT",
        "strategy": "intraday",
        "horizon": "1h",
        "aggregate_sample_count": 80,
        "aggregate_labeled_count": 80,
        "aggregate_score": 62.75,
        "evaluation_bucket": "aggregate_metric_promising_observation",
        "watchlist_tier": "paper_replay_promising_watchlist",
        "watchlist_priority": "high",
        "production_live_selection_allowed": False,
        "mapper_live_path_allowed": False,
        "engine_live_path_allowed": False,
        "no_order_execution": True,
        "no_synthetic_fill": True,
        "no_pnl_claim": True,
        "order_id": None,
        "fill_id": None,
        "entry_price": None,
        "exit_price": None,
        "realized_pnl": None,
        "unrealized_pnl": None,
        "exact_outcome_used": False,
        "aggregate_metric_only": True,
    }
    row.update(overrides)
    return row


def _summary(
    *,
    display_name: str,
    window_hours: int,
    max_rows: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    promising_count = sum(
        row.get("watchlist_tier") == "paper_replay_promising_watchlist"
        for row in rows
    )
    standard_count = sum(
        row.get("watchlist_tier") == "paper_replay_standard_watchlist"
        for row in rows
    )
    return {
        "configuration": {
            "display_name": display_name,
            "window_hours": window_hours,
            "max_rows": max_rows,
        },
        "watchlist_row_count": len(rows),
        "promising_watchlist_count": promising_count,
        "standard_watchlist_count": standard_count,
        "paper_replay_candidate_count": len(rows),
        "production_candidate_count": 0,
        "watchlist_rows": rows,
    }


def _standard_row(**overrides: Any) -> dict[str, Any]:
    return _watchlist_row(
        aggregate_score=54.5,
        evaluation_bucket="aggregate_metric_watchlist_observation",
        watchlist_tier="paper_replay_standard_watchlist",
        watchlist_priority="medium",
        **overrides,
    )


def test_two_rows_same_observation_across_configurations_become_one_retained_row() -> None:
    summaries = [
        _summary(
            display_name="144h_5000",
            window_hours=144,
            max_rows=5000,
            rows=[
                _watchlist_row(
                    paper_replay_watchlist_id="watchlist:144",
                    aggregate_metric_evaluation_id="evaluation:144",
                )
            ],
        ),
        _summary(
            display_name="336h_10000",
            window_hours=336,
            max_rows=10000,
            rows=[
                _watchlist_row(
                    paper_replay_watchlist_id="watchlist:336",
                    aggregate_metric_evaluation_id="evaluation:336",
                    aggregate_sample_count=120,
                    aggregate_labeled_count=100,
                    aggregate_score=70.0,
                )
            ],
        ),
    ]

    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=summaries
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["retention_observation_key"] == "BTCUSDT:intraday:1h"
    assert row["observed_configuration_count"] == 2
    assert row["source_watchlist_row_count"] == 2
    assert row["source_paper_replay_watchlist_ids"] == [
        "watchlist:144",
        "watchlist:336",
    ]
    assert row["max_aggregate_score"] == 70.0
    assert row["max_aggregate_sample_count"] == 120
    assert row["aggregate_score_values"] == [62.75, 70.0]


def test_retained_group_with_promising_source_is_high_priority() -> None:
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=[
            _summary(
                display_name="144h_5000",
                window_hours=144,
                max_rows=5000,
                rows=[_standard_row(paper_replay_watchlist_id="standard")],
            ),
            _summary(
                display_name="336h_10000",
                window_hours=336,
                max_rows=10000,
                rows=[_watchlist_row(paper_replay_watchlist_id="promising")],
            ),
        ]
    )

    row = rows[0]
    assert row["retention_tier"] == report_module.RETAINED_PROMISING_TIER
    assert row["retention_priority"] == "high"
    assert row["best_watchlist_tier"] == "paper_replay_promising_watchlist"
    assert row["best_watchlist_priority"] == "high"


def test_retained_group_with_only_standard_sources_is_medium_priority() -> None:
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=[
            _summary(
                display_name="144h_5000",
                window_hours=144,
                max_rows=5000,
                rows=[_standard_row(paper_replay_watchlist_id="standard:144")],
            ),
            _summary(
                display_name="336h_10000",
                window_hours=336,
                max_rows=10000,
                rows=[_standard_row(paper_replay_watchlist_id="standard:336")],
            ),
        ]
    )

    row = rows[0]
    assert row["retention_tier"] == report_module.RETAINED_STANDARD_TIER
    assert row["retention_priority"] == "medium"


def test_single_window_promising_row_is_medium_priority() -> None:
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=[
            _summary(
                display_name="336h_10000",
                window_hours=336,
                max_rows=10000,
                rows=[_watchlist_row()],
            )
        ]
    )

    row = rows[0]
    assert row["retention_tier"] == report_module.SINGLE_WINDOW_PROMISING_TIER
    assert row["retention_priority"] == "medium"


def test_single_window_standard_row_is_low_priority() -> None:
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=[
            _summary(
                display_name="336h_10000",
                window_hours=336,
                max_rows=10000,
                rows=[_standard_row()],
            )
        ]
    )

    row = rows[0]
    assert row["retention_tier"] == report_module.SINGLE_WINDOW_STANDARD_TIER
    assert row["retention_priority"] == "low"


def test_source_watchlist_ids_and_lineage_ids_are_preserved_in_lists() -> None:
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=[
            _summary(
                display_name="144h_5000",
                window_hours=144,
                max_rows=5000,
                rows=[
                    _watchlist_row(
                        paper_replay_watchlist_id="watchlist:2",
                        aggregate_metric_evaluation_id="evaluation:2",
                        source_outcome_attachment_source_id="attachment:2",
                        source_outcome_tracking_id="tracking:2",
                        source_journal_entry_id="journal:2",
                        source_paper_replay_candidate_id="candidate:2",
                    ),
                    _watchlist_row(
                        paper_replay_watchlist_id="watchlist:1",
                        aggregate_metric_evaluation_id="evaluation:1",
                        source_outcome_attachment_source_id="attachment:1",
                        source_outcome_tracking_id="tracking:1",
                        source_journal_entry_id="journal:1",
                        source_paper_replay_candidate_id="candidate:1",
                    ),
                ],
            )
        ]
    )

    row = rows[0]
    assert row["source_paper_replay_watchlist_ids"] == [
        "watchlist:1",
        "watchlist:2",
    ]
    assert row["source_aggregate_metric_evaluation_ids"] == [
        "evaluation:1",
        "evaluation:2",
    ]
    assert row["source_outcome_attachment_source_ids"] == [
        "attachment:1",
        "attachment:2",
    ]
    assert row["source_outcome_tracking_ids"] == ["tracking:1", "tracking:2"]
    assert row["source_journal_entry_ids"] == ["journal:1", "journal:2"]
    assert row["source_paper_replay_candidate_ids"] == [
        "candidate:1",
        "candidate:2",
    ]


def test_retention_rows_are_non_live_and_cannot_enter_mapper_engine_or_execution_path() -> None:
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=[
            _summary(
                display_name="336h_10000",
                window_hours=336,
                max_rows=10000,
                rows=[_watchlist_row()],
            )
        ]
    )
    row = rows[0]
    invariants = report_module.build_retention_safety_invariant_summary(rows)

    assert row["production_live_selection_allowed"] is False
    assert row["mapper_live_path_allowed"] is False
    assert row["engine_live_path_allowed"] is False
    assert row["retention_is_live_edge_selection"] is False
    assert invariants["all_rows_production_live_selection_allowed_is_false"] is True
    assert invariants["all_rows_mapper_live_path_allowed_is_false"] is True
    assert invariants["all_rows_engine_live_path_allowed_is_false"] is True
    assert invariants["retention_rows_are_not_live_edge_selections"] is True


def test_no_order_fill_price_or_pnl_fields_are_populated() -> None:
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=[
            _summary(
                display_name="336h_10000",
                window_hours=336,
                max_rows=10000,
                rows=[_watchlist_row()],
            )
        ]
    )
    row = rows[0]
    invariants = report_module.build_retention_safety_invariant_summary(rows)

    assert row["no_order_execution"] is True
    assert row["no_synthetic_fill"] is True
    assert row["no_pnl_claim"] is True
    assert row["order_id"] is None
    assert row["fill_id"] is None
    assert row["entry_price"] is None
    assert row["exit_price"] is None
    assert row["realized_pnl"] is None
    assert row["unrealized_pnl"] is None
    assert row["exact_outcome_used"] is False
    assert row["aggregate_metric_only"] is True
    assert invariants["all_rows_no_order_execution_is_true"] is True
    assert invariants["all_rows_no_synthetic_fill_is_true"] is True
    assert invariants["all_rows_no_pnl_claim_is_true"] is True
    assert invariants["no_order_or_fill_ids_are_present"] is True
    assert invariants["no_price_or_pnl_fields_are_present"] is True
    assert invariants["exact_outcomes_are_not_used"] is True
    assert invariants["aggregate_metrics_remain_aggregate_only"] is True


def test_final_assessment_recommends_review_packet_when_retained_observations_exist() -> None:
    summaries = [
        _summary(
            display_name="144h_5000",
            window_hours=144,
            max_rows=5000,
            rows=[_standard_row(paper_replay_watchlist_id="standard:144")],
        ),
        _summary(
            display_name="336h_10000",
            window_hours=336,
            max_rows=10000,
            rows=[_standard_row(paper_replay_watchlist_id="standard:336")],
        ),
    ]
    rows = report_module.build_retention_rows(
        source_watchlist_configuration_summaries=summaries
    )
    final = report_module.build_final_assessment(
        retention_rows=rows,
        source_watchlist_configuration_summaries=summaries,
    )

    assert final["retention_rows_present"] is True
    assert final["retention_row_count"] == 1
    assert final["retained_observation_count"] == 1
    assert final["single_window_observation_count"] == 0
    assert final["retained_standard_count"] == 1
    assert final["production_candidate_count"] == 0
    assert (
        final["recommended_next_stage"]
        == "design_retained_watchlist_review_packet_report"
    )

    invariant_summary = final["retention_safety_invariant_summary"]
    assert (
        invariant_summary[
            "all_retention_rows_preserve_source_paper_replay_watchlist_ids"
        ]
        is True
    )
    assert (
        invariant_summary[
            "all_retention_rows_preserve_source_aggregate_metric_evaluation_ids"
        ]
        is True
    )
    assert (
        invariant_summary[
            "all_retention_rows_preserve_source_outcome_attachment_source_ids"
        ]
        is True
    )
    assert (
        invariant_summary[
            "all_retention_rows_preserve_source_outcome_tracking_ids"
        ]
        is True
    )
    assert (
        invariant_summary["all_retention_rows_preserve_source_journal_entry_ids"]
        is True
    )
    assert (
        invariant_summary[
            "all_retention_rows_preserve_source_paper_replay_candidate_ids"
        ]
        is True
    )


def test_build_report_uses_source_watchlist_builder(monkeypatch: Any) -> None:
    source_report = {
        "inputs": {"input_path": "input.jsonl", "output_dir": "out"},
        "final_assessment": {"watchlist_row_count": 1},
        "configurations_evaluated": [],
        "configuration_summaries": [
            _summary(
                display_name="336h_10000",
                window_hours=336,
                max_rows=10000,
                rows=[_watchlist_row()],
            )
        ],
    }
    calls: list[dict[str, Any]] = []

    def fake_build_report(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return source_report

    monkeypatch.setattr(report_module.watchlist_report, "build_report", fake_build_report)

    report = report_module.build_report(
        input_path=Path("source.jsonl"),
        output_dir=Path("reports"),
        configurations=None,
    )

    assert calls == [
        {
            "input_path": Path("source.jsonl"),
            "output_dir": Path("reports"),
            "configurations": None,
        }
    ]
    assert report["report_type"] == report_module.REPORT_TYPE
    assert report["retention_rows"][0]["source_report_type"] == (
        "selected_strategy_edge_candidate_paper_replay_watchlist_report"
    )


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report
        is report_module.run_selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report
    )
