from __future__ import annotations

from src.research.edge_selection_engine import run_edge_selection_engine
from src.research.edge_selection_schema_validator import validate_shadow_output


def test_no_candidates_abstains() -> None:
    payload = _base_mapper_payload(candidates=[])

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "abstain"
    assert result["reason_codes"] == ["NO_CANDIDATES_AVAILABLE"]
    assert result["ranking"] == []


def test_insufficient_data_candidate_is_blocked() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "selected_candidate_strength": "insufficient_data",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "flat",
            }
        ]
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "abstain"
    assert result["ranking"][0]["candidate_status"] == "blocked"
    assert "CANDIDATE_STRENGTH_INSUFFICIENT_DATA" in result["ranking"][0]["reason_codes"]


def test_unstable_candidate_is_blocked() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "unstable",
                "drift_direction": "increase",
            }
        ]
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "abstain"
    assert result["ranking"][0]["candidate_status"] == "blocked"
    assert "CANDIDATE_STABILITY_UNSTABLE" in result["ranking"][0]["reason_codes"]


def test_incomplete_candidate_is_blocked() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "flat",
            }
        ]
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "abstain"
    assert result["ranking"][0]["candidate_status"] == "blocked"
    assert "CANDIDATE_IDENTITY_INCOMPLETE" in result["ranking"][0]["reason_codes"]


def test_low_sample_candidate_is_blocked() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "flat",
                "latest_sample_size": 10,
                "cumulative_sample_size": 50,
            }
        ]
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "abstain"
    assert result["ranking"][0]["candidate_status"] == "blocked"
    assert "CANDIDATE_LATEST_SAMPLE_TOO_LOW" in result["ranking"][0]["reason_codes"]
    assert "CANDIDATE_CUMULATIVE_SAMPLE_TOO_LOW" in result["ranking"][0]["reason_codes"]


def test_weak_single_horizon_candidate_is_penalized() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "SOLUSDT",
                "strategy": "breakout",
                "horizon": "15m",
                "selected_candidate_strength": "weak",
                "selected_stability_label": "single_horizon_only",
                "drift_direction": "flat",
                "edge_stability_score": 1.2,
            }
        ]
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "abstain"
    assert result["ranking"][0]["candidate_status"] == "penalized"
    assert "CANDIDATE_STRENGTH_WEAK" in result["ranking"][0]["reason_codes"]
    assert "CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY" in result["ranking"][0]["reason_codes"]
    assert "CANDIDATE_EDGE_STABILITY_SCORE_LOW" in result["ranking"][0]["reason_codes"]


def test_strongest_candidate_selected_when_clearly_better() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "increase",
                "edge_stability_score": 4.8,
                "score_delta": 0.3,
                "source_preference": "latest",
                "latest_sample_size": 55,
                "cumulative_sample_size": 220,
                "symbol_cumulative_support": 400,
                "strategy_cumulative_support": 320,
            },
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "4h",
                "selected_candidate_strength": "moderate",
                "selected_stability_label": "single_horizon_only",
                "drift_direction": "flat",
                "edge_stability_score": 2.6,
                "score_delta": 0.1,
                "source_preference": "latest",
                "latest_sample_size": 25,
                "cumulative_sample_size": 80,
                "symbol_cumulative_support": 170,
                "strategy_cumulative_support": 150,
            },
        ]
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "selected"
    assert result["selected_symbol"] == "BTCUSDT"
    assert result["selected_strategy"] == "swing"
    assert result["selected_horizon"] == "4h"
    assert result["reason_codes"] == ["CLEAR_TOP_CANDIDATE"]
    assert result["ranking"][0]["candidate_status"] == "eligible"


def test_tied_candidates_abstain() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "ADAUSDT",
                "strategy": "swing",
                "horizon": "1h",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "flat",
                "edge_stability_score": 3.5,
            },
            {
                "symbol": "BNBUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "flat",
                "edge_stability_score": 3.5,
            },
        ]
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "abstain"
    assert result["reason_codes"] == ["TOP_CANDIDATES_TIED"]
    assert len(result["ranking"]) == 2


def test_output_shape_includes_ranking_and_top_level_status_fields() -> None:
    payload = _base_mapper_payload(
        candidates=[
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "increase",
                "edge_stability_score": 4.0,
                "score_delta": 0.4,
            }
        ]
    )

    result = run_edge_selection_engine(payload)
    validation = validate_shadow_output(result)

    assert validation.is_valid is True
    assert result["mode"] == "shadow"
    assert isinstance(result["ranking"], list)
    assert "selection_status" in result
    assert "reason_codes" in result
    assert "candidates_considered" in result
    assert "latest_window_record_count" in result
    assert "cumulative_record_count" in result
    assert "selection_explanation" in result


def test_invalid_mapper_payload_returns_blocked_status() -> None:
    payload = _base_mapper_payload(
        ok=False,
        errors=["upstream validation failed"],
        candidates=[
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "increase",
            }
        ],
    )

    result = run_edge_selection_engine(payload)

    assert result["selection_status"] == "blocked"
    assert result["reason_codes"] == ["UPSTREAM_INPUT_INVALID"]
    assert result["ranking"] == []


def _base_mapper_payload(
    *,
    ok: bool = True,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    candidates: list[dict] | None = None,
) -> dict:
    return {
        "ok": ok,
        "generated_at": "2026-03-16T00:00:00+00:00",
        "latest_window_record_count": 124,
        "cumulative_record_count": 3578,
        "candidates": candidates or [],
        "errors": errors or [],
        "warnings": warnings or [],
        "history_line_count": 5,
    }
