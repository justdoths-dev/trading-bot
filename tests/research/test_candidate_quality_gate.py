from __future__ import annotations

import json
from pathlib import Path

from src.research.candidate_quality_gate import (
    apply_candidate_quality_gate,
    compute_candidate_metrics,
    determine_drop_reason,
    is_selected_record,
)


def _candidate(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
) -> dict[str, str]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
    }


def _record(
    *,
    selected_symbol: str = "BTCUSDT",
    selected_strategy: str = "swing",
    selected_horizon: str = "4h",
    return_value: float | str | None = 1.0,
    return_field_horizon: str = "4h",
    selection_status: str = "selected",
    include_edge_selection_output: bool = True,
    include_top_level_selected_fields: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": "ETHUSDT",
        "future_return_15m": None,
        "future_return_1h": None,
        "future_return_4h": None,
    }

    if include_edge_selection_output:
        payload["edge_selection_output"] = {
            "selection_status": selection_status,
            "selected_symbol": selected_symbol,
            "selected_strategy": selected_strategy,
            "selected_horizon": selected_horizon,
        }

    if include_top_level_selected_fields:
        payload["selected_symbol"] = selected_symbol
        payload["selected_strategy"] = selected_strategy
        payload["selected_horizon"] = selected_horizon

    payload[f"future_return_{return_field_horizon}"] = return_value
    return payload


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def test_is_selected_record_accepts_modern_selected_row() -> None:
    record = _record()
    assert is_selected_record(record) is True


def test_is_selected_record_rejects_abstain_row() -> None:
    record = _record(selection_status="abstain")
    assert is_selected_record(record) is False


def test_is_selected_record_accepts_legacy_top_level_selected_fields() -> None:
    record = _record(
        include_edge_selection_output=False,
        include_top_level_selected_fields=True,
    )
    assert is_selected_record(record) is True


def test_compute_candidate_metrics_requires_exact_selected_identity_match() -> None:
    candidate = _candidate()
    records = [
        _record(return_value=1.0),
        _record(return_value=3.0),
        _record(selected_strategy="intraday", return_value=100.0),
        _record(selected_symbol="ETHUSDT", return_value=100.0),
        {
            "symbol": "BTCUSDT",
            "selected_strategy": "swing",
            "future_return_4h": 100.0,
        },
    ]

    metrics = compute_candidate_metrics(candidate, records)

    assert metrics == {
        "positive_rate_pct": 100.0,
        "median_return_pct": 2.0,
        "sample_count": 2,
    }


def test_compute_candidate_metrics_excludes_horizon_mismatch() -> None:
    candidate = _candidate(horizon="4h")
    records = [
        _record(selected_horizon="1h", return_value=9.0, return_field_horizon="4h"),
        _record(selected_horizon="4h", return_value=1.5, return_field_horizon="4h"),
    ]

    metrics = compute_candidate_metrics(candidate, records)

    assert metrics == {
        "positive_rate_pct": 100.0,
        "median_return_pct": 1.5,
        "sample_count": 1,
    }


def test_compute_candidate_metrics_excludes_abstain_rows() -> None:
    candidate = _candidate()
    records = [
        _record(selection_status="abstain", return_value=100.0),
        _record(selection_status="selected", return_value=2.0),
    ]

    metrics = compute_candidate_metrics(candidate, records)

    assert metrics == {
        "positive_rate_pct": 100.0,
        "median_return_pct": 2.0,
        "sample_count": 1,
    }


def test_compute_candidate_metrics_supports_legacy_top_level_selected_fields() -> None:
    candidate = _candidate()
    records = [
        _record(
            include_edge_selection_output=False,
            include_top_level_selected_fields=True,
            return_value=1.0,
        ),
        _record(
            include_edge_selection_output=False,
            include_top_level_selected_fields=True,
            return_value="3.0",
        ),
    ]

    metrics = compute_candidate_metrics(candidate, records)

    assert metrics == {
        "positive_rate_pct": 100.0,
        "median_return_pct": 2.0,
        "sample_count": 2,
    }


def test_compute_candidate_metrics_ignores_malformed_or_non_selected_rows() -> None:
    candidate = _candidate()
    records = [
        {"edge_selection_output": {"selection_status": "selected"}, "future_return_4h": 100.0},
        {"selected_symbol": "BTCUSDT", "selected_strategy": "swing", "future_return_4h": 100.0},
        {"selected_symbol": "BTCUSDT", "selected_horizon": "4h", "future_return_4h": 100.0},
        _record(return_value=2.0),
    ]

    metrics = compute_candidate_metrics(candidate, records)

    assert metrics == {
        "positive_rate_pct": 100.0,
        "median_return_pct": 2.0,
        "sample_count": 1,
    }


def test_determine_drop_reason_reports_near_floor_median_failure() -> None:
    reason = determine_drop_reason(
        {
            "sample_count": 20,
            "median_return_pct": 0.0,
            "positive_rate_pct": 55.0,
        }
    )
    assert reason == "near_floor_rescue_median_not_positive"


def test_determine_drop_reason_reports_near_floor_positive_rate_failure() -> None:
    reason = determine_drop_reason(
        {
            "sample_count": 20,
            "median_return_pct": 0.1,
            "positive_rate_pct": 49.999,
        }
    )
    assert reason == "near_floor_rescue_positive_rate_below_minimum"


def test_apply_candidate_quality_gate_drops_for_insufficient_sample(tmp_path: Path) -> None:
    candidate = _candidate()
    path = _write_jsonl(
        tmp_path / "trade_analysis.jsonl",
        [_record(return_value=1.0) for _ in range(19)],
    )

    result = apply_candidate_quality_gate([candidate], trade_analysis_path=path)

    assert result["total_candidates"] == 1
    assert result["strict_kept_count"] == 0
    assert result["strict_dropped_count"] == 1
    assert result["fallback_applied"] is True
    assert result["fallback_restored_count"] == 1
    assert result["final_kept_count"] == 1

    assert result["strict_dropped_candidates"][0]["reason"] == "sample_count_below_minimum"
    assert result["fallback_restored_candidates"] == [candidate]
    assert result["final_kept_candidates"] == [candidate]

    # compatibility
    assert result["kept_candidates"] == [candidate]


def test_apply_candidate_quality_gate_rescues_near_floor_positive_candidates(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    returns = [1.0] * 10 + [0.0] * 10
    path = _write_jsonl(
        tmp_path / "trade_analysis.jsonl",
        [_record(return_value=value) for value in returns],
    )

    result = apply_candidate_quality_gate([candidate], trade_analysis_path=path)

    assert result["strict_kept_count"] == 1
    assert result["strict_dropped_count"] == 0
    assert result["fallback_applied"] is False
    assert result["fallback_restored_count"] == 0
    assert result["final_kept_count"] == 1
    assert result["strict_kept_candidates"] == [candidate]
    assert result["final_kept_candidates"] == [candidate]


def test_apply_candidate_quality_gate_does_not_rescue_near_floor_non_positive_median(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    returns = [1.0] * 10 + [-1.0] * 10
    path = _write_jsonl(
        tmp_path / "trade_analysis.jsonl",
        [_record(return_value=value) for value in returns],
    )

    result = apply_candidate_quality_gate([candidate], trade_analysis_path=path)

    assert result["strict_kept_count"] == 0
    assert result["strict_dropped_count"] == 1
    assert result["fallback_applied"] is True
    assert result["fallback_restored_count"] == 1
    assert result["final_kept_count"] == 1
    assert (
        result["strict_dropped_candidates"][0]["reason"]
        == "near_floor_rescue_median_not_positive"
    )


def test_apply_candidate_quality_gate_does_not_rescue_below_near_floor_band(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    path = _write_jsonl(
        tmp_path / "trade_analysis.jsonl",
        [_record(return_value=1.0) for _ in range(19)],
    )

    result = apply_candidate_quality_gate([candidate], trade_analysis_path=path)

    assert result["strict_kept_count"] == 0
    assert result["strict_dropped_count"] == 1
    assert result["fallback_applied"] is True
    assert result["fallback_restored_count"] == 1
    assert result["final_kept_count"] == 1
    assert result["strict_dropped_candidates"][0]["reason"] == "sample_count_below_minimum"


def test_apply_candidate_quality_gate_drops_for_negative_median(tmp_path: Path) -> None:
    candidate = _candidate()
    returns = [-2.0] * 16 + [1.0] * 14
    path = _write_jsonl(
        tmp_path / "trade_analysis.jsonl",
        [_record(return_value=value) for value in returns],
    )

    result = apply_candidate_quality_gate([candidate], trade_analysis_path=path)

    assert result["strict_kept_count"] == 0
    assert result["strict_dropped_count"] == 1
    assert result["fallback_applied"] is True
    assert result["fallback_restored_count"] == 1
    assert result["final_kept_count"] == 1

    assert result["strict_dropped_candidates"][0]["reason"] == "median_return_pct_negative"
    assert result["final_kept_candidates"] == [candidate]


def test_apply_candidate_quality_gate_drops_for_low_positive_rate(tmp_path: Path) -> None:
    candidate = _candidate()
    returns = [1.0] * 17 + [0.0] * 23
    path = _write_jsonl(
        tmp_path / "trade_analysis.jsonl",
        [_record(return_value=value) for value in returns],
    )

    result = apply_candidate_quality_gate([candidate], trade_analysis_path=path)

    assert result["strict_kept_count"] == 0
    assert result["strict_dropped_count"] == 1
    assert result["fallback_applied"] is True
    assert result["fallback_restored_count"] == 1
    assert result["final_kept_count"] == 1

    assert result["strict_dropped_candidates"][0]["reason"] == "positive_rate_pct_below_minimum"


def test_apply_candidate_quality_gate_preserves_normal_thresholds_outside_rescue_path(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    returns = [1.0] * 14 + [0.0] * 16
    path = _write_jsonl(
        tmp_path / "trade_analysis.jsonl",
        [_record(return_value=value) for value in returns],
    )

    result = apply_candidate_quality_gate([candidate], trade_analysis_path=path)

    assert result["strict_kept_count"] == 1
    assert result["strict_dropped_count"] == 0
    assert result["fallback_applied"] is False
    assert result["fallback_restored_count"] == 0
    assert result["final_kept_count"] == 1
    assert result["strict_kept_candidates"] == [candidate]


def test_apply_candidate_quality_gate_keeps_and_drops_mixed_candidates(
    tmp_path: Path,
) -> None:
    btc_candidate = _candidate(symbol="BTCUSDT")
    eth_candidate = _candidate(symbol="ETHUSDT")

    records: list[dict[str, object]] = []
    records.extend(_record(selected_symbol="BTCUSDT", return_value=1.0) for _ in range(30))
    records.extend(_record(selected_symbol="ETHUSDT", return_value=-1.0) for _ in range(30))

    path = _write_jsonl(tmp_path / "trade_analysis.jsonl", records)

    result = apply_candidate_quality_gate(
        [btc_candidate, eth_candidate],
        trade_analysis_path=path,
    )

    assert result["total_candidates"] == 2
    assert result["strict_kept_count"] == 1
    assert result["strict_dropped_count"] == 1
    assert result["fallback_applied"] is False
    assert result["fallback_restored_count"] == 0
    assert result["final_kept_count"] == 1

    assert result["strict_kept_candidates"] == [btc_candidate]
    assert result["final_kept_candidates"] == [btc_candidate]

    assert result["strict_dropped_candidates"][0]["candidate"]["symbol"] == "ETHUSDT"


def test_apply_candidate_quality_gate_restores_original_candidates_when_all_drop(
    tmp_path: Path,
) -> None:
    first_candidate = _candidate(symbol="BTCUSDT")
    second_candidate = _candidate(symbol="ETHUSDT")

    records = [_record(selected_symbol="BTCUSDT", return_value=-1.0) for _ in range(30)]
    records.extend(_record(selected_symbol="ETHUSDT", return_value=-1.0) for _ in range(30))
    path = _write_jsonl(tmp_path / "trade_analysis.jsonl", records)

    result = apply_candidate_quality_gate(
        [first_candidate, second_candidate],
        trade_analysis_path=path,
    )

    assert result["strict_kept_count"] == 0
    assert result["strict_dropped_count"] == 2
    assert result["fallback_applied"] is True
    assert result["fallback_restored_count"] == 2
    assert result["final_kept_count"] == 2

    assert result["fallback_restored_candidates"] == [first_candidate, second_candidate]
    assert result["final_kept_candidates"] == [first_candidate, second_candidate]