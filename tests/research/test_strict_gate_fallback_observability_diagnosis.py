from __future__ import annotations

from src.research.strict_gate_fallback_observability_diagnosis import build_report


def _row_with_quality_gate(
    *,
    total_candidates: int,
    strict_kept_count: int,
    strict_dropped_count: int,
    fallback_applied: bool,
    fallback_restored_count: int,
    final_kept_count: int,
    strict_dropped_candidates: list[dict] | None = None,
    fallback_restored_candidates: list[dict] | None = None,
    final_kept_candidates: list[dict] | None = None,
    include_selection_output: bool = True,
) -> dict:
    row = {
        "edge_selection_mapper_payload": {
            "candidate_quality_gate": {
                "total_candidates": total_candidates,
                "strict_kept_count": strict_kept_count,
                "strict_dropped_count": strict_dropped_count,
                "strict_dropped_candidates": strict_dropped_candidates or [],
                "fallback_applied": fallback_applied,
                "fallback_restored_count": fallback_restored_count,
                "fallback_restored_candidates": fallback_restored_candidates or [],
                "final_kept_count": final_kept_count,
                "final_kept_candidates": final_kept_candidates or [],
            }
        }
    }
    if include_selection_output:
        row["edge_selection_output"] = {"selection_status": "selected"}
    return row


def _candidate(symbol: str, strategy: str, horizon: str) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
    }


def _dropped_candidate(symbol: str, strategy: str, horizon: str, reason: str) -> dict:
    return {
        "candidate": _candidate(symbol, strategy, horizon),
        "reason": reason,
    }


def test_strict_pass_only_row_is_counted_correctly() -> None:
    rows = [
        _row_with_quality_gate(
            total_candidates=2,
            strict_kept_count=2,
            strict_dropped_count=0,
            fallback_applied=False,
            fallback_restored_count=0,
            final_kept_count=2,
            final_kept_candidates=[
                _candidate("BTCUSDT", "swing", "4h"),
                _candidate("ETHUSDT", "intraday", "1h"),
            ],
        )
    ]

    report = build_report(rows)
    summary = report["summary"]

    assert summary["total_rows_examined"] == 1
    assert summary["overall_counts"]["strict_pass_rows"] == 1
    assert summary["overall_counts"]["strict_fail_rows"] == 0
    assert summary["overall_counts"]["fallback_applied_rows"] == 0
    assert summary["overall_counts"]["fallback_only_rows"] == 0
    assert summary["overall_counts"]["strict_full_drop_rows"] == 0


def test_strict_full_fail_with_fallback_restore_is_counted_correctly() -> None:
    rows = [
        _row_with_quality_gate(
            total_candidates=4,
            strict_kept_count=0,
            strict_dropped_count=4,
            fallback_applied=True,
            fallback_restored_count=4,
            final_kept_count=4,
            strict_dropped_candidates=[
                _dropped_candidate("BTCUSDT", "swing", "4h", "MIN_SAMPLE_COUNT"),
                _dropped_candidate("ETHUSDT", "swing", "4h", "MIN_SAMPLE_COUNT"),
                _dropped_candidate("SOLUSDT", "intraday", "1h", "MEDIAN_EDGE"),
                _dropped_candidate("BNBUSDT", "intraday", "15m", "POSITIVE_RATE"),
            ],
            fallback_restored_candidates=[
                _candidate("BTCUSDT", "swing", "4h"),
                _candidate("ETHUSDT", "swing", "4h"),
                _candidate("SOLUSDT", "intraday", "1h"),
                _candidate("BNBUSDT", "intraday", "15m"),
            ],
            final_kept_candidates=[
                _candidate("BTCUSDT", "swing", "4h"),
                _candidate("ETHUSDT", "swing", "4h"),
                _candidate("SOLUSDT", "intraday", "1h"),
                _candidate("BNBUSDT", "intraday", "15m"),
            ],
        )
    ]

    report = build_report(rows)
    summary = report["summary"]

    assert summary["overall_counts"]["strict_pass_rows"] == 0
    assert summary["overall_counts"]["strict_fail_rows"] == 1
    assert summary["overall_counts"]["fallback_applied_rows"] == 1
    assert summary["overall_counts"]["fallback_only_rows"] == 1
    assert summary["overall_counts"]["strict_full_drop_rows"] == 1


def test_mixed_row_is_counted_correctly() -> None:
    rows = [
        _row_with_quality_gate(
            total_candidates=3,
            strict_kept_count=1,
            strict_dropped_count=2,
            fallback_applied=True,
            fallback_restored_count=2,
            final_kept_count=3,
            strict_dropped_candidates=[
                _dropped_candidate("ETHUSDT", "swing", "4h", "MIN_SAMPLE_COUNT"),
                _dropped_candidate("SOLUSDT", "intraday", "1h", "POSITIVE_RATE"),
            ],
            fallback_restored_candidates=[
                _candidate("ETHUSDT", "swing", "4h"),
                _candidate("SOLUSDT", "intraday", "1h"),
            ],
            final_kept_candidates=[
                _candidate("BTCUSDT", "swing", "4h"),
                _candidate("ETHUSDT", "swing", "4h"),
                _candidate("SOLUSDT", "intraday", "1h"),
            ],
        )
    ]

    report = build_report(rows)
    summary = report["summary"]

    assert summary["overall_counts"]["strict_pass_rows"] == 1
    assert summary["overall_counts"]["mixed_rows"] == 1
    assert summary["overall_counts"]["fallback_applied_rows"] == 1
    assert summary["overall_counts"]["fallback_only_rows"] == 0


def test_missing_quality_gate_block_is_handled_safely() -> None:
    rows = [
        {"edge_selection_output": {"selection_status": "selected"}},
        _row_with_quality_gate(
            total_candidates=1,
            strict_kept_count=1,
            strict_dropped_count=0,
            fallback_applied=False,
            fallback_restored_count=0,
            final_kept_count=1,
            final_kept_candidates=[_candidate("BTCUSDT", "swing", "4h")],
        ),
    ]

    report = build_report(rows)
    summary = report["summary"]

    assert report["rows_missing_quality_gate"] == 1
    assert summary["total_rows_examined"] == 1
    assert summary["overall_counts"]["strict_pass_rows"] == 1


def test_drop_reason_aggregation_counts_reasons_correctly() -> None:
    rows = [
        _row_with_quality_gate(
            total_candidates=2,
            strict_kept_count=0,
            strict_dropped_count=2,
            fallback_applied=True,
            fallback_restored_count=2,
            final_kept_count=2,
            strict_dropped_candidates=[
                _dropped_candidate("BTCUSDT", "swing", "4h", "MIN_SAMPLE_COUNT"),
                _dropped_candidate("ETHUSDT", "swing", "4h", "MIN_SAMPLE_COUNT"),
            ],
            fallback_restored_candidates=[
                _candidate("BTCUSDT", "swing", "4h"),
                _candidate("ETHUSDT", "swing", "4h"),
            ],
            final_kept_candidates=[
                _candidate("BTCUSDT", "swing", "4h"),
                _candidate("ETHUSDT", "swing", "4h"),
            ],
        ),
        _row_with_quality_gate(
            total_candidates=1,
            strict_kept_count=0,
            strict_dropped_count=1,
            fallback_applied=True,
            fallback_restored_count=1,
            final_kept_count=1,
            strict_dropped_candidates=[
                _dropped_candidate("SOLUSDT", "intraday", "1h", "POSITIVE_RATE"),
            ],
            fallback_restored_candidates=[
                _candidate("SOLUSDT", "intraday", "1h"),
            ],
            final_kept_candidates=[
                _candidate("SOLUSDT", "intraday", "1h"),
            ],
        ),
    ]

    report = build_report(rows)
    reasons = report["summary"]["drop_reason_distribution"]

    assert reasons[0]["reason"] == "MIN_SAMPLE_COUNT"
    assert reasons[0]["count"] == 2

    reason_map = {item["reason"]: item["count"] for item in reasons}
    assert reason_map["POSITIVE_RATE"] == 1


def test_identity_level_survival_accounting_infers_strict_survivors_correctly() -> None:
    rows = [
        _row_with_quality_gate(
            total_candidates=3,
            strict_kept_count=1,
            strict_dropped_count=2,
            fallback_applied=True,
            fallback_restored_count=2,
            final_kept_count=3,
            strict_dropped_candidates=[
                _dropped_candidate("ETHUSDT", "swing", "4h", "MIN_SAMPLE_COUNT"),
                _dropped_candidate("SOLUSDT", "intraday", "1h", "POSITIVE_RATE"),
            ],
            fallback_restored_candidates=[
                _candidate("ETHUSDT", "swing", "4h"),
                _candidate("SOLUSDT", "intraday", "1h"),
            ],
            final_kept_candidates=[
                _candidate("BTCUSDT", "swing", "4h"),
                _candidate("ETHUSDT", "swing", "4h"),
                _candidate("SOLUSDT", "intraday", "1h"),
            ],
        )
    ]

    report = build_report(rows)
    identities = report["summary"]["identity_level_summary"]
    identity_map = {
        (item["symbol"], item["strategy"], item["horizon"]): item
        for item in identities
    }

    btc = identity_map[("BTCUSDT", "swing", "4h")]
    eth = identity_map[("ETHUSDT", "swing", "4h")]
    sol = identity_map[("SOLUSDT", "intraday", "1h")]

    assert btc["strict_survived_rows"] == 1
    assert btc["fallback_restored_rows"] == 0
    assert btc["strict_survival_rate"] == 1.0

    assert eth["strict_survived_rows"] == 0
    assert eth["strict_dropped_rows"] == 1
    assert eth["fallback_restored_rows"] == 1

    assert sol["strict_survived_rows"] == 0
    assert sol["strict_dropped_rows"] == 1
    assert sol["fallback_restored_rows"] == 1