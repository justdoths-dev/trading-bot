from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.research import future_return_labeler
from src.research.future_return_labeling_common import (
    build_future_update_for_row,
    build_future_update_from_prices,
    has_future_fields_for_horizon,
    horizon_to_timedelta,
    is_mature_for_horizon,
    iter_supported_horizons,
    parse_logged_at_to_utc,
)


def test_parse_logged_at_to_utc_handles_iso_z_naive_and_invalid_values() -> None:
    assert parse_logged_at_to_utc("2026-04-08T12:34:56Z") == datetime(
        2026, 4, 8, 12, 34, 56, tzinfo=UTC
    )
    assert parse_logged_at_to_utc("2026-04-08T12:34:56") == datetime(
        2026, 4, 8, 12, 34, 56, tzinfo=UTC
    )
    assert parse_logged_at_to_utc("2026-04-08T21:34:56+09:00") == datetime(
        2026, 4, 8, 12, 34, 56, tzinfo=UTC
    )
    assert parse_logged_at_to_utc("") is None
    assert parse_logged_at_to_utc("not-a-timestamp") is None
    assert parse_logged_at_to_utc(None) is None


def test_horizon_mapping_and_iteration_match_expected_contract() -> None:
    assert iter_supported_horizons() == ("15m", "1h", "4h")
    assert horizon_to_timedelta("15m").total_seconds() == 900
    assert horizon_to_timedelta("1h").total_seconds() == 3600
    assert horizon_to_timedelta("4h").total_seconds() == 14400


def test_is_mature_for_horizon_honors_exact_boundary() -> None:
    row_ts = datetime(2026, 4, 8, 0, 0, tzinfo=UTC)

    assert is_mature_for_horizon(
        row_ts,
        datetime(2026, 4, 8, 1, 0, tzinfo=UTC),
        "1h",
    )
    assert not is_mature_for_horizon(
        row_ts,
        datetime(2026, 4, 8, 0, 59, 59, tzinfo=UTC),
        "1h",
    )


def test_has_future_fields_for_horizon_requires_nonempty_return_and_label() -> None:
    assert not has_future_fields_for_horizon({}, "15m")
    assert not has_future_fields_for_horizon({"future_return_15m": 1.0}, "15m")
    assert not has_future_fields_for_horizon({"future_label_15m": "up"}, "15m")
    assert not has_future_fields_for_horizon(
        {"future_return_15m": None, "future_label_15m": "up"},
        "15m",
    )
    assert not has_future_fields_for_horizon(
        {"future_return_15m": 0.0, "future_label_15m": ""},
        "15m",
    )
    assert not has_future_fields_for_horizon(
        {"future_return_15m": 0.0, "future_label_15m": "   "},
        "15m",
    )
    assert has_future_fields_for_horizon(
        {
            "future_return_15m": 0.0,
            "future_label_15m": "flat",
        },
        "15m",
    )


def test_build_future_update_for_row_uses_row_entry_price_and_explicit_future_price() -> None:
    row = {
        "logged_at": "2026-04-08T00:00:00Z",
        "symbol": "BTCUSDT",
        "risk": {"entry_price": 100.0},
    }

    assert build_future_update_for_row(
        row=row,
        future_price=105.0,
        horizon="15m",
    ) == {
        "future_return_15m": 5.0,
        "future_label_15m": "up",
    }


def test_common_future_update_matches_live_labeler_formula_and_thresholds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    record = {
        "logged_at": "2026-04-08T00:00:00Z",
        "symbol": "BTCUSDT",
        "risk": {"entry_price": 100.0},
    }
    log_path = tmp_path / "trade_analysis.jsonl"
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        future_return_labeler.ccxt,
        "binance",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        future_return_labeler,
        "_fetch_horizon_prices",
        lambda **_kwargs: {
            "15m": 105.0,
            "1h": 90.0,
            "4h": 100.1,
        },
    )

    future_return_labeler.label_dataset(log_path=log_path)

    relabeled = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert {
        "future_return_15m": relabeled["future_return_15m"],
        "future_label_15m": relabeled["future_label_15m"],
    } == build_future_update_from_prices(
        entry_price=100.0,
        future_price=105.0,
        horizon="15m",
    )
    assert {
        "future_return_1h": relabeled["future_return_1h"],
        "future_label_1h": relabeled["future_label_1h"],
    } == build_future_update_from_prices(
        entry_price=100.0,
        future_price=90.0,
        horizon="1h",
    )
    assert {
        "future_return_4h": relabeled["future_return_4h"],
        "future_label_4h": relabeled["future_label_4h"],
    } == build_future_update_from_prices(
        entry_price=100.0,
        future_price=100.1,
        horizon="4h",
    )


def test_label_dataset_does_not_treat_placeholder_keys_as_fully_labeled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    record = {
        "logged_at": "2026-04-08T00:00:00Z",
        "symbol": "BTCUSDT",
        "risk": {"entry_price": 100.0},
        "future_return_15m": 999.0,
        "future_label_15m": "",
        "future_return_1h": None,
        "future_label_1h": "up",
        "future_return_4h": 1.23,
        "future_label_4h": "flat",
    }
    log_path = tmp_path / "trade_analysis.jsonl"
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        future_return_labeler.ccxt,
        "binance",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        future_return_labeler,
        "_fetch_horizon_prices",
        lambda **_kwargs: {
            "15m": 105.0,
            "1h": 110.0,
            "4h": 120.0,
        },
    )

    result = future_return_labeler.label_dataset(log_path=log_path)
    relabeled = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert result["updated_records"] == 1
    assert relabeled["future_label_15m"] == "up"
    assert relabeled["future_return_1h"] == 10.0
