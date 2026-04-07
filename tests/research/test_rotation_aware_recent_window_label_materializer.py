from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from src.research import rotation_aware_recent_window_label_materializer as materializer


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_materializer_reads_mixed_rotation_sources_and_matches_recent_window_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logs_dir = tmp_path / "logs"
    input_path = logs_dir / "trade_analysis.jsonl"
    output_path = logs_dir / "research_reports" / "latest" / "recent.jsonl"

    _write_gzip_jsonl(
        logs_dir / "trade_analysis.jsonl.2.gz",
        [
            {
                "id": 1,
                "logged_at": "2026-04-06T22:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 100.0},
            }
        ],
    )
    _write_jsonl(
        logs_dir / "trade_analysis.jsonl.1",
        [
            {
                "id": 2,
                "logged_at": "2026-04-07T00:00:00Z",
                "symbol": "ETHUSDT",
                "risk": {"entry_price": 200.0},
            },
            {
                "id": 3,
                "logged_at": "2026-04-07T01:00:00Z",
                "symbol": "XRPUSDT",
                "risk": {"entry_price": 300.0},
            },
        ],
    )
    _write_jsonl(
        input_path,
        [
            {
                "id": 4,
                "logged_at": "2026-04-07T02:00:00Z",
                "symbol": "SOLUSDT",
                "risk": {"entry_price": 400.0},
            }
        ],
    )
    _write_jsonl(
        logs_dir / "trade_analysis_cumulative.jsonl",
        [
            {
                "id": 999,
                "logged_at": "2026-04-07T02:30:00Z",
                "symbol": "DOGEUSDT",
                "risk": {"entry_price": 999.0},
            }
        ],
    )

    monkeypatch.setattr(materializer.ccxt, "binance", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        materializer,
        "fetch_horizon_prices",
        lambda **_kwargs: None,
    )

    summary = materializer.materialize_recent_window_labels(
        input_path=input_path,
        output_path=output_path,
        max_age_hours=2,
        max_rows=2,
    )

    records = _read_jsonl(output_path)
    assert [record["id"] for record in records] == [3, 4]
    assert summary["source_file_count"] == 3
    assert summary["total_rows"] == 2
    assert summary["rows_updated"] == 0
    assert summary["rows_preserved"] == 2
    assert summary["mature_unlabeled_seen"] == 1
    assert summary["mature_unlabeled_horizon_count"] == 2
    assert summary["rows_still_missing_after_attempt"] == 1
    assert summary["invalid_rows_skipped_for_update"] == 0
    assert summary["fetch_failures"] == 1
    assert summary["per_horizon"] == {
        "15m": {
            "mature_unlabeled_seen": 1,
            "updated": 0,
            "still_missing_after_attempt": 1,
        },
        "1h": {
            "mature_unlabeled_seen": 1,
            "updated": 0,
            "still_missing_after_attempt": 1,
        },
        "4h": {
            "mature_unlabeled_seen": 0,
            "updated": 0,
            "still_missing_after_attempt": 0,
        },
    }


def test_materializer_updates_mature_unlabeled_rows_with_production_equivalent_fetches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logs_dir = tmp_path / "logs"
    input_path = logs_dir / "trade_analysis.jsonl"
    output_path = logs_dir / "research_reports" / "latest" / "recent.jsonl"

    _write_gzip_jsonl(
        logs_dir / "trade_analysis.jsonl.1.gz",
        [
            {
                "id": 1,
                "logged_at": "2026-04-08T00:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 100.0},
                "note": "seed",
            },
            {
                "id": 2,
                "logged_at": "2026-04-08T02:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 115.0},
                "future_return_15m": 999.0,
                "future_label_15m": "keep",
                "future_return_1h": 998.0,
                "future_label_1h": "keep",
                "future_return_4h": 997.0,
                "future_label_4h": "keep",
            },
        ],
    )
    _write_jsonl(
        input_path,
        [
            {
                "id": 3,
                "logged_at": "2026-04-08T03:50:00Z",
                "symbol": "ETHUSDT",
                "risk": {"entry_price": 210.0},
            },
            {
                "id": 4,
                "logged_at": "2026-04-08T04:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 120.0},
            },
        ],
    )

    calls: list[tuple[str, str]] = []

    def _fake_fetch_horizon_prices(**kwargs):
        calls.append(
            (
                kwargs["symbol"],
                kwargs["logged_at"].isoformat(),
            )
        )
        return {
            "15m": 105.0,
            "1h": 110.0,
            "4h": 120.0,
        }

    monkeypatch.setattr(materializer.ccxt, "binance", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(materializer, "fetch_horizon_prices", _fake_fetch_horizon_prices)

    summary = materializer.materialize_recent_window_labels(
        input_path=input_path,
        output_path=output_path,
        max_age_hours=36,
        max_rows=2500,
    )

    records = _read_jsonl(output_path)
    by_id = {record["id"]: record for record in records}

    assert calls == [("BTC/USDT", "2026-04-08T00:00:00+00:00")]
    assert by_id[1] == {
        "id": 1,
        "logged_at": "2026-04-08T00:00:00Z",
        "symbol": "BTCUSDT",
        "risk": {"entry_price": 100.0},
        "note": "seed",
        "future_return_15m": 5.0,
        "future_label_15m": "up",
        "future_return_1h": 10.0,
        "future_label_1h": "up",
        "future_return_4h": 20.0,
        "future_label_4h": "up",
    }
    assert by_id[2]["future_return_15m"] == 999.0
    assert by_id[2]["future_label_15m"] == "keep"
    assert by_id[2]["future_return_1h"] == 998.0
    assert by_id[2]["future_label_1h"] == "keep"
    assert by_id[2]["future_return_4h"] == 997.0
    assert by_id[2]["future_label_4h"] == "keep"
    assert by_id[3] == {
        "id": 3,
        "logged_at": "2026-04-08T03:50:00Z",
        "symbol": "ETHUSDT",
        "risk": {"entry_price": 210.0},
    }
    assert by_id[4] == {
        "id": 4,
        "logged_at": "2026-04-08T04:00:00Z",
        "symbol": "BTCUSDT",
        "risk": {"entry_price": 120.0},
    }

    assert summary == {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_file_count": 2,
        "total_rows": 4,
        "rows_updated": 1,
        "rows_preserved": 3,
        "mature_unlabeled_seen": 1,
        "mature_unlabeled_horizon_count": 3,
        "rows_still_missing_after_attempt": 0,
        "invalid_rows_skipped_for_update": 0,
        "fetch_failures": 0,
        "per_horizon": {
            "15m": {
                "mature_unlabeled_seen": 1,
                "updated": 1,
                "still_missing_after_attempt": 0,
            },
            "1h": {
                "mature_unlabeled_seen": 1,
                "updated": 1,
                "still_missing_after_attempt": 0,
            },
            "4h": {
                "mature_unlabeled_seen": 1,
                "updated": 1,
                "still_missing_after_attempt": 0,
            },
        },
    }


def test_materializer_preserves_labeled_rows_and_skips_not_yet_mature_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logs_dir = tmp_path / "logs"
    input_path = logs_dir / "trade_analysis.jsonl"
    output_path = logs_dir / "research_reports" / "latest" / "recent.jsonl"

    _write_jsonl(
        input_path,
        [
            {
                "id": 1,
                "logged_at": "2026-04-08T00:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 100.0},
                "future_return_15m": 5.0,
                "future_label_15m": "up",
                "future_return_1h": 10.0,
                "future_label_1h": "up",
                "future_return_4h": 20.0,
                "future_label_4h": "up",
            },
            {
                "id": 2,
                "logged_at": "2026-04-08T03:50:00Z",
                "symbol": "ETHUSDT",
                "risk": {"entry_price": 210.0},
            },
            {
                "id": 3,
                "logged_at": "2026-04-08T04:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 120.0},
            },
        ],
    )

    calls: list[str] = []

    def _unexpected_fetch(**kwargs):
        calls.append(kwargs["symbol"])
        return {
            "15m": 101.0,
            "1h": 102.0,
            "4h": 103.0,
        }

    monkeypatch.setattr(materializer.ccxt, "binance", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(materializer, "fetch_horizon_prices", _unexpected_fetch)

    summary = materializer.materialize_recent_window_labels(
        input_path=input_path,
        output_path=output_path,
        max_age_hours=36,
        max_rows=2500,
    )

    records = _read_jsonl(output_path)
    assert records == [
        {
            "id": 1,
            "logged_at": "2026-04-08T00:00:00Z",
            "symbol": "BTCUSDT",
            "risk": {"entry_price": 100.0},
            "future_return_15m": 5.0,
            "future_label_15m": "up",
            "future_return_1h": 10.0,
            "future_label_1h": "up",
            "future_return_4h": 20.0,
            "future_label_4h": "up",
        },
        {
            "id": 2,
            "logged_at": "2026-04-08T03:50:00Z",
            "symbol": "ETHUSDT",
            "risk": {"entry_price": 210.0},
        },
        {
            "id": 3,
            "logged_at": "2026-04-08T04:00:00Z",
            "symbol": "BTCUSDT",
            "risk": {"entry_price": 120.0},
        },
    ]
    assert calls == []
    assert summary["rows_updated"] == 0
    assert summary["mature_unlabeled_seen"] == 0
    assert summary["rows_still_missing_after_attempt"] == 0


def test_materializer_treats_partial_placeholder_values_as_unlabeled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logs_dir = tmp_path / "logs"
    input_path = logs_dir / "trade_analysis.jsonl"
    output_path = logs_dir / "research_reports" / "latest" / "recent.jsonl"

    _write_jsonl(
        input_path,
        [
            {
                "id": 1,
                "logged_at": "2026-04-08T00:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 100.0},
                "future_return_15m": 999.0,
                "future_label_15m": "",
            },
            {
                "id": 2,
                "logged_at": "2026-04-08T01:00:00Z",
                "symbol": "BTCUSDT",
                "risk": {"entry_price": 101.0},
            },
        ],
    )

    monkeypatch.setattr(materializer.ccxt, "binance", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        materializer,
        "fetch_horizon_prices",
        lambda **_kwargs: {
            "15m": 105.0,
            "1h": 110.0,
            "4h": 120.0,
        },
    )

    summary = materializer.materialize_recent_window_labels(
        input_path=input_path,
        output_path=output_path,
        max_age_hours=36,
        max_rows=2500,
    )

    records = _read_jsonl(output_path)
    assert records[0]["future_return_15m"] == 5.0
    assert records[0]["future_label_15m"] == "up"
    assert summary["rows_updated"] == 1
