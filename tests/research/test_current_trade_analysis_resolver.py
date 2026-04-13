from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from src.research.inputs.current_trade_analysis_resolver import (
    discover_current_trade_analysis_files,
)


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


def test_discover_current_trade_analysis_files_includes_rotations_and_symbol_shards(
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    _write_gzip_jsonl(logs_dir / "trade_analysis.jsonl.3.gz", [{"id": 1}])
    _write_jsonl(logs_dir / "trade_analysis.jsonl.2", [{"id": 2}])
    _write_jsonl(logs_dir / "trade_analysis.jsonl.1", [{"id": 3}])
    _write_jsonl(logs_dir / "trade_analysis.jsonl", [])
    _write_jsonl(logs_dir / "trade_analysis_btcusdt.jsonl", [{"id": 4}])
    _write_jsonl(logs_dir / "trade_analysis_ethusdt.jsonl", [{"id": 5}])
    _write_jsonl(logs_dir / "trade_analysis_cumulative.jsonl", [{"id": 999}])
    _write_jsonl(logs_dir / "trade_analysis_cumulative_snapshot.jsonl", [{"id": 1000}])

    source_files = discover_current_trade_analysis_files(
        logs_dir,
        include_rotated_base=True,
    )

    assert source_files == [
        logs_dir / "trade_analysis.jsonl.3.gz",
        logs_dir / "trade_analysis.jsonl.2",
        logs_dir / "trade_analysis.jsonl.1",
        logs_dir / "trade_analysis.jsonl",
        logs_dir / "trade_analysis_btcusdt.jsonl",
        logs_dir / "trade_analysis_ethusdt.jsonl",
    ]


def test_discover_current_trade_analysis_files_can_limit_to_active_current_sources(
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    _write_jsonl(logs_dir / "trade_analysis.jsonl.1", [{"id": 1}])
    _write_jsonl(logs_dir / "trade_analysis.jsonl", [])
    _write_jsonl(logs_dir / "trade_analysis_btcusdt.jsonl", [{"id": 2}])
    _write_jsonl(logs_dir / "trade_analysis_ethusdt.jsonl", [{"id": 3}])
    _write_jsonl(logs_dir / "trade_analysis_cumulative.jsonl", [{"id": 999}])

    source_files = discover_current_trade_analysis_files(
        logs_dir,
        include_rotated_base=False,
    )

    assert source_files == [
        logs_dir / "trade_analysis.jsonl",
        logs_dir / "trade_analysis_btcusdt.jsonl",
        logs_dir / "trade_analysis_ethusdt.jsonl",
    ]
