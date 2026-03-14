from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from src.research.cumulative_dataset_builder import build_cumulative_dataset


def test_merges_active_and_rotated_plain_file(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "trade_analysis.jsonl.1").write_text('{"id": 1}\n', encoding="utf-8")
    (logs_dir / "trade_analysis.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
    output_path = logs_dir / "trade_analysis_cumulative.jsonl"

    summary = build_cumulative_dataset(logs_dir=logs_dir, output_path=output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == '{"id": 1}\n{"id": 2}\n'
    assert summary["lines_written"] == 2
    assert summary["output_path"] == str(output_path)
    assert str(logs_dir / "trade_analysis.jsonl.1") in summary["files_read"]
    assert str(logs_dir / "trade_analysis.jsonl") in summary["files_read"]


def test_merges_active_and_gz_rotated_file(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    gz_path = logs_dir / "trade_analysis.jsonl.2.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write('{"id": 10}\n')
    (logs_dir / "trade_analysis.jsonl").write_text('{"id": 11}\n', encoding="utf-8")
    output_path = logs_dir / "trade_analysis_cumulative.jsonl"

    summary = build_cumulative_dataset(logs_dir=logs_dir, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == '{"id": 10}\n{"id": 11}\n'
    assert summary["lines_written"] == 2
    assert str(gz_path) in summary["files_read"]


def test_orders_multiple_rotations_oldest_to_active(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    with gzip.open(logs_dir / "trade_analysis.jsonl.2.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"id": 100}\n')

    (logs_dir / "trade_analysis.jsonl.1").write_text('{"id": 101}\n', encoding="utf-8")
    (logs_dir / "trade_analysis.jsonl").write_text('{"id": 102}\n', encoding="utf-8")
    output_path = logs_dir / "trade_analysis_cumulative.jsonl"

    build_cumulative_dataset(logs_dir=logs_dir, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == (
        '{"id": 100}\n{"id": 101}\n{"id": 102}\n'
    )


def test_handles_missing_rotated_files_safely(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    active_path = logs_dir / "trade_analysis.jsonl"
    active_path.write_text('{"id": 20}\n', encoding="utf-8")
    output_path = logs_dir / "trade_analysis_cumulative.jsonl"

    summary = build_cumulative_dataset(logs_dir=logs_dir, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == '{"id": 20}\n'
    assert summary["files_read"] == [str(active_path)]
    assert summary["lines_written"] == 1


def test_writes_valid_output_file(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "trade_analysis.jsonl.3.gz").write_bytes(gzip.compress(b'{"id": 30}\n'))
    (logs_dir / "trade_analysis.jsonl.1").write_text('{"id": 31}\n', encoding="utf-8")
    (logs_dir / "trade_analysis.jsonl").write_text('{"id": 32}\n', encoding="utf-8")
    output_path = logs_dir / "trade_analysis_cumulative.jsonl"

    build_cumulative_dataset(logs_dir=logs_dir, output_path=output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"id": 30}', '{"id": 31}', '{"id": 32}']


def test_does_not_crash_on_empty_input_set(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    output_path = logs_dir / "trade_analysis_cumulative.jsonl"

    summary = build_cumulative_dataset(logs_dir=logs_dir, output_path=output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""
    assert summary["files_read"] == []
    assert summary["lines_written"] == 0
    assert summary["output_path"] == str(output_path)


def test_raises_if_output_overlaps_with_input_file(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    active_path = logs_dir / "trade_analysis.jsonl"
    active_path.write_text('{"id": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        build_cumulative_dataset(
            logs_dir=logs_dir,
            output_path=active_path,
        )
