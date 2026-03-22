from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.research.rotation_aware_future_return_relabeler import (
    build_rotation_aware_future_return_relabel_dataset,
    discover_experiment_source_files,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]



def test_discovers_mixed_plain_and_gzip_sources_in_historical_order(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    with gzip.open(logs_dir / "trade_analysis.jsonl.2.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"id": 1}\n')
    (logs_dir / "trade_analysis.jsonl.1").write_text('{"id": 2}\n', encoding="utf-8")
    (logs_dir / "trade_analysis.jsonl").write_text('{"id": 3}\n', encoding="utf-8")

    source_files = discover_experiment_source_files(logs_dir)

    assert source_files == [
        logs_dir / "trade_analysis.jsonl.2.gz",
        logs_dir / "trade_analysis.jsonl.1",
        logs_dir / "trade_analysis.jsonl",
    ]



def test_build_preserves_historical_order_and_writes_deterministic_output(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    with gzip.open(logs_dir / "trade_analysis.jsonl.2.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"id": 100, "future_return_15m": 0.12}\n')
    (logs_dir / "trade_analysis.jsonl.1").write_text(
        '{"id": 101, "future_return_15m": -0.12}\n',
        encoding="utf-8",
    )
    (logs_dir / "trade_analysis.jsonl").write_text(
        '{"id": 102, "future_return_15m": 0.00}\n',
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate.jsonl"

    summary = build_rotation_aware_future_return_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
    )

    records = _read_jsonl(output_path)
    assert [record["id"] for record in records] == [100, 101, 102]
    assert [record["future_label_15m"] for record in records] == ["up", "down", "flat"]
    assert summary["records_written"] == 3



def test_relabeling_logic_with_up_down_flat_and_exact_boundary(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "trade_analysis.jsonl").write_text(
        "\n".join(
            [
                '{"id": 1, "future_return_15m": 0.06}',
                '{"id": 2, "future_return_15m": -0.06}',
                '{"id": 3, "future_return_15m": 0.01}',
                '{"id": 4, "future_return_15m": 0.05}',
                '{"id": 5, "future_return_15m": -0.05}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate.jsonl"

    build_rotation_aware_future_return_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
        threshold_15m=0.05,
    )

    records = _read_jsonl(output_path)
    assert [record["future_label_15m"] for record in records] == [
        "up",
        "down",
        "flat",
        "flat",
        "flat",
    ]



def test_reuses_existing_numeric_future_return_values_without_rewriting_them(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "trade_analysis.jsonl").write_text(
        '{"id": 1, "future_return_15m": "0.12", "future_label_15m": "old"}\n',
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate.jsonl"

    build_rotation_aware_future_return_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
    )

    records = _read_jsonl(output_path)
    assert records[0]["future_return_15m"] == "0.12"
    assert records[0]["future_label_15m"] == "up"



def test_records_without_numeric_future_return_remain_safe_and_writable(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "trade_analysis.jsonl").write_text(
        '{"id": 1, "future_return_15m": "n/a", "future_label_15m": "keep"}\n',
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate.jsonl"

    summary = build_rotation_aware_future_return_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
    )

    records = _read_jsonl(output_path)
    assert records[0]["future_return_15m"] == "n/a"
    assert records[0]["future_label_15m"] == "keep"
    assert summary["records_with_numeric_return_15m"] == 0
    assert summary["relabeled_count_15m"] == 0



def test_invalid_json_lines_are_counted_and_skipped(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "trade_analysis.jsonl").write_text(
        '{"id": 1, "future_return_15m": 0.10}\nnot-json\n\n{"id": 2, "future_return_15m": -0.10}\n',
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate.jsonl"

    summary = build_rotation_aware_future_return_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
    )

    records = _read_jsonl(output_path)
    assert [record["id"] for record in records] == [1, 2]
    assert summary["total_records_seen"] == 2
    assert summary["records_written"] == 2
    assert summary["invalid_json_line_count"] == 1
    assert summary["blank_line_count"] == 1



def test_summary_structure_contains_required_fields(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "trade_analysis.jsonl").write_text(
        '{"id": 1, "future_return_15m": 0.12, "future_return_1h": 0.00, "future_return_4h": -0.20}\n',
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate.jsonl"

    summary = build_rotation_aware_future_return_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
    )

    assert summary["source_file_count"] == 1
    assert summary["total_records_seen"] == 1
    assert summary["records_written"] == 1
    assert summary["records_with_numeric_return_15m"] == 1
    assert summary["records_with_numeric_return_1h"] == 1
    assert summary["records_with_numeric_return_4h"] == 1
    assert summary["relabeled_count_15m"] == 1
    assert summary["relabeled_count_1h"] == 1
    assert summary["relabeled_count_4h"] == 1
    assert summary["output_path"] == str(output_path)
    assert summary["thresholds_pct"] == {
        "15m": 0.05,
        "1h": 0.1,
        "4h": 0.15,
    }
    assert isinstance(summary["source_files"], list)
    assert "generated_at" in summary
