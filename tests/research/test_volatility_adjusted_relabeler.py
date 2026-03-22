from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from src.research.experimental_labeling.volatility_adjusted_relabeler import (
    VolatilityAdjustedRelabelError,
    build_volatility_adjusted_relabel_dataset,
    discover_volatility_adjusted_source_files,
    relabel_trade_analysis_row,
)
from src.research.volatility_adjusted_relabel_report import (
    build_volatility_adjusted_relabel_summary,
    load_jsonl_records,
    run_volatility_adjusted_relabel_report,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_relabel_timestamp(records: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []

    for record in records:
        cloned = json.loads(json.dumps(record))
        experimental_labeling = cloned.get("experimental_labeling")
        if isinstance(experimental_labeling, dict):
            experimental_labeling.pop("relabel_timestamp", None)
        normalized.append(cloned)

    return normalized


def test_discovers_rotated_sources_in_deterministic_order(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    with gzip.open(logs_dir / "trade_analysis.jsonl.2.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"id": 1}\n')
    (logs_dir / "trade_analysis.jsonl.1").write_text('{"id": 2}\n', encoding="utf-8")
    (logs_dir / "trade_analysis.jsonl").write_text('{"id": 3}\n', encoding="utf-8")

    source_files = discover_volatility_adjusted_source_files(logs_dir)

    assert source_files == [
        logs_dir / "trade_analysis.jsonl.2.gz",
        logs_dir / "trade_analysis.jsonl.1",
        logs_dir / "trade_analysis.jsonl",
    ]


def test_relabel_trade_analysis_row_preserves_schema_and_future_returns() -> None:
    row = {
        "id": 100,
        "symbol": "BTCUSDT",
        "future_return_15m": 0.40,
        "future_return_1h": -0.60,
        "future_return_4h": 0.10,
        "future_label_15m": "old",
        "future_label_1h": "old",
        "future_label_4h": "old",
        "risk": {
            "atr_value": 1.0,
            "entry_price": 100.0,
        },
        "custom": {"keep": True},
    }

    relabeled = relabel_trade_analysis_row(row)["record"]

    assert relabeled["id"] == 100
    assert relabeled["symbol"] == "BTCUSDT"
    assert relabeled["future_return_15m"] == 0.40
    assert relabeled["future_return_1h"] == -0.60
    assert relabeled["future_return_4h"] == 0.10
    assert relabeled["custom"] == {"keep": True}
    assert relabeled["future_label_15m"] == "up"
    assert relabeled["future_label_1h"] == "down"
    assert relabeled["future_label_4h"] == "flat"

    metadata = relabeled["experimental_labeling"]
    assert metadata["labeling_method"] == "candidate_b_volatility_adjusted_v1"
    assert "relabel_timestamp" in metadata


def test_relabel_adds_experimental_metadata_and_fallback_flag() -> None:
    row = {
        "id": 101,
        "future_return_15m": 0.01,
        "future_return_1h": 0.01,
        "future_return_4h": 0.01,
        "risk": {
            "atr_value": "n/a",
            "entry_price": 100.0,
        },
    }

    relabeled = relabel_trade_analysis_row(row)
    metadata = relabeled["record"]["experimental_labeling"]

    assert metadata["labeling_method"] == "candidate_b_volatility_adjusted_v1"
    assert metadata["atr_pct"] == 0.0
    assert metadata["used_fallback_atr_pct"] is True
    assert metadata["thresholds"] == {
        "15m": 0.05,
        "1h": 0.1,
        "4h": 0.15,
    }
    assert relabeled["used_fallback_atr_pct"] is True


def test_dataset_build_is_rotation_aware_and_deterministic(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    with gzip.open(logs_dir / "trade_analysis.jsonl.1.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            '{"id": 1, "future_return_15m": 0.40, "future_return_1h": 0.20, "future_return_4h": -0.80, "risk": {"atr_value": 1.0, "entry_price": 100.0}}\n'
        )
    (logs_dir / "trade_analysis.jsonl").write_text(
        "\n".join(
            [
                '{"id": 2, "future_return_15m": 0.01, "future_return_1h": -0.05, "future_return_4h": 0.20, "risk": {"atr_value": "n/a", "entry_price": 100.0}}',
                '{"id": 3, "future_return_15m": "n/a", "future_return_1h": 0.60, "future_return_4h": -0.30, "future_label_15m": "keep", "risk": {"atr_value": 2.0, "entry_price": 100.0}}',
            ]
        ) + "\n",
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate_b.jsonl"

    first_summary = build_volatility_adjusted_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
    )
    first_records = _read_jsonl(output_path)

    second_summary = build_volatility_adjusted_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
    )
    second_records = _read_jsonl(output_path)

    assert [record["id"] for record in first_records] == [1, 2, 3]
    assert _strip_relabel_timestamp(first_records) == _strip_relabel_timestamp(second_records)
    assert first_summary["fallback_row_count"] == 1
    assert second_summary["fallback_row_count"] == 1
    assert first_records[0]["future_label_15m"] == "up"
    assert first_records[0]["future_label_4h"] == "down"
    assert first_records[1]["future_label_15m"] == "flat"
    assert first_records[1]["experimental_labeling"]["used_fallback_atr_pct"] is True
    assert first_records[2]["future_return_15m"] == "n/a"
    assert first_records[2]["future_label_15m"] == "keep"
    assert first_records[2]["future_label_1h"] == "flat"
    assert first_records[2]["future_label_4h"] == "flat"


def test_output_path_overlap_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    source_path = logs_dir / "trade_analysis.jsonl"
    source_path.write_text('{"id": 1}\n', encoding="utf-8")

    with pytest.raises(VolatilityAdjustedRelabelError, match="Output path must not overlap"):
        build_volatility_adjusted_relabel_dataset(
            input_dir=logs_dir,
            output_path=source_path,
        )


def test_report_build_summarizes_thresholds_and_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "candidate_b.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "future_label_15m": "up",
                        "future_label_1h": "flat",
                        "future_label_4h": "down",
                        "experimental_labeling": {
                            "labeling_method": "candidate_b_volatility_adjusted_v1",
                            "relabel_timestamp": "2026-03-23T00:00:00+00:00",
                            "atr_pct": 1.0,
                            "used_fallback_atr_pct": False,
                            "thresholds": {"15m": 0.35, "1h": 0.5, "4h": 0.65},
                        },
                    }
                ),
                json.dumps(
                    {
                        "future_label_15m": "flat",
                        "future_label_1h": "down",
                        "future_label_4h": "up",
                        "experimental_labeling": {
                            "labeling_method": "candidate_b_volatility_adjusted_v1",
                            "relabel_timestamp": "2026-03-23T00:00:01+00:00",
                            "atr_pct": 0.0,
                            "used_fallback_atr_pct": True,
                            "thresholds": {"15m": 0.05, "1h": 0.1, "4h": 0.15},
                        },
                    }
                ),
                json.dumps(
                    {
                        "future_label_15m": "down",
                        "future_label_1h": "down",
                        "future_label_4h": "down",
                        "experimental_labeling": {
                            "labeling_method": "some_other_method",
                            "thresholds": {"15m": 9.99, "1h": 9.99, "4h": 9.99},
                        },
                    }
                ),
            ]
        ) + "\n",
        encoding="utf-8",
    )

    records = load_jsonl_records(dataset_path)
    summary = build_volatility_adjusted_relabel_summary(records, input_path=dataset_path)

    assert summary["dataset_overview"] == {
        "total_row_count": 2,
        "fallback_row_count": 1,
        "fallback_row_ratio": 0.5,
    }
    assert summary["threshold_statistics_by_horizon"]["15m"] == {
        "min": 0.05,
        "max": 0.35,
        "mean": 0.2,
        "median": 0.2,
    }
    assert summary["label_distribution_counts_by_horizon"]["4h"] == {
        "up": 1,
        "down": 1,
        "flat": 0,
    }
    assert summary["label_distribution_ratios_by_horizon"]["1h"] == {
        "up": 0.0,
        "down": 0.5,
        "flat": 0.5,
    }


def test_report_runner_writes_json_and_markdown(tmp_path: Path) -> None:
    dataset_path = tmp_path / "candidate_b.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "future_label_15m": "flat",
                "future_label_1h": "flat",
                "future_label_4h": "flat",
                "experimental_labeling": {
                    "labeling_method": "candidate_b_volatility_adjusted_v1",
                    "relabel_timestamp": "2026-03-23T00:00:00+00:00",
                    "atr_pct": 0.0,
                    "used_fallback_atr_pct": True,
                    "thresholds": {"15m": 0.05, "1h": 0.1, "4h": 0.15},
                },
            }
        ) + "\n",
        encoding="utf-8",
    )
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"

    result = run_volatility_adjusted_relabel_report(
        input_path=dataset_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert result["summary"]["dataset_overview"]["fallback_row_count"] == 1
    assert json_output.exists()
    assert markdown_output.exists()

