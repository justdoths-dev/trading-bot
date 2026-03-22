from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from src.research.asymmetric_threshold_relabel_report import (
    build_asymmetric_threshold_relabel_summary,
    load_jsonl_records,
    run_asymmetric_threshold_relabel_report,
)
from src.research.experimental_labeling.asymmetric_threshold_relabeler import (
    AsymmetricThresholdRelabelError,
    build_asymmetric_threshold_relabel_dataset,
    discover_asymmetric_threshold_source_files,
    relabel_trade_analysis_row,
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

    source_files = discover_asymmetric_threshold_source_files(logs_dir)

    assert source_files == [
        logs_dir / "trade_analysis.jsonl.2.gz",
        logs_dir / "trade_analysis.jsonl.1",
        logs_dir / "trade_analysis.jsonl",
    ]


def test_relabel_trade_analysis_row_preserves_future_returns_and_rewrites_only_labels() -> None:
    row = {
        "id": 100,
        "symbol": "BTCUSDT",
        "future_return_15m": 0.05,
        "future_return_1h": -0.11,
        "future_return_4h": 0.18,
        "future_label_15m": "old_up",
        "future_label_1h": "old_down",
        "future_label_4h": "old_flat",
        "custom": {"keep": True},
    }

    relabeled = relabel_trade_analysis_row(row, variant_name="c2_moderate")
    record = relabeled["record"]

    assert record["id"] == 100
    assert record["symbol"] == "BTCUSDT"
    assert record["future_return_15m"] == 0.05
    assert record["future_return_1h"] == -0.11
    assert record["future_return_4h"] == 0.18
    assert record["custom"] == {"keep": True}
    assert record["future_label_15m"] == "up"
    assert record["future_label_1h"] == "flat"
    assert record["future_label_4h"] == "up"


def test_relabel_preserves_non_numeric_future_label_fields() -> None:
    row = {
        "future_return_15m": "n/a",
        "future_return_1h": 0.10,
        "future_return_4h": -0.30,
        "future_label_15m": "keep_me",
        "future_label_1h": "replace_me",
        "future_label_4h": "replace_me",
    }

    relabeled = relabel_trade_analysis_row(row, variant_name="c1_conservative")
    record = relabeled["record"]

    assert record["future_label_15m"] == "keep_me"
    assert record["future_label_1h"] == "up"
    assert record["future_label_4h"] == "down"
    assert relabeled["label_rebuilt"] == {"15m": False, "1h": True, "4h": True}


def test_relabel_adds_comparison_friendly_metadata() -> None:
    row = {
        "future_return_15m": 0.01,
        "future_return_1h": -0.20,
        "future_return_4h": 0.14,
    }

    relabeled = relabel_trade_analysis_row(
        row,
        variant_name="c3_stronger",
        source_path=Path("logs/trade_analysis.jsonl"),
    )
    metadata = relabeled["record"]["experimental_labeling"]

    assert metadata["labeling_method"] == "candidate_c_asymmetric_threshold_v1"
    assert metadata["variant"] == "c3_stronger"
    assert metadata["thresholds"] == {
        "15m": {"up": 0.05, "down": 0.07},
        "1h": {"up": 0.1, "down": 0.13},
        "4h": {"up": 0.15, "down": 0.2},
    }
    assert metadata["source_path"] == "logs/trade_analysis.jsonl"
    assert metadata["numeric_return_available_by_horizon"] == {
        "15m": True,
        "1h": True,
        "4h": True,
    }
    assert metadata["label_rebuilt_by_horizon"] == {
        "15m": True,
        "1h": True,
        "4h": True,
    }
    assert "relabel_timestamp" in metadata


def test_unknown_variant_handling_is_explicit() -> None:
    with pytest.raises(ValueError, match="Unknown asymmetric threshold variant"):
        relabel_trade_analysis_row({"future_return_15m": 0.1}, variant_name="unknown")


def test_dataset_build_preserves_row_count_and_is_rotation_aware(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    with gzip.open(logs_dir / "trade_analysis.jsonl.1.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            '{"id": 1, "future_return_15m": 0.055, "future_return_1h": -0.11, "future_return_4h": -0.30}\n'
        )
    (logs_dir / "trade_analysis.jsonl").write_text(
        "\n".join(
            [
                '{"id": 2, "future_return_15m": 0.01, "future_return_1h": -0.20, "future_return_4h": 0.20}',
                '{"id": 3, "future_return_15m": "n/a", "future_return_1h": 0.10, "future_return_4h": -0.18, "future_label_15m": "keep"}',
            ]
        ) + "\n",
        encoding="utf-8",
    )
    output_path = logs_dir / "experiments" / "candidate_c.jsonl"

    first_summary = build_asymmetric_threshold_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
        variant_name="c2_moderate",
    )
    first_records = _read_jsonl(output_path)

    second_summary = build_asymmetric_threshold_relabel_dataset(
        input_dir=logs_dir,
        output_path=output_path,
        variant_name="c2_moderate",
    )
    second_records = _read_jsonl(output_path)

    assert [record["id"] for record in first_records] == [1, 2, 3]
    assert len(first_records) == 3
    assert _strip_relabel_timestamp(first_records) == _strip_relabel_timestamp(second_records)
    assert first_summary["records_written"] == 3
    assert second_summary["records_written"] == 3
    assert first_records[0]["future_label_15m"] == "up"
    assert first_records[0]["future_label_1h"] == "flat"
    assert first_records[0]["future_label_4h"] == "down"
    assert first_records[1]["future_label_1h"] == "down"
    assert first_records[2]["future_label_15m"] == "keep"
    assert first_records[2]["future_label_1h"] == "up"


def test_output_path_overlap_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    source_path = logs_dir / "trade_analysis.jsonl"
    source_path.write_text('{"id": 1}\n', encoding="utf-8")

    with pytest.raises(AsymmetricThresholdRelabelError, match="Output path must not overlap"):
        build_asymmetric_threshold_relabel_dataset(
            input_dir=logs_dir,
            output_path=source_path,
            variant_name="c2_moderate",
        )


def test_report_build_summarizes_variant_thresholds_and_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "candidate_c.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "future_label_15m": "up",
                        "future_label_1h": "flat",
                        "future_label_4h": "down",
                        "experimental_labeling": {
                            "labeling_method": "candidate_c_asymmetric_threshold_v1",
                            "variant": "c2_moderate",
                            "thresholds": {
                                "15m": {"up": 0.05, "down": 0.065},
                                "1h": {"up": 0.10, "down": 0.125},
                                "4h": {"up": 0.15, "down": 0.19},
                            },
                            "label_rebuilt_by_horizon": {
                                "15m": True,
                                "1h": True,
                                "4h": True,
                            },
                        },
                    }
                ),
                "",
                "not-json",
                json.dumps(
                    {
                        "future_label_15m": "flat",
                        "future_label_1h": "down",
                        "future_label_4h": "up",
                        "experimental_labeling": {
                            "labeling_method": "candidate_c_asymmetric_threshold_v1",
                            "variant": "c2_moderate",
                            "thresholds": {
                                "15m": {"up": 0.05, "down": 0.065},
                                "1h": {"up": 0.10, "down": 0.125},
                                "4h": {"up": 0.15, "down": 0.19},
                            },
                            "label_rebuilt_by_horizon": {
                                "15m": True,
                                "1h": True,
                                "4h": True,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "future_label_15m": "down",
                        "future_label_1h": "down",
                        "future_label_4h": "down",
                        "experimental_labeling": {
                            "labeling_method": "candidate_c_asymmetric_threshold_v1",
                            "variant": "c1_conservative",
                            "thresholds": {
                                "15m": {"up": 0.05, "down": 0.06},
                                "1h": {"up": 0.10, "down": 0.12},
                                "4h": {"up": 0.15, "down": 0.18},
                            },
                            "label_rebuilt_by_horizon": {
                                "15m": True,
                                "1h": True,
                                "4h": True,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "future_label_15m": "down",
                        "future_label_1h": "up",
                        "future_label_4h": "flat",
                        "experimental_labeling": {
                            "labeling_method": "candidate_c_asymmetric_threshold_v1",
                            "variant": "c2_moderate",
                            "thresholds": {
                                "15m": {"up": 0.05, "down": 0.065},
                                "1h": {"up": 0.10, "down": 0.125},
                                "4h": {"up": 0.15, "down": 0.19},
                            },
                            "label_rebuilt_by_horizon": {
                                "15m": False,
                                "1h": False,
                                "4h": False,
                            },
                        },
                    }
                ),
                "[]",
            ]
        ) + "\n",
        encoding="utf-8",
    )

    records, instrumentation = load_jsonl_records(dataset_path)
    summary = build_asymmetric_threshold_relabel_summary(
        records,
        input_path=dataset_path,
        output_path=tmp_path / "summary.json",
        variant_name="c2_moderate",
        parser_instrumentation=instrumentation,
    )

    assert summary["dataset_overview"] == {
        "total_row_count": 3,
        "rebuilt_row_count_by_horizon": {
            "15m": 2,
            "1h": 2,
            "4h": 2,
        },
    }
    assert summary["parser_instrumentation"] == {
        "blank_line_count": 1,
        "invalid_json_line_count": 1,
        "non_object_line_count": 1,
    }
    assert summary["thresholds_by_horizon"] == {
        "15m": {"up": 0.05, "down": 0.065},
        "1h": {"up": 0.1, "down": 0.125},
        "4h": {"up": 0.15, "down": 0.19},
    }
    assert summary["label_distribution_counts_by_horizon"]["1h"] == {
        "up": 0,
        "down": 1,
        "flat": 1,
    }
    assert summary["label_distribution_ratios_by_horizon"]["4h"] == {
        "up": 0.5,
        "down": 0.5,
        "flat": 0.0,
    }


def test_report_ignores_preserved_labels_when_not_rebuilt(tmp_path: Path) -> None:
    dataset_path = tmp_path / "candidate_c.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "future_label_15m": "down",
                        "future_label_1h": "up",
                        "future_label_4h": "flat",
                        "experimental_labeling": {
                            "labeling_method": "candidate_c_asymmetric_threshold_v1",
                            "variant": "c2_moderate",
                            "label_rebuilt_by_horizon": {
                                "15m": False,
                                "1h": False,
                                "4h": False,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "future_label_15m": "up",
                        "future_label_1h": "flat",
                        "future_label_4h": "down",
                        "experimental_labeling": {
                            "labeling_method": "candidate_c_asymmetric_threshold_v1",
                            "variant": "c2_moderate",
                            "label_rebuilt_by_horizon": {
                                "15m": True,
                                "1h": True,
                                "4h": True,
                            },
                        },
                    }
                ),
            ]
        ) + "\n",
        encoding="utf-8",
    )

    records, instrumentation = load_jsonl_records(dataset_path)
    summary = build_asymmetric_threshold_relabel_summary(
        records,
        input_path=dataset_path,
        output_path=tmp_path / "summary.json",
        variant_name="c2_moderate",
        parser_instrumentation=instrumentation,
    )

    assert summary["dataset_overview"]["total_row_count"] == 2
    assert summary["dataset_overview"]["rebuilt_row_count_by_horizon"] == {
        "15m": 1,
        "1h": 1,
        "4h": 1,
    }
    assert summary["label_distribution_counts_by_horizon"] == {
        "15m": {"up": 1, "down": 0, "flat": 0},
        "1h": {"up": 0, "down": 0, "flat": 1},
        "4h": {"up": 0, "down": 1, "flat": 0},
    }


def test_report_runner_writes_json_and_markdown(tmp_path: Path) -> None:
    dataset_path = tmp_path / "candidate_c.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "future_label_15m": "flat",
                "future_label_1h": "flat",
                "future_label_4h": "flat",
                "experimental_labeling": {
                    "labeling_method": "candidate_c_asymmetric_threshold_v1",
                    "variant": "c2_moderate",
                    "thresholds": {
                        "15m": {"up": 0.05, "down": 0.065},
                        "1h": {"up": 0.10, "down": 0.125},
                        "4h": {"up": 0.15, "down": 0.19},
                    },
                    "label_rebuilt_by_horizon": {
                        "15m": True,
                        "1h": True,
                        "4h": True,
                    },
                },
            }
        ) + "\n",
        encoding="utf-8",
    )
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"

    result = run_asymmetric_threshold_relabel_report(
        input_path=dataset_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
        variant_name="c2_moderate",
    )

    assert result["summary"]["dataset_overview"]["total_row_count"] == 1
    assert result["summary"]["dataset_overview"]["rebuilt_row_count_by_horizon"] == {
        "15m": 1,
        "1h": 1,
        "4h": 1,
    }
    assert json_output.exists()
    assert markdown_output.exists()
