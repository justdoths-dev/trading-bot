from __future__ import annotations

from pathlib import Path

import pytest

from src.research.edge_selection_shadow_writer import (
    read_edge_selection_shadow_outputs,
    write_edge_selection_shadow_output,
)


def test_valid_shadow_output_is_appended_to_jsonl_file(tmp_path: Path) -> None:
    output_path = tmp_path / "logs" / "edge_selection_shadow" / "edge_selection_shadow.jsonl"
    payload = _valid_shadow_payload()

    written_path = write_edge_selection_shadow_output(payload, output_path=output_path)

    assert written_path == output_path
    records = read_edge_selection_shadow_outputs(output_path)
    assert records == [payload]


def test_multiple_writes_append_multiple_lines_without_overwriting(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "edge_selection_shadow.jsonl"
    first_payload = _valid_shadow_payload(selected_symbol="BTCUSDT")
    second_payload = _valid_shadow_payload(selected_symbol="ETHUSDT")

    write_edge_selection_shadow_output(first_payload, output_path=output_path)
    write_edge_selection_shadow_output(second_payload, output_path=output_path)

    records = read_edge_selection_shadow_outputs(output_path)
    assert records == [first_payload, second_payload]
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_missing_output_directory_is_created_automatically(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "missing" / "edge_selection_shadow.jsonl"

    write_edge_selection_shadow_output(_valid_shadow_payload(), output_path=output_path)

    assert output_path.exists() is True
    assert output_path.parent.exists() is True


def test_invalid_shadow_output_raises_value_error(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "edge_selection_shadow.jsonl"
    payload = _valid_shadow_payload()
    payload["mode"] = "live"

    with pytest.raises(ValueError, match="Invalid shadow output payload"):
        write_edge_selection_shadow_output(payload, output_path=output_path)


def test_read_helper_returns_parsed_records_correctly(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "edge_selection_shadow.jsonl"
    payload = _valid_shadow_payload(selected_symbol="SOLUSDT", selected_strategy="breakout")

    write_edge_selection_shadow_output(payload, output_path=output_path)

    records = read_edge_selection_shadow_outputs(output_path)

    assert records[0]["selected_symbol"] == "SOLUSDT"
    assert records[0]["selected_strategy"] == "breakout"
    assert records[0]["ranking"][0]["symbol"] == "SOLUSDT"


def test_read_helper_skips_blank_lines(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "edge_selection_shadow.jsonl"
    payload = _valid_shadow_payload()

    write_edge_selection_shadow_output(payload, output_path=output_path)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")

    records = read_edge_selection_shadow_outputs(output_path)

    assert records == [payload]


def test_read_helper_raises_for_non_file_path(tmp_path: Path) -> None:
    directory_path = tmp_path / "shadow_dir"
    directory_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="is not a file"):
        read_edge_selection_shadow_outputs(directory_path)


def test_read_helper_raises_for_invalid_json_line(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "edge_selection_shadow.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('{"valid": true}\n{not-json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        read_edge_selection_shadow_outputs(output_path)


def test_writer_appends_trailing_newline(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "edge_selection_shadow.jsonl"

    write_edge_selection_shadow_output(_valid_shadow_payload(), output_path=output_path)

    content = output_path.read_text(encoding="utf-8")
    assert content.endswith("\n")


def _valid_shadow_payload(
    *,
    selected_symbol: str = "BTCUSDT",
    selected_strategy: str = "swing",
) -> dict:
    return {
        "generated_at": "2026-03-16T00:00:00+00:00",
        "mode": "shadow",
        "selection_status": "selected",
        "reason_codes": ["CLEAR_TOP_CANDIDATE"],
        "candidates_considered": 1,
        "selected_symbol": selected_symbol,
        "selected_strategy": selected_strategy,
        "selected_horizon": "4h",
        "selection_score": 8.5,
        "selection_confidence": 0.9,
        "ranking": [
            {
                "rank": 1,
                "symbol": selected_symbol,
                "strategy": selected_strategy,
                "horizon": "4h",
                "candidate_status": "eligible",
                "selection_score": 8.5,
                "selection_confidence": 0.9,
                "reason_codes": ["ELIGIBLE_CONSERVATIVE_PASS"],
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "drift_direction": "increase",
                "score_delta": 0.3,
                "source_preference": "latest",
                "edge_stability_score": 4.8,
                "selected_visible_horizons": ["1h", "4h"],
                "latest_sample_size": 24,
                "cumulative_sample_size": 110,
                "symbol_cumulative_support": 180,
                "strategy_cumulative_support": 141,
                "stability_gate_pass": True,
                "drift_blocked": False,
            }
        ],
    }
