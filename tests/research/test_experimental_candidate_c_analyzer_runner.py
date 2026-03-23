from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.research import experimental_candidate_c_analyzer_runner


def test_runner_uses_existing_research_analyzer_with_direct_input_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    input_path = tmp_path / "candidate_c.jsonl"
    input_path.write_text('{"ok": true}\n', encoding="utf-8")
    output_dir = tmp_path / "reports"

    def _fake_run_research_analyzer(*, input_path: Path, output_dir: Path) -> dict[str, Any]:
        captured["input_path"] = input_path
        captured["output_dir"] = output_dir
        return {
            "dataset_overview": {"total_records": 10},
            "strategy_lab": {"dataset_rows": 8},
        }

    monkeypatch.setattr(
        experimental_candidate_c_analyzer_runner,
        "run_research_analyzer",
        _fake_run_research_analyzer,
    )

    result = experimental_candidate_c_analyzer_runner.run_experimental_candidate_c_analyzer(
        input_path=input_path,
        output_dir=output_dir,
    )

    assert captured["input_path"] == input_path
    assert captured["output_dir"] == output_dir
    assert result["summary_json"] == str(output_dir / "summary.json")
    assert result["summary_md"] == str(output_dir / "summary.md")
    assert result["dataset_overview"]["total_records"] == 10
    assert result["strategy_lab"]["dataset_rows"] == 8


def test_runner_raises_clearly_when_input_dataset_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "missing_candidate_c.jsonl"
    output_dir = tmp_path / "reports"

    with pytest.raises(FileNotFoundError, match="Candidate C analyzer input dataset not found"):
        experimental_candidate_c_analyzer_runner.run_experimental_candidate_c_analyzer(
            input_path=input_path,
            output_dir=output_dir,
        )


def test_default_paths_match_candidate_c2_experiment_location() -> None:
    assert experimental_candidate_c_analyzer_runner.DEFAULT_INPUT_PATH == Path(
        "logs/experiments/trade_analysis_relabel_candidate_c_c2_moderate.jsonl"
    )
    assert experimental_candidate_c_analyzer_runner.DEFAULT_OUTPUT_DIR == Path(
        "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate"
    )