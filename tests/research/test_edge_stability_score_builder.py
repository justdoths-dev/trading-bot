from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.research import edge_stability_score_builder
from src.research.edge_stability_score_builder import build_edge_stability_scores


def _comparison_payload() -> dict[str, Any]:
    return {
        "edge_candidates_comparison": {
            "15m": {
                "latest_candidate_strength": "moderate",
                "cumulative_candidate_strength": "moderate",
                "latest_top_strategy_group": "swing",
                "cumulative_top_strategy_group": "swing",
                "latest_top_symbol_group": "BTCUSDT",
                "cumulative_top_symbol_group": "BTCUSDT",
                "latest_top_alignment_state_group": "aligned",
                "cumulative_top_alignment_state_group": "aligned",
            },
            "1h": {
                "latest_candidate_strength": "moderate",
                "cumulative_candidate_strength": "moderate",
                "latest_top_strategy_group": "swing",
                "cumulative_top_strategy_group": "swing",
                "latest_top_symbol_group": "BTCUSDT",
                "cumulative_top_symbol_group": "BTCUSDT",
                "latest_top_alignment_state_group": "aligned",
                "cumulative_top_alignment_state_group": "aligned",
            },
            "4h": {
                "latest_candidate_strength": "insufficient_data",
                "cumulative_candidate_strength": "weak",
                "latest_top_strategy_group": "n/a",
                "cumulative_top_strategy_group": "trend",
                "latest_top_symbol_group": "n/a",
                "cumulative_top_symbol_group": "ETHUSDT",
                "latest_top_alignment_state_group": "n/a",
                "cumulative_top_alignment_state_group": "mixed",
            },
        },
        "edge_stability_comparison": {
            "strategy": {
                "latest_stability_label": "multi_horizon_confirmed",
                "cumulative_stability_label": "multi_horizon_confirmed",
                "latest_group": "swing",
                "cumulative_group": "swing",
                "latest_visible_horizons": ["15m", "1h"],
                "cumulative_visible_horizons": ["15m", "1h"],
            },
            "symbol": {
                "latest_stability_label": "single_horizon_only",
                "cumulative_stability_label": "single_horizon_only",
                "latest_group": "BTCUSDT",
                "cumulative_group": "BTCUSDT",
                "latest_visible_horizons": ["15m"],
                "cumulative_visible_horizons": ["15m"],
            },
            "alignment_state": {
                "latest_stability_label": "single_horizon_only",
                "cumulative_stability_label": "unstable",
                "latest_group": "aligned",
                "cumulative_group": "aligned",
                "latest_visible_horizons": ["15m"],
                "cumulative_visible_horizons": ["15m", "1h"],
            },
        },
    }


def test_build_edge_stability_scores_happy_path(tmp_path: Path) -> None:
    input_path = tmp_path / "comparison.json"
    output_dir = tmp_path / "edge_scores"
    input_path.write_text(json.dumps(_comparison_payload()), encoding="utf-8")

    result = build_edge_stability_scores(input_path=input_path, output_dir=output_dir)

    assert "edge_stability_scores" in result
    assert "strategy" in result["edge_stability_scores"]
    assert result["score_summary"]["top_strategy"]["group"] == "swing"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()


def test_build_edge_stability_scores_handles_missing_sections_safely(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "comparison.json"
    output_dir = tmp_path / "edge_scores"
    input_path.write_text(json.dumps({}), encoding="utf-8")

    result = build_edge_stability_scores(input_path=input_path, output_dir=output_dir)

    assert result["edge_stability_scores"]["strategy"] == []
    assert result["edge_stability_scores"]["symbol"] == []
    assert result["score_summary"]["top_strategy"] == {
        "group": "n/a",
        "score": 0.0,
        "source_preference": "n/a",
    }


def test_score_calculation_matches_rule(tmp_path: Path) -> None:
    input_path = tmp_path / "comparison.json"
    output_dir = tmp_path / "edge_scores"
    input_path.write_text(json.dumps(_comparison_payload()), encoding="utf-8")

    result = build_edge_stability_scores(input_path=input_path, output_dir=output_dir)

    strategy_item = result["edge_stability_scores"]["strategy"][0]
    assert strategy_item["group"] == "swing"
    assert strategy_item["score"] == 5.0
    assert strategy_item["score_components"] == {
        "candidate_strength_weight": 2.0,
        "stability_label_weight": 2.0,
        "horizon_bonus": 1.0,
    }


def test_markdown_and_json_outputs_are_written(tmp_path: Path) -> None:
    input_path = tmp_path / "comparison.json"
    output_dir = tmp_path / "edge_scores"
    input_path.write_text(json.dumps(_comparison_payload()), encoding="utf-8")

    build_edge_stability_scores(input_path=input_path, output_dir=output_dir)

    assert (output_dir / "summary.json").read_text(encoding="utf-8")
    assert (output_dir / "summary.md").read_text(encoding="utf-8")


def test_output_wording_remains_observational_only(tmp_path: Path) -> None:
    input_path = tmp_path / "comparison.json"
    output_dir = tmp_path / "edge_scores"
    input_path.write_text(json.dumps(_comparison_payload()), encoding="utf-8")

    result = build_edge_stability_scores(input_path=input_path, output_dir=output_dir)
    markdown = (output_dir / "summary.md").read_text(encoding="utf-8").lower()
    joined = json.dumps(result).lower()

    assert "buy" not in markdown
    assert "sell" not in markdown
    assert "recommended" not in markdown
    assert "best trade" not in markdown
    assert "buy" not in joined
    assert "sell" not in joined


def test_top_item_summary_is_present(tmp_path: Path) -> None:
    input_path = tmp_path / "comparison.json"
    output_dir = tmp_path / "edge_scores"
    input_path.write_text(json.dumps(_comparison_payload()), encoding="utf-8")

    result = build_edge_stability_scores(input_path=input_path, output_dir=output_dir)

    assert result["score_summary"] == {
        "top_strategy": {
            "group": "swing",
            "score": 5.0,
            "source_preference": "cumulative",
        },
        "top_symbol": {
            "group": "BTCUSDT",
            "score": 3.5,
            "source_preference": "cumulative",
        },
        "top_alignment_state": {
            "group": "aligned",
            "score": 3.5,
            "source_preference": "latest",
        },
    }


def test_default_paths_resolve_correctly() -> None:
    base_dir = Path(edge_stability_score_builder.__file__).resolve().parents[2] / "logs"

    assert edge_stability_score_builder._default_input_path() == (
        base_dir / "research_reports" / "comparison" / "summary.json"
    )
    assert edge_stability_score_builder._default_output_dir() == (
        base_dir / "research_reports" / "edge_scores"
    )
