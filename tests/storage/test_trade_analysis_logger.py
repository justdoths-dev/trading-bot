from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.storage.trade_analysis_logger import (
    TradeAnalysisLogger,
    TradeAnalysisLoggerConfig,
)


def _build_strategy_result() -> dict[str, object]:
    return {
        "selected_strategy": "momentum_breakout",
        "selected_result": {
            "strategy": "momentum_breakout",
            "bias": "bullish",
            "signal": "long",
            "confidence": 0.82,
            "reason": "Momentum aligned across intraday and swing frames.",
            "timeframe_summary": {
                "5m": {"bias": "bullish"},
                "15m": {"bias": "bullish"},
                "1h": {"bias": "bullish"},
                "4h": {"bias": "neutral"},
            },
        },
        "scalping_result": {
            "strategy": "scalp",
            "signal": "long",
            "confidence": 0.71,
        },
        "intraday_result": {
            "strategy": "intraday",
            "signal": "long",
            "confidence": 0.83,
        },
        "swing_result": {
            "strategy": "swing",
            "signal": "hold",
            "confidence": 0.54,
        },
    }


def _build_risk_result() -> dict[str, object]:
    return {
        "execution_allowed": True,
        "entry_price": 62500.0,
        "stop_loss": 61850.0,
        "take_profit": 63800.0,
        "risk_per_unit": 650.0,
        "reward_per_unit": 1300.0,
        "risk_reward_ratio": 2.0,
        "atr_value": 240.5,
        "volatility_state": "stable",
        "reason": "Risk-reward is acceptable.",
    }


def _build_execution_result() -> dict[str, object]:
    return {
        "action": "buy",
        "symbol": "BTCUSDT",
        "execution_mode": "paper",
        "signal": "long",
        "bias": "bullish",
        "execution_allowed": True,
        "entry_price": 62500.0,
        "stop_loss": 61850.0,
        "take_profit": 63800.0,
        "reason": "Plan approved by risk engine.",
    }


def _build_ai_result() -> dict[str, object]:
    return {
        "source": "openai",
        "model": "gpt-test",
        "environment": "paper",
        "generated_at": "2026-03-26T00:00:00+00:00",
        "analysis": {
            "market_structure": "Bullish continuation",
            "rule_engine_assessment": "Aligned with breakout bias",
            "key_bottlenecks": ["Need sustained volume"],
            "long_scenario": "Continuation above intraday high",
            "short_scenario": "Invalidation below support",
            "final_stance": "long",
            "stance_reason": "Trend continuation remains intact.",
            "telegram_briefing": ["Momentum favors continuation."],
        },
    }


def _build_logger(tmp_path: Path) -> TradeAnalysisLogger:
    return TradeAnalysisLogger(
        config=TradeAnalysisLoggerConfig(
            log_dir=str(tmp_path),
            filename="trade_analysis.jsonl",
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_log_initializes_replay_ready_fields_to_null(tmp_path: Path) -> None:
    logger = _build_logger(tmp_path)

    record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    assert record["edge_selection_mapper_payload"] is None
    assert record["edge_selection_output"] is None
    assert record["edge_selection_metadata"] is None

    persisted_records = _read_jsonl(logger.log_path)
    assert persisted_records == [record]


def test_enrich_latest_record_writes_replay_ready_payloads_and_preserves_fields(
    tmp_path: Path,
) -> None:
    logger = _build_logger(tmp_path)
    base_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    mapper_payload = {
        "generated_at": datetime(2026, 3, 26, 0, 0, tzinfo=timezone.utc),
        "reports_dir": Path("/tmp/reports"),
        "ranked_candidates": [
            {"symbol": "BTCUSDT", "score": 0.91},
            {"symbol": "ETHUSDT", "score": 0.88},
        ],
        "tags": {"momentum", "trend"},
    }
    edge_output = {
        "selection_status": "selected",
        "selected_symbol": "BTCUSDT",
        "supporting_paths": (Path("/tmp/a.json"), Path("/tmp/b.json")),
    }
    metadata = {
        "mapper_version": "edge_selection_input_mapper_v1",
        "engine_version": "edge_selection_engine_v1",
        "replay_ready": True,
        "shadow_status": "success",
        "mapper_generated_at": datetime(2026, 3, 26, 0, 5, tzinfo=timezone.utc),
    }

    enriched = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload=mapper_payload,
        edge_selection_output=edge_output,
        edge_selection_metadata=metadata,
    )

    assert enriched["symbol"] == base_record["symbol"]
    assert enriched["selected_strategy"] == base_record["selected_strategy"]
    assert enriched["risk"] == base_record["risk"]
    assert enriched["execution"] == base_record["execution"]
    assert enriched["ai"] == base_record["ai"]
    assert enriched["alignment"] == base_record["alignment"]

    assert enriched["edge_selection_mapper_payload"]["generated_at"] == "2026-03-26T00:00:00+00:00"
    assert enriched["edge_selection_mapper_payload"]["reports_dir"] == "/tmp/reports"
    assert enriched["edge_selection_mapper_payload"]["ranked_candidates"] == [
        {"symbol": "BTCUSDT", "score": 0.91},
        {"symbol": "ETHUSDT", "score": 0.88},
    ]
    assert sorted(enriched["edge_selection_mapper_payload"]["tags"]) == ["momentum", "trend"]

    assert enriched["edge_selection_output"] == {
        "selection_status": "selected",
        "selected_symbol": "BTCUSDT",
        "supporting_paths": ["/tmp/a.json", "/tmp/b.json"],
    }

    assert enriched["edge_selection_metadata"] == {
        "mapper_version": "edge_selection_input_mapper_v1",
        "engine_version": "edge_selection_engine_v1",
        "replay_ready": True,
        "shadow_status": "success",
        "mapper_generated_at": "2026-03-26T00:05:00+00:00",
    }

    persisted_records = _read_jsonl(logger.log_path)
    assert persisted_records == [enriched]


def test_enrich_latest_record_keeps_mapper_and_output_null_on_failure(tmp_path: Path) -> None:
    logger = _build_logger(tmp_path)
    base_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    enriched = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload=None,
        edge_selection_output=None,
        edge_selection_metadata={
            "shadow_status": "failed",
            "replay_ready": False,
            "error_type": "RuntimeError",
            "error_message": "forced shadow failure",
        },
    )

    assert enriched["edge_selection_mapper_payload"] is None
    assert enriched["edge_selection_output"] is None
    assert enriched["edge_selection_metadata"]["shadow_status"] == "failed"
    assert enriched["edge_selection_metadata"]["replay_ready"] is False
    assert enriched["edge_selection_metadata"]["error_type"] == "RuntimeError"
    assert enriched["edge_selection_metadata"]["error_message"] == "forced shadow failure"


def test_enrich_latest_record_stores_ai_scaffold_shadow_annotation(tmp_path: Path) -> None:
    logger = _build_logger(tmp_path)
    base_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    ai_scaffold_shadow = {
        "enabled": True,
        "source": "ai_scaffold_static_mock",
        "generated_at": datetime(2026, 3, 26, 0, 10, tzinfo=timezone.utc),
        "annotation_mode": "read_only_shadow",
        "decision_impact": False,
        "request": {
            "symbol": "btcusdt",
            "strategy_context": {
                "strategy": "momentum_breakout",
                "bias": "long",
                "setup_state": "ready",
                "selection_state": "selected",
            },
        },
        "response": {
            "bias": "long",
            "confidence": "medium",
            "regime_label": "directional_trend",
            "reason": "Directional inputs are aligned to the long side in shadow analysis.",
            "timeframe_summary": {
                "1h": {
                    "bias": "long",
                    "momentum_state": "supportive",
                    "volatility_state": "contained",
                    "trigger_state": "armed",
                }
            },
        },
        "error": None,
    }

    enriched = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload=None,
        edge_selection_output=None,
        edge_selection_metadata=None,
        ai_scaffold_shadow=ai_scaffold_shadow,
    )

    assert "ai_scaffold_shadow" in enriched
    assert enriched["ai_scaffold_shadow"] == {
        "enabled": True,
        "source": "ai_scaffold_static_mock",
        "generated_at": "2026-03-26T00:10:00+00:00",
        "annotation_mode": "read_only_shadow",
        "decision_impact": False,
        "request": {
            "symbol": "btcusdt",
            "strategy_context": {
                "strategy": "momentum_breakout",
                "bias": "long",
                "setup_state": "ready",
                "selection_state": "selected",
            },
        },
        "response": {
            "bias": "long",
            "confidence": "medium",
            "regime_label": "directional_trend",
            "reason": "Directional inputs are aligned to the long side in shadow analysis.",
            "timeframe_summary": {
                "1h": {
                    "bias": "long",
                    "momentum_state": "supportive",
                    "volatility_state": "contained",
                    "trigger_state": "armed",
                }
            },
        },
        "error": None,
    }

    persisted_records = _read_jsonl(logger.log_path)
    assert persisted_records == [enriched]


def test_enrich_latest_record_removes_stale_ai_scaffold_shadow_when_none(tmp_path: Path) -> None:
    logger = _build_logger(tmp_path)
    base_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    first = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload=None,
        edge_selection_output=None,
        edge_selection_metadata=None,
        ai_scaffold_shadow={
            "enabled": True,
            "source": "ai_scaffold_static_mock",
            "generated_at": "2026-03-26T00:10:00+00:00",
            "annotation_mode": "read_only_shadow",
            "decision_impact": False,
            "request": {"symbol": "btcusdt"},
            "response": {
                "bias": "long",
                "reason": "Directional inputs are aligned to the long side in shadow analysis.",
                "confidence": "medium",
                "timeframe_summary": {},
            },
            "error": None,
        },
    )
    assert "ai_scaffold_shadow" in first

    second = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload=None,
        edge_selection_output=None,
        edge_selection_metadata=None,
        ai_scaffold_shadow=None,
    )

    assert "ai_scaffold_shadow" not in second
    persisted_records = _read_jsonl(logger.log_path)
    assert persisted_records == [second]


def test_enrich_latest_record_is_idempotent_for_same_payload(tmp_path: Path) -> None:
    logger = _build_logger(tmp_path)
    base_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    metadata = {
        "mapper_version": "edge_selection_input_mapper_v1",
        "engine_version": "edge_selection_engine_v1",
        "replay_ready": True,
        "shadow_status": "success",
    }

    first = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload={"generated_at": "2026-03-26T00:00:00+00:00"},
        edge_selection_output={"selection_status": "selected"},
        edge_selection_metadata=metadata,
    )

    second = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload={"generated_at": "2026-03-26T00:00:00+00:00"},
        edge_selection_output={"selection_status": "selected"},
        edge_selection_metadata=metadata,
    )

    assert first == second
    persisted_records = _read_jsonl(logger.log_path)
    assert persisted_records == [second]


def test_enrich_latest_record_targets_latest_matching_row_only(tmp_path: Path) -> None:
    logger = _build_logger(tmp_path)

    first_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )
    second_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    enriched = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=second_record["logged_at"],
        edge_selection_mapper_payload={"generated_at": "2026-03-26T00:00:00+00:00"},
        edge_selection_output={"selection_status": "selected"},
        edge_selection_metadata={"shadow_status": "success", "replay_ready": True},
    )

    records = _read_jsonl(logger.log_path)

    assert records[0]["logged_at"] == first_record["logged_at"]
    assert records[0]["edge_selection_mapper_payload"] is None
    assert records[0]["edge_selection_output"] is None
    assert records[0]["edge_selection_metadata"] is None

    assert records[1] == enriched
    assert records[1]["logged_at"] == second_record["logged_at"]
    assert records[1]["edge_selection_metadata"]["shadow_status"] == "success"


def test_ai_scaffold_shadow_is_the_only_difference_between_disabled_and_enabled_records(
    tmp_path: Path,
) -> None:
    logger = _build_logger(tmp_path)

    base_record = logger.log(
        symbol="BTCUSDT",
        strategy_result=_build_strategy_result(),
        risk_result=_build_risk_result(),
        execution_result=_build_execution_result(),
        ai_result=_build_ai_result(),
    )

    disabled_enriched = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload={"generated_at": "2026-03-26T00:00:00+00:00"},
        edge_selection_output={"selection_status": "selected"},
        edge_selection_metadata={"shadow_status": "success", "replay_ready": True},
        ai_scaffold_shadow=None,
    )

    enabled_shadow = {
        "enabled": True,
        "source": "ai_scaffold_static_mock",
        "generated_at": "2026-03-26T00:10:00+00:00",
        "annotation_mode": "read_only_shadow",
        "decision_impact": False,
        "request": {
            "symbol": "btcusdt",
            "strategy_context": {
                "strategy": "momentum_breakout",
                "bias": "long",
                "setup_state": "ready",
                "selection_state": "selected",
            },
        },
        "response": {
            "bias": "long",
            "confidence": "medium",
            "regime_label": "directional_trend",
            "reason": "Directional inputs are aligned to the long side in shadow analysis.",
            "timeframe_summary": {
                "1h": {
                    "bias": "long",
                    "momentum_state": "supportive",
                    "volatility_state": "contained",
                    "trigger_state": "armed",
                }
            },
        },
        "error": None,
    }

    enabled_enriched = logger.enrich_latest_record(
        symbol="BTCUSDT",
        logged_at=base_record["logged_at"],
        edge_selection_mapper_payload={"generated_at": "2026-03-26T00:00:00+00:00"},
        edge_selection_output={"selection_status": "selected"},
        edge_selection_metadata={"shadow_status": "success", "replay_ready": True},
        ai_scaffold_shadow=enabled_shadow,
    )

    enabled_without_shadow = dict(enabled_enriched)
    enabled_without_shadow.pop("ai_scaffold_shadow", None)

    assert enabled_without_shadow == disabled_enriched
    assert enabled_enriched["ai_scaffold_shadow"]["decision_impact"] is False
