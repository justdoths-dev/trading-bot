from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.trading_pipeline_service import (
    FORCE_SHADOW_FAILURE_ENV_VAR,
    TradingPipelineService,
)


class _FakeLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def enrich_latest_record(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        record = {
            "logged_at": kwargs["logged_at"],
            "symbol": kwargs["symbol"],
            "edge_selection_mapper_payload": kwargs["edge_selection_mapper_payload"],
            "edge_selection_output": kwargs["edge_selection_output"],
            "edge_selection_metadata": kwargs["edge_selection_metadata"],
        }
        if kwargs.get("ai_scaffold_shadow") is not None:
            record["ai_scaffold_shadow"] = kwargs["ai_scaffold_shadow"]
        return record


class _FakePipeline:
    def __init__(self) -> None:
        self.logger = _FakeLogger()
        self.run_calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        run_ai: bool,
        ai_result_override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.run_calls.append(
            {
                "run_ai": run_ai,
                "ai_result_override": ai_result_override,
            }
        )
        return {
            "symbol": "BTCUSDT",
            "strategy_result": {"selected_strategy": "momentum_breakout"},
            "risk_result": {"execution_allowed": True},
            "execution_result": {"action": "buy"},
            "ai_output": {
                "result": {
                    "source": "openai",
                    "analysis": {"final_stance": "long"},
                }
            },
            "log_record": {
                "logged_at": "2026-03-26T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "timeframe_summary": {
                    "1h": {
                        "bias": "long",
                        "momentum_state": "supportive",
                        "volatility_state": "contained",
                        "trigger_state": "armed",
                    },
                    "4h": {
                        "bias": "long",
                        "momentum_state": "supportive",
                        "volatility_state": "contained",
                        "trigger_state": "watching",
                    },
                },
                "rule_engine": {
                    "strategy": "momentum_breakout",
                    "signal": "long",
                    "bias": "bullish",
                },
                "risk": {
                    "execution_allowed": True,
                    "risk_reward_ratio": 2.0,
                    "volatility_state": "normal",
                },
            },
            "telegram_send_result": {"sent": False},
        }


class _ExplodingAnnotator:
    enabled = True

    def annotate(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError("forced ai scaffold failure")


def test_run_shadow_observation_passes_quality_gated_candidates_to_engine(monkeypatch) -> None:
    service = TradingPipelineService()
    mapper_payload = {
        "generated_at": "2026-03-26T00:00:00+00:00",
        "candidates": [
            {"symbol": "BTCUSDT", "strategy": "swing", "horizon": "4h"},
            {"symbol": "ETHUSDT", "strategy": "swing", "horizon": "4h"},
        ],
    }
    gate_result = {
        "input_path_used": "logs/trade_analysis_cumulative.jsonl",
        "total_candidates": 2,
        "strict_kept_candidates": [
            {"symbol": "BTCUSDT", "strategy": "swing", "horizon": "4h"}
        ],
        "strict_kept_count": 1,
        "strict_dropped_candidates": [
            {
                "candidate": {"symbol": "ETHUSDT", "strategy": "swing", "horizon": "4h"},
                "reason": "median_return_pct_negative",
                "metrics": {
                    "positive_rate_pct": 40.0,
                    "median_return_pct": -0.2,
                    "sample_count": 30,
                },
            }
        ],
        "strict_dropped_count": 1,
        "fallback_applied": False,
        "fallback_restored_candidates": [],
        "fallback_restored_count": 0,
        "final_kept_candidates": [
            {"symbol": "BTCUSDT", "strategy": "swing", "horizon": "4h"}
        ],
        "final_kept_count": 1,
        # compatibility
        "kept_candidates": [{"symbol": "BTCUSDT", "strategy": "swing", "horizon": "4h"}],
        "dropped_candidates": [
            {
                "candidate": {"symbol": "ETHUSDT", "strategy": "swing", "horizon": "4h"},
                "reason": "median_return_pct_negative",
                "metrics": {
                    "positive_rate_pct": 40.0,
                    "median_return_pct": -0.2,
                    "sample_count": 30,
                },
            }
        ],
        "kept_count": 1,
        "dropped_count": 1,
    }
    shadow_output = {
        "selection_status": "selected",
        "selected_symbol": "BTCUSDT",
        "candidates_considered": 1,
    }
    output_path = Path("logs/research_reports/edge_selection_shadow.json")
    captured_payloads: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "src.services.trading_pipeline_service.map_edge_selection_input",
        lambda _: mapper_payload,
    )
    monkeypatch.setattr(
        "src.services.trading_pipeline_service.apply_candidate_quality_gate",
        lambda candidates: gate_result,
    )

    def _run_engine(payload: dict[str, Any]) -> dict[str, Any]:
        captured_payloads.append(payload)
        return shadow_output

    monkeypatch.setattr(
        "src.services.trading_pipeline_service.run_edge_selection_engine",
        _run_engine,
    )
    monkeypatch.setattr(
        "src.services.trading_pipeline_service.write_edge_selection_shadow_output",
        lambda _: output_path,
    )
    monkeypatch.setattr(service, "_notify_shadow_events", lambda **_: None)

    result = service._run_shadow_observation(trigger_symbol="BTCUSDT")
    metadata = result["edge_selection_metadata"]
    gated_payload = result["edge_selection_mapper_payload"]

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["candidates"] == gate_result["final_kept_candidates"]

    gate_snapshot = captured_payloads[0]["candidate_quality_gate"]
    assert gate_snapshot["strict_kept_count"] == 1
    assert gate_snapshot["strict_dropped_count"] == 1
    assert gate_snapshot["fallback_restored_count"] == 0
    assert gate_snapshot["final_kept_count"] == 1
    assert gate_snapshot["fallback_applied"] is False

    assert gated_payload["candidates"] == gate_result["final_kept_candidates"]
    assert gated_payload["candidate_quality_gate"]["input_path_used"] == gate_result["input_path_used"]
    assert gated_payload["candidate_quality_gate"]["strict_dropped_candidates"] == gate_result[
        "strict_dropped_candidates"
    ]

    assert result["edge_selection_output"] == shadow_output
    assert metadata["replay_ready"] is True
    assert metadata["shadow_status"] == "success"
    assert metadata["shadow_output_path"] == str(output_path)
    assert metadata["mapper_version"] == "edge_selection_input_mapper_v1"
    assert metadata["quality_gate_version"] == "candidate_quality_gate_v1"
    assert metadata["engine_version"] == "edge_selection_engine_v1"
    assert metadata["trigger_symbol"] == "BTCUSDT"
    assert metadata["selection_status"] == "selected"

    assert metadata["quality_gate_total_candidates"] == 2
    assert metadata["quality_gate_strict_kept_count"] == 1
    assert metadata["quality_gate_strict_dropped_count"] == 1
    assert metadata["quality_gate_fallback_applied"] is False
    assert metadata["quality_gate_fallback_restored_count"] == 0
    assert metadata["quality_gate_final_kept_count"] == 1

    # compatibility
    assert metadata["quality_gate_kept_count"] == 1
    assert metadata["quality_gate_dropped_count"] == 1


def test_run_shadow_observation_forced_failure_returns_failed_metadata(monkeypatch) -> None:
    service = TradingPipelineService()
    monkeypatch.setenv(FORCE_SHADOW_FAILURE_ENV_VAR, "true")

    result = service._run_shadow_observation(trigger_symbol="BTCUSDT")
    metadata = result["edge_selection_metadata"]

    assert result["edge_selection_mapper_payload"] is None
    assert result["edge_selection_output"] is None
    assert metadata["replay_ready"] is False
    assert metadata["shadow_status"] == "failed"
    assert metadata["error_type"] == "RuntimeError"
    assert FORCE_SHADOW_FAILURE_ENV_VAR in metadata["error_message"]
    assert metadata["trigger_symbol"] == "BTCUSDT"


def test_run_returns_main_pipeline_result_even_when_shadow_fails(monkeypatch) -> None:
    service = TradingPipelineService()
    fake_pipeline = _FakePipeline()

    monkeypatch.setattr(
        service,
        "_get_pipeline",
        lambda *, symbol, send_telegram: fake_pipeline,
    )
    monkeypatch.setenv(FORCE_SHADOW_FAILURE_ENV_VAR, "1")

    result = service.run(symbol="BTCUSDT", run_ai=True, send_telegram=False)
    call = fake_pipeline.logger.calls[0]
    metadata = call["edge_selection_metadata"]

    assert result["symbol"] == "BTCUSDT"
    assert result["strategy_result"]["selected_strategy"] == "momentum_breakout"
    assert result["risk_result"]["execution_allowed"] is True
    assert result["execution_result"]["action"] == "buy"
    assert result["ai_output"]["result"]["source"] == "openai"

    assert result["log_record"]["edge_selection_mapper_payload"] is None
    assert result["log_record"]["edge_selection_output"] is None
    assert result["log_record"]["edge_selection_metadata"]["replay_ready"] is False
    assert result["log_record"]["edge_selection_metadata"]["shadow_status"] == "failed"

    assert fake_pipeline.run_calls == [{"run_ai": True, "ai_result_override": None}]
    assert call["symbol"] == "BTCUSDT"
    assert call["edge_selection_mapper_payload"] is None
    assert call["edge_selection_output"] is None
    assert metadata["replay_ready"] is False
    assert metadata["shadow_status"] == "failed"
    assert metadata["error_type"] == "RuntimeError"


def test_run_with_ai_scaffold_disabled_preserves_existing_behavior(monkeypatch) -> None:
    service = TradingPipelineService(ai_scaffold_shadow_enabled=False)
    fake_pipeline = _FakePipeline()
    shadow_context = {
        "edge_selection_mapper_payload": {"generated_at": "2026-03-26T00:00:00+00:00"},
        "edge_selection_output": {"selection_status": "selected"},
        "edge_selection_metadata": {"shadow_status": "success", "replay_ready": True},
    }

    monkeypatch.setattr(
        service,
        "_get_pipeline",
        lambda *, symbol, send_telegram: fake_pipeline,
    )
    monkeypatch.setattr(
        service,
        "_run_shadow_observation",
        lambda trigger_symbol: shadow_context,
    )

    result = service.run(symbol="BTCUSDT", run_ai=True, send_telegram=False)
    call = fake_pipeline.logger.calls[0]

    assert "ai_scaffold_shadow" not in result["log_record"]
    assert call.get("ai_scaffold_shadow") is None
    assert result["strategy_result"] == {"selected_strategy": "momentum_breakout"}
    assert result["risk_result"] == {"execution_allowed": True}
    assert result["execution_result"] == {"action": "buy"}
    assert result["ai_output"]["result"]["source"] == "openai"


def test_run_with_ai_scaffold_enabled_adds_read_only_annotation(monkeypatch) -> None:
    service = TradingPipelineService(ai_scaffold_shadow_enabled=True)
    fake_pipeline = _FakePipeline()
    shadow_context = {
        "edge_selection_mapper_payload": {"generated_at": "2026-03-26T00:00:00+00:00"},
        "edge_selection_output": {"selection_status": "selected"},
        "edge_selection_metadata": {"shadow_status": "success", "replay_ready": True},
    }

    monkeypatch.setattr(
        service,
        "_get_pipeline",
        lambda *, symbol, send_telegram: fake_pipeline,
    )
    monkeypatch.setattr(
        service,
        "_run_shadow_observation",
        lambda trigger_symbol: shadow_context,
    )

    result = service.run(symbol="BTCUSDT", run_ai=True, send_telegram=False)
    annotation = result["log_record"]["ai_scaffold_shadow"]

    assert annotation["enabled"] is True
    assert annotation["source"] == "ai_scaffold_static_mock"
    assert annotation["annotation_mode"] == "read_only_shadow"
    assert annotation["decision_impact"] is False
    assert annotation["request"]["symbol"] == "btcusdt"
    assert annotation["request"]["strategy_context"]["selection_state"] == "selected"
    assert "alignment_state" not in annotation["request"]["strategy_context"]
    assert annotation["response"]["bias"] == "long"
    assert annotation["response"]["reason"] == (
        "Directional inputs are aligned to the long side in shadow analysis."
    )
    assert annotation["response"]["confidence"] == "high"
    assert annotation["response"]["timeframe_summary"] == {
        "1h": {
            "bias": "long",
            "momentum_state": "supportive",
            "volatility_state": "contained",
            "trigger_state": "armed",
        },
        "4h": {
            "bias": "long",
            "momentum_state": "supportive",
            "volatility_state": "contained",
            "trigger_state": "watching",
        },
    }
    assert annotation["response"]["model_version"] == "static_mock_v1"
    assert annotation["error"] is None


def test_run_with_ai_scaffold_failure_is_captured_without_breaking_pipeline(monkeypatch) -> None:
    service = TradingPipelineService(
        ai_scaffold_shadow_enabled=True,
        ai_scaffold_shadow_annotator=_ExplodingAnnotator(),
    )
    fake_pipeline = _FakePipeline()
    shadow_context = {
        "edge_selection_mapper_payload": {"generated_at": "2026-03-26T00:00:00+00:00"},
        "edge_selection_output": {"selection_status": "selected"},
        "edge_selection_metadata": {"shadow_status": "success", "replay_ready": True},
    }

    monkeypatch.setattr(
        service,
        "_get_pipeline",
        lambda *, symbol, send_telegram: fake_pipeline,
    )
    monkeypatch.setattr(
        service,
        "_run_shadow_observation",
        lambda trigger_symbol: shadow_context,
    )

    result = service.run(symbol="BTCUSDT", run_ai=True, send_telegram=False)
    annotation = result["log_record"]["ai_scaffold_shadow"]

    assert result["strategy_result"] == {"selected_strategy": "momentum_breakout"}
    assert result["risk_result"] == {"execution_allowed": True}
    assert result["execution_result"] == {"action": "buy"}
    assert annotation["enabled"] is True
    assert annotation["decision_impact"] is False
    assert annotation["response"] == {}
    assert annotation["error"] == {
        "type": "RuntimeError",
        "message": "forced ai scaffold failure",
    }


def test_rule_engine_outputs_remain_identical_with_ai_scaffold_enabled(monkeypatch) -> None:
    fake_pipeline_disabled = _FakePipeline()
    fake_pipeline_enabled = _FakePipeline()

    disabled_service = TradingPipelineService(ai_scaffold_shadow_enabled=False)
    enabled_service = TradingPipelineService(ai_scaffold_shadow_enabled=True)

    shadow_context = {
        "edge_selection_mapper_payload": {"generated_at": "2026-03-26T00:00:00+00:00"},
        "edge_selection_output": {"selection_status": "selected"},
        "edge_selection_metadata": {"shadow_status": "success", "replay_ready": True},
    }

    monkeypatch.setattr(
        disabled_service,
        "_get_pipeline",
        lambda *, symbol, send_telegram: fake_pipeline_disabled,
    )
    monkeypatch.setattr(
        enabled_service,
        "_get_pipeline",
        lambda *, symbol, send_telegram: fake_pipeline_enabled,
    )
    monkeypatch.setattr(
        disabled_service,
        "_run_shadow_observation",
        lambda trigger_symbol: shadow_context,
    )
    monkeypatch.setattr(
        enabled_service,
        "_run_shadow_observation",
        lambda trigger_symbol: shadow_context,
    )

    disabled_result = disabled_service.run(
        symbol="BTCUSDT",
        run_ai=True,
        send_telegram=False,
    )
    enabled_result = enabled_service.run(
        symbol="BTCUSDT",
        run_ai=True,
        send_telegram=False,
    )

    assert enabled_result["strategy_result"] == disabled_result["strategy_result"]
    assert enabled_result["risk_result"] == disabled_result["risk_result"]
    assert enabled_result["execution_result"] == disabled_result["execution_result"]
    assert enabled_result["ai_output"] == disabled_result["ai_output"]
    assert "ai_scaffold_shadow" not in disabled_result["log_record"]
    assert enabled_result["log_record"]["ai_scaffold_shadow"]["decision_impact"] is False
