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
        return {
            "logged_at": kwargs["logged_at"],
            "symbol": kwargs["symbol"],
            "edge_selection_mapper_payload": kwargs["edge_selection_mapper_payload"],
            "edge_selection_output": kwargs["edge_selection_output"],
            "edge_selection_metadata": kwargs["edge_selection_metadata"],
        }


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
            },
            "telegram_send_result": {"sent": False},
        }


def test_run_shadow_observation_returns_success_context(monkeypatch) -> None:
    service = TradingPipelineService()
    mapper_payload = {
        "generated_at": "2026-03-26T00:00:00+00:00",
        "candidates": [{"symbol": "BTCUSDT", "score": 0.91}],
    }
    shadow_output = {
        "selection_status": "selected",
        "selected_symbol": "BTCUSDT",
        "candidates_considered": 2,
    }
    output_path = Path("logs/research_reports/edge_selection_shadow.json")

    monkeypatch.setattr(
        "src.services.trading_pipeline_service.map_edge_selection_input",
        lambda _: mapper_payload,
    )
    monkeypatch.setattr(
        "src.services.trading_pipeline_service.run_edge_selection_engine",
        lambda payload: shadow_output if payload is mapper_payload else {},
    )
    monkeypatch.setattr(
        "src.services.trading_pipeline_service.write_edge_selection_shadow_output",
        lambda _: output_path,
    )
    monkeypatch.setattr(service, "_notify_shadow_events", lambda **_: None)

    result = service._run_shadow_observation(trigger_symbol="BTCUSDT")
    metadata = result["edge_selection_metadata"]

    assert result["edge_selection_mapper_payload"] == mapper_payload
    assert result["edge_selection_output"] == shadow_output
    assert metadata["replay_ready"] is True
    assert metadata["shadow_status"] == "success"
    assert metadata["shadow_output_path"] == str(output_path)
    assert metadata["mapper_version"] == "edge_selection_input_mapper_v1"
    assert metadata["engine_version"] == "edge_selection_engine_v1"
    assert metadata["trigger_symbol"] == "BTCUSDT"
    assert metadata["selection_status"] == "selected"


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