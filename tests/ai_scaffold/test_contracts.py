from __future__ import annotations

from dataclasses import asdict

from src.ai_scaffold.contracts import (
    AIInterpretationRequest,
    AIInterpretationResponse,
    RiskContextSummary,
    StrategyContextSummary,
    TimeframeSummary,
)


def test_contracts_expose_expected_request_and_response_fields() -> None:
    request = AIInterpretationRequest(
        symbol="BTCUSDT",
        as_of="2026-04-06T00:00:00Z",
        timeframes=[
            TimeframeSummary(
                timeframe="1h",
                market_bias="long",
                momentum_state="supportive",
                volatility_state="contained",
                trigger_state="armed",
            )
        ],
        strategy_context=StrategyContextSummary(
            strategy_name="trend_follow",
            directional_bias="long",
            setup_state="confirmed",
            alignment_state="aligned",
        ),
        risk_context=RiskContextSummary(
            execution_allowed=True,
            risk_reward_state="acceptable",
            exposure_state="light",
            volatility_risk_state="normal",
        ),
        prompt_version="ai_scaffold_prompt_v1",
    )
    response = AIInterpretationResponse(
        bias="long",
        confidence="high",
        regime_label="directional_trend",
        reasoning=["strategy_bias=long"],
        caution_flags=["elevated_volatility"],
        recommended_action="observe_long_setup",
        model_version="static_mock_v1",
        prompt_version="ai_scaffold_prompt_v1",
    )

    request_payload = asdict(request)
    response_payload = asdict(response)

    assert request_payload["symbol"] == "BTCUSDT"
    assert request_payload["timeframes"][0]["timeframe"] == "1h"
    assert request_payload["strategy_context"]["directional_bias"] == "long"
    assert request_payload["risk_context"]["execution_allowed"] is True

    assert response_payload["bias"] == "long"
    assert response_payload["confidence"] == "high"
    assert response_payload["reasoning"] == ["strategy_bias=long"]
    assert response_payload["caution_flags"] == ["elevated_volatility"]
    assert response_payload["model_version"] == "static_mock_v1"
