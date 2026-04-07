from __future__ import annotations

from src.ai_scaffold.contracts import (
    AIInterpretationRequest,
    RiskContextSummary,
    StrategyContextSummary,
    TimeframeSummary,
)
from src.ai_scaffold.interpreter import StaticMockInterpreter


def test_static_mock_interpreter_returns_expected_structure() -> None:
    interpreter = StaticMockInterpreter(model_version="static_mock_v1")

    request = AIInterpretationRequest(
        symbol="BTCUSDT",
        timeframes=[
            TimeframeSummary(
                timeframe="1h",
                market_bias="long",
                momentum_state="supportive",
                volatility_state="contained",
                trigger_state="armed",
            ),
            TimeframeSummary(
                timeframe="4h",
                market_bias="long",
                momentum_state="supportive",
                volatility_state="contained",
                trigger_state="watching",
            ),
        ],
        strategy_context=StrategyContextSummary(
            strategy_name="trend_follow",
            directional_bias="long",
            setup_state="confirmed",
            selection_state="selected",
        ),
        risk_context=RiskContextSummary(
            execution_allowed=True,
            risk_reward_state="acceptable",
            exposure_state="light",
            volatility_risk_state="normal",
        ),
        prompt_version="ai_scaffold_prompt_v1",
    )

    response = interpreter.interpret(request)

    assert response.bias == "long"
    assert response.confidence == "high"
    assert response.regime_label == "directional_trend"
    assert response.reason == "Directional inputs are aligned to the long side in shadow analysis."
    assert response.timeframe_summary == {
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

    assert response.reasoning == [
        "strategy_bias=long",
        "timeframe_biases=['long', 'long']",
        "timeframe_conflict=false",
        "execution_allowed=true",
        "regime=directional_trend",
    ]

    assert response.caution_flags == []
    assert response.recommended_action == "observe_long_setup"
    assert response.model_version == "static_mock_v1"
    assert response.prompt_version == "ai_scaffold_prompt_v1"


def test_static_mock_interpreter_is_deterministic_for_blocked_or_conflicted_inputs() -> None:
    interpreter = StaticMockInterpreter()

    request = AIInterpretationRequest(
        symbol="BTCUSDT",
        timeframes=[
            TimeframeSummary(
                timeframe="1h",
                market_bias="long",
                volatility_state="expanding",
            ),
            TimeframeSummary(
                timeframe="4h",
                market_bias="short",
                volatility_state="contained",
            ),
        ],
        strategy_context=StrategyContextSummary(
            strategy_name="mean_revert",
            directional_bias="short",
            setup_state="forming",
            selection_state="abstain",
        ),
        risk_context=RiskContextSummary(
            execution_allowed=False,
            risk_reward_state="weak",
            exposure_state="full",
            volatility_risk_state="high",
        ),
    )

    first_response = interpreter.interpret(request)
    second_response = interpreter.interpret(request)

    assert first_response == second_response

    assert first_response.bias == "neutral"
    assert first_response.confidence == "low"
    assert first_response.regime_label == "volatile_expansion"
    assert (
        first_response.reason
        == "Timeframe signals conflict, so the shadow view remains neutral."
    )
    assert first_response.timeframe_summary == {
        "1h": {
            "bias": "long",
            "momentum_state": "unknown",
            "volatility_state": "expanding",
            "trigger_state": "unknown",
        },
        "4h": {
            "bias": "short",
            "momentum_state": "unknown",
            "volatility_state": "contained",
            "trigger_state": "unknown",
        },
    }

    assert first_response.caution_flags == [
        "execution_blocked",
        "timeframe_conflict",
        "setup_not_confirmed",
        "risk_reward_unclear",
        "elevated_volatility",
    ]

    assert first_response.recommended_action == "hold"
