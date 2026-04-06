from __future__ import annotations

from src.ai_scaffold.payload_builder import (
    build_ai_interpretation_request,
    build_payload,
)


# =========================================================
# 기존 핵심 테스트 (절대 유지)
# =========================================================

def test_payload_builder_generates_deterministic_compact_payload() -> None:
    timeframe_summaries = {
        "4h": {
            "bias": "LONG",
            "momentum": "Supportive",
            "volatility": "Contained",
            "trigger": "Watching",
            "ignored_numeric_indicator": 73.42,
        },
        "1h": {
            "bias": "long",
            "momentum": "supportive",
            "volatility": "contained",
            "trigger": "armed",
        },
    }

    first_payload = build_payload(
        symbol="BTCUSDT",
        timeframe_summaries=timeframe_summaries,
        strategy_context={
            "strategy": "Trend Follow",
            "bias": "long",
            "setup_state": "confirmed",
            "alignment_state": "aligned",
        },
        risk_context={
            "execution_allowed": True,
            "risk_reward_state": "acceptable",
            "exposure_state": "light",
            "volatility_risk_state": "normal",
        },
        as_of="2026-04-06T00:00:00Z",
    )
    second_payload = build_payload(
        symbol="BTCUSDT",
        timeframe_summaries=timeframe_summaries,
        strategy_context={
            "strategy": "Trend Follow",
            "bias": "long",
            "setup_state": "confirmed",
            "alignment_state": "aligned",
        },
        risk_context={
            "execution_allowed": True,
            "risk_reward_state": "acceptable",
            "exposure_state": "light",
            "volatility_risk_state": "normal",
        },
        as_of="2026-04-06T00:00:00Z",
    )

    assert first_payload == second_payload
    assert [item["timeframe"] for item in first_payload["timeframes"]] == ["1h", "4h"]

    assert first_payload["timeframes"][0] == {
        "timeframe": "1h",
        "market_bias": "long",
        "momentum_state": "supportive",
        "volatility_state": "contained",
        "trigger_state": "armed",
    }

    # 핵심: 허용되지 않은 필드 제거 검증
    assert "ignored_numeric_indicator" not in first_payload["timeframes"][1]


def test_payload_builder_handles_missing_fields_safely() -> None:
    request = build_ai_interpretation_request(
        symbol="",
        timeframe_summaries=[{"timeframe": "15m"}],
        strategy_context={},
        risk_context=None,
        as_of="",
    )

    assert request.symbol == "UNKNOWN"
    assert request.as_of is None
    assert request.timeframes[0].market_bias == "neutral"
    assert request.timeframes[0].momentum_state == "unknown"
    assert request.strategy_context.strategy_name == "unknown"
    assert request.risk_context.execution_allowed is False


# =========================================================
# 추가 테스트 (2차 개선)
# =========================================================

def test_payload_builder_handles_timeframe_ordering_correctly() -> None:
    payload = build_payload(
        symbol="BTCUSDT",
        timeframe_summaries={
            "15m": {"bias": "long"},
            "1h": {"bias": "long"},
            "4h": {"bias": "long"},
        },
    )

    assert [tf["timeframe"] for tf in payload["timeframes"]] == ["15m", "1h", "4h"]


def test_payload_builder_handles_invalid_bias_gracefully() -> None:
    payload = build_payload(
        symbol="BTCUSDT",
        timeframe_summaries={
            "1h": {"bias": "LONG_BIAS"},
        },
    )

    assert payload["timeframes"][0]["market_bias"] == "long"


def test_payload_builder_handles_none_inputs_safely() -> None:
    payload = build_payload(
        symbol=None,
        timeframe_summaries=None,
        strategy_context=None,
        risk_context=None,
        as_of=None,
    )

    assert payload["symbol"] == "UNKNOWN"
    assert payload["timeframes"] == []