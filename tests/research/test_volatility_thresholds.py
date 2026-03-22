from __future__ import annotations

import pytest

from src.research.experimental_labeling.volatility_adjusted_config import (
    HorizonVolatilityThreshold,
    VolatilityAdjustedThresholdConfig,
    get_candidate_b_v1_config,
)
from src.research.experimental_labeling.volatility_thresholds import (
    compute_candidate_b_v1_threshold_metadata,
    compute_candidate_b_v1_thresholds,
    compute_horizon_threshold_pct,
    compute_normalized_atr_pct,
    compute_volatility_adjusted_thresholds,
    resolve_atr_pct_with_fallback,
)


def test_candidate_b_v1_config_is_valid() -> None:
    config = get_candidate_b_v1_config()
    config.validate()


def test_compute_normalized_atr_pct_from_risk_block() -> None:
    row = {
        "risk": {
            "atr_value": 2.5,
            "entry_price": 100.0,
        }
    }

    assert compute_normalized_atr_pct(row) == 2.5


def test_compute_normalized_atr_pct_from_string_inputs() -> None:
    row = {
        "risk": {
            "atr_value": "1.25",
            "entry_price": "100",
        }
    }

    assert compute_normalized_atr_pct(row) == 1.25


def test_compute_thresholds_for_each_horizon() -> None:
    row = {
        "risk": {
            "atr_value": 1.0,
            "entry_price": 100.0,
        }
    }

    assert compute_candidate_b_v1_thresholds(row) == {
        "15m": 0.35,
        "1h": 0.5,
        "4h": 0.65,
    }


def test_min_threshold_floor_behavior() -> None:
    row = {
        "risk": {
            "atr_value": 0.01,
            "entry_price": 100.0,
        }
    }

    assert compute_candidate_b_v1_thresholds(row) == {
        "15m": 0.05,
        "1h": 0.1,
        "4h": 0.15,
    }


def test_fallback_behavior_for_missing_or_invalid_inputs() -> None:
    missing_row = {"risk": {"entry_price": 100.0}}
    zero_entry_row = {"risk": {"atr_value": 1.5, "entry_price": 0.0}}
    negative_atr_row = {"risk": {"atr_value": -1.5, "entry_price": 100.0}}
    bool_row = {"risk": {"atr_value": True, "entry_price": 100.0}}

    assert compute_normalized_atr_pct(missing_row) is None
    assert compute_normalized_atr_pct(zero_entry_row) is None
    assert compute_normalized_atr_pct(negative_atr_row) is None
    assert compute_normalized_atr_pct(bool_row) is None

    assert compute_candidate_b_v1_thresholds(missing_row) == {
        "15m": 0.05,
        "1h": 0.1,
        "4h": 0.15,
    }
    assert compute_candidate_b_v1_thresholds(zero_entry_row) == {
        "15m": 0.05,
        "1h": 0.1,
        "4h": 0.15,
    }
    assert compute_candidate_b_v1_thresholds(negative_atr_row) == {
        "15m": 0.05,
        "1h": 0.1,
        "4h": 0.15,
    }


def test_custom_fallback_atr_pct_is_used_when_inputs_are_invalid() -> None:
    row = {
        "risk": {
            "atr_value": "n/a",
            "entry_price": 100.0,
        }
    }
    config = VolatilityAdjustedThresholdConfig(
        horizon_settings={
            "15m": HorizonVolatilityThreshold(min_threshold_pct=0.05, atr_multiplier=0.35),
            "1h": HorizonVolatilityThreshold(min_threshold_pct=0.10, atr_multiplier=0.50),
            "4h": HorizonVolatilityThreshold(min_threshold_pct=0.15, atr_multiplier=0.65),
        },
        fallback_atr_pct=2.0,
    )

    atr_pct, used_fallback = resolve_atr_pct_with_fallback(
        row,
        fallback_atr_pct=config.fallback_atr_pct,
    )
    thresholds = compute_volatility_adjusted_thresholds(row, config)

    assert atr_pct == 2.0
    assert used_fallback is True
    assert thresholds == {
        "15m": 0.7,
        "1h": 1.0,
        "4h": 1.3,
    }


def test_compute_horizon_threshold_pct_is_deterministic() -> None:
    custom_horizon = HorizonVolatilityThreshold(
        min_threshold_pct=0.05,
        atr_multiplier=0.35,
    )

    first = compute_horizon_threshold_pct(1.25, custom_horizon)
    second = compute_horizon_threshold_pct(1.25, custom_horizon)

    assert first == second
    assert first == 0.4375


def test_threshold_metadata_contains_expected_fields() -> None:
    row = {
        "risk": {
            "atr_value": 1.0,
            "entry_price": 100.0,
        }
    }

    metadata = compute_candidate_b_v1_threshold_metadata(row)

    assert metadata["atr_pct"] == 1.0
    assert metadata["used_fallback_atr_pct"] is False
    assert metadata["thresholds"] == {
        "15m": 0.35,
        "1h": 0.5,
        "4h": 0.65,
    }


def test_invalid_config_raises_for_missing_horizon() -> None:
    config = VolatilityAdjustedThresholdConfig(
        horizon_settings={
            "15m": HorizonVolatilityThreshold(min_threshold_pct=0.05, atr_multiplier=0.35),
            "1h": HorizonVolatilityThreshold(min_threshold_pct=0.10, atr_multiplier=0.50),
        },
        fallback_atr_pct=0.0,
    )

    with pytest.raises(ValueError, match="Missing horizon settings"):
        compute_volatility_adjusted_thresholds({}, config)
