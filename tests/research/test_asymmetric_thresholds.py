from __future__ import annotations

import pytest

from src.research.experimental_labeling.asymmetric_threshold_config import (
    AsymmetricThresholdVariantConfig,
    DEFAULT_VARIANT_NAME,
    HorizonAsymmetricThreshold,
    get_asymmetric_threshold_variant,
)
from src.research.experimental_labeling.asymmetric_thresholds import (
    build_threshold_map,
    compute_asymmetric_threshold_labels,
    compute_variant_threshold_map,
    label_future_return_asymmetric,
)


def test_default_variant_is_valid() -> None:
    config = get_asymmetric_threshold_variant(DEFAULT_VARIANT_NAME)
    config.validate()


def test_unknown_variant_raises_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown asymmetric threshold variant"):
        get_asymmetric_threshold_variant("does_not_exist")


def test_inclusive_boundary_behavior_is_explicit() -> None:
    assert label_future_return_asymmetric(0.05, up_threshold_pct=0.05, down_threshold_pct=0.065) == "up"
    assert label_future_return_asymmetric(-0.065, up_threshold_pct=0.05, down_threshold_pct=0.065) == "down"
    assert label_future_return_asymmetric(0.049999, up_threshold_pct=0.05, down_threshold_pct=0.065) == "flat"
    assert label_future_return_asymmetric(-0.064999, up_threshold_pct=0.05, down_threshold_pct=0.065) == "flat"


def test_asymmetric_behavior_reduces_downside_labeling_vs_upside() -> None:
    assert label_future_return_asymmetric(-0.055, up_threshold_pct=0.05, down_threshold_pct=0.065) == "flat"
    assert label_future_return_asymmetric(0.055, up_threshold_pct=0.05, down_threshold_pct=0.065) == "up"


def test_compute_threshold_map_for_named_variant() -> None:
    assert compute_variant_threshold_map("c2_moderate") == {
        "15m": {"up": 0.05, "down": 0.065},
        "1h": {"up": 0.1, "down": 0.125},
        "4h": {"up": 0.15, "down": 0.19},
    }


def test_compute_asymmetric_threshold_labels_handles_missing_returns() -> None:
    row = {
        "future_return_15m": 0.06,
        "future_return_1h": -0.11,
        "future_return_4h": "n/a",
    }

    result = compute_asymmetric_threshold_labels(
        row,
        get_asymmetric_threshold_variant("c1_conservative"),
    )

    assert result == {
        "15m": "up",
        "1h": "flat",
        "4h": None,
    }


def test_invalid_config_raises_when_down_threshold_is_less_than_up() -> None:
    invalid = AsymmetricThresholdVariantConfig(
        variant_name="broken",
        horizon_settings={
            "15m": HorizonAsymmetricThreshold(up_threshold_pct=0.05, down_threshold_pct=0.04),
            "1h": HorizonAsymmetricThreshold(up_threshold_pct=0.10, down_threshold_pct=0.10),
            "4h": HorizonAsymmetricThreshold(up_threshold_pct=0.15, down_threshold_pct=0.15),
        },
    )

    with pytest.raises(ValueError, match="down_threshold_pct must be >= up_threshold_pct"):
        build_threshold_map(invalid)
