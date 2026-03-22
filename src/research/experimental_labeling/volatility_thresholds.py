from __future__ import annotations

from typing import Any

from src.research.experimental_labeling.volatility_adjusted_config import (
    CANDIDATE_B_V1_CONFIG,
    TARGET_HORIZONS,
    HorizonVolatilityThreshold,
    VolatilityAdjustedThresholdConfig,
)


def _safe_dict(value: Any) -> dict[str, Any]:
    """Return the input when it is a dict; otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    """Safely coerce supported scalar values to float."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None

    return None


def extract_row_atr_inputs(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    Extract ATR inputs from a trade_analysis row.

    Primary source:
    - row["risk"]["atr_value"]
    - row["risk"]["entry_price"]

    Secondary fallback:
    - row["atr_value"]
    - row["entry_price"]
    """
    risk = _safe_dict(row.get("risk"))
    atr_value = _safe_float(risk.get("atr_value", row.get("atr_value")))
    entry_price = _safe_float(risk.get("entry_price", row.get("entry_price")))
    return atr_value, entry_price


def compute_normalized_atr_pct(row: dict[str, Any]) -> float | None:
    """
    Return normalized ATR percent:

        atr_pct = atr_value / entry_price * 100

    Returns None when inputs are missing or invalid.
    """
    atr_value, entry_price = extract_row_atr_inputs(row)

    if atr_value is None or entry_price is None:
        return None
    if atr_value < 0.0 or entry_price <= 0.0:
        return None

    return round((atr_value / entry_price) * 100.0, 6)


def resolve_atr_pct_with_fallback(
    row: dict[str, Any],
    *,
    fallback_atr_pct: float,
) -> tuple[float, bool]:
    """
    Resolve normalized ATR percent with fallback.

    Returns:
    - resolved_atr_pct
    - used_fallback flag
    """
    atr_pct = compute_normalized_atr_pct(row)
    if atr_pct is not None:
        return atr_pct, False

    safe_fallback = max(0.0, float(fallback_atr_pct))
    return safe_fallback, True


def compute_horizon_threshold_pct(
    atr_pct: float,
    horizon_config: HorizonVolatilityThreshold,
) -> float:
    """
    Compute one horizon threshold using:

        threshold = max(min_threshold_pct, atr_pct * atr_multiplier)
    """
    threshold = max(
        float(horizon_config.min_threshold_pct),
        float(atr_pct) * float(horizon_config.atr_multiplier),
    )
    return round(threshold, 6)


def compute_volatility_adjusted_thresholds(
    row: dict[str, Any],
    config: VolatilityAdjustedThresholdConfig = CANDIDATE_B_V1_CONFIG,
) -> dict[str, float]:
    """Return volatility-adjusted thresholds for all configured horizons."""
    config.validate()

    atr_pct, _ = resolve_atr_pct_with_fallback(
        row,
        fallback_atr_pct=config.fallback_atr_pct,
    )

    thresholds: dict[str, float] = {}
    for horizon in TARGET_HORIZONS:
        horizon_config = config.horizon_settings[horizon]
        thresholds[horizon] = compute_horizon_threshold_pct(atr_pct, horizon_config)

    return thresholds


def compute_candidate_b_v1_thresholds(row: dict[str, Any]) -> dict[str, float]:
    """Convenience wrapper for Candidate B v1 volatility-adjusted thresholds."""
    return compute_volatility_adjusted_thresholds(row, CANDIDATE_B_V1_CONFIG)


def compute_candidate_b_v1_threshold_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """
    Return threshold metadata for diagnosis and future relabel reports.

    This helper is useful for the later relabel/report stage without changing
    the threshold computation contract.
    """
    config = CANDIDATE_B_V1_CONFIG
    config.validate()

    atr_pct, used_fallback = resolve_atr_pct_with_fallback(
        row,
        fallback_atr_pct=config.fallback_atr_pct,
    )
    thresholds = compute_volatility_adjusted_thresholds(row, config)

    return {
        "atr_pct": atr_pct,
        "used_fallback_atr_pct": used_fallback,
        "thresholds": thresholds,
    }