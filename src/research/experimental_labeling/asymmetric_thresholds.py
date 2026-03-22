from __future__ import annotations

from typing import Any

from src.research.experimental_labeling.asymmetric_threshold_config import (
    DEFAULT_VARIANT_NAME,
    AsymmetricThresholdVariantConfig,
    TARGET_HORIZONS,
    get_asymmetric_threshold_variant,
)

TARGET_LABELS = ("up", "down", "flat")


def _safe_float(value: Any) -> float | None:
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


def label_future_return_asymmetric(
    future_return: float,
    *,
    up_threshold_pct: float,
    down_threshold_pct: float,
) -> str:
    """Label one future return using inclusive asymmetric thresholds."""
    if future_return >= up_threshold_pct:
        return "up"
    if future_return <= -down_threshold_pct:
        return "down"
    return "flat"


def build_threshold_map(
    config: AsymmetricThresholdVariantConfig,
) -> dict[str, dict[str, float]]:
    """Materialize threshold settings in a report-friendly dict form."""
    config.validate()
    return {
        horizon: {
            "up": round(config.horizon_settings[horizon].up_threshold_pct, 6),
            "down": round(config.horizon_settings[horizon].down_threshold_pct, 6),
        }
        for horizon in TARGET_HORIZONS
    }


def compute_asymmetric_threshold_labels(
    row: dict[str, Any],
    config: AsymmetricThresholdVariantConfig,
) -> dict[str, str | None]:
    """Return per-horizon labels for numeric future returns in one row."""
    thresholds = build_threshold_map(config)
    labels: dict[str, str | None] = {}

    for horizon in TARGET_HORIZONS:
        future_return = _safe_float(row.get(f"future_return_{horizon}"))
        if future_return is None:
            labels[horizon] = None
            continue

        threshold_pair = thresholds[horizon]
        labels[horizon] = label_future_return_asymmetric(
            future_return,
            up_threshold_pct=threshold_pair["up"],
            down_threshold_pct=threshold_pair["down"],
        )

    return labels


def compute_variant_threshold_map(
    variant_name: str = DEFAULT_VARIANT_NAME,
) -> dict[str, dict[str, float]]:
    """Convenience wrapper for report/relabel modules using a named variant."""
    return build_threshold_map(get_asymmetric_threshold_variant(variant_name))
