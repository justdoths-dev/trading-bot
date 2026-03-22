from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

TARGET_HORIZONS = ("15m", "1h", "4h")
DEFAULT_VARIANT_NAME = "c2_moderate"


@dataclass(frozen=True)
class HorizonAsymmetricThreshold:
    """Research-only asymmetric thresholds for one prediction horizon."""

    up_threshold_pct: float
    down_threshold_pct: float


@dataclass(frozen=True)
class AsymmetricThresholdVariantConfig:
    """Named configuration for Candidate C asymmetric threshold relabeling."""

    variant_name: str
    horizon_settings: Mapping[str, HorizonAsymmetricThreshold]

    def validate(self) -> None:
        missing = [horizon for horizon in TARGET_HORIZONS if horizon not in self.horizon_settings]
        extra = [horizon for horizon in self.horizon_settings if horizon not in TARGET_HORIZONS]

        if missing:
            raise ValueError(f"Missing horizon settings: {missing}")
        if extra:
            raise ValueError(f"Unexpected horizon settings: {extra}")

        for horizon, settings in self.horizon_settings.items():
            if settings.up_threshold_pct < 0.0:
                raise ValueError(
                    f"up_threshold_pct must be >= 0.0 for horizon={horizon}"
                )
            if settings.down_threshold_pct < 0.0:
                raise ValueError(
                    f"down_threshold_pct must be >= 0.0 for horizon={horizon}"
                )
            if settings.down_threshold_pct < settings.up_threshold_pct:
                raise ValueError(
                    f"down_threshold_pct must be >= up_threshold_pct for horizon={horizon}"
                )


ASYMMETRIC_THRESHOLD_VARIANTS: dict[str, AsymmetricThresholdVariantConfig] = {
    "c1_conservative": AsymmetricThresholdVariantConfig(
        variant_name="c1_conservative",
        horizon_settings={
            "15m": HorizonAsymmetricThreshold(up_threshold_pct=0.05, down_threshold_pct=0.06),
            "1h": HorizonAsymmetricThreshold(up_threshold_pct=0.10, down_threshold_pct=0.12),
            "4h": HorizonAsymmetricThreshold(up_threshold_pct=0.15, down_threshold_pct=0.18),
        },
    ),
    "c2_moderate": AsymmetricThresholdVariantConfig(
        variant_name="c2_moderate",
        horizon_settings={
            "15m": HorizonAsymmetricThreshold(up_threshold_pct=0.05, down_threshold_pct=0.065),
            "1h": HorizonAsymmetricThreshold(up_threshold_pct=0.10, down_threshold_pct=0.125),
            "4h": HorizonAsymmetricThreshold(up_threshold_pct=0.15, down_threshold_pct=0.19),
        },
    ),
    "c3_stronger": AsymmetricThresholdVariantConfig(
        variant_name="c3_stronger",
        horizon_settings={
            "15m": HorizonAsymmetricThreshold(up_threshold_pct=0.05, down_threshold_pct=0.07),
            "1h": HorizonAsymmetricThreshold(up_threshold_pct=0.10, down_threshold_pct=0.13),
            "4h": HorizonAsymmetricThreshold(up_threshold_pct=0.15, down_threshold_pct=0.20),
        },
    ),
}


def get_asymmetric_threshold_variant(
    variant_name: str = DEFAULT_VARIANT_NAME,
) -> AsymmetricThresholdVariantConfig:
    """Return one validated Candidate C asymmetric threshold variant."""
    config = ASYMMETRIC_THRESHOLD_VARIANTS.get(variant_name)
    if config is None:
        supported = ", ".join(sorted(ASYMMETRIC_THRESHOLD_VARIANTS))
        raise ValueError(
            f"Unknown asymmetric threshold variant: {variant_name}. Supported variants: {supported}"
        )

    config.validate()
    return config
