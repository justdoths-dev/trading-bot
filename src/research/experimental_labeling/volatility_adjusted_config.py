from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

TARGET_HORIZONS = ("15m", "1h", "4h")


@dataclass(frozen=True)
class HorizonVolatilityThreshold:
    """Configuration for one horizon's ATR-adjusted threshold."""

    min_threshold_pct: float
    atr_multiplier: float


@dataclass(frozen=True)
class VolatilityAdjustedThresholdConfig:
    """Centralized threshold configuration for research-only relabel experiments."""

    horizon_settings: Mapping[str, HorizonVolatilityThreshold]
    fallback_atr_pct: float = 0.0

    def validate(self) -> None:
        """Validate config integrity for all target horizons."""
        missing = [h for h in TARGET_HORIZONS if h not in self.horizon_settings]
        extra = [h for h in self.horizon_settings if h not in TARGET_HORIZONS]

        if missing:
            raise ValueError(f"Missing horizon settings: {missing}")
        if extra:
            raise ValueError(f"Unexpected horizon settings: {extra}")

        for horizon, setting in self.horizon_settings.items():
            if setting.min_threshold_pct < 0.0:
                raise ValueError(
                    f"min_threshold_pct must be >= 0.0 for horizon={horizon}"
                )
            if setting.atr_multiplier < 0.0:
                raise ValueError(
                    f"atr_multiplier must be >= 0.0 for horizon={horizon}"
                )

        if self.fallback_atr_pct < 0.0:
            raise ValueError("fallback_atr_pct must be >= 0.0")


CANDIDATE_B_V1_CONFIG = VolatilityAdjustedThresholdConfig(
    horizon_settings={
        "15m": HorizonVolatilityThreshold(
            min_threshold_pct=0.05,
            atr_multiplier=0.35,
        ),
        "1h": HorizonVolatilityThreshold(
            min_threshold_pct=0.10,
            atr_multiplier=0.50,
        ),
        "4h": HorizonVolatilityThreshold(
            min_threshold_pct=0.15,
            atr_multiplier=0.65,
        ),
    },
    fallback_atr_pct=0.0,
)


def get_candidate_b_v1_config() -> VolatilityAdjustedThresholdConfig:
    """Return the research-only Candidate B v1 threshold configuration."""
    CANDIDATE_B_V1_CONFIG.validate()
    return CANDIDATE_B_V1_CONFIG
