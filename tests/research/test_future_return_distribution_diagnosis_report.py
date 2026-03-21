from __future__ import annotations

from src.research.future_return_distribution_diagnosis_report import (
    CURRENT_FLAT_THRESHOLD_PCT,
    ReturnSeries,
    _build_horizon_overview,
    _build_observations,
    _count_flat_band,
    _percentile,
    _safe_float,
    _threshold_sweep,
)


def test_safe_float_handles_common_inputs() -> None:
    assert _safe_float(None) is None
    assert _safe_float(True) is None
    assert _safe_float(False) is None
    assert _safe_float(1) == 1.0
    assert _safe_float(1.25) == 1.25
    assert _safe_float(" 2.5 ") == 2.5
    assert _safe_float("3.0%") == 3.0
    assert _safe_float("") is None
    assert _safe_float("abc") is None


def test_percentile_interpolates_and_handles_empty_series() -> None:
    values = [0.0, 10.0, 20.0, 30.0]

    assert _percentile([], 50.0) is None
    assert _percentile(values, 0.0) == 0.0
    assert _percentile(values, 25.0) == 7.5
    assert _percentile(values, 50.0) == 15.0
    assert _percentile(values, 75.0) == 22.5
    assert _percentile(values, 100.0) == 30.0



def test_count_flat_band_includes_threshold_boundaries() -> None:
    values = [-0.20, -0.19, 0.0, 0.19, 0.20, 0.21]

    assert _count_flat_band(values, CURRENT_FLAT_THRESHOLD_PCT) == 5
    assert _count_flat_band(values, 0.19) == 3



def test_threshold_sweep_is_deterministic_for_toy_input() -> None:
    values = [-0.50, -0.20, 0.0, 0.20, 0.50]

    result = _threshold_sweep(values)

    assert result["0.05"] == {
        "flat_band_count": 1,
        "flat_band_ratio_pct": 20.0,
    }
    assert result["0.20"] == {
        "flat_band_count": 3,
        "flat_band_ratio_pct": 60.0,
    }
    assert result["0.50"] == {
        "flat_band_count": 5,
        "flat_band_ratio_pct": 100.0,
    }



def test_build_horizon_overview_returns_nullable_distribution_stats_for_empty_series() -> None:
    overview = _build_horizon_overview(ReturnSeries(horizon="15m", values=[]))

    assert overview["sample_count"] == 0
    assert overview["min_return_pct"] is None
    assert overview["max_return_pct"] is None
    assert overview["avg_return_pct"] is None
    assert overview["median_return_pct"] is None
    assert overview["p05"] is None
    assert overview["p95"] is None
    assert overview["current_flat_band_count"] == 0
    assert overview["current_flat_band_ratio_pct"] == 0.0
    assert overview["positive_return_ratio_pct"] == 0.0
    assert overview["non_positive_return_ratio_pct"] == 0.0



def test_build_horizon_overview_computes_expected_distribution_values() -> None:
    series = ReturnSeries(horizon="15m", values=[-0.40, -0.10, 0.0, 0.20, 0.60])

    overview = _build_horizon_overview(series)

    assert overview["sample_count"] == 5
    assert overview["min_return_pct"] == -0.4
    assert overview["max_return_pct"] == 0.6
    assert overview["avg_return_pct"] == 0.06
    assert overview["median_return_pct"] == 0.0
    assert overview["p25"] == -0.1
    assert overview["p75"] == 0.2
    assert overview["current_flat_band_count"] == 3
    assert overview["current_flat_band_ratio_pct"] == 60.0
    assert overview["positive_return_ratio_pct"] == 40.0
    assert overview["non_positive_return_ratio_pct"] == 60.0



def test_build_observations_emits_15m_heavy_flat_concentration_labels() -> None:
    horizon_overview = {
        "15m": {
            "current_flat_band_ratio_pct": 42.0,
        },
        "1h": {
            "current_flat_band_ratio_pct": 24.0,
        },
        "4h": {
            "current_flat_band_ratio_pct": 18.0,
        },
    }
    by_symbol = {
        "15m": [
            {
                "symbol": "BTCUSDT",
                "sample_count": 30,
                "current_flat_band_ratio_pct": 62.0,
            },
            {
                "symbol": "ETHUSDT",
                "sample_count": 25,
                "current_flat_band_ratio_pct": 55.0,
            },
        ],
        "1h": [],
        "4h": [],
    }

    result = _build_observations(horizon_overview=horizon_overview, by_symbol=by_symbol)

    assert result == [
        "current_threshold_likely_too_wide_for_15m",
        "symbol_level_flat_concentration_detected",
        "horizon_specific_threshold_review_recommended",
    ]



def test_build_observations_emits_broad_flat_pressure_label() -> None:
    horizon_overview = {
        "15m": {"current_flat_band_ratio_pct": 28.0},
        "1h": {"current_flat_band_ratio_pct": 31.0},
        "4h": {"current_flat_band_ratio_pct": 27.0},
    }
    by_symbol = {
        "15m": [],
        "1h": [],
        "4h": [],
    }

    result = _build_observations(horizon_overview=horizon_overview, by_symbol=by_symbol)

    assert result == ["flat_pressure_is_broad_across_horizons"]
