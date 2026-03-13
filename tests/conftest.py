from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def valid_research_record() -> dict[str, Any]:
    return {
        "logged_at": "2026-03-10T12:00:00+00:00",
        "symbol": "BTCUSDT",
        "selected_strategy": "momentum_breakout",
        "rule_engine": {
            "bias": "bullish",
            "signal": "long",
        },
        "risk": {
            "execution_allowed": True,
        },
        "execution": {
            "action": "long",
        },
        "ai_output": {
            "summary": "AI agrees with the long setup.",
        },
        "future_return_15m": 0.8,
        "future_return_1h": 1.4,
        "future_return_4h": -0.3,
        "future_label_15m": "up",
        "future_label_1h": "up",
        "future_label_4h": "down",
    }


@pytest.fixture
def write_jsonl(tmp_path: Path):
    def _write_jsonl(records: list[dict[str, Any]], filename: str = "trade_analysis.jsonl") -> Path:
        path = tmp_path / filename
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")
        return path

    return _write_jsonl
