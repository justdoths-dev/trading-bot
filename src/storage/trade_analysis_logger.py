from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TradeAnalysisLoggerConfig:
    """Configuration for trade analysis logging."""
    log_dir: str = "logs"
    filename: str = "trade_analysis.jsonl"
    utc_timestamp: bool = True


class TradeAnalysisLogger:
    """
    Persist trading analysis results to a JSONL file.

    One execution = one JSON object line.
    This keeps logging append-only, easy to inspect, and easy to migrate later.
    """

    def __init__(self, config: TradeAnalysisLoggerConfig | None = None) -> None:
        self.config = config or TradeAnalysisLoggerConfig()
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / self.config.filename

    def log(
        self,
        symbol: str,
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
        ai_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a normalized log record and append it to the JSONL file.

        strategy_result must contain:
        {
            "selected_strategy": str,
            "selected_result": dict,
            "scalping_result": dict,
            "intraday_result": dict,
            "swing_result": dict
        }

        Returns the saved record for optional debug printing.
        """
        record = self._build_record(
            symbol=symbol,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_result,
        )
        self._append_record(record)
        return record

    def _build_record(
        self,
        symbol: str,
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
        ai_result: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = ai_result.get("analysis", {}) or {}

        selected_strategy = strategy_result.get("selected_strategy")
        selected_result = strategy_result.get("selected_result", {}) or {}

        scalping_result = strategy_result.get("scalping_result", {}) or {}
        intraday_result = strategy_result.get("intraday_result", {}) or {}
        swing_result = strategy_result.get("swing_result", {}) or {}

        record = {
            "logged_at": self._now_iso(),
            "symbol": symbol,
            "selected_strategy": selected_strategy,
            "scalping_result": {
                "strategy": scalping_result.get("strategy"),
                "signal": scalping_result.get("signal"),
                "confidence": scalping_result.get("confidence"),
            },
            "intraday_result": {
                "strategy": intraday_result.get("strategy"),
                "signal": intraday_result.get("signal"),
                "confidence": intraday_result.get("confidence"),
            },
            "swing_result": {
                "strategy": swing_result.get("strategy"),
                "signal": swing_result.get("signal"),
                "confidence": swing_result.get("confidence"),
            },
            "rule_engine": {
                "strategy": selected_result.get("strategy"),
                "bias": selected_result.get("bias"),
                "signal": selected_result.get("signal"),
                "confidence": selected_result.get("confidence"),
                "reason": selected_result.get("reason"),
            },
            "risk": {
                "execution_allowed": risk_result.get("execution_allowed"),
                "entry_price": risk_result.get("entry_price"),
                "stop_loss": risk_result.get("stop_loss"),
                "take_profit": risk_result.get("take_profit"),
                "risk_per_unit": risk_result.get("risk_per_unit"),
                "reward_per_unit": risk_result.get("reward_per_unit"),
                "risk_reward_ratio": risk_result.get("risk_reward_ratio"),
                "atr_value": risk_result.get("atr_value"),
                "volatility_state": risk_result.get("volatility_state"),
                "reason": risk_result.get("reason"),
            },
            "execution": {
                "action": execution_result.get("action"),
                "symbol": execution_result.get("symbol"),
                "execution_mode": execution_result.get("execution_mode"),
                "signal": execution_result.get("signal"),
                "bias": execution_result.get("bias"),
                "execution_allowed": execution_result.get("execution_allowed"),
                "entry_price": execution_result.get("entry_price"),
                "stop_loss": execution_result.get("stop_loss"),
                "take_profit": execution_result.get("take_profit"),
                "reason": execution_result.get("reason"),
            },
            "ai": {
                "source": ai_result.get("source"),
                "model": ai_result.get("model"),
                "environment": ai_result.get("environment"),
                "generated_at": ai_result.get("generated_at"),
                "market_structure": analysis.get("market_structure"),
                "rule_engine_assessment": analysis.get("rule_engine_assessment"),
                "key_bottlenecks": analysis.get("key_bottlenecks", []),
                "long_scenario": analysis.get("long_scenario"),
                "short_scenario": analysis.get("short_scenario"),
                "final_stance": analysis.get("final_stance"),
                "stance_reason": analysis.get("stance_reason"),
                "telegram_briefing": analysis.get("telegram_briefing", []),
            },
        }

        record["alignment"] = self._build_alignment(record)

        return record

    def _build_alignment(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Compare execution action and AI final stance at a simple semantic level.
        """
        execution_action = record["execution"].get("action")
        ai_stance = record["ai"].get("final_stance")

        execution_to_ai = {
            "buy": "long",
            "sell": "short",
            "hold": "hold",
        }

        normalized_execution_stance = execution_to_ai.get(execution_action, "hold")
        aligned = normalized_execution_stance == ai_stance

        return {
            "execution_action": execution_action,
            "ai_final_stance": ai_stance,
            "normalized_execution_stance": normalized_execution_stance,
            "is_aligned": aligned,
        }

    def _append_record(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _now_iso(self) -> str:
        if self.config.utc_timestamp:
            return datetime.now(timezone.utc).isoformat()
        return datetime.now().isoformat()