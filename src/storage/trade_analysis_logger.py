from __future__ import annotations

import json
import threading
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
        self._lock = threading.Lock()

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
        with self._lock:
            self._append_record(record)
        return record

    def enrich_latest_record(
        self,
        *,
        symbol: str,
        logged_at: str,
        edge_selection_mapper_payload: dict[str, Any] | None = None,
        edge_selection_output: dict[str, Any] | None = None,
        edge_selection_metadata: dict[str, Any] | None = None,
        ai_scaffold_shadow: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add replay-ready edge-selection context to the latest matching log row."""
        with self._lock:
            if not self.log_path.exists() or not self.log_path.is_file():
                raise FileNotFoundError(
                    f"Trade analysis log does not exist: {self.log_path}"
                )

            with self.log_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()

            target_index: int | None = None
            target_record: dict[str, Any] | None = None

            for index in range(len(lines) - 1, -1, -1):
                content = lines[index].strip()
                if not content:
                    continue

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue

                if not isinstance(parsed, dict):
                    continue

                if self._record_matches_identity(
                    parsed,
                    symbol=symbol,
                    logged_at=logged_at,
                ):
                    target_index = index
                    target_record = parsed
                    break

            if target_index is None or target_record is None:
                raise ValueError(
                    "Could not find the latest trade-analysis record to enrich for "
                    f"symbol={symbol} logged_at={logged_at}."
                )

            updated_record = dict(target_record)
            updated_record["edge_selection_mapper_payload"] = (
                self._build_edge_selection_mapper_payload_snapshot(
                    edge_selection_mapper_payload
                )
            )
            updated_record["edge_selection_output"] = (
                self._build_edge_selection_output_snapshot(edge_selection_output)
            )
            updated_record["edge_selection_metadata"] = (
                self._build_edge_selection_metadata_snapshot(edge_selection_metadata)
            )

            if ai_scaffold_shadow is not None:
                updated_record["ai_scaffold_shadow"] = (
                    self._build_ai_scaffold_shadow_snapshot(ai_scaffold_shadow)
                )
            else:
                updated_record.pop("ai_scaffold_shadow", None)

            lines[target_index] = json.dumps(updated_record, ensure_ascii=False) + "\n"

            with self.log_path.open("w", encoding="utf-8") as handle:
                handle.writelines(lines)

            return updated_record

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

        rule_engine_bias = self._value_or_unknown(selected_result.get("bias"))
        rule_engine_reason = self._value_or_unknown(selected_result.get("reason"))

        timeframe_summary = self._build_timeframe_summary(strategy_result)
        timeframe_summary_text = self._build_timeframe_summary_text(strategy_result)

        record = {
            "logged_at": self._now_iso(),
            "symbol": symbol,
            "selected_strategy": selected_strategy,
            "bias": rule_engine_bias,
            "reason": rule_engine_reason,
            "timeframe_summary": timeframe_summary,
            "timeframe_summary_text": timeframe_summary_text,
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
                "interpretation_bias": analysis.get("interpretation_bias"),
                "confidence_note": analysis.get("confidence_note"),
                "caution_flags": analysis.get("caution_flags", []),
                "execution_note": analysis.get("execution_note"),
                "telegram_briefing": analysis.get("telegram_briefing", []),
                "decision_policy": {
                    "rule_engine_authoritative": True,
                    "ai_role": "read_only_interpreter",
                    "ai_must_not_override_rule_engine": True,
                    "ai_must_not_modify_ranking_or_selection": True,
                    "ai_must_not_trigger_execution": True,
                },
            },
            "edge_selection_mapper_payload": None,
            "edge_selection_output": None,
            "edge_selection_metadata": None,
        }

        record["alignment"] = self._build_alignment(record)

        return record

    def _build_alignment(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Compare execution action and AI interpretation bias at a simple semantic level.

        This is intentionally coarse and informational only.
        """
        execution_action = record["execution"].get("action")
        interpretation_bias = record["ai"].get("interpretation_bias")

        execution_to_bias = {
            "buy": "long_bias",
            "sell": "short_bias",
            "hold": "neutral_bias",
        }

        normalized_execution_bias = execution_to_bias.get(execution_action, "neutral_bias")
        aligned = normalized_execution_bias == interpretation_bias

        return {
            "execution_action": execution_action,
            "ai_interpretation_bias": interpretation_bias,
            "normalized_execution_bias": normalized_execution_bias,
            "is_aligned": aligned,
        }

    def _build_timeframe_summary(self, strategy_result: dict[str, Any]) -> dict[str, Any]:
        """
        Preserve the detector-based structured timeframe summary.

        This should remain a dict so that research/diagnostics can access:
        - context_layer
        - setup_layer
        - trigger_layer
        - timeframe-specific states
        """
        selected_result = strategy_result.get("selected_result", {}) or {}
        timeframe_summary_raw = selected_result.get("timeframe_summary", {})

        if not isinstance(timeframe_summary_raw, dict):
            return {"legacy_value": timeframe_summary_raw}

        safe_summary = self._json_safe_copy(timeframe_summary_raw) or {}

        for timeframe in ("1d", "4h", "1h", "15m", "5m", "1m"):
            timeframe_value = safe_summary.get(timeframe)
            if isinstance(timeframe_value, dict):
                timeframe_value["bollinger_20_2"] = self._build_bollinger_snapshot(timeframe_value)

        return safe_summary

    def _build_bollinger_snapshot(self, timeframe_row: dict[str, Any]) -> dict[str, Any]:
        return {
            "middle": self._safe_float(timeframe_row.get("bb_middle_20")),
            "std": self._safe_float(timeframe_row.get("bb_std_20")),
            "upper": self._safe_float(timeframe_row.get("bb_upper_20_2")),
            "lower": self._safe_float(timeframe_row.get("bb_lower_20_2")),
            "bandwidth": self._safe_float(timeframe_row.get("bb_bandwidth_20_2")),
            "percent_b": self._safe_float(timeframe_row.get("bb_percent_b_20_2")),
        }

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return None

        if value_float != value_float:  # NaN check
            return None

        return round(value_float, 6)

    def _build_timeframe_summary_text(self, strategy_result: dict[str, Any]) -> str:
        """
        Build a legacy-friendly human-readable timeframe summary string.

        This is kept as a separate field for grep/debug convenience, while
        timeframe_summary itself stays structured.
        """
        selected_result = strategy_result.get("selected_result", {}) or {}
        timeframe_summary_raw = selected_result.get("timeframe_summary", {})

        timeframe_bias = {
            "5m": "unknown",
            "15m": "unknown",
            "1h": "unknown",
            "4h": "unknown",
        }

        if isinstance(timeframe_summary_raw, dict):
            for timeframe in timeframe_bias:
                raw_value = timeframe_summary_raw.get(timeframe)
                timeframe_bias[timeframe] = self._normalize_timeframe_bias(raw_value)

        intraday_signal = (strategy_result.get("intraday_result", {}) or {}).get("signal")
        scalping_signal = (strategy_result.get("scalping_result", {}) or {}).get("signal")
        swing_signal = (strategy_result.get("swing_result", {}) or {}).get("signal")

        if timeframe_bias["5m"] == "unknown":
            timeframe_bias["5m"] = self._signal_to_bias(intraday_signal)
            if timeframe_bias["5m"] == "unknown":
                timeframe_bias["5m"] = self._signal_to_bias(scalping_signal)

        if timeframe_bias["15m"] == "unknown":
            timeframe_bias["15m"] = self._signal_to_bias(intraday_signal)

        if timeframe_bias["1h"] == "unknown":
            timeframe_bias["1h"] = self._signal_to_bias(intraday_signal)
            if timeframe_bias["1h"] == "unknown":
                timeframe_bias["1h"] = self._signal_to_bias(swing_signal)

        if timeframe_bias["4h"] == "unknown":
            timeframe_bias["4h"] = self._signal_to_bias(swing_signal)

        return (
            f"5m {timeframe_bias['5m']} / "
            f"15m {timeframe_bias['15m']} / "
            f"1h {timeframe_bias['1h']} / "
            f"4h {timeframe_bias['4h']}"
        )

    def _normalize_timeframe_bias(self, raw_value: Any) -> str:
        if isinstance(raw_value, dict):
            for key in ("bias", "signal", "state", "trend"):
                if key in raw_value:
                    return self._normalize_timeframe_bias(raw_value.get(key))
            return "unknown"

        if raw_value is None:
            return "unknown"

        text = str(raw_value).strip().lower()
        if not text:
            return "unknown"

        alias_map = {
            "bullish": "bullish",
            "bearish": "bearish",
            "neutral": "neutral",
            "long": "bullish",
            "short": "bearish",
            "hold": "neutral",
        }
        return alias_map.get(text, text)

    def _signal_to_bias(self, signal: Any) -> str:
        if signal is None:
            return "unknown"

        normalized = str(signal).strip().lower()
        if normalized == "long":
            return "bullish"
        if normalized == "short":
            return "bearish"
        if normalized == "hold":
            return "neutral"
        return "unknown"

    def _value_or_unknown(self, value: Any) -> Any:
        return value if value is not None else "unknown"

    def _append_record(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    def _build_edge_selection_mapper_payload_snapshot(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return self._json_safe_copy(payload)

    def _build_edge_selection_output_snapshot(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return self._json_safe_copy(payload)

    def _build_edge_selection_metadata_snapshot(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return self._json_safe_copy(payload)

    def _build_ai_scaffold_shadow_snapshot(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return self._json_safe_copy(payload)

    def _json_safe_copy(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, dict):
            return {str(key): self._json_safe_copy(item) for key, item in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._json_safe_copy(item) for item in value]

        return str(value)

    def _record_matches_identity(
        self,
        record: dict[str, Any],
        *,
        symbol: str,
        logged_at: str,
    ) -> bool:
        return record.get("symbol") == symbol and record.get("logged_at") == logged_at

    def _now_iso(self) -> str:
        if self.config.utc_timestamp:
            return datetime.now(timezone.utc).isoformat()
        return datetime.now().isoformat()
