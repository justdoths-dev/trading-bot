from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "research_reports" / "latest"
DEFAULT_MAX_MISMATCH_EXAMPLES = 3

_DIRECTIONAL_ACTION_TO_BIAS = {
    "observe_long_setup": "long",
    "observe_short_setup": "short",
}
_NEUTRAL_ACTIONS = {"hold", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only comparison report for rule outputs vs AI shadow annotations."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Explicit trade-analysis JSONL input path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and Markdown report outputs.",
    )
    parser.add_argument(
        "--max-mismatch-examples",
        type=int,
        default=DEFAULT_MAX_MISMATCH_EXAMPLES,
        help="Maximum number of representative mismatch examples to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ai_shadow_comparison_report(
        input_path=args.input_path,
        output_dir=resolve_output_dir(args.output_dir),
        max_mismatch_examples=args.max_mismatch_examples,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def run_ai_shadow_comparison_report(
    *,
    input_path: Path | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_mismatch_examples: int = DEFAULT_MAX_MISMATCH_EXAMPLES,
) -> dict[str, Any]:
    resolved_input = resolve_input_path(input_path)
    loaded = load_trade_analysis_rows(resolved_input)
    summary = build_ai_shadow_comparison_summary(
        records=loaded["records"],
        input_path=resolved_input,
        data_quality=loaded["data_quality"],
        max_mismatch_examples=max_mismatch_examples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ai_shadow_comparison_report.json"
    md_path = output_dir / "ai_shadow_comparison_report.md"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_ai_shadow_comparison_markdown(summary),
        encoding="utf-8",
    )

    return {
        "input_path": str(resolved_input),
        "output_dir": str(output_dir),
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "summary": summary,
    }


def resolve_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def resolve_input_path(explicit_input_path: Path | None) -> Path:
    if explicit_input_path is None:
        raise ValueError("input_path is required for ai shadow comparison report.")

    resolved = explicit_input_path.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved = resolved.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Input path is not a file: {resolved}")

    return resolved


def load_trade_analysis_rows(input_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    data_quality = {
        "input_exists": input_path.exists(),
        "input_is_file": input_path.is_file() if input_path.exists() else False,
        "total_lines": 0,
        "valid_records": 0,
        "blank_lines": 0,
        "malformed_lines": 0,
        "non_object_lines": 0,
        "malformed_line_numbers": [],
    }

    if not input_path.exists() or not input_path.is_file():
        return {"records": records, "data_quality": data_quality}

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            data_quality["total_lines"] += 1
            line = raw_line.strip()
            if not line:
                data_quality["blank_lines"] += 1
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                data_quality["malformed_lines"] += 1
                data_quality["malformed_line_numbers"].append(line_number)
                continue

            if not isinstance(payload, dict):
                data_quality["malformed_lines"] += 1
                data_quality["non_object_lines"] += 1
                data_quality["malformed_line_numbers"].append(line_number)
                continue

            record = dict(payload)
            record["_line_number"] = line_number
            records.append(record)
            data_quality["valid_records"] += 1

    return {"records": records, "data_quality": data_quality}


def build_ai_shadow_comparison_summary(
    *,
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
    max_mismatch_examples: int = DEFAULT_MAX_MISMATCH_EXAMPLES,
) -> dict[str, Any]:
    comparable_rows = 0
    rows_with_shadow = 0
    rows_without_shadow = 0
    rows_with_shadow_error = 0
    rows_with_empty_shadow_response = 0

    bias_alignment_counts: Counter[str] = Counter()
    execution_context_counts: Counter[str] = Counter()
    recommended_action_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    regime_label_counts: Counter[str] = Counter()
    mismatch_flag_counts: Counter[str] = Counter()
    mismatch_rows_count = 0

    mismatch_examples: list[dict[str, Any]] = []

    for record in records:
        shadow = record.get("ai_scaffold_shadow")
        if not isinstance(shadow, dict):
            rows_without_shadow += 1
            continue

        rows_with_shadow += 1

        has_shadow_error = shadow.get("error") not in (None, {})
        if has_shadow_error:
            rows_with_shadow_error += 1

        response = shadow.get("response")
        if not isinstance(response, dict) or not response:
            if not has_shadow_error:
                rows_with_empty_shadow_response += 1
            continue

        comparable_rows += 1

        rule_bias = _extract_rule_bias(record)
        shadow_bias = _normalize_bias(response.get("bias"))
        bias_alignment = _classify_bias_alignment(rule_bias, shadow_bias)
        bias_alignment_counts[bias_alignment] += 1

        execution_allowed = _extract_execution_allowed(record)
        recommended_action = _normalize_text(response.get("recommended_action"), "unknown")
        recommended_action_counts[recommended_action] += 1

        execution_context = _classify_execution_context(
            execution_allowed=execution_allowed,
            recommended_action=recommended_action,
        )
        execution_context_counts[execution_context] += 1

        confidence_counts[_normalize_text(response.get("confidence"), "unknown")] += 1
        regime_label_counts[_normalize_text(response.get("regime_label"), "unknown")] += 1

        mismatch_flags = _build_mismatch_flags(
            rule_bias=rule_bias,
            shadow_bias=shadow_bias,
            execution_context=execution_context,
        )
        for flag in mismatch_flags:
            mismatch_flag_counts[flag] += 1

        if mismatch_flags:
            mismatch_rows_count += 1

        if mismatch_flags and len(mismatch_examples) < max(0, max_mismatch_examples):
            mismatch_examples.append(
                {
                    "line_number": _safe_int(record.get("_line_number")),
                    "logged_at": _coerce_text(record.get("logged_at"), "unknown"),
                    "symbol": _normalize_symbol(record.get("symbol")),
                    "selected_strategy": _coerce_text(
                        record.get("selected_strategy"),
                        "unknown",
                    ),
                    "rule_bias": rule_bias,
                    "shadow_bias": shadow_bias,
                    "execution_allowed": execution_allowed,
                    "recommended_action": recommended_action,
                    "confidence": _normalize_text(response.get("confidence"), "unknown"),
                    "regime_label": _normalize_text(response.get("regime_label"), "unknown"),
                    "mismatch_flags": mismatch_flags,
                }
            )

    total_rows = len(records)
    rows_skipped_unavailable = total_rows - comparable_rows

    summary = {
        "metadata": {
            "generated_at": _utc_now_iso(),
            "report_type": "ai_shadow_comparison_report",
            "input_path": str(input_path),
            "total_rows_inspected": total_rows,
            "rows_with_ai_scaffold_shadow": rows_with_shadow,
            "rows_with_comparison_available": comparable_rows,
            "rows_skipped_unavailable": rows_skipped_unavailable,
            "max_mismatch_examples": max(0, max_mismatch_examples),
            "data_quality": data_quality,
        },
        "comparison_availability": {
            "rows_without_ai_scaffold_shadow": rows_without_shadow,
            "rows_with_shadow_error": rows_with_shadow_error,
            "rows_with_empty_shadow_response": rows_with_empty_shadow_response,
            "bias_comparison_available": comparable_rows,
            "execution_comparison_available": comparable_rows,
        },
        "bias_alignment_summary": {
            "aligned": bias_alignment_counts.get("aligned", 0),
            "mismatched": bias_alignment_counts.get("mismatched", 0),
            "rule_neutral_or_unknown": bias_alignment_counts.get(
                "rule_neutral_or_unknown",
                0,
            ),
            "shadow_neutral_or_unknown": bias_alignment_counts.get(
                "shadow_neutral_or_unknown",
                0,
            ),
            "unavailable": bias_alignment_counts.get("unavailable", 0),
        },
        "execution_context_summary": {
            "allowed_and_shadow_directional": execution_context_counts.get(
                "allowed_and_shadow_directional",
                0,
            ),
            "allowed_and_shadow_neutral": execution_context_counts.get(
                "allowed_and_shadow_neutral",
                0,
            ),
            "blocked_and_shadow_neutral": execution_context_counts.get(
                "blocked_and_shadow_neutral",
                0,
            ),
            "blocked_but_shadow_directional": execution_context_counts.get(
                "blocked_but_shadow_directional",
                0,
            ),
            "unavailable": execution_context_counts.get("unavailable", 0),
        },
        "recommended_action_summary": _distribution_rows(
            recommended_action_counts,
            key_name="recommended_action",
        ),
        "confidence_summary": _distribution_rows(
            confidence_counts,
            key_name="confidence",
        ),
        "regime_label_summary": _distribution_rows(
            regime_label_counts,
            key_name="regime_label",
        ),
        "mismatch_summary": {
            "rows_with_mismatch_flags": mismatch_rows_count,
            "mismatch_flag_counts": _distribution_rows(
                mismatch_flag_counts,
                key_name="mismatch_flag",
            ),
        },
        "representative_mismatch_examples": mismatch_examples,
    }
    summary["interpretation"] = _build_interpretation(summary)
    return summary


def render_ai_shadow_comparison_markdown(summary: dict[str, Any]) -> str:
    metadata = summary.get("metadata", {})
    availability = summary.get("comparison_availability", {})
    bias_summary = summary.get("bias_alignment_summary", {})
    execution_summary = summary.get("execution_context_summary", {})
    interpretation = summary.get("interpretation", {})

    lines = [
        "# AI Shadow Comparison Report",
        "",
        f"Generated at: {metadata.get('generated_at', 'unknown')}",
        f"Input path: {metadata.get('input_path', 'unknown')}",
        "",
        "## Coverage",
        "",
        f"- total_rows_inspected: {metadata.get('total_rows_inspected', 0)}",
        f"- rows_with_ai_scaffold_shadow: {metadata.get('rows_with_ai_scaffold_shadow', 0)}",
        f"- rows_with_comparison_available: {metadata.get('rows_with_comparison_available', 0)}",
        f"- rows_skipped_unavailable: {metadata.get('rows_skipped_unavailable', 0)}",
        f"- rows_without_ai_scaffold_shadow: {availability.get('rows_without_ai_scaffold_shadow', 0)}",
        f"- rows_with_shadow_error: {availability.get('rows_with_shadow_error', 0)}",
        f"- rows_with_empty_shadow_response: {availability.get('rows_with_empty_shadow_response', 0)}",
        "",
        "## Bias Alignment",
        "",
        f"- aligned: {bias_summary.get('aligned', 0)}",
        f"- mismatched: {bias_summary.get('mismatched', 0)}",
        f"- rule_neutral_or_unknown: {bias_summary.get('rule_neutral_or_unknown', 0)}",
        f"- shadow_neutral_or_unknown: {bias_summary.get('shadow_neutral_or_unknown', 0)}",
        f"- unavailable: {bias_summary.get('unavailable', 0)}",
        "",
        "## Execution Context",
        "",
        f"- allowed_and_shadow_directional: {execution_summary.get('allowed_and_shadow_directional', 0)}",
        f"- allowed_and_shadow_neutral: {execution_summary.get('allowed_and_shadow_neutral', 0)}",
        f"- blocked_and_shadow_neutral: {execution_summary.get('blocked_and_shadow_neutral', 0)}",
        f"- blocked_but_shadow_directional: {execution_summary.get('blocked_but_shadow_directional', 0)}",
        f"- unavailable: {execution_summary.get('unavailable', 0)}",
        "",
        "## Recommended Action Summary",
        "",
    ]

    lines.extend(
        _render_distribution_rows(
            summary.get("recommended_action_summary", []),
            "recommended_action",
        )
    )
    lines.extend(
        [
            "",
            "## Confidence Summary",
            "",
        ]
    )
    lines.extend(
        _render_distribution_rows(
            summary.get("confidence_summary", []),
            "confidence",
        )
    )
    lines.extend(
        [
            "",
            "## Regime Label Summary",
            "",
        ]
    )
    lines.extend(
        _render_distribution_rows(
            summary.get("regime_label_summary", []),
            "regime_label",
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {interpretation.get('coverage_summary', 'No coverage summary available.')}",
            f"- {interpretation.get('bias_summary', 'No bias summary available.')}",
            f"- {interpretation.get('execution_summary', 'No execution summary available.')}",
            f"- {interpretation.get('mismatch_summary', 'No mismatch summary available.')}",
            "",
            "## Representative Mismatch Examples",
            "",
        ]
    )

    mismatch_examples = summary.get("representative_mismatch_examples", [])
    if mismatch_examples:
        for example in mismatch_examples:
            lines.append(
                "- "
                f"line={example.get('line_number', 0)} "
                f"symbol={example.get('symbol', 'UNKNOWN')} "
                f"strategy={example.get('selected_strategy', 'unknown')} "
                f"rule_bias={example.get('rule_bias', 'unknown')} "
                f"shadow_bias={example.get('shadow_bias', 'unknown')} "
                f"execution_allowed={example.get('execution_allowed', 'unknown')} "
                f"recommended_action={example.get('recommended_action', 'unknown')} "
                f"flags={example.get('mismatch_flags', [])}"
            )
    else:
        lines.append("- none")

    lines.append("")
    return "\n".join(lines)


def _render_distribution_rows(rows: list[dict[str, Any]], key_name: str) -> list[str]:
    if not rows:
        return ["- none"]

    rendered: list[str] = []
    for row in rows:
        rendered.append(
            f"- {row.get(key_name, 'unknown')}: {row.get('count', 0)} ({row.get('rate', 0.0)})"
        )
    return rendered


def _build_interpretation(summary: dict[str, Any]) -> dict[str, str]:
    metadata = summary.get("metadata", {})
    bias_summary = summary.get("bias_alignment_summary", {})
    execution_summary = summary.get("execution_context_summary", {})
    mismatch_summary = summary.get("mismatch_summary", {})

    comparable_rows = metadata.get("rows_with_comparison_available", 0)
    total_rows = metadata.get("total_rows_inspected", 0)

    coverage_summary = (
        f"{comparable_rows} of {total_rows} inspected rows contained comparable AI shadow output."
    )

    bias_summary_text = (
        f"Bias aligned in {bias_summary.get('aligned', 0)} rows and mismatched in "
        f"{bias_summary.get('mismatched', 0)} rows."
    )

    execution_summary_text = (
        f"Execution-blocked rows with directional shadow actions occurred "
        f"{execution_summary.get('blocked_but_shadow_directional', 0)} times."
    )

    mismatch_rows = mismatch_summary.get("rows_with_mismatch_flags", 0)
    mismatch_summary_text = (
        f"{mismatch_rows} representative row(s) contained at least one tracked mismatch flag."
    )

    return {
        "coverage_summary": coverage_summary,
        "bias_summary": bias_summary_text,
        "execution_summary": execution_summary_text,
        "mismatch_summary": mismatch_summary_text,
    }


def _distribution_rows(counter: Counter[str], *, key_name: str) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows: list[dict[str, Any]] = []

    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                key_name: key,
                "count": count,
                "rate": _ratio(count, total),
            }
        )

    return rows


def _classify_bias_alignment(rule_bias: str, shadow_bias: str) -> str:
    if rule_bias == "unknown" or shadow_bias == "unknown":
        return "unavailable"
    if rule_bias == "neutral":
        return "rule_neutral_or_unknown"
    if shadow_bias == "neutral":
        return "shadow_neutral_or_unknown"
    if rule_bias == shadow_bias:
        return "aligned"
    return "mismatched"


def _classify_execution_context(
    *,
    execution_allowed: bool | None,
    recommended_action: str,
) -> str:
    if execution_allowed is None:
        return "unavailable"

    shadow_action_bias = _action_to_bias(recommended_action)
    if execution_allowed:
        if shadow_action_bias in {"long", "short"}:
            return "allowed_and_shadow_directional"
        return "allowed_and_shadow_neutral"

    if shadow_action_bias in {"long", "short"}:
        return "blocked_but_shadow_directional"
    return "blocked_and_shadow_neutral"


def _build_mismatch_flags(
    *,
    rule_bias: str,
    shadow_bias: str,
    execution_context: str,
) -> list[str]:
    flags: list[str] = []

    if _classify_bias_alignment(rule_bias, shadow_bias) == "mismatched":
        flags.append("bias_mismatch")

    if execution_context == "blocked_but_shadow_directional":
        flags.append("directional_shadow_when_execution_blocked")

    return flags


def _extract_rule_bias(record: dict[str, Any]) -> str:
    rule_engine = record.get("rule_engine")
    if isinstance(rule_engine, dict):
        for key in ("signal", "bias"):
            normalized = _normalize_bias(rule_engine.get(key))
            if normalized != "unknown":
                return normalized

    normalized = _normalize_bias(record.get("bias"))
    if normalized != "unknown":
        return normalized

    return "unknown"


def _extract_execution_allowed(record: dict[str, Any]) -> bool | None:
    risk = record.get("risk")
    if isinstance(risk, dict) and isinstance(risk.get("execution_allowed"), bool):
        return risk.get("execution_allowed")

    execution = record.get("execution")
    if isinstance(execution, dict) and isinstance(execution.get("execution_allowed"), bool):
        return execution.get("execution_allowed")

    return None


def _normalize_bias(value: Any) -> str:
    text = _normalize_text(value, "unknown")
    if text in {"long", "bullish", "buy", "observe_long_setup"}:
        return "long"
    if text in {"short", "bearish", "sell", "observe_short_setup"}:
        return "short"
    if text in {"neutral", "hold", "wait"}:
        return "neutral"
    return "unknown"


def _action_to_bias(recommended_action: str) -> str:
    if recommended_action in _DIRECTIONAL_ACTION_TO_BIAS:
        return _DIRECTIONAL_ACTION_TO_BIAS[recommended_action]
    if recommended_action in _NEUTRAL_ACTIONS:
        return "neutral"
    return "unknown"


def _normalize_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.lower()


def _coerce_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text


def _normalize_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    return text or "UNKNOWN"


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()