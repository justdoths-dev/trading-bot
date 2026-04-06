from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_REPORT_ROOT = Path("logs/research_reports/strict_gate_fallback_observability")
LATEST_DIR = Path("logs/research_reports/latest")
LATEST_JSON_NAME = "strict_gate_fallback_observability_diagnosis.json"
LATEST_MD_NAME = "strict_gate_fallback_observability_diagnosis.md"


@dataclass(frozen=True)
class IdentityKey:
    symbol: str
    strategy: str
    horizon: str

    def as_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "horizon": self.horizon,
        }


@dataclass
class ParsedQualityGateRow:
    row_index: int
    used_compatibility_fields: bool
    total_candidates: int
    strict_kept_count: int
    strict_dropped_count: int
    fallback_applied: bool
    fallback_restored_count: int
    final_kept_count: int
    strict_dropped_candidates: list[dict[str, Any]]
    fallback_restored_candidates: list[dict[str, Any]]
    final_kept_candidates: list[dict[str, Any]]
    has_selection_output: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _normalize_candidate_payload(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    candidate = item.get("candidate")
    if isinstance(candidate, dict):
        return candidate
    return item


def _candidate_identity(item: Any) -> IdentityKey | None:
    candidate = _normalize_candidate_payload(item)
    symbol = str(candidate.get("symbol", "")).strip().upper()
    strategy = str(candidate.get("strategy", "")).strip().lower()
    horizon = str(candidate.get("horizon", "")).strip().lower()

    if not symbol or not strategy or not horizon:
        return None

    return IdentityKey(symbol=symbol, strategy=strategy, horizon=horizon)


def _load_jsonl_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _extract_quality_gate_block(row: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    mapper_payload = row.get("edge_selection_mapper_payload")
    if not isinstance(mapper_payload, dict):
        return None, False

    gate = mapper_payload.get("candidate_quality_gate")
    if isinstance(gate, dict):
        explicit_keys = {
            "total_candidates",
            "strict_kept_count",
            "strict_dropped_count",
            "fallback_applied",
            "fallback_restored_count",
            "final_kept_count",
        }
        if explicit_keys.intersection(gate.keys()):
            return gate, False

        compatibility_keys = {"kept_count", "dropped_count", "dropped_candidates"}
        if compatibility_keys.intersection(gate.keys()):
            return gate, True

    return None, False


def _parse_quality_gate_row(row: dict[str, Any], row_index: int) -> ParsedQualityGateRow | None:
    gate, compatibility_only = _extract_quality_gate_block(row)
    if gate is None:
        return None

    if compatibility_only:
        total_candidates = _safe_int(gate.get("kept_count")) + _safe_int(gate.get("dropped_count"))
        strict_kept_count = _safe_int(gate.get("kept_count"))
        strict_dropped_count = _safe_int(gate.get("dropped_count"))
        fallback_applied = False
        fallback_restored_count = 0
        final_kept_count = strict_kept_count
        strict_dropped_candidates = list(gate.get("dropped_candidates") or [])
        fallback_restored_candidates: list[dict[str, Any]] = []
        final_kept_candidates: list[dict[str, Any]] = []
    else:
        total_candidates = _safe_int(gate.get("total_candidates"))
        strict_kept_count = _safe_int(gate.get("strict_kept_count"))
        strict_dropped_count = _safe_int(gate.get("strict_dropped_count"))
        fallback_applied = _safe_bool(gate.get("fallback_applied"))
        fallback_restored_count = _safe_int(gate.get("fallback_restored_count"))
        final_kept_count = _safe_int(gate.get("final_kept_count"))
        strict_dropped_candidates = list(gate.get("strict_dropped_candidates") or [])
        fallback_restored_candidates = list(gate.get("fallback_restored_candidates") or [])
        final_kept_candidates = list(gate.get("final_kept_candidates") or [])

    return ParsedQualityGateRow(
        row_index=row_index,
        used_compatibility_fields=compatibility_only,
        total_candidates=total_candidates,
        strict_kept_count=strict_kept_count,
        strict_dropped_count=strict_dropped_count,
        fallback_applied=fallback_applied,
        fallback_restored_count=fallback_restored_count,
        final_kept_count=final_kept_count,
        strict_dropped_candidates=strict_dropped_candidates,
        fallback_restored_candidates=fallback_restored_candidates,
        final_kept_candidates=final_kept_candidates,
        has_selection_output=isinstance(row.get("edge_selection_output"), dict),
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _mean_int(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _sorted_counter_items(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    items = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        items.append(
            {
                "reason": key,
                "count": count,
                "ratio": _ratio(count, total),
            }
        )
    return items


def _infer_strict_survivor_identities(parsed: ParsedQualityGateRow) -> set[IdentityKey]:
    final_identities = {
        identity
        for identity in (_candidate_identity(item) for item in parsed.final_kept_candidates)
        if identity is not None
    }
    restored_identities = {
        identity
        for identity in (_candidate_identity(item) for item in parsed.fallback_restored_candidates)
        if identity is not None
    }
    return final_identities - restored_identities


def _build_identity_summary(parsed_rows: list[ParsedQualityGateRow]) -> list[dict[str, Any]]:
    stats: dict[IdentityKey, dict[str, int]] = defaultdict(
        lambda: {
            "rows_seen": 0,
            "strict_survived_rows": 0,
            "strict_dropped_rows": 0,
            "fallback_restored_rows": 0,
        }
    )

    for parsed in parsed_rows:
        row_seen_identities: set[IdentityKey] = set()
        strict_survivor_identities = _infer_strict_survivor_identities(parsed)

        dropped_identities = {
            identity
            for identity in (_candidate_identity(item) for item in parsed.strict_dropped_candidates)
            if identity is not None
        }
        restored_identities = {
            identity
            for identity in (_candidate_identity(item) for item in parsed.fallback_restored_candidates)
            if identity is not None
        }
        final_identities = {
            identity
            for identity in (_candidate_identity(item) for item in parsed.final_kept_candidates)
            if identity is not None
        }

        row_seen_identities |= strict_survivor_identities
        row_seen_identities |= dropped_identities
        row_seen_identities |= restored_identities
        row_seen_identities |= final_identities

        for identity in row_seen_identities:
            stats[identity]["rows_seen"] += 1

        for identity in strict_survivor_identities:
            stats[identity]["strict_survived_rows"] += 1

        for identity in dropped_identities:
            stats[identity]["strict_dropped_rows"] += 1

        for identity in restored_identities:
            stats[identity]["fallback_restored_rows"] += 1

    results: list[dict[str, Any]] = []
    for identity, counters in sorted(
        stats.items(),
        key=lambda item: (
            -item[1]["rows_seen"],
            item[0].symbol,
            item[0].strategy,
            item[0].horizon,
        ),
    ):
        rows_seen = counters["rows_seen"]
        strict_survived_rows = counters["strict_survived_rows"]
        results.append(
            {
                **identity.as_dict(),
                **counters,
                "strict_survival_rate": _ratio(strict_survived_rows, rows_seen),
            }
        )
    return results


def build_summary(parsed_rows: list[ParsedQualityGateRow], *, label: str = "full_sample") -> dict[str, Any]:
    rows_examined = len(parsed_rows)

    strict_pass_rows = sum(1 for row in parsed_rows if row.strict_kept_count > 0)
    strict_fail_rows = sum(1 for row in parsed_rows if row.strict_kept_count == 0)
    fallback_applied_rows = sum(1 for row in parsed_rows if row.fallback_applied)
    fallback_only_rows = sum(
        1
        for row in parsed_rows
        if row.strict_kept_count == 0 and row.fallback_restored_count > 0
    )
    mixed_rows = sum(
        1
        for row in parsed_rows
        if row.strict_kept_count > 0 and row.strict_dropped_count > 0
    )
    strict_full_drop_rows = sum(
        1
        for row in parsed_rows
        if row.strict_kept_count == 0 and row.strict_dropped_count > 0
    )
    rows_with_selection_output = sum(1 for row in parsed_rows if row.has_selection_output)
    compatibility_rows = sum(1 for row in parsed_rows if row.used_compatibility_fields)

    drop_reason_counter: Counter[str] = Counter()
    for row in parsed_rows:
        for dropped in row.strict_dropped_candidates:
            reason = str(dropped.get("reason", "UNKNOWN")).strip() or "UNKNOWN"
            drop_reason_counter[reason] += 1

    summary = {
        "label": label,
        "total_rows_examined": rows_examined,
        "rows_with_quality_gate": rows_examined,
        "rows_with_selection_output": rows_with_selection_output,
        "rows_using_compatibility_fields": compatibility_rows,
        "overall_counts": {
            "strict_pass_rows": strict_pass_rows,
            "strict_fail_rows": strict_fail_rows,
            "fallback_applied_rows": fallback_applied_rows,
            "fallback_only_rows": fallback_only_rows,
            "mixed_rows": mixed_rows,
            "strict_full_drop_rows": strict_full_drop_rows,
        },
        "aggregate_gate_metrics": {
            "avg_total_candidates": _mean_int([row.total_candidates for row in parsed_rows]),
            "avg_strict_kept_count": _mean_int([row.strict_kept_count for row in parsed_rows]),
            "avg_strict_dropped_count": _mean_int([row.strict_dropped_count for row in parsed_rows]),
            "avg_fallback_restored_count": _mean_int([row.fallback_restored_count for row in parsed_rows]),
            "avg_final_kept_count": _mean_int([row.final_kept_count for row in parsed_rows]),
        },
        "ratios": {
            "strict_pass_row_ratio": _ratio(strict_pass_rows, rows_examined),
            "strict_fail_row_ratio": _ratio(strict_fail_rows, rows_examined),
            "fallback_applied_ratio": _ratio(fallback_applied_rows, rows_examined),
            "fallback_only_ratio": _ratio(fallback_only_rows, rows_examined),
            "strict_full_drop_row_ratio": _ratio(strict_full_drop_rows, rows_examined),
        },
        "drop_reason_distribution": _sorted_counter_items(drop_reason_counter),
        "identity_level_summary": _build_identity_summary(parsed_rows),
    }
    return summary


def build_report(
    rows: list[dict[str, Any]],
    *,
    recent_rows: int | None = None,
    window_sizes: list[int] | None = None,
) -> dict[str, Any]:
    selected_rows = rows[-recent_rows:] if recent_rows and recent_rows > 0 else rows

    parsed_rows: list[ParsedQualityGateRow] = []
    missing_quality_gate_rows = 0
    for index, row in enumerate(selected_rows, start=1):
        parsed = _parse_quality_gate_row(row, row_index=index)
        if parsed is None:
            missing_quality_gate_rows += 1
            continue
        parsed_rows.append(parsed)

    default_windows = [25, 50, 100]
    effective_windows = window_sizes if window_sizes is not None else default_windows

    recent_windows: list[dict[str, Any]] = []
    for window_size in effective_windows:
        if window_size <= 0:
            continue
        window_rows = parsed_rows[-window_size:]
        if not window_rows:
            continue
        recent_windows.append(build_summary(window_rows, label=f"last_{len(window_rows)}"))

    report = {
        "generated_at": _utc_now_iso(),
        "input_rows_read": len(rows),
        "recent_rows_limit": recent_rows,
        "rows_after_recent_filter": len(selected_rows),
        "rows_missing_quality_gate": missing_quality_gate_rows,
        "summary": build_summary(parsed_rows),
        "recent_window_summaries": recent_windows,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    counts = summary["overall_counts"]
    metrics = summary["aggregate_gate_metrics"]
    ratios = summary["ratios"]
    drop_reasons = summary["drop_reason_distribution"][:10]
    identities = summary["identity_level_summary"][:20]

    lines: list[str] = []
    lines.append("# Strict Gate Fallback Observability Diagnosis")
    lines.append("")
    lines.append("## Run")
    lines.append(f"- generated_at: {report['generated_at']}")
    lines.append(f"- input_rows_read: {report['input_rows_read']}")
    lines.append(f"- recent_rows_limit: {report['recent_rows_limit']}")
    lines.append(f"- rows_after_recent_filter: {report['rows_after_recent_filter']}")
    lines.append(f"- rows_missing_quality_gate: {report['rows_missing_quality_gate']}")
    lines.append("")
    lines.append("## Overall Counts")
    lines.append(f"- total_rows_examined: {summary['total_rows_examined']}")
    lines.append(f"- rows_with_quality_gate: {summary['rows_with_quality_gate']}")
    lines.append(f"- rows_with_selection_output: {summary['rows_with_selection_output']}")
    lines.append(f"- rows_using_compatibility_fields: {summary['rows_using_compatibility_fields']}")
    lines.append(f"- strict_pass_rows: {counts['strict_pass_rows']}")
    lines.append(f"- strict_fail_rows: {counts['strict_fail_rows']}")
    lines.append(f"- fallback_applied_rows: {counts['fallback_applied_rows']}")
    lines.append(f"- fallback_only_rows: {counts['fallback_only_rows']}")
    lines.append(f"- mixed_rows: {counts['mixed_rows']}")
    lines.append(f"- strict_full_drop_rows: {counts['strict_full_drop_rows']}")
    lines.append("")
    lines.append("## Aggregate Gate Metrics")
    lines.append(f"- avg_total_candidates: {metrics['avg_total_candidates']}")
    lines.append(f"- avg_strict_kept_count: {metrics['avg_strict_kept_count']}")
    lines.append(f"- avg_strict_dropped_count: {metrics['avg_strict_dropped_count']}")
    lines.append(f"- avg_fallback_restored_count: {metrics['avg_fallback_restored_count']}")
    lines.append(f"- avg_final_kept_count: {metrics['avg_final_kept_count']}")
    lines.append("")
    lines.append("## Ratios")
    lines.append(f"- strict_pass_row_ratio: {ratios['strict_pass_row_ratio']}")
    lines.append(f"- strict_fail_row_ratio: {ratios['strict_fail_row_ratio']}")
    lines.append(f"- fallback_applied_ratio: {ratios['fallback_applied_ratio']}")
    lines.append(f"- fallback_only_ratio: {ratios['fallback_only_ratio']}")
    lines.append(f"- strict_full_drop_row_ratio: {ratios['strict_full_drop_row_ratio']}")
    lines.append("")
    lines.append("## Top Drop Reasons")
    if drop_reasons:
        for item in drop_reasons:
            lines.append(
                f"- reason={item['reason']}: count={item['count']}, ratio={item['ratio']}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Identity-Level Summary (Top 20 by rows_seen)")
    if identities:
        for item in identities:
            lines.append(
                "- "
                f"{item['symbol']} / {item['strategy']} / {item['horizon']}: "
                f"rows_seen={item['rows_seen']}, "
                f"strict_survived_rows={item['strict_survived_rows']}, "
                f"strict_dropped_rows={item['strict_dropped_rows']}, "
                f"fallback_restored_rows={item['fallback_restored_rows']}, "
                f"strict_survival_rate={item['strict_survival_rate']}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Recent Window Summaries")
    if report["recent_window_summaries"]:
        for window in report["recent_window_summaries"]:
            window_counts = window["overall_counts"]
            window_ratios = window["ratios"]
            lines.append(f"### {window['label']}")
            lines.append(f"- total_rows_examined: {window['total_rows_examined']}")
            lines.append(f"- strict_pass_rows: {window_counts['strict_pass_rows']}")
            lines.append(f"- strict_fail_rows: {window_counts['strict_fail_rows']}")
            lines.append(f"- fallback_applied_rows: {window_counts['fallback_applied_rows']}")
            lines.append(f"- fallback_only_rows: {window_counts['fallback_only_rows']}")
            lines.append(f"- strict_full_drop_rows: {window_counts['strict_full_drop_rows']}")
            lines.append(f"- strict_pass_row_ratio: {window_ratios['strict_pass_row_ratio']}")
            lines.append(f"- fallback_applied_ratio: {window_ratios['fallback_applied_ratio']}")
            lines.append("")
    else:
        lines.append("- none")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report_files(
    *,
    report: dict[str, Any],
    markdown: str,
    report_root: Path,
    write_latest_copy: bool,
) -> tuple[Path, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = report_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "strict_gate_fallback_observability_diagnosis.json"
    md_path = run_dir / "strict_gate_fallback_observability_diagnosis.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    if write_latest_copy:
        LATEST_DIR.mkdir(parents=True, exist_ok=True)
        (LATEST_DIR / LATEST_JSON_NAME).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (LATEST_DIR / LATEST_MD_NAME).write_text(markdown, encoding="utf-8")

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize strict gate survival and fallback dependency from trade_analysis JSONL rows."
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to trade_analysis JSONL input. Default: logs/trade_analysis.jsonl",
    )
    parser.add_argument(
        "--recent-rows",
        type=int,
        default=None,
        help="Limit analysis to the most recent N rows.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Also write JSON/Markdown copies to logs/research_reports/latest/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)

    rows = _load_jsonl_rows(input_path)
    report = build_report(rows, recent_rows=args.recent_rows)
    markdown = render_markdown(report)
    json_path, md_path = write_report_files(
        report=report,
        markdown=markdown,
        report_root=DEFAULT_REPORT_ROOT,
        write_latest_copy=args.write_latest_copy,
    )

    print(json.dumps(
        {
            "input_path": str(input_path.resolve()),
            "rows_read": len(rows),
            "json_report": str(json_path.resolve()),
            "markdown_report": str(md_path.resolve()),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()