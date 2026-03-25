from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOGS_ROOT = REPO_ROOT / "logs"
VALID_MODES = ("current", "historical", "experiment")
PRIMARY_CURRENT_RELATIVE_PATH = "trade_analysis.jsonl"
PRIMARY_CUMULATIVE_RELATIVE_PATH = "trade_analysis_cumulative.jsonl"


def discover_jsonl_layout(
    logs_root: Path | None = None,
) -> dict[str, Any]:
    """Discover and classify JSONL files under logs/ with deterministic ordering."""
    resolved_logs_root = Path(logs_root) if logs_root is not None else DEFAULT_LOGS_ROOT
    files = sorted(
        resolved_logs_root.rglob("*.jsonl") if resolved_logs_root.exists() else [],
        key=lambda path: _relative_path(path, resolved_logs_root),
    )

    discovered_files: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    for path in files:
        classification = classify_jsonl_file(path, resolved_logs_root)
        classification_counts[classification["category"]] += 1
        stat = path.stat()
        discovered_files.append(
            {
                "path": str(path),
                "relative_path": _relative_path(path, resolved_logs_root),
                "category": classification["category"],
                "category_reason": classification["reason"],
                "line_count": _count_jsonl_lines(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "logs_root": str(resolved_logs_root),
        "file_count": len(discovered_files),
        "classification_summary": [
            {"category": category, "count": count}
            for category, count in sorted(classification_counts.items(), key=lambda item: item[0])
        ],
        "files": discovered_files,
    }


def classify_jsonl_file(path: Path, logs_root: Path | None = None) -> dict[str, str]:
    resolved_logs_root = Path(logs_root) if logs_root is not None else DEFAULT_LOGS_ROOT
    relative_path = _relative_path(path, resolved_logs_root)

    if relative_path == PRIMARY_CURRENT_RELATIVE_PATH:
        return {
            "category": "primary_production_current",
            "reason": "active production trade-analysis file",
        }
    if relative_path == PRIMARY_CUMULATIVE_RELATIVE_PATH:
        return {
            "category": "primary_production_cumulative",
            "reason": "canonical cumulative production trade-analysis file",
        }
    if relative_path.startswith("archive/"):
        return {
            "category": "archive_historical_candidate",
            "reason": "historical archive trade-analysis candidate",
        }
    if relative_path.startswith("backups/"):
        return {
            "category": "backup_snapshot_candidate",
            "reason": "backup snapshot excluded by default to avoid double counting",
        }
    if relative_path.startswith("experiments/"):
        return {
            "category": "experiment_input",
            "reason": "experiment input file",
        }
    if relative_path.startswith("research_reports/"):
        return {
            "category": "derived_report_artifact",
            "reason": "report or derived research artifact",
        }
    if relative_path.startswith("edge_selection_shadow/"):
        return {
            "category": "derived_shadow_output",
            "reason": "derived edge-selection shadow output",
        }
    return {
        "category": "unclassified_jsonl",
        "reason": "unclassified JSONL outside the known raw trade-analysis layout",
    }


def resolve_jsonl_inputs(
    mode: str,
    *,
    logs_root: Path | None = None,
    include_archive_candidates: bool = False,
    include_current_with_historical: bool = False,
) -> dict[str, Any]:
    normalized_mode = _normalize_mode(mode)
    discovery = discover_jsonl_layout(logs_root)
    discovery_relative_paths = {str(item["relative_path"]) for item in discovery["files"]}

    decisions: list[dict[str, Any]] = []
    for file_info in discovery["files"]:
        decision = _decision_for_file(
            file_info,
            normalized_mode,
            include_archive_candidates=include_archive_candidates,
            include_current_with_historical=include_current_with_historical,
            discovery_relative_paths=discovery_relative_paths,
        )
        decisions.append({**file_info, **decision})

    included = [item for item in decisions if item["decision"] == "included"]
    included.sort(key=lambda item: _included_sort_key(item, normalized_mode))

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": normalized_mode,
        "logs_root": discovery["logs_root"],
        "include_archive_candidates": include_archive_candidates,
        "include_current_with_historical": include_current_with_historical,
        "record_ordering": "file_order_only",
        "included_files": [item["path"] for item in included],
        "included_relative_paths": [item["relative_path"] for item in included],
        "latest_file": _latest_file(included),
        "decision_summary": [
            {
                "decision": decision,
                "count": sum(1 for item in decisions if item["decision"] == decision),
            }
            for decision in ("included", "excluded")
        ],
        "decisions": decisions,
    }


def load_resolved_jsonl_records(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Load records using deterministic file-order only.

    This intentionally does not attempt record-level timestamp ordering or dedup.
    Future extensions can add timestamp-aware merging after overlap validation exists.
    """
    records: list[dict[str, Any]] = []
    for path in selection.get("included_files", []):
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                content = line.strip()
                if not content:
                    continue
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in {file_path} line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Expected JSON object in {file_path} line {line_number}"
                    )
                records.append(parsed)
    return records


def render_resolution_text(selection: dict[str, Any]) -> str:
    lines = [
        f"mode={selection.get('mode')}",
        f"logs_root={selection.get('logs_root')}",
        f"include_archive_candidates={selection.get('include_archive_candidates')}",
        f"include_current_with_historical={selection.get('include_current_with_historical')}",
        f"record_ordering={selection.get('record_ordering')}",
        f"latest_file={selection.get('latest_file') or 'n/a'}",
        "",
        "Decision Summary:",
    ]
    for row in selection.get("decision_summary", []):
        lines.append(f"- {row.get('decision')}: {row.get('count')}")
    lines.append("")
    lines.append("Decisions:")
    for decision in selection.get("decisions", []):
        lines.append(
            f"- [{decision.get('decision')}] {decision.get('relative_path')} "
            f"({decision.get('category')}): {decision.get('decision_reason')}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _decision_for_file(
    file_info: dict[str, Any],
    mode: str,
    *,
    include_archive_candidates: bool,
    include_current_with_historical: bool,
    discovery_relative_paths: set[str],
) -> dict[str, str]:
    relative_path = str(file_info["relative_path"])
    category = str(file_info["category"])

    if category in {"derived_report_artifact", "derived_shadow_output"}:
        return {
            "decision": "excluded",
            "decision_reason": "derived artifact excluded from raw input resolution",
        }
    if category == "backup_snapshot_candidate":
        return {
            "decision": "excluded",
            "decision_reason": "backup snapshot excluded by default to avoid overlap and double counting",
        }
    if category == "archive_historical_candidate":
        if mode == "historical" and include_archive_candidates:
            return {
                "decision": "included",
                "decision_reason": "explicit historical archive inclusion requested",
            }
        return {
            "decision": "excluded",
            "decision_reason": "archive candidate surfaced for inspection but excluded from default resolution",
        }
    if category == "experiment_input":
        if mode == "experiment":
            return {
                "decision": "included",
                "decision_reason": "experiment mode includes experiment inputs",
            }
        return {
            "decision": "excluded",
            "decision_reason": "experiment inputs excluded outside experiment mode",
        }
    if category == "primary_production_current":
        if mode == "current":
            return {
                "decision": "included",
                "decision_reason": "current mode prefers the active production trade-analysis file",
            }
        if mode == "historical":
            if include_current_with_historical:
                return {
                    "decision": "included",
                    "decision_reason": "explicit historical inclusion requested for current production file; dedup and overlap validation are not yet applied",
                }
            if PRIMARY_CUMULATIVE_RELATIVE_PATH in discovery_relative_paths:
                return {
                    "decision": "excluded",
                    "decision_reason": "excluded by default because overlap with cumulative is assumed but not yet validated",
                }
            return {
                "decision": "included",
                "decision_reason": "historical mode fallback because cumulative production file is unavailable",
            }
        return {
            "decision": "excluded",
            "decision_reason": "experiment mode excludes active production inputs by default",
        }
    if category == "primary_production_cumulative":
        if mode == "historical":
            return {
                "decision": "included",
                "decision_reason": "historical mode prefers the canonical cumulative production source",
            }
        return {
            "decision": "excluded",
            "decision_reason": "cumulative production file excluded outside historical mode to avoid overlap",
        }
    return {
        "decision": "excluded",
        "decision_reason": "unclassified JSONL excluded from safe default resolution",
    }


def _included_sort_key(file_info: dict[str, Any], mode: str) -> tuple[int, str]:
    relative_path = str(file_info["relative_path"])
    if mode == "current":
        priority = 0 if relative_path == PRIMARY_CURRENT_RELATIVE_PATH else 1
    elif mode == "historical":
        if relative_path == PRIMARY_CUMULATIVE_RELATIVE_PATH:
            priority = 0
        elif relative_path == PRIMARY_CURRENT_RELATIVE_PATH:
            priority = 1
        else:
            priority = 2
    else:
        priority = 0
    return (priority, relative_path)


def _latest_file(included: list[dict[str, Any]]) -> str | None:
    if not included:
        return None
    latest = max(
        included,
        key=lambda item: (item["modified_at"], item["relative_path"]),
    )
    return str(latest["path"])


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _relative_path(path: Path, logs_root: Path) -> str:
    return path.relative_to(logs_root).as_posix()


def _normalize_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError(
            f"Unsupported mode: {value}. Expected one of {', '.join(VALID_MODES)}"
        )
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve reproducible JSONL inputs for historical rerun preparation."
    )
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument(
        "--include-archives",
        action="store_true",
        help="Include logs/archive/** in historical mode. Disabled by default for safety.",
    )
    parser.add_argument(
        "--include-current-with-historical",
        action="store_true",
        help="Include logs/trade_analysis.jsonl alongside cumulative in historical mode without dedup.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full resolver payload as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selection = resolve_jsonl_inputs(
        args.mode,
        logs_root=args.logs_root,
        include_archive_candidates=args.include_archives,
        include_current_with_historical=args.include_current_with_historical,
    )
    if args.json:
        print(json.dumps(selection, indent=2, ensure_ascii=False))
    else:
        print(render_resolution_text(selection), end="")


if __name__ == "__main__":
    main()
