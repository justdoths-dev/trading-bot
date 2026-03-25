from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.inputs.jsonl_input_resolver import (
    DEFAULT_LOGS_ROOT,
    REPO_ROOT,
    VALID_MODES,
    resolve_jsonl_inputs,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "research_reports" / "latest"


def create_jsonl_snapshot_manifest(
    mode: str,
    *,
    logs_root: Path | None = None,
    output_path: Path | None = None,
    include_archive_candidates: bool = False,
    include_current_with_historical: bool = False,
) -> dict[str, Any]:
    selection = resolve_jsonl_inputs(
        mode,
        logs_root=logs_root or DEFAULT_LOGS_ROOT,
        include_archive_candidates=include_archive_candidates,
        include_current_with_historical=include_current_with_historical,
    )

    files: list[dict[str, Any]] = []
    total_lines = 0
    for raw_path in selection["included_files"]:
        path = Path(raw_path)
        line_count = _count_jsonl_lines(path)
        total_lines += line_count
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "file_size_bytes": stat.st_size,
                "line_count": line_count,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            }
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": selection["mode"],
        "logs_root": selection["logs_root"],
        "file_count": len(files),
        "include_archive_candidates": selection["include_archive_candidates"],
        "include_current_with_historical": selection["include_current_with_historical"],
        "included_relative_paths": list(selection["included_relative_paths"]),
        "decision_summary": list(selection["decision_summary"]),
        "file_list": [file_info["path"] for file_info in files],
        "files": files,
        "total_lines": total_lines,
    }

    final_output = output_path or (
        DEFAULT_OUTPUT_DIR / f"jsonl_snapshot_manifest_{selection['mode']}.json"
    )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "manifest": manifest,
        "output_path": str(final_output),
        "selection": selection,
    }


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible JSONL snapshot manifest for rerun preparation."
    )
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--output", type=Path, default=None)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = create_jsonl_snapshot_manifest(
        args.mode,
        logs_root=args.logs_root,
        output_path=args.output,
        include_archive_candidates=args.include_archives,
        include_current_with_historical=args.include_current_with_historical,
    )
    print(result["output_path"])


if __name__ == "__main__":
    main()
