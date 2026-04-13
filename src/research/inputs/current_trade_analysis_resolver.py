from __future__ import annotations

import re
from pathlib import Path

BASE_TRADE_ANALYSIS_FILENAME = "trade_analysis.jsonl"
_ROTATED_BASE_PATTERN = re.compile(r"^trade_analysis\.jsonl\.(\d+)(?:\.gz)?$")
_SYMBOL_SHARD_PATTERN = re.compile(r"^trade_analysis_([A-Za-z0-9]+)\.jsonl$")


def discover_current_trade_analysis_files(
    logs_dir: Path,
    *,
    include_rotated_base: bool,
) -> list[Path]:
    resolved_logs_dir = Path(logs_dir)
    if not resolved_logs_dir.exists() or not resolved_logs_dir.is_dir():
        return []

    rotated_base_paths: list[tuple[int, Path]] = []
    active_base_path: Path | None = None
    shard_paths: list[Path] = []

    for path in resolved_logs_dir.iterdir():
        if not path.is_file():
            continue

        if path.name == BASE_TRADE_ANALYSIS_FILENAME:
            active_base_path = path
            continue

        if include_rotated_base:
            rotation_index = _extract_rotation_index(path)
            if rotation_index is not None:
                rotated_base_paths.append((rotation_index, path))
                continue

        if _is_current_symbol_shard(path):
            shard_paths.append(path)

    rotated_base_paths.sort(key=lambda item: (-item[0], item[1].name))
    shard_paths.sort(key=lambda path: path.name.lower())

    source_files = [path for _, path in rotated_base_paths]
    if active_base_path is not None:
        source_files.append(active_base_path)
    source_files.extend(shard_paths)

    return source_files


def _extract_rotation_index(path: Path) -> int | None:
    match = _ROTATED_BASE_PATTERN.match(path.name)
    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_current_symbol_shard(path: Path) -> bool:
    match = _SYMBOL_SHARD_PATTERN.match(path.name)
    if not match:
        return False

    shard_name = match.group(1).lower()
    return shard_name != "cumulative"
