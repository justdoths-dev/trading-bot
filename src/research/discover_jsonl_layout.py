from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.inputs.jsonl_input_resolver import DEFAULT_LOGS_ROOT, discover_jsonl_layout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and classify JSONL files under logs/."
    )
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the discovery payload as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = discover_jsonl_layout(args.logs_root)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"logs_root={payload['logs_root']}")
    print(f"file_count={payload['file_count']}")
    print("")
    print("Classification Summary:")
    for row in payload["classification_summary"]:
        print(f"- {row['category']}: {row['count']}")
    print("")
    print("Files:")
    for file_info in payload["files"]:
        print(f"- relative_path={file_info['relative_path']}")
        print(f"  category={file_info['category']}")
        print(f"  line_count={file_info['line_count']}")
        print(f"  size_bytes={file_info['size_bytes']}")
        print(f"  modified_at={file_info['modified_at']}")
        print(f"  category_reason={file_info['category_reason']}")


if __name__ == "__main__":
    main()
