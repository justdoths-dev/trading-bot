from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_LOGS_ROOT",
    "VALID_MODES",
    "classify_jsonl_file",
    "discover_jsonl_layout",
    "load_resolved_jsonl_records",
    "render_resolution_text",
    "resolve_jsonl_inputs",
    "create_jsonl_snapshot_manifest",
]


def __getattr__(name: str) -> Any:
    if name == "create_jsonl_snapshot_manifest":
        module = import_module("src.research.inputs.jsonl_snapshot_manifest")
        return getattr(module, name)

    if name in {
        "DEFAULT_LOGS_ROOT",
        "VALID_MODES",
        "classify_jsonl_file",
        "discover_jsonl_layout",
        "load_resolved_jsonl_records",
        "render_resolution_text",
        "resolve_jsonl_inputs",
    }:
        module = import_module("src.research.inputs.jsonl_input_resolver")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
