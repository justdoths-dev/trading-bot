from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any


def reexport_module(namespace: dict[str, Any], target: str) -> ModuleType:
    module = import_module(target)
    module_name = namespace.get("__name__")
    if isinstance(module_name, str) and module_name != "__main__":
        sys.modules[module_name] = module

    skipped_names = {
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }
    exported_names = [name for name in vars(module) if name not in skipped_names]

    for name in exported_names:
        namespace[name] = getattr(module, name)

    namespace["__doc__"] = module.__doc__
    namespace["__all__"] = list(getattr(module, "__all__", exported_names))
    return module


def run_module_main(target: str) -> None:
    module = import_module(target)
    main = getattr(module, "main", None)
    if callable(main):
        main()
