"""Run-scheme plugins.

Importing this package loads every module next to it, other than ``base`` and
``registry``. A new scheme is a module that calls ``register_scheme``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .registry import get_scheme, register_scheme, scheme_names, unregister_scheme

__all__ = [
    "get_scheme",
    "register_scheme",
    "scheme_names",
    "unregister_scheme",
]


def _load_plugins() -> None:
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        if path.stem.startswith("_") or path.stem in {"base", "registry"}:
            continue
        importlib.import_module(f"{__name__}.{path.stem}")


_load_plugins()
