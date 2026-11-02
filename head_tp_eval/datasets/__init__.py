"""Dataset plugins.

Importing this package loads every module next to it, other than ``base`` and
``registry``. A new dataset is a module that calls ``register_dataset``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .registry import (
    dataset_names,
    get_dataset,
    register_dataset,
    unregister_dataset,
)

__all__ = [
    "dataset_names",
    "get_dataset",
    "register_dataset",
    "unregister_dataset",
]


def _load_plugins() -> None:
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        if path.stem.startswith("_") or path.stem in {"base", "registry"}:
            continue
        importlib.import_module(f"{__name__}.{path.stem}")


_load_plugins()
