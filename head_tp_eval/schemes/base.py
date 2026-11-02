"""Scheme plugin contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RunSpec:
    """One client run against an already-started server configuration."""

    config: str
    dataset: str
    dataset_args: dict[str, Any]
    result_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_args", dict(self.dataset_args))

    def as_params(self) -> Mapping[str, Any]:
        return self.dataset_args


class Scheme(Protocol):
    """Expand one invocation into the runs that make up an experiment."""

    name: str

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add flags that exist only for this scheme."""

    def expand(self, args: argparse.Namespace) -> list[RunSpec]:
        """Return the runs selected by ``args``."""
