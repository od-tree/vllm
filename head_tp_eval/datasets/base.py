"""Dataset plugin contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..trace import TraceRow


@dataclass(frozen=True)
class TraceBuild:
    """Rows plus whether the arrival process reused a finite timestamp list."""

    rows: list[TraceRow]
    cycled: bool = False


class Dataset(Protocol):
    """One way to build a timed trace.

    ``add_arguments`` flags must be unique to this dataset, because the trace
    command installs them only after ``--dataset`` selects the plugin.
    ``build`` accepts a plain mapping so a scheme can call it without a parser.
    """

    name: str

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add this dataset's flags to the trace command."""

    def build(self, params: Mapping[str, Any]) -> TraceBuild:
        """Return the trace described by ``params``."""
