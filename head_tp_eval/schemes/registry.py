"""Registry of eval run schemes."""

from __future__ import annotations

from .base import Scheme

_SCHEMES: dict[str, Scheme] = {}


def register_scheme(scheme: Scheme) -> None:
    existing = _SCHEMES.get(scheme.name)
    if existing is not None and type(existing) is not type(scheme):
        raise ValueError(f"scheme {scheme.name!r} is already registered")
    _SCHEMES[scheme.name] = scheme


def unregister_scheme(name: str) -> None:
    try:
        del _SCHEMES[name]
    except KeyError as exc:
        raise ValueError(f"scheme {name!r} is not registered") from exc


def get_scheme(name: str) -> Scheme:
    try:
        return _SCHEMES[name]
    except KeyError as exc:
        known = ", ".join(scheme_names()) or "(none)"
        raise ValueError(f"unknown scheme {name!r}; known: {known}") from exc


def scheme_names() -> list[str]:
    return sorted(_SCHEMES)
