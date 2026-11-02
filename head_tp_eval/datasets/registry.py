"""Registry of trace datasets."""

from __future__ import annotations

from .base import Dataset

_DATASETS: dict[str, Dataset] = {}


def register_dataset(dataset: Dataset) -> None:
    existing = _DATASETS.get(dataset.name)
    if existing is not None and type(existing) is not type(dataset):
        raise ValueError(f"dataset {dataset.name!r} is already registered")
    _DATASETS[dataset.name] = dataset


def unregister_dataset(name: str) -> None:
    try:
        del _DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"dataset {name!r} is not registered") from exc


def get_dataset(name: str) -> Dataset:
    try:
        return _DATASETS[name]
    except KeyError as exc:
        known = ", ".join(dataset_names()) or "(none)"
        raise ValueError(f"unknown dataset {name!r}; known: {known}") from exc


def dataset_names() -> list[str]:
    return sorted(_DATASETS)
