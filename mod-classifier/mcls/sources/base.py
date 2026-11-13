"""The interface every dataset loader implements."""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from mcls.contracts import Recording
from mcls.taxonomy import Taxonomy, load

#: Raw downloads live here unless MCLS_RAW points elsewhere (a CRC node's /tmp).
DEFAULT_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def raw_root() -> Path:
    return Path(os.environ.get("MCLS_RAW", DEFAULT_RAW))


class Source(ABC):
    """One dataset.

    A loader knows its file format and nothing else. It yields `Recording`s
    whose `read(start, count)` returns complex64 samples at the native rate;
    resampling, burst finding and windowing belong to the front end.
    """

    #: Name used in the taxonomy, the manifest and on the command line.
    name: str = ""

    def __init__(self, root: Path | None = None, taxonomy: Taxonomy | None = None):
        self.root = Path(root) if root is not None else raw_root() / self.name
        self.taxonomy = taxonomy or load()

    @abstractmethod
    def recordings(self) -> Iterator[Recording]:
        """Every recording found under `self.root`, each one validated."""

    def check(self) -> list[str]:
        """Problems that stop this source being used (missing files, bad sizes). Empty when ready."""
        return [] if self.root.exists() else [f"{self.root} does not exist"]


REGISTRY: dict[str, type[Source]] = {}

#: Loader modules under mcls/sources/; importing one registers its Source(s).
LOADERS = ("tampere", "dronerf", "nist_wifi_bt", "sigmf", "gateman", "triangular", "drexel", "rtl433")


def register(cls: type[Source]) -> type[Source]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name")
    REGISTRY[cls.name] = cls
    return cls


def get(name: str, **kwargs) -> Source:
    # Import the loaders so they register themselves.
    for module in LOADERS:
        importlib.import_module(f"mcls.sources.{module}")

    if name not in REGISTRY:
        raise KeyError(f"no source named {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
