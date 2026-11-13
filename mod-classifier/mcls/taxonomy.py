"""Read configs/taxonomy.toml: the class set and the label of every file.

A file's label comes from its dataset's overrides (the first whose `match` is a
substring of the file name) or, failing that, the dataset default. Two optional
additions, both backwards compatible:

- `emitter_id` on an override: the transmitting device (e.g. the drone model)
  when the class is physical (`wifi`, `bluetooth`) but the emitter matters.
- `exclude = true` on an override, or `default_exclude = true` on a dataset:
  the file has no class in the current set and must not be used. `label_for`
  then returns a Label with `excluded=True` and label `EXCLUDED`, which is not
  a class, so a loader that forgets to check fails at manifest validation
  instead of training on it. An exclusion needs a `note` saying why.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from mcls.contracts import LABEL_STRENGTHS

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "taxonomy.toml"

#: Label of an excluded file. Never a class.
EXCLUDED = "excluded"


@dataclass(frozen=True)
class Label:
    label: str
    strength: str
    note: str | None = None
    emitter_id: str | None = None
    excluded: bool = False


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    license: str
    is_real: bool
    url: str


class Taxonomy:
    def __init__(self, data: dict):
        classes = data.get("classes", {})
        self.v1: tuple[str, ...] = tuple(classes.get("v1", ()))
        self.v2: tuple[str, ...] = tuple(classes.get("v2", ()))
        if not self.v1:
            raise ValueError("taxonomy: [classes] v1 is empty")
        overlap = set(self.v1) & set(self.v2)
        if overlap:
            raise ValueError(f"taxonomy: classes in both v1 and v2: {sorted(overlap)}")
        if EXCLUDED in self.all_classes:
            raise ValueError(f"taxonomy: {EXCLUDED!r} is reserved and cannot be a class")
        # Named subsets: every other key of [classes], a full class list drawn from v1 + v2.
        self.subsets: dict[str, tuple[str, ...]] = {}
        for key, value in classes.items():
            if key in ("v1", "v2"):
                continue
            subset = tuple(value)
            unknown = [c for c in subset if c not in self.all_classes]
            if not subset or unknown or len(set(subset)) != len(subset):
                raise ValueError(f"taxonomy: [classes] {key} must list distinct v1/v2 classes; unknown: {unknown}")
            self.subsets[key] = subset
        self._datasets = data.get("datasets", {})
        for name in self._datasets:
            self._check_dataset(name)

    @property
    def all_classes(self) -> tuple[str, ...]:
        return self.v1 + self.v2

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(self._datasets)

    @property
    def class_set_names(self) -> tuple[str, ...]:
        return ("v1", "v2", *self.subsets)

    def classes(self, version: str = "v1") -> tuple[str, ...]:
        """The class list a model trains on: "v1", "v2" for v1 plus v2, or a named subset."""
        if version == "v1":
            return self.v1
        if version == "v2":
            return self.all_classes
        if version in self.subsets:
            return self.subsets[version]
        raise ValueError(f"unknown class version {version!r} (known: {', '.join(self.class_set_names)})")

    def dataset(self, name: str) -> DatasetInfo:
        entry = self._entry(name)
        return DatasetInfo(name, entry["license"], bool(entry["is_real"]), entry.get("url", ""))

    def label_for(self, dataset: str, file_name: str) -> Label:
        """The label of one file: the first override whose `match` is in the name, else the default."""
        entry = self._entry(dataset)
        for override in entry.get("overrides", []):
            if override["match"] in file_name:
                if override.get("exclude", False):
                    return Label(EXCLUDED, "weak", override["note"], override.get("emitter_id"), excluded=True)
                return Label(override["label"], override["strength"], override.get("note"), override.get("emitter_id"))
        if entry.get("default_exclude", False):
            return Label(EXCLUDED, "weak", entry["note"], excluded=True)
        return Label(entry["default_label"], entry["default_strength"])

    def _entry(self, name: str) -> dict:
        if name not in self._datasets:
            raise KeyError(f"taxonomy has no dataset {name!r}")
        return self._datasets[name]

    def _check_dataset(self, name: str) -> None:
        entry = self._datasets[name]
        for key in ("license", "is_real"):
            if key not in entry:
                raise ValueError(f"taxonomy: dataset {name!r} is missing {key}")
        pairs = []
        if entry.get("default_exclude", False):
            if not entry.get("note"):
                raise ValueError(f"taxonomy: dataset {name!r} excludes its default without a note")
        else:
            for key in ("default_label", "default_strength"):
                if key not in entry:
                    raise ValueError(f"taxonomy: dataset {name!r} is missing {key}")
            pairs.append((entry["default_label"], entry["default_strength"]))
        for override in entry.get("overrides", []):
            if not override.get("match"):
                raise ValueError(f"taxonomy: dataset {name!r} has an override without `match`")
            if override.get("exclude", False):
                if not override.get("note"):
                    raise ValueError(f"taxonomy: dataset {name!r} excludes {override['match']!r} without a note")
                continue
            for key in ("label", "strength"):
                if key not in override:
                    raise ValueError(f"taxonomy: dataset {name!r} override {override['match']!r} is missing {key}")
            pairs.append((override["label"], override["strength"]))
        for label, strength in pairs:
            if label not in self.all_classes:
                raise ValueError(f"taxonomy: dataset {name!r} uses unknown class {label!r}")
            if strength not in LABEL_STRENGTHS:
                raise ValueError(f"taxonomy: dataset {name!r} uses unknown strength {strength!r}")


def load(path: Path = DEFAULT_PATH) -> Taxonomy:
    with open(path, "rb") as handle:
        return Taxonomy(tomllib.load(handle))
