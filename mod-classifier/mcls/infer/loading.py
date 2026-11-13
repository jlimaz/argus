"""Open a trained model as a `Classifier`, without importing torch until needed.

`path` may be:

- a run folder (`runs/<name>/`): `checkpoint.pt` (Engineer 7, neural) is used
  when present, else `model.pkl` (Engineer 6, baselines);
- a `.pt` checkpoint: `mcls.models.classifier.load_classifier` (needs torch);
- a `.pkl` baseline: `mcls.baselines.load` on its folder.

The calibration.json for a model sits in the same run folder; see
`calibration_for`.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcls.contracts import Classifier


def _check(classifier, path: Path) -> Classifier:
    if not isinstance(classifier, Classifier) or not classifier.classes:
        raise TypeError(f"{path}: the loaded model does not offer .classes and .classify (mcls.contracts.Classifier)")
    return classifier


def load_classifier(path: str | Path, device: str = "cpu", calibration: str | Path | None = None) -> Classifier:
    """The model at `path`. `calibration` is passed to Engineer 7's loader, so its `classify` is calibrated
    too; `mcls.infer` itself uses only `logits` and applies the temperature once."""
    path = Path(path)
    if path.is_dir():
        if (path / "checkpoint.pt").is_file():
            path = path / "checkpoint.pt"
        elif (path / "model.pkl").is_file():
            path = path / "model.pkl"
        else:
            raise FileNotFoundError(f"{path} holds neither checkpoint.pt nor model.pkl")
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix == ".pt":
        try:
            from mcls.models.classifier import load_classifier as load_checkpoint  # Engineer 7; imports torch
        except ImportError as err:
            raise ImportError(f"{path.name} is a PyTorch checkpoint; this needs torch and mcls.models ({err})") from err
        kwargs = {"device": device}
        if calibration is not None:
            kwargs["calibration"] = Path(calibration)
        return _check(load_checkpoint(path, **kwargs), path)
    if path.suffix == ".pkl":
        from mcls.baselines import load as load_baseline  # Engineer 6

        return _check(load_baseline(path.parent), path)
    raise ValueError(f"{path.name}: expected a run folder, a .pt checkpoint or a model.pkl")


def calibration_for(model_path: str | Path) -> Path | None:
    """The calibration.json next to a model (its run folder), or None."""
    path = Path(model_path)
    folder = path if path.is_dir() else path.parent
    candidate = folder / "calibration.json"
    return candidate if candidate.is_file() else None


def run_identity(model_path: str | Path) -> tuple[str | None, str | None]:
    """(run folder name, manifest_sha256) of the run a model belongs to, from its meta.json; Nones if absent."""
    path = Path(model_path)
    folder = path if path.is_dir() else path.parent
    meta = folder / "meta.json"
    if not meta.is_file():
        return None, None
    data = json.loads(meta.read_text())
    return folder.resolve().name, data.get("manifest_sha256")
