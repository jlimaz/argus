"""Classify the bursts in an IQ recording with every wave 5 model, and pick one model per burst.

Same code path as training: bursts from `pipeline.features.detect_bursts`, windows
from `mcls.infer.core.burst_windows` (the front end that cut the training windows),
per-model logits, each model's own temperature and unknown threshold
(`calibration.json`), averaged over the burst's windows (`mcls.infer.core.aggregate`).

Model choice per burst: `runs/crc/routing_w5.json` (scripts/build_routing.py), which
maps a band to the model with the best validation accuracy there. The burst's band
comes from its measured centre frequency, which is known before any model runs.

The 2D CNN runs from `model.onnx` with onnxruntime, so no torch is needed;
kNN and LightGBM load from their `model.pkl` (scikit-learn / lightgbm pinned to the
versions they were trained with). Band priors are not applied, to match how the
routing was scored.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UI_DIR = Path(__file__).resolve().parent
ROOT = UI_DIR.parent  # mod-classifier
ARGUS = ROOT.parent  # the repo, for pipeline/
for path in (str(ROOT), str(ARGUS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from mcls.contracts import WINDOW_SAMPLES, band_of  # noqa: E402
from mcls.infer.calibration import load_calibration  # noqa: E402
from mcls.infer.core import UNKNOWN, aggregate, burst_windows, window_logits  # noqa: E402
from pipeline.features import detect_bursts  # noqa: E402

RUNS = ROOT / "runs" / "crc"
ROUTING = RUNS / "routing_w5.json"
#: The UI reads at most this many samples (a longer file is cut, and says so).
MAX_SAMPLES = 40_000_000
#: Bursts classified per file, in time order (the rest are listed as skipped).
MAX_BURSTS = 40
FORMATS = {
    "cu8": (np.uint8, 127.5, 127.5),
    "ci8": (np.int8, 0.0, 128.0),
    "ci16_le": (np.dtype("<i2"), 0.0, 32768.0),
    "ci16_be": (np.dtype(">i2"), 0.0, 32768.0),
    "cf32_le": (np.dtype("<f4"), 0.0, 1.0),
}
SUFFIX_FORMATS = {".cu8": "cu8", ".cs8": "ci8", ".ci8": "ci8", ".cs16": "ci16_le", ".ci16": "ci16_le",
                  ".cf32": "cf32_le", ".fc32": "cf32_le"}
SIGMF_TYPES = {"cu8": "cu8", "cu8_le": "cu8", "ci8": "ci8", "ci8_le": "ci8", "ci16_le": "ci16_le",
               "ci16_be": "ci16_be", "cf32_le": "cf32_le"}


class InputError(ValueError):
    """A problem with the uploaded recording, worded for the person who uploaded it."""


class OnnxClassifier:
    """The 2D CNN exported by scripts/export_onnx.py: (n, 4096) complex windows -> (n, K) logits."""

    def __init__(self, run: Path):
        import onnxruntime as ort

        info = json.loads((run / "model.onnx.json").read_text())
        self.classes = tuple(info["classes"])
        self.session = ort.InferenceSession(str(run / "model.onnx"), providers=["CPUExecutionProvider"])

    def logits(self, windows: np.ndarray) -> np.ndarray:
        x = np.stack([windows.real, windows.imag], axis=1).astype(np.float32)
        out = [self.session.run(["logits"], {"iq": x[i : i + 256]})[0] for i in range(0, len(x), 256)]
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), dtype=np.float32)

    def classify(self, windows: np.ndarray) -> np.ndarray:
        z = self.logits(windows)
        z = z - z.max(axis=1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


@dataclass
class Model:
    name: str
    label: str
    classifier: object
    temperature: float
    unknown_threshold: float | None


@dataclass
class Bank:
    """Every model the UI runs, plus the routing that picks one per burst."""

    models: list[Model]
    routing: dict
    classes: tuple[str, ...] = field(default=())

    def by_name(self, name: str) -> Model:
        return next(m for m in self.models if m.name == name)


LABELS = {"cnn2d_v1_w5": "2D CNN", "baseline_knn_v1_w5": "kNN", "baseline_lightgbm_v1_w5": "LightGBM"}


def load_bank(routing_path: Path = ROUTING, runs: Path = RUNS) -> Bank:
    if not routing_path.is_file():
        raise FileNotFoundError(f"{routing_path} is missing: run scripts/build_routing.py first")
    routing = json.loads(routing_path.read_text())
    models = []
    for entry in routing["models"]:
        run = runs / entry["name"]
        if (run / "model.onnx").is_file():
            classifier = OnnxClassifier(run)
        else:
            from mcls.baselines import load as load_baseline

            classifier = load_baseline(run)
        calibration = load_calibration(run / "calibration.json", tuple(classifier.classes))
        models.append(Model(entry["name"], LABELS.get(entry["name"], entry["name"]), classifier,
                            calibration.temperature, calibration.unknown_threshold))
    classes = tuple(models[0].classifier.classes)
    for model in models[1:]:
        if tuple(model.classifier.classes) != classes:
            raise ValueError(f"{model.name}: class order differs from {models[0].name}")
    return Bank(models, routing, classes)


# ---------------------------------------------------------------------------
# Reading the upload
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    iq: np.ndarray
    sample_rate_hz: float
    center_frequency_hz: float
    source: str
    datatype: str
    cut: bool  # True when the file was longer than MAX_SAMPLES


def to_complex(raw: bytes, datatype: str, max_samples: int = MAX_SAMPLES) -> tuple[np.ndarray, bool]:
    if datatype not in FORMATS:
        raise InputError(f"Sample format {datatype!r} is not supported. Use one of {', '.join(FORMATS)}.")
    dtype, offset, scale = FORMATS[datatype]
    values = np.frombuffer(raw, dtype=dtype)
    values = values[: len(values) - len(values) % 2]
    cut = len(values) // 2 > max_samples
    values = values[: 2 * max_samples].astype(np.float32)
    values = (values - offset) / scale
    if not np.isfinite(values).all():
        raise InputError("The file holds values that are not finite numbers: is the sample format right?")
    return (values[0::2] + 1j * values[1::2]).astype(np.complex64), cut


def read_upload(files: dict[str, bytes], sample_rate_hz: float | None, center_frequency_hz: float | None,
                datatype: str | None = None) -> Recording:
    """`files`: name -> bytes. One raw file (.cu8 / .cs8 / .cs16 / .cf32), or a .sigmf-meta + .sigmf-data pair."""
    metas = [n for n in files if n.endswith(".sigmf-meta")]
    datas = [n for n in files if n.endswith(".sigmf-data")]
    if metas or datas:
        if len(metas) != 1 or len(datas) != 1:
            raise InputError("A SigMF recording needs exactly one .sigmf-meta and one .sigmf-data file.")
        try:
            meta = json.loads(files[metas[0]])
        except json.JSONDecodeError as exc:
            raise InputError(f"{metas[0]} is not valid JSON: {exc.msg}.") from exc
        info = meta.get("global") or {}
        kind = SIGMF_TYPES.get(str(info.get("core:datatype")))
        if kind is None:
            raise InputError(f"SigMF datatype {info.get('core:datatype')!r} is not supported.")
        rate = sample_rate_hz or info.get("core:sample_rate")
        captures = meta.get("captures") or [{}]
        centre = center_frequency_hz if center_frequency_hz is not None else captures[0].get("core:frequency")
        if centre is not None and 0 < float(centre) < 1e5:
            centre = float(centre) * 1e6  # some datasets (LoRaIQ) write MHz where SigMF says Hz
        iq, cut = to_complex(files[datas[0]], kind)
        name, datatype = datas[0], kind
    else:
        if len(files) != 1:
            raise InputError("Upload one raw IQ file, or a .sigmf-meta and .sigmf-data pair.")
        name, raw = next(iter(files.items()))
        datatype = datatype or SUFFIX_FORMATS.get(Path(name).suffix.lower())
        if datatype is None:
            raise InputError(f"{name}: the file ending does not say the sample format; choose it in the form.")
        rate, centre = sample_rate_hz, center_frequency_hz
        iq, cut = to_complex(raw, datatype)
    if not rate or float(rate) <= 0:
        raise InputError("The sample rate is needed (raw files do not record it).")
    if centre is None or float(centre) < 0:
        raise InputError("The centre frequency is needed (raw files do not record it).")
    if len(iq) < WINDOW_SAMPLES:
        raise InputError(f"The recording is only {len(iq)} samples long; at least {WINDOW_SAMPLES} are needed.")
    return Recording(iq, float(rate), float(centre), name, datatype, cut)


# ---------------------------------------------------------------------------
# Classifying
# ---------------------------------------------------------------------------


def route_for(bank: Bank, band: str) -> tuple[str, str]:
    """(model name, why) for a band, from the routing table; the fallback model when the band was never seen."""
    table = bank.routing["routes"]["band"]
    fallback = bank.routing["fallback_model"]
    if band in table:
        entry = table[band]
        acc = entry["val_accuracy"][entry["model"]]
        return entry["model"], (f"{entry['reason']} for band {band}: {acc:.1%} on {entry['val_windows']:,} "
                                f"validation windows from {entry['val_recordings']} recordings")
    return fallback, f"band {band} had no validation data, so the overall best model ({LABELS.get(fallback, fallback)})"


def classify(bank: Bank, rec: Recording, max_bursts: int = MAX_BURSTS) -> dict:
    bursts = detect_bursts(rec.iq, rec.sample_rate_hz, rec.center_frequency_hz)
    rows = []
    for index, burst in enumerate(bursts[:max_bursts]):
        windows = burst_windows(rec.iq, rec.sample_rate_hz, rec.center_frequency_hz, burst)
        always_on = False
        if len(windows) == 0:
            # The activity test rejects every window of an always-on signal (a jammer); retry without it.
            windows = burst_windows(rec.iq, rec.sample_rate_hz, rec.center_frequency_hz, burst, min_active=0.0)
            always_on = len(windows) > 0
        band = band_of(burst["center_frequency_hz"])
        chosen, why = route_for(bank, band)
        per_model = {}
        for model in bank.models:
            if len(windows):
                logits = window_logits(model.classifier, windows)
            else:
                logits = np.zeros((0, len(bank.classes)))
            result = aggregate(logits, bank.classes, temperature=model.temperature,
                               unknown_threshold=model.unknown_threshold)
            per_model[model.name] = {"label": model.label, "class": result.signal_class, "confidence": result.confidence,
                                     "probabilities": result.probs}
        final = per_model[chosen]
        answers = {v["class"] for v in per_model.values()}
        rows.append({
            "index": index,
            "start_s": burst["start_s"],
            "end_s": burst["end_s"],
            "center_frequency_hz": burst["center_frequency_hz"],
            "bandwidth_hz": burst["bandwidth_hz"],
            "snr_db": burst["snr_db"],
            "band": band,
            "windows": int(len(windows)),
            "always_on": always_on,
            "chosen_model": chosen,
            "chosen_label": LABELS.get(chosen, chosen),
            "why": why,
            "class": final["class"] if len(windows) else UNKNOWN,
            "confidence": final["confidence"],
            "models": per_model,
            "agreement": "all agree" if len(answers) == 1 else "models disagree: check by hand",
        })
    return {
        "source": rec.source,
        "datatype": rec.datatype,
        "sample_rate_hz": rec.sample_rate_hz,
        "center_frequency_hz": rec.center_frequency_hz,
        "duration_s": len(rec.iq) / rec.sample_rate_hz,
        "cut": rec.cut,
        "bursts_found": len(bursts),
        "bursts_classified": len(rows),
        "bursts": rows,
    }


def spectrogram_png(rec: Recording, bursts: list[dict]) -> str:
    """A spectrogram of the recording with the classified bursts marked, as a base64 PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iq = rec.iq
    rate = rec.sample_rate_hz
    # At most 800 FFT frames, spread evenly over the whole file: no decimation, so no aliasing,
    # and the time axis is in seconds of the original recording.
    nfft = 512 if len(iq) >= 512 * 64 else 128
    frames = max(1, min(800, len(iq) // nfft))
    starts = np.linspace(0, len(iq) - nfft, frames).astype(int)
    window = np.hanning(nfft).astype(np.float32)
    spectra = np.fft.fftshift(np.fft.fft(np.stack([iq[s : s + nfft] for s in starts]) * window, axis=1), axes=1)
    db = 10 * np.log10(np.abs(spectra) ** 2 + 1e-20)
    lo, hi = np.percentile(db, [5, 99.5])
    duration = len(iq) / rate
    f_lo = (rec.center_frequency_hz - rate / 2) / 1e6
    f_hi = (rec.center_frequency_hz + rate / 2) / 1e6
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=110)
    ax.imshow(db.T, origin="lower", aspect="auto", cmap="magma", vmin=lo, vmax=hi,
              extent=(0, duration, f_lo, f_hi), interpolation="nearest")
    for b in bursts:
        ax.axvspan(b["start_s"], b["end_s"], facecolor="none", edgecolor="#9cc3ff", lw=0.9)
        ax.text(b["start_s"], ax.get_ylim()[1], f" {b['index'] + 1}", va="top", ha="left", fontsize=8, color="white")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (MHz)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def summary(bank: Bank) -> dict:
    """What the UI shows before any upload: the models, their test scores and the routing table."""
    test = bank.routing["test"]
    return {
        "classes": list(bank.classes),
        "models": [{"name": m.name, "label": m.label} for m in bank.models],
        "strategies": [{"name": k, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in test.items()],
        "routes": [{"band": band, "model": LABELS.get(e["model"], e["model"]), "val_windows": e["val_windows"],
                    "val_recordings": e["val_recordings"], "reason": e["reason"]}
                   for band, e in bank.routing["routes"]["band"].items()],
        "fallback": LABELS.get(bank.routing["fallback_model"], bank.routing["fallback_model"]),
        "per_class_f1": test.get("route by band", {}).get("f1", {}),
        "note": bank.routing.get("note", ""),
    }
