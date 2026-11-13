"""Turn a capture into a data file the Library accepts.

    python -m pipeline.ingest sweep.csv -o signals.jsonl
    python -m pipeline.ingest capture.cs8 --sample-rate 20M --center-freq 2.437G -o signals.jsonl
    python -m pipeline.ingest recording.sigmf-meta -o signals.jsonl

A `.csv` is read as rtl_power or hackrf_sweep output; `.sigmf-meta`, `.sigmf-data`
or `.sigmf` as a SigMF recording; `.cu8`, `.cs8`, `.ci16` and `.cf32` as raw IQ.
Every row has center_frequency_hz, bandwidth_hz and modulation ("unknown"), so the
file can be added under Data and classified as it is. The extra measurements
(power, noise floor, SNR, duty cycle, ...) ride along; the classifier ignores them.

    python -m pipeline.ingest capture.cs8 --sample-rate 20M --center-freq 2.437G \
        --classify --model mod-classifier/runs/<name> -o signals.jsonl

With `--classify` (IQ recordings only, off by default), each burst also goes
through the signal classifier in `mod-classifier/` (`mcls.infer`), which adds
`signal_class` (a class name, or "unknown" when the top probability is under
the calibrated threshold), `class_confidence` and `class_probs`. `modulation`
stays "unknown". Without the flag nothing changes.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

from pipeline.inputs import InputError, combine_observations
from pipeline.iq import FORMATS, IQError, datatype_for, open_samples, read_raw, read_sigmf, sigmf_paths, to_complex
from pipeline.sweep import THRESHOLD_DB, SweepError, detect_signals, read_sweep

# What the dashboard accepts per data file (frontend/lib/laya-files.ts).
LIBRARY_LIMIT_BYTES = 2 * 1024 * 1024

_UNITS = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}


def hertz(text: str) -> float:
    """'2.437G', '20M', '915e6' or '433920000' in Hz."""
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([kmg]?)(?:hz)?\s*", text.lower())
    if not match:
        raise argparse.ArgumentTypeError(f"{text!r} is not a frequency, such as 2.437G or 20M.")
    return float(match.group(1)) * _UNITS[match.group(2)]


def ingest(path: Path, sample_rate_hz: float | None = None, center_frequency_hz: float | None = None,
           datatype: str | None = None, threshold_db: float = THRESHOLD_DB) -> list[dict]:
    """Observation rows for one capture file, checked against the Library's rules."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".sigmf-meta", ".sigmf-data", ".sigmf")):
        rows = read_sigmf(path)
    elif name.endswith(".csv"):
        rows = detect_signals(read_sweep(path.read_text()), threshold_db)
        for row in rows:
            row["source"] = path.name
    else:
        rows = read_raw(path, sample_rate_hz, center_frequency_hz, datatype)
    for index, row in enumerate(rows):
        row["id"] = index
    if rows:
        # The same check the Library runs, so a file written here is never refused there.
        combine_observations([(path.name, json.dumps(rows))], limit=len(rows))
    return rows


# The signal classifier lives in mod-classifier/ (package `mcls`), next to pipeline/.
MOD_CLASSIFIER = Path(__file__).resolve().parents[1] / "mod-classifier"
# Classifier windows: 4096 samples at 20 MS/s, at most 8 per burst, hop 2048 (mcls.infer).
_WINDOW_S = 4096 / 20e6
_BURST_READ_S = (4096 + 7 * 2048) / 20e6


def _infer():
    """`mcls.infer`, imported only when --classify is used."""
    if str(MOD_CLASSIFIER) not in sys.path:
        sys.path.insert(0, str(MOD_CLASSIFIER))
    import mcls.infer

    return mcls.infer


def _iq_source(path: Path, sample_rate_hz, center_frequency_hz, datatype):
    """(interleaved values, datatype, sample rate, tuned centre at a sample index) for a raw or SigMF file."""
    name = path.name.lower()
    if name.endswith((".sigmf-meta", ".sigmf-data", ".sigmf")):
        meta_path, data_path = sigmf_paths(path)
        meta = json.loads(meta_path.read_text())
        info = meta.get("global") or {}
        datatype = {"ci8_le": "ci8", "cu8_le": "cu8"}.get(info.get("core:datatype"), info.get("core:datatype"))
        captures = sorted(meta.get("captures") or [], key=lambda c: c.get("core:sample_start", 0))

        def centre(sample: int) -> float:
            current = captures[0]
            for capture in captures:
                if capture.get("core:sample_start", 0) <= sample:
                    current = capture
            return float(current["core:frequency"])

        return open_samples(data_path, datatype), datatype, float(info["core:sample_rate"]), centre
    datatype = datatype or datatype_for(path)
    return open_samples(path, datatype), datatype, float(sample_rate_hz), lambda sample: float(center_frequency_hz)


def classify_rows(path: Path, rows: list[dict], model: Path, calibration: Path | None = None,
                  priors: Path | None = None, sample_rate_hz: float | None = None,
                  center_frequency_hz: float | None = None, datatype: str | None = None) -> list[dict]:
    """Add signal_class, class_confidence and class_probs to IQ burst rows (in place; also returned)."""
    infer = _infer()
    from mcls.infer.loading import calibration_for, run_identity

    calibration = calibration or calibration_for(model)
    if calibration is None:
        raise IQError(f"No calibration.json next to {model}; pass --calibration. "
                      "Its threshold decides when a burst is 'unknown'.")
    classifier = infer.load_classifier(model, calibration=calibration)
    run, manifest_sha256 = run_identity(model)
    # Refuses a calibration.json fitted for another model, class list, manifest or run.
    cal = infer.load_calibration(calibration, classifier.classes, manifest_sha256=manifest_sha256, run=run)
    band_priors = infer.load_priors(priors)
    values, datatype, rate, centre_at = _iq_source(Path(path), sample_rate_hz, center_frequency_hz, datatype)
    if datatype not in FORMATS:
        raise IQError(f"{Path(path).name}: the sample format {datatype!r} is not supported.")
    total = len(values) // 2
    pad = math.ceil(_WINDOW_S * rate) + 64
    for row in rows:
        start = int(round(row["start_s"] * rate))
        duration = row.get("burst_duration_s")
        end = start + math.ceil((duration if duration is not None else _BURST_READ_S) * rate)
        lo, hi = max(0, start - pad), min(total, end + pad)
        burst = {
            "start_s": (start - lo) / rate,
            "end_s": None if duration is None else (end - lo) / rate,
            "center_frequency_hz": row["center_frequency_hz"],
        }
        iq = to_complex(values[2 * lo : 2 * hi], datatype)
        result = infer.classify_bursts(iq, rate, centre_at(start), [burst], classifier, calibration=cal,
                                       priors=band_priors)[0]
        row.update(result.row_fields())
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn a sweep or IQ capture into Argus observation rows.")
    parser.add_argument("capture", type=Path, help="a .csv sweep, a SigMF recording, or a raw .cu8/.cs8/.ci16/.cf32 file")
    parser.add_argument("-o", "--output", type=Path, help="JSONL file to write (default: print to stdout)")
    parser.add_argument("--sample-rate", type=hertz, help="raw IQ only: samples per second, e.g. 20M")
    parser.add_argument("--center-freq", type=hertz, help="raw IQ only: the frequency the radio was tuned to, e.g. 2.437G")
    parser.add_argument("--format", dest="datatype", help="raw IQ only: cu8, ci8, ci16_le or cf32_le, when the suffix does not say")
    parser.add_argument("--threshold-db", type=float, default=THRESHOLD_DB,
                        help=f"sweeps only: how far above the noise floor a signal must rise (default {THRESHOLD_DB:g} dB)")
    parser.add_argument("--classify", action="store_true",
                        help="IQ only: add signal_class, class_confidence and class_probs from the signal classifier")
    parser.add_argument("--model", type=Path, help="with --classify: a mod-classifier run folder, checkpoint.pt or model.pkl")
    parser.add_argument("--calibration", type=Path,
                        help="with --classify: calibration.json (default: the one in the model's run folder)")
    parser.add_argument("--priors", type=Path, help="with --classify: band priors TOML (default: mod-classifier/configs/band_priors.toml)")
    args = parser.parse_args(argv)
    if args.classify and args.model is None:
        parser.error("--classify needs --model")
    if args.classify and args.capture.name.lower().endswith(".csv"):
        parser.error("--classify needs an IQ recording; a sweep has no samples to classify")

    if not args.capture.exists():
        print(f"{args.capture} does not exist.", file=sys.stderr)
        return 1
    try:
        rows = ingest(args.capture, args.sample_rate, args.center_freq, args.datatype, args.threshold_db)
    except (SweepError, IQError, InputError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.classify and rows:
        try:
            classify_rows(args.capture, rows, args.model, args.calibration, args.priors,
                          args.sample_rate, args.center_freq, args.datatype)
        except (IQError, OSError, ValueError, ImportError, TypeError) as exc:
            print(f"--classify: {exc}", file=sys.stderr)
            return 1

    text = "".join(json.dumps(row) + "\n" for row in rows)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {len(rows)} signals to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    if not rows:
        print("No signal rose above the noise floor.", file=sys.stderr)
    if len(text.encode()) > LIBRARY_LIMIT_BYTES:
        print("The output is over the Library's 2 MB limit; split it before adding it under Data.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
