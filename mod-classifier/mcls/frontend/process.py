"""One recording in, shards and rows out: survey, channelize, find bursts, cut, write.

The recording is read in chunks (never whole). The 20 MS/s stream is cut into
detection blocks of DETECT_BLOCK samples; a burst still running at the end of a
block that started in the block's second half is carried whole into the next
block, so it is detected and windowed once. Bursts longer than half a block are
split at the block edge (both pieces are windowed and get separate burst ids;
detect_bursts marks them truncated).

`WindowEngine` holds that logic once. `process_recording` (training shards) and
`windows_from_iq` (inference) both drive it, so inference cuts exactly the
windows training saw.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, WINDOW_SAMPLES, ManifestRow, Recording, band_of
from mcls.frontend import windows as W
from mcls.frontend.channelize import Channelizer, choose_offset
from mcls.frontend.shards import MAX_WINDOWS_PER_SHARD, ShardWriter

#: Native samples read per call to Recording.read (16 MB of complex64).
READ_CHUNK = 1 << 21
#: 20 MS/s samples per detect_bursts call: about 0.21 s, as pipeline/iq.py.
DETECT_BLOCK = 1 << 22
BACKGROUND = "background"
#: Labels of always-on signals. Burst detection treats a signal that fills every
#: block as the noise floor, so the activity test rejects all its windows (GATEMAN
#: on CRC, Nov 12: 575,718 of 575,718 rejected per file). Their windows are spread
#: evenly over each block instead, with no detection.
CONTINUOUS = frozenset({"gnss_jammer"})
#: Least active share of a window, per label, where the default (FrontendParams.min_active)
#: is too strict. Radar pulses last 4-50 us, under 25% of a 205 us window, so the
#: default rejected nearly every Drexel radar window (CRC, Nov 12). Windows still
#: come only from detected bursts, so they sit on the pulses.
MIN_ACTIVE_BY_LABEL = {"pulsed_radar": 0.01}


@dataclass
class FrontendParams:
    max_windows_per_burst: int = W.MAX_PER_BURST
    max_windows_per_recording: int | None = None
    limit_seconds: float | None = None
    min_active: float = W.MIN_ACTIVE
    read_chunk: int = READ_CHUNK
    detect_block: int = DETECT_BLOCK
    max_windows_per_shard: int = MAX_WINDOWS_PER_SHARD


def stem_for(source_file: str) -> str:
    """'GPSL1+AMtone/x.bin' -> 'GPSL1+AMtone__x' (path separators flattened, last suffix dropped)."""
    flat = source_file.replace("\\", "/").replace("/", "__")
    stem = Path(flat).stem
    return stem or flat


def burst_id(dataset: str, stem: str, index: int) -> str:
    """'<dataset>/<stem>/b<index:06d>': shared by every window of one detected burst."""
    return f"{dataset}/{stem}/b{index:06d}"


def _finite(value, digits):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def base_row(rec: Recording, centre_hz: float) -> dict:
    """The ManifestRow fields shared by every window of a recording, in contract order, without split."""
    return {
        "window_id": "", "shard": "", "offset": 0, "length": WINDOW_SAMPLES,
        "dataset": rec.dataset, "recording_id": rec.recording_id,
        "label": rec.label, "label_strength": rec.label_strength,
        "native_rate_hz": float(rec.sample_rate_hz), "center_frequency_hz": float(centre_hz),
        "band": band_of(centre_hz), "is_real": bool(rec.is_real), "license": rec.license,
        "emitter_id": rec.emitter_id, "snr_db_est": None, "bandwidth_hz": None, "duty_cycle": None,
        "burst_id": None,
    }


class WindowEngine:
    """Detection blocks, carry-over, burst numbering and window cutting over a 20 MS/s stream.

    Feed 20 MS/s samples with `feed`, then call `finish` once. Each call returns
    (windows (n, 4096) unit RMS complex64, list of (absolute 20 MS/s start, burst index or None)).
    `self.bursts[k]` is burst k: the detect_bursts dict with "start"/"end" made
    absolute 20 MS/s indices and "burst_index" = k. Bursts are numbered in time
    order over the whole stream, once each (a carried burst is counted only in
    the block that holds it whole). Bursts that yield no window keep their number.
    """

    def __init__(self, centre_hz: float, background: bool, params: FrontendParams, continuous: bool = False,
                 min_active: float | None = None):
        self.centre_hz = centre_hz
        self.background = background
        self.continuous = continuous
        self.min_active = params.min_active if min_active is None else min_active
        self.params = params
        self.bursts: list[dict] = []
        self.windows = 0
        self.rejected = 0
        self.blocks = 0
        self.done = False
        self.detect_s = 0.0
        self.window_s = 0.0
        self._buf = np.zeros(0, dtype=SAMPLE_DTYPE)
        self._abs = 0  # 20 MS/s index of self._buf[0]

    def _handle(self, block: np.ndarray, final: bool) -> tuple[int, np.ndarray, list]:
        t0 = time.perf_counter()
        bursts = [] if self.continuous else W.detect(block, self.centre_hz)
        self.detect_s += time.perf_counter() - t0
        consume = len(block)
        if not final:
            carried = [b for b in bursts if b["end"] >= len(block) - 1 and b["start"] > len(block) // 2]
            if carried:
                consume = min(b["start"] for b in carried)
                bursts = [b for b in bursts if b["start"] < consume]
        t0 = time.perf_counter()
        self.blocks += 1
        first = len(self.bursts)
        for i, b in enumerate(bursts):
            b["burst_index"] = first + i
        if self.background or self.continuous:  # continuous: no bursts, so windows spread over the block
            wins = W.idle_windows(block[:consume], bursts, self.params.max_windows_per_burst)
        else:
            wins, rejected = W.burst_windows(block, bursts, self.params.max_windows_per_burst,
                                             self.min_active, floor=W.noise_floor(block))
            self.rejected += rejected
        cap = self.params.max_windows_per_recording
        if cap is not None and self.windows + len(wins) >= cap:
            wins = wins[: cap - self.windows]
            self.done = True
        samples = W.cut(block, wins)
        info = [(self._abs + w.start, None if w.burst is None else w.burst["burst_index"]) for w in wins]
        for b in bursts:  # absolute positions, once windowing has used the block-relative ones
            b["start"] += self._abs
            b["end"] += self._abs
            self.bursts.append(b)
        self.windows += len(wins)
        self.window_s += time.perf_counter() - t0
        return consume, samples, info

    def _drain(self, final: bool) -> tuple[np.ndarray, list]:
        out, info = [], []
        block_len = self.params.detect_block
        while not self.done:
            if len(self._buf) < (WINDOW_SAMPLES if final else block_len):
                break
            last = final and len(self._buf) <= block_len
            used, samples, i = self._handle(self._buf[:block_len], final=last)
            out.append(samples)
            info += i
            self._buf, self._abs = self._buf[used:], self._abs + used
            if last:
                break
        windows = np.concatenate(out) if out else np.zeros((0, WINDOW_SAMPLES), dtype=SAMPLE_DTYPE)
        return windows, info

    def feed(self, y: np.ndarray) -> tuple[np.ndarray, list]:
        self._buf = np.concatenate([self._buf, np.asarray(y, dtype=SAMPLE_DTYPE)])
        return self._drain(final=False)

    def finish(self) -> tuple[np.ndarray, list]:
        return self._drain(final=True)


def burst_stats(burst: dict, ch: Channelizer, native_rate_hz: float) -> dict:
    """One burst: 20 MS/s span, native span and times (resampler delay corrected), detect_bursts features."""
    native_start = float(ch.native_index(burst["start"]))
    native_end = float(ch.native_index(burst["end"]))
    return {
        "burst_index": burst["burst_index"],
        "start": int(burst["start"]), "end": int(burst["end"]),
        "native_start": native_start, "native_end": native_end,
        "start_s": native_start / native_rate_hz, "end_s": native_end / native_rate_hz,
        "snr_db": _finite(burst.get("snr_db"), 2),
        "bandwidth_hz": _finite(burst.get("bandwidth_hz"), 0),
        "duty_cycle": _finite(burst.get("duty_cycle"), 5),
        "center_frequency_hz": burst.get("center_frequency_hz"),
        "pulses": burst.get("pulses"),
        "truncated": burst.get("truncated"),
        "hopping": burst.get("hopping"),
        "pulse_interval_s": burst.get("pulse_interval_s"),
        "burst_duration_s": burst.get("burst_duration_s"),
    }


@dataclass
class FrontEndResult:
    """What `windows_from_iq` returns. Window i came from burst `window_burst[i]` (None: background)."""

    windows: np.ndarray  # (n, 4096) complex64, unit RMS
    window_burst: list  # per window: burst index into `bursts`, or None
    window_start_20: np.ndarray  # int64: start of each window in the 20 MS/s stream
    window_start_native: np.ndarray  # float64: native sample index of each window start (delay corrected)
    bursts: list = field(default_factory=list)  # burst_stats dicts; bursts[k]["burst_index"] == k
    burst_ids: list = field(default_factory=list)  # per window: "<id_prefix>/b<k:06d>" or None
    slice_offset_hz: float = 0.0
    slice_center_hz: float = 0.0
    samples_20msps: int = 0
    windows_rejected_inactive: int = 0


def windows_from_iq(iq: np.ndarray, sample_rate_hz: float, center_frequency_hz: float, *,
                    offset_hz: float | None = None, background: bool = False,
                    params: FrontendParams | None = None, id_prefix: str | None = None) -> FrontEndResult:
    """Native-rate IQ in; the windows, burst ids and burst stats preprocess would produce for it.

    Same code path as preprocess: slice survey (unless `offset_hz` is given),
    Channelizer, detection blocks with carry-over, activity test, per-burst cap,
    unit RMS. `background` cuts idle windows instead. `id_prefix`
    ("<dataset>/<stem>") fills `burst_ids`. `params.limit_seconds` is ignored:
    pass only the samples you want.
    """
    params = params or FrontendParams()
    iq = np.asarray(iq, dtype=SAMPLE_DTYPE)
    if offset_hz is None:
        offset_hz = choose_offset(lambda s, c: iq[s : s + c], len(iq), sample_rate_hz) if len(iq) else 0.0
    ch = Channelizer(sample_rate_hz, offset_hz)
    centre = center_frequency_hz + ch.offset_hz
    engine = WindowEngine(centre, background, params)
    parts, info, produced = [], [], 0

    def take(result):
        parts.append(result[0])
        info.extend(result[1])

    for start in range(0, len(iq), params.read_chunk):
        if engine.done:
            break
        y = ch.process(iq[start : start + params.read_chunk])
        produced += len(y)
        take(engine.feed(y))
    if not engine.done:
        y = ch.flush()
        produced += len(y)
        take(engine.feed(y))
        take(engine.finish())
    windows = np.concatenate(parts) if parts else np.zeros((0, WINDOW_SAMPLES), dtype=SAMPLE_DTYPE)
    starts = np.array([s for s, _ in info], dtype=np.int64)
    which = [k for _, k in info]
    return FrontEndResult(
        windows=windows,
        window_burst=which,
        window_start_20=starts,
        window_start_native=np.asarray(ch.native_index(starts), dtype=np.float64),
        bursts=[burst_stats(b, ch, sample_rate_hz) for b in engine.bursts],
        burst_ids=[None if k is None or id_prefix is None else f"{id_prefix}/b{k:06d}" for k in which],
        slice_offset_hz=ch.offset_hz,
        slice_center_hz=centre,
        samples_20msps=produced,
        windows_rejected_inactive=engine.rejected,
    )


def process_recording(rec: Recording, out_dir: Path, stem: str | None = None,
                      params: FrontendParams | None = None) -> dict:
    """Write `<stem>.sNNN.npy` and `<stem>.sNNN.rows.jsonl`; return the recording's stats."""
    params = params or FrontendParams()
    stem = stem or stem_for(rec.source_file)
    t_begin = time.perf_counter()
    timing = {"survey_s": 0.0, "read_s": 0.0, "channelize_s": 0.0, "detect_s": 0.0, "window_s": 0.0}

    n_total = rec.num_samples
    if params.limit_seconds is not None:
        n_total = min(n_total, max(1, int(round(params.limit_seconds * rec.sample_rate_hz))))

    t = time.perf_counter()
    offset = choose_offset(rec.read, n_total, rec.sample_rate_hz)
    timing["survey_s"] = time.perf_counter() - t
    ch = Channelizer(rec.sample_rate_hz, offset)
    centre = rec.center_frequency_hz + ch.offset_hz
    template = base_row(rec, centre)
    engine = WindowEngine(centre, rec.label == BACKGROUND, params, continuous=rec.label in CONTINUOUS,
                          min_active=MIN_ACTIVE_BY_LABEL.get(rec.label))
    writer = ShardWriter(out_dir, stem, params.max_windows_per_shard)
    produced = 0

    def write(result) -> None:
        samples, info = result
        rows = []
        for _, k in info:
            row = dict(template)
            row["window_id"] = f"{rec.dataset}/{stem}/{writer.total + len(rows):06d}"
            if k is not None:
                b = engine.bursts[k]
                row["snr_db_est"] = _finite(b.get("snr_db"), 2)
                row["bandwidth_hz"] = _finite(b.get("bandwidth_hz"), 0)
                row["duty_cycle"] = _finite(b.get("duty_cycle"), 5)
                row["burst_id"] = burst_id(rec.dataset, stem, k)
            rows.append(row)
        writer.add(samples, rows)

    start = 0
    try:
        while start < n_total and not engine.done:
            count = min(params.read_chunk, n_total - start)
            t0 = time.perf_counter()
            x = rec.read(start, count)
            timing["read_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            y = ch.process(x)
            timing["channelize_s"] += time.perf_counter() - t0
            start += count
            produced += len(y)
            write(engine.feed(y))
        if not engine.done:
            t0 = time.perf_counter()
            y = ch.flush()
            timing["channelize_s"] += time.perf_counter() - t0
            produced += len(y)
            write(engine.feed(y))
            write(engine.finish())
    finally:
        writer.close()
    timing["detect_s"] = engine.detect_s
    timing["window_s"] = engine.window_s

    return {
        "dataset": rec.dataset,
        "recording_id": rec.recording_id,
        "source_file": rec.source_file,
        "stem": stem,
        "label": rec.label,
        "native_rate_hz": rec.sample_rate_hz,
        "recording_center_hz": rec.center_frequency_hz,
        "slice_offset_hz": ch.offset_hz,
        "slice_center_hz": centre,
        "resampler": {"up": ch.up, "down": ch.down, "taps": ch.taps},
        "native_samples_read": int(start),
        "seconds_processed": start / rec.sample_rate_hz,
        "samples_20msps": int(produced),
        "detect_blocks": engine.blocks,
        "bursts": len(engine.bursts),
        "windows": engine.windows,
        "windows_rejected_inactive": engine.rejected,
        "shards": writer.shards,
        "wall_s": round(time.perf_counter() - t_begin, 3),
        "timing_s": {k: round(v, 3) for k, v in timing.items()},
        "params": asdict(params),
    }


def check_rows(rows: list[dict]) -> None:
    """Raise ValueError unless every row is a valid ManifestRow once a split is added."""
    for row in rows:
        if "split" in row:
            raise ValueError(f"{row.get('window_id')}: rows files carry no split")
        ManifestRow.from_dict({**row, "split": "train"}).validate()
