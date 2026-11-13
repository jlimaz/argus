"""Burst and idle windows, normalization, the two model views, shard files."""

import json

import numpy as np
import pytest

from mcls.contracts import WINDOW_SAMPLES, check_windows
from mcls.frontend import windows as W
from mcls.frontend.shards import HEADER_BYTES, ShardWriter, npy_header
from mcls.frontend.views import iq_view, stft_view

RATE = 20e6


def bursty(spans, total, snr_db=25.0, seed=0, gaps=None):
    """Unit-power noise floor 10^(-snr/10) plus unit-power band-limited bursts at `spans`.

    `gaps`: (start, end) spans switched off again inside the bursts (pulse gaps).
    """
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(total) + 1j * rng.standard_normal(total)) / np.sqrt(2) * 10 ** (-snr_db / 20)
    for a, b in spans:
        s = rng.standard_normal(b - a) + 1j * rng.standard_normal(b - a)
        s = np.convolve(s, np.ones(4) / 2, mode="same")  # about 5 MHz wide
        x[a:b] += s / np.sqrt(np.mean(np.abs(s) ** 2))
    for a, b in gaps or []:
        x[a:b] = (rng.standard_normal(b - a) + 1j * rng.standard_normal(b - a)) / np.sqrt(2) * 10 ** (-snr_db / 20)
    return x.astype(np.complex64)


def test_windows_sit_inside_bursts_and_respect_the_cap():
    # Two bursts more than 20 ms (400,000 samples) apart, so detect_bursts keeps them separate.
    spans = [(50_000, 150_000), (700_000, 720_000)]
    x = bursty(spans, 1_000_000)
    bursts = W.detect(x, 2.44e9)
    assert len(bursts) == 2
    wins, rejected = W.burst_windows(x, bursts, max_per_burst=5)
    assert rejected == 0
    per_burst = {}
    for w in wins:
        per_burst[id(w.burst)] = per_burst.get(id(w.burst), 0) + 1
        a, b = w.burst["start"], w.burst["end"]
        assert a <= w.start and w.start + WINDOW_SAMPLES <= b
    assert sorted(per_burst.values()) == [5, 5]
    # Spread evenly: the windows of the first burst cover most of it.
    first = sorted(w.start for w in wins if w.burst is bursts[0])
    assert first[-1] - first[0] > 0.8 * (100_000 - WINDOW_SAMPLES)
    # 50% overlap without a cap: (length - 4096) // 2048 + 1 windows.
    wins_all, _ = W.burst_windows(x, bursts[1:], max_per_burst=10_000)
    assert len(wins_all) == (bursts[1]["end"] - bursts[1]["start"] - WINDOW_SAMPLES) // W.HOP + 1


def test_quiet_gaps_inside_a_burst_are_rejected():
    # One burst of two pulses 5 ms apart: detect_bursts groups them; the gap must give no windows.
    x = bursty([(100_000, 140_000)], 600_000, gaps=[(110_000, 130_000)])
    bursts = W.detect(x, 2.44e9)
    assert len(bursts) == 1 and bursts[0]["pulses"] == 2
    wins, rejected = W.burst_windows(x, bursts, max_per_burst=1000)
    assert rejected > 0
    for w in wins:
        assert w.start + WINDOW_SAMPLES <= 110_000 + WINDOW_SAMPLES * (1 - W.MIN_ACTIVE) or w.start >= 130_000 - WINDOW_SAMPLES * (1 - W.MIN_ACTIVE)


def test_short_burst_gets_one_centred_window():
    x = bursty([(300_000, 303_000)], 600_000)
    bursts = W.detect(x, 2.44e9)
    wins, _ = W.burst_windows(x, bursts)
    assert len(wins) == 1
    assert abs(wins[0].start + WINDOW_SAMPLES // 2 - 301_500) < 200


def test_idle_windows_avoid_bursts():
    x = bursty([(100_000, 200_000)], 400_000)
    bursts = W.detect(x, 2.44e9)
    wins = W.idle_windows(x, bursts, max_per_stretch=1000)
    assert wins
    for w in wins:
        assert w.burst is None
        assert w.start + WINDOW_SAMPLES <= bursts[0]["start"] - W.IDLE_MARGIN or w.start >= bursts[0]["end"] + W.IDLE_MARGIN
    assert len(W.idle_windows(x, bursts, max_per_stretch=3)) == 6  # two stretches, three each


def test_normalize_and_cut():
    x = bursty([(0, 50_000)], 50_000) * 37.0
    w = W.cut(x, [W.Window(0, None), W.Window(10_000, None)])
    check_windows(w)
    assert np.allclose(np.sqrt(np.mean(np.abs(w) ** 2, axis=1)), 1.0, atol=1e-5)
    zeros = W.normalize(np.zeros((2, WINDOW_SAMPLES), dtype=np.complex64))
    assert np.all(zeros == 0)


def test_windows_for_inference():
    x = bursty([(50_000, 90_000)], 300_000)
    windows, bursts, index = W.windows_for_inference(x, 2.44e9)
    check_windows(windows)
    assert len(bursts) == 1 and len(index) == len(windows) > 0 and set(index) == {0}


def test_views():
    rng = np.random.default_rng(0)
    w = W.normalize(rng.standard_normal((3, WINDOW_SAMPLES)) + 1j * rng.standard_normal((3, WINDOW_SAMPLES)))
    iq = iq_view(w)
    assert iq.shape == (3, 2, WINDOW_SAMPLES) and iq.dtype == np.float32
    assert np.array_equal(iq[:, 0], w.real) and np.array_equal(iq[:, 1], w.imag)
    s = stft_view(w)
    assert s.shape == (3, 1, 64, 64) and s.dtype == np.float32 and np.isfinite(s).all()
    # A tone at +2.5 MHz lights frequency row 32 + 2.5/20*64 = 40, in every time column.
    t = np.exp(2j * np.pi * 2.5e6 / RATE * np.arange(WINDOW_SAMPLES)).astype(np.complex64)[None]
    st = stft_view(t)[0, 0]
    assert np.all(np.argmax(st, axis=0) == 40)
    with pytest.raises(ValueError):
        stft_view(np.zeros((1, 100), dtype=np.complex64))


def test_npy_header_is_valid(tmp_path):
    for n in (0, 4096, 10**12):
        h = npy_header(n)
        assert len(h) == HEADER_BYTES
    data = (np.arange(3 * WINDOW_SAMPLES) * (1 + 1j)).astype(np.complex64)
    path = tmp_path / "a.npy"
    path.write_bytes(npy_header(len(data)) + data.astype("<c8").tobytes())
    assert np.array_equal(np.load(path, mmap_mode="r"), data)
    np.save(tmp_path / "ref.npy", data)  # numpy's own header for the same array also loads identically
    assert np.array_equal(np.load(tmp_path / "ref.npy"), np.load(path))


def test_shard_writer_rolls_and_rows_match(tmp_path):
    rng = np.random.default_rng(1)
    w = W.normalize(rng.standard_normal((7, WINDOW_SAMPLES)) + 1j * rng.standard_normal((7, WINDOW_SAMPLES)))
    writer = ShardWriter(tmp_path, "rec", max_windows=3)
    writer.add(w[:4], [{"window_id": f"x/{i}"} for i in range(4)])
    writer.add(w[4:], [{"window_id": f"x/{i}"} for i in range(4, 7)])
    writer.close()
    assert writer.shards == ["rec.s000.npy", "rec.s001.npy", "rec.s002.npy"]
    k = 0
    for shard in writer.shards:
        data = np.load(tmp_path / shard, mmap_mode="r")
        rows = [json.loads(line) for line in open(tmp_path / shard.replace(".npy", ".rows.jsonl"))]
        assert data.dtype == np.complex64 and data.shape == (len(rows) * WINDOW_SAMPLES,)
        for row in rows:
            assert row["window_id"] == f"x/{k}" and row["shard"] == shard and row["length"] == WINDOW_SAMPLES
            assert np.array_equal(data[row["offset"] : row["offset"] + row["length"]], w[k])
            k += 1
    assert k == 7
