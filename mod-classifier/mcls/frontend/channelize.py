"""Pick the 20 MHz slice with the most energy, shift it to baseband, resample to 20 MS/s.

Recordings run from 10 to 200 MS/s; the HackRF the model will serve sees 20 MHz.
`choose_offset` takes a coarse averaged spectrum over the recording and returns
the centre offset of the best 20 MHz slice. `Channelizer` then mixes that offset
down and resamples, chunk by chunk, with output identical to one-shot processing.

The offset is snapped to a grid of sample_rate / GRID_BINS, so the mixer phase
k * n / GRID_BINS is computed exactly in integers from the absolute sample index.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, SAMPLE_RATE_HZ
from mcls.frontend.resample import RationalResampler, rational

#: Coarse STFT size used to choose the slice; also the mixer frequency grid.
GRID_BINS = 1024
#: Width of the slice the model sees, Hz.
SLICE_HZ = SAMPLE_RATE_HZ
#: How many stretches of the recording the coarse spectrum averages, and their length.
SURVEY_READS = 32
SURVEY_SAMPLES = 1 << 16


def coarse_spectrum(read: Callable[[int, int], np.ndarray], num_samples: int,
                    reads: int = SURVEY_READS, samples: int = SURVEY_SAMPLES) -> np.ndarray:
    """Average Hann-windowed power spectrum (GRID_BINS bins, fftshifted: bin GRID_BINS/2 is DC).

    Reads `reads` stretches of `samples` samples spread evenly over the recording,
    in increasing order (cheap for forward-only readers such as gzip streams).
    """
    samples = min(samples, num_samples)
    starts = np.unique(np.linspace(0, num_samples - samples, max(1, reads)).astype(np.int64))
    window = np.hanning(GRID_BINS).astype(np.float32)
    psd = np.zeros(GRID_BINS)
    frames = 0
    for start in starts:
        x = read(int(start), samples)
        x = x - x.mean()  # a receiver's DC offset is not signal
        usable = (len(x) // GRID_BINS) * GRID_BINS
        if usable == 0:
            x = np.concatenate([x, np.zeros(GRID_BINS - len(x), dtype=x.dtype)])
            usable = GRID_BINS
        blocks = x[:usable].reshape(-1, GRID_BINS) * window
        psd += (np.abs(np.fft.fft(blocks, axis=1)) ** 2).sum(axis=0)
        frames += blocks.shape[0]
    return np.fft.fftshift(psd / max(frames, 1))


def best_slice_bin(spectrum: np.ndarray, sample_rate_hz: float, slice_hz: float = SLICE_HZ) -> int:
    """Signed grid bin (offset / (fs / GRID_BINS)) of the slice centre holding the most energy.

    The DC bin and its two neighbours each side (the Hann main lobe) are replaced
    by the mean of the next bins out, so a receiver's DC spike does not pull the
    slice to the centre.
    """
    bins = len(spectrum)
    s = spectrum.astype(np.float64).copy()
    mid = bins // 2
    s[mid - 2 : mid + 3] = 0.5 * (s[mid - 3] + s[mid + 3])
    width = int(round(slice_hz / sample_rate_hz * bins))
    if width >= bins:
        return 0
    cumulative = np.concatenate([[0.0], np.cumsum(s)])
    energy = cumulative[width:] - cumulative[:-width]  # energy[i] = sum s[i : i + width]
    best = int(np.argmax(energy))
    # Slice centre bin for a window starting at `best` of `width` bins.
    centre = best + width // 2
    return centre - mid


def choose_offset(read: Callable[[int, int], np.ndarray], num_samples: int, sample_rate_hz: float,
                  slice_hz: float = SLICE_HZ) -> float:
    """Offset (Hz, relative to the recording centre) of the 20 MHz slice with most energy.

    0.0 when the recording is no wider than the slice.
    """
    if sample_rate_hz <= slice_hz:
        return 0.0
    spectrum = coarse_spectrum(read, num_samples)
    return best_slice_bin(spectrum, sample_rate_hz, slice_hz) * sample_rate_hz / GRID_BINS


class Channelizer:
    """Mix `offset_hz` to DC and resample to 20 MS/s, chunk by chunk.

    `offset_hz` is snapped to the grid fs / GRID_BINS; read the snapped value
    from `self.offset_hz`. Chunked output equals one-shot output bit for bit.
    """

    def __init__(self, sample_rate_hz: float, offset_hz: float = 0.0, rate_out: float = SAMPLE_RATE_HZ):
        self.sample_rate_hz = float(sample_rate_hz)
        self.grid_bin = int(round(offset_hz / self.sample_rate_hz * GRID_BINS))
        self.offset_hz = self.grid_bin * self.sample_rate_hz / GRID_BINS
        self.up, self.down = rational(self.sample_rate_hz, rate_out)
        self.passthrough = self.up == self.down
        self.resampler = None if self.passthrough else RationalResampler(self.up, self.down)
        self._n = 0  # absolute index of the next input sample
        k = np.arange(GRID_BINS)
        self._lut = np.exp(-2j * np.pi * k / GRID_BINS).astype(np.complex64).astype(np.complex128)

    @property
    def taps(self) -> int:
        return 0 if self.resampler is None else len(self.resampler.taps)

    def native_index(self, m):
        """Native input sample index (float) that 20 MS/s output sample `m` stands for.

        Corrects the resampler's group delay ((taps - 1) / 2 upsampled samples)
        and its dropped start-up transient; identity for passthrough.
        """
        r = self.resampler
        if r is None:
            return np.asarray(m, dtype=np.float64) * 1.0
        return (r._first_j + np.asarray(m, dtype=np.float64) * r.down - r.delay) / r.up

    def _mix(self, chunk: np.ndarray) -> np.ndarray:
        # complex128: products of float32 values are exact in float64, so the
        # result does not depend on SIMD/FMA code paths (chunk alignment).
        n = np.arange(self._n, self._n + len(chunk), dtype=np.int64)
        self._n += len(chunk)
        if self.grid_bin == 0:
            return chunk
        return chunk.astype(np.complex128) * self._lut[(n * self.grid_bin) % GRID_BINS]

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = self._mix(np.asarray(chunk, dtype=SAMPLE_DTYPE))
        if self.passthrough:
            return x.astype(SAMPLE_DTYPE, copy=False)
        return self.resampler.process(x)

    def flush(self) -> np.ndarray:
        if self.passthrough:
            return np.zeros(0, dtype=SAMPLE_DTYPE)
        return self.resampler.flush()


def to_20msps(iq: np.ndarray, sample_rate_hz: float, offset_hz: float | None = None) -> np.ndarray:
    """One shot: choose the slice (unless given), mix it to DC and resample to 20 MS/s."""
    iq = np.asarray(iq, dtype=SAMPLE_DTYPE)
    if offset_hz is None:
        offset_hz = choose_offset(lambda s, c: iq[s : s + c], len(iq), sample_rate_hz)
    ch = Channelizer(sample_rate_hz, offset_hz)
    return np.concatenate([ch.process(iq), ch.flush()])
