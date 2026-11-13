"""Rational resampling (up by L, low-pass, down by M) in numpy only.

The filter is a Kaiser-windowed sinc FIR designed at the upsampled rate L x fs_in.
The output is the polyphase result y[m] = sum_k h[k] u[mM - k], where u is the
input with L - 1 zeros between samples; it is computed by FFT overlap-save on
blocks whose boundaries are fixed in absolute sample positions, so feeding the
signal in any chunk sizes gives bit-for-bit the same output as one call.

scipy is not used (it is not installed on the laptop). Error bounds are measured
in tests/test_frontend_resample.py.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

#: Stopband attenuation of the designed filter, dB.
ATTENUATION_DB = 80.0
#: Flat passband as a share of the narrower Nyquist (input or output).
PASSBAND_SHARE = 0.8
#: FFT size used for overlap-save blocks.
FFT_SIZE = 1 << 16


def rational(rate_in: float, rate_out: float, max_denominator: int = 1000) -> tuple[int, int]:
    """(up, down) with rate_out / rate_in == up / down; raises if no small exact ratio exists."""
    ratio = Fraction(rate_out / rate_in).limit_denominator(max_denominator)
    if abs(float(ratio) * rate_in - rate_out) > 1e-6 * rate_out:
        raise ValueError(f"no rational ratio up/down <= {max_denominator} maps {rate_in} to {rate_out}")
    return ratio.numerator, ratio.denominator


def design_lowpass(up: int, down: int, attenuation_db: float = ATTENUATION_DB,
                   passband_share: float = PASSBAND_SHARE) -> np.ndarray:
    """Kaiser windowed-sinc low-pass for an up/down resampler, gain `up`, odd length.

    Frequencies are normalized to the input rate (fs_in = 1, so the upsampled rate is `up`).
    Passband edge: passband_share x the narrower Nyquist; stopband edge: that Nyquist.
    """
    nyquist = min(1.0, up / down) / 2  # narrower of input and output Nyquist, in units of fs_in
    f_pass = passband_share * nyquist
    f_stop = nyquist
    fs_up = float(up)
    delta_w = 2 * math.pi * (f_stop - f_pass) / fs_up
    taps = int(math.ceil((attenuation_db - 8.0) / (2.285 * delta_w))) + 1
    taps += 1 - taps % 2  # odd: integer group delay
    if attenuation_db > 50:
        beta = 0.1102 * (attenuation_db - 8.7)
    elif attenuation_db >= 21:
        beta = 0.5842 * (attenuation_db - 21) ** 0.4 + 0.07886 * (attenuation_db - 21)
    else:
        beta = 0.0
    cutoff = (f_pass + f_stop) / 2 / fs_up  # cycles per upsampled sample
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(taps, beta)
    h *= up / h.sum()  # unity DC gain after zero stuffing
    return h


class RationalResampler:
    """Stateful up/down resampler. `process` chunks in, `flush` once at the end.

    Output sample m sits at upsampled index j = m * down, i.e. input time
    (m * down - delay) / up input samples, with delay = (len(taps) - 1) / 2.
    The first outputs whose filter span reaches before the first input sample
    (the start-up transient) are dropped: output 0 is at j0 = first multiple of
    `down` that is >= len(taps) - 1.
    """

    def __init__(self, up: int, down: int, taps: np.ndarray | None = None, fft_size: int = FFT_SIZE):
        if up < 1 or down < 1:
            raise ValueError("up and down must be positive")
        g = math.gcd(up, down)
        self.up, self.down = up // g, down // g
        self.taps = np.asarray(design_lowpass(self.up, self.down) if taps is None else taps, dtype=np.float64)
        ntaps = len(self.taps)
        self.fft_size = max(fft_size, 1 << int(math.ceil(math.log2(4 * ntaps))))
        self.block = self.fft_size - (ntaps - 1)  # valid outputs (upsampled) per FFT
        self._H = np.fft.fft(self.taps, self.fft_size)
        self.delay = (ntaps - 1) / 2
        # Upsampled stream state: `self._buf` holds u[self._buf_start : ...].
        self._buf = np.zeros(ntaps - 1, dtype=np.complex128)  # zero history before the first sample
        self._buf_start = -(ntaps - 1)
        self._next_block = 0  # absolute upsampled index of the next block's first output
        self._first_j = -(-(ntaps - 1) // self.down) * self.down
        self._flushed = False

    @property
    def identity(self) -> bool:
        return self.up == 1 and self.down == 1 and len(self.taps) == 1

    def _upsample(self, x: np.ndarray) -> np.ndarray:
        if self.up == 1:
            return x.astype(np.complex128)
        u = np.zeros(len(x) * self.up, dtype=np.complex128)
        u[:: self.up] = x
        return u

    def _run_blocks(self, end_j: int, final: bool) -> np.ndarray:
        """Filter every complete block (or the last partial one when final) and pick outputs."""
        ntaps = len(self.taps)
        out = []
        buf_end = self._buf_start + len(self._buf)
        while True:
            j0 = self._next_block
            j1 = j0 + self.block
            if j1 > end_j:
                if not final or j0 >= end_j:
                    break
                j1 = end_j
            seg = self._buf[j0 - (ntaps - 1) - self._buf_start : j1 - self._buf_start]
            frame = np.zeros(self.fft_size, dtype=np.complex128)
            frame[: len(seg)] = seg
            v = np.fft.ifft(np.fft.fft(frame) * self._H)[ntaps - 1 : ntaps - 1 + (j1 - j0)]
            first = max(j0, self._first_j)
            first = -(-first // self.down) * self.down
            if first < j1:
                out.append(v[first - j0 :: self.down])
            self._next_block = j1
        keep_from = self._next_block - (ntaps - 1)
        if keep_from > self._buf_start:
            self._buf = self._buf[keep_from - self._buf_start :]
            self._buf_start = keep_from
        assert self._buf_start + len(self._buf) == buf_end
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, dtype=np.complex64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if self._flushed:
            raise RuntimeError("process() after flush()")
        u = self._upsample(np.asarray(chunk))
        self._buf = np.concatenate([self._buf, u])
        return self._run_blocks(self._buf_start + len(self._buf), final=False)

    def flush(self) -> np.ndarray:
        """Outputs still held back (the last partial block). Call once, at the end."""
        if self._flushed:
            return np.zeros(0, dtype=np.complex64)
        self._flushed = True
        return self._run_blocks(self._buf_start + len(self._buf), final=True)


def resample(x: np.ndarray, up: int, down: int, taps: np.ndarray | None = None) -> np.ndarray:
    """One-shot resampling (same output as RationalResampler on any chunking)."""
    r = RationalResampler(up, down, taps)
    return np.concatenate([r.process(x), r.flush()])
