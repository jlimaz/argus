"""DSP front end: raw native-rate samples to unit-RMS 4096-sample windows at 20 MS/s.

- `resample`: numpy rational windowed-sinc resampler (chunked == one-shot)
- `channelize`: choose the 20 MHz slice with most energy, mix to DC, resample
- `windows`: burst / idle windows via pipeline.features.detect_bursts, normalize
- `views`: I/Q (2 x 4096) and log STFT (1 x 64 x 64) model inputs
- `shards`: streaming .npy shards and their JSONL rows
- `process`: one Recording to shards, rows and stats (used by `python -m mcls.preprocess`)
"""
