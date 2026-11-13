# Burst Router

A local web page that classifies the bursts in an IQ recording. Every burst goes
through the three wave 5 models (2D CNN, kNN, LightGBM). The model that did best on
**validation** data for the burst's frequency band makes the call, and the other two
are shown beside it, so a disagreement is visible.

## Start it

```bash
cd mod-classifier
.venv-ui/bin/uvicorn ui.app:app --port 8765 --app-dir .
```

Then open http://localhost:8765. It opens on a test sample already classified.

First time on a new machine, build the environment (versions pinned to the ones the
models were trained with on CRC; no torch):

```bash
python3.12 -m venv .venv-ui && .venv-ui/bin/pip install "numpy==2.5.4" "scipy==1.18.1" "scikit-learn==1.9.1" "lightgbm==4.7.0" "h5py==3.16.0" "onnxruntime==1.31.1" fastapi uvicorn python-multipart matplotlib pytest httpx
```

It needs the wave 5 runs in `runs/crc/` (copied from CRC), `runs/crc/routing_w5.json`
(`scripts/build_routing.py`) and `runs/crc/cnn2d_v1_w5/model.onnx`
(`scripts/export_onnx.py`, run on CRC).

## What it does with a recording

1. Reads it: a raw `.cu8`, `.cs8`, `.cs16` or `.cf32` file (with its sample rate and
   centre frequency), or a SigMF `.sigmf-meta` + `.sigmf-data` pair. At most 40
   million samples are read.
2. Finds bursts with `pipeline.features.detect_bursts`, the detector the app uses.
3. Cuts each burst into 4096-sample windows at 20 MS/s with the same front end that
   cut the training windows. If the activity test rejects every window (an
   always-on signal such as a jammer), it retries without the test and marks the
   burst "always-on signal".
4. Runs all three models, each with its own calibration (temperature, and
   "unknown" below its threshold), averaged over the burst's windows.
5. Picks the model for the burst's band from the routing table and reports its
   class and confidence, plus every model's answer and whether they agree.

## How the model is chosen

`scripts/build_routing.py` reads each model's predictions on the validation split,
groups the windows by band, and picks the model with the best validation accuracy
in each band. A band with fewer than 2 validation recordings or 200 windows (or a
band never seen) goes to the overall best model, the 2D CNN. The test split is
used only to score the result:

| Strategy (test, 34,488 windows, 7 classes) | Accuracy | Macro-F1 |
|---|---|---|
| Route by band (used here) | 0.925 | 0.883 |
| 2D CNN alone | 0.914 | 0.871 |
| Average of all three | 0.915 | 0.864 |
| kNN alone | 0.861 | 0.825 |
| LightGBM alone | 0.827 | 0.758 |
| Oracle: always a model that is right (a ceiling, not reachable) | 0.944 | 0.907 |

## Test samples

`ui/samples/` (git-ignored, about 49 MB) holds short slices of real recordings from
the **test** split: DJI and Yuneec drone links (Tampere), two GNSS jammers (GATEMAN,
Triangular), Parrot Wi-Fi (DroneRF), two rtl_433 sensors and a LoRa frame.
`samples.json` gives each one's source, true class and licence.
`scripts/fetch_ui_samples.py` fetches them again (run it on CRC). NIST Wi-Fi and
Bluetooth are missing: their captures sit in a 21.6 GB HDF5 file that cannot be
sliced remotely.

## Limits

- Trained and tested on lab recordings only; nothing has been measured on the
  team's own radios.
- `pulsed_radar` was trained on simulated waveforms only and has no test data.
- `other_ism` comes from rtl_433_tests, which states no licence (used by Odon's
  decision on Nov 12; check with the team before sharing a model trained on it).
- Each band in the routing table comes mostly from one dataset, so a route may
  reflect that dataset as much as the signal type.
- The Yuneec Typhoon H 5.7 GHz link is labelled `drone_link` but every model calls
  it `wifi`; the label itself is still open (see `configs/taxonomy.toml`).
- When the models disagree, check the burst by hand.
