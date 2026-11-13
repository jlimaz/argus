# mod-classifier

Argus's signal classifier for raw IQ recordings: short windows of samples in, a signal class and a confidence out. It is separate from the Laya model in the main app, which reads three text fields per signal. It classifies a waveform; it never decodes content.

**State (Nov 13, 2026):** wave 5 is trained and its weights are committed. Current numbers, data gaps and next steps are in [STATUS.md](STATUS.md).

| Read | For |
| --- | --- |
| [STATUS.md](STATUS.md) | What is trained, how good it is, what the data lacks, what comes next |
| [ui/README.md](ui/README.md) | The Burst Router page: upload a recording, see each burst classified |
| [crc/README.md](crc/README.md) | Preparing data and training on Notre Dame's CRC |
| [DATA_ACQUISITION.md](DATA_ACQUISITION.md), [LICENCES.md](LICENCES.md) | Where each dataset comes from, how it is fetched, and its licence |
| [PLAN.md](PLAN.md) | The original design (Nov 11). Read it for the reasoning, not for the current state |
| [docs/history/](docs/history/) | The Nov 11 handoff and the Nov 12-13 overnight log, kept unchanged as a record |

## What it classifies

Eight classes (class set `v2c` in [configs/taxonomy.toml](configs/taxonomy.toml)): `drone_link`, `wifi`, `bluetooth`, `background`, `gnss_jammer`, `lora`, `pulsed_radar`, `other_ism`. A burst whose confidence falls under the model's threshold is reported as `unknown`.

Three wave 5 models are in `runs/crc/`, each with its `calibration.json`, `metrics.json`, `config.toml` and `meta.json`:

| Model | Run folder | Test accuracy | Test macro-F1 |
| --- | --- | --- | --- |
| 2D CNN on a spectrogram (also as ONNX) | `cnn2d_v1_w5` | 0.914 | 0.871 |
| kNN on 19 measurements | `baseline_knn_v1_w5` | 0.861 | 0.825 |
| LightGBM on the same measurements | `baseline_lightgbm_v1_w5` | 0.827 | 0.758 |

`runs/crc/routing_w5.json` says which model decides for each frequency band. All of this is measured on public lab recordings only, and `pulsed_radar` has no test recordings; see the limits in [STATUS.md](STATUS.md).

## Layout

| Path | What |
| --- | --- |
| `mcls/contracts.py` | The shared formats: `Recording`, `ManifestRow`, `Classifier`, the 20 MS/s, 4096-sample complex64 window, and `band_of()` |
| `mcls/taxonomy.py`, `configs/taxonomy.toml` | Class sets (v1, v2, v2a to v2c) and the label, strength and licence of every dataset's files |
| `mcls/acquire.py`, `mcls/fetch.py`, `configs/catalog.toml` | Downloads: only files listed in the catalog, resumable, checksum-verified |
| `mcls/sources/` | One loader per dataset format (Tampere, DroneRF, NIST Wi-Fi/BT, SigMF, GATEMAN, Triangular L1, Drexel radar, rtl_433) |
| `mcls/frontend/`, `mcls/preprocess.py` | Resample, find bursts, cut and normalize windows. Inference uses the same code |
| `mcls/dataset/`, `mcls/manifest.py` | The JSONL manifest, splits by recording (never by window), the leakage check, samplers and augmentation |
| `mcls/baselines/`, `mcls/models/`, `mcls/train/` | kNN, LightGBM and softmax baselines; the 1D, 2D and fused CNNs; training |
| `mcls/evaluate/` | Metrics, calibration (temperature and the `unknown` threshold) |
| `mcls/infer/` | Classify bursts with a trained run; used by the UI and by `python -m pipeline.ingest --classify` |
| `configs/` | Run configs, checked key by key so a typo fails before the job is queued |
| `crc/` | CRC setup, data preparation and training jobs |
| `scripts/` | Helpers: `build_routing.py`, `export_onnx.py`, `fetch_ui_samples.py`, file lists, shard merging |
| `ui/` | The Burst Router (FastAPI and one static page) |
| `runs/crc/` | The trained runs, committed on purpose. Everything else under `runs/` is git-ignored |
| `tests/` | pytest; torch, scikit-learn and LightGBM tests are skipped where those are not installed |

## Use the trained models

Start the Burst Router as described in [ui/README.md](ui/README.md). It needs its own environment, `.venv-ui`, with library versions pinned to the ones used in training, because the kNN and LightGBM models are pickles.

From the main app's command line, add the class to each burst found in an IQ file:

```bash
# from the repository root
python -m pipeline.ingest capture.cu8 --sample-rate 20M --center-freq 2.437G \
    --classify --model mod-classifier/runs/crc/baseline_lightgbm_v1_w5 -o signals.jsonl
```

## Run the tests

```bash
cd mod-classifier && ../.venv/bin/python -m pytest tests -q
```

The tests must run from `mod-classifier/` (`conftest.py` puts it on the import path). The repository CI does not run them yet; [crc/CI_PROPOSAL.md](crc/CI_PROPOSAL.md) is the draft for adding them.

## Retrain

Training runs on CRC, not on a laptop: see [crc/README.md](crc/README.md). The prepared windows and `data/manifest_w5.jsonl` are not in git; what is needed to get them is listed in [STATUS.md](STATUS.md#6-replicating-on-crc).

## Not in git

`data/` (raw recordings, windows, manifests), `runs/` other than `runs/crc/`, `.venv-ui/` and `ui/samples/`.
