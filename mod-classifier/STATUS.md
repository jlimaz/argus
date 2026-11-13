# mod-classifier: status and next steps

Written by Gabriel on 2026-11-13 from `main` at `49467c6`, as a review of the work Odon merged in PR #7. Update this file when a new wave is trained or the data changes.

**Decision (2026-11-13):** next work continues from `mod-classifier/`. Gabriel's earlier branches `feat/crc-nist-radar` and `feat/tf-detector` are superseded; they were never pushed and are kept as local tags `archive/crc-nist-radar` and `archive/tf-detector` in his checkout.

## 1. What Odon pushed

| PR | Merged | What |
|---|---|---|
| #4 `feat/capture-ingest` | 2026-11-11 | `pipeline/sweep.py`, `pipeline/iq.py`, `pipeline/ingest.py`: turn rtl_power / hackrf_sweep CSVs and cu8 / cs8 / SigMF recordings into Argus data files (`modulation: "unknown"`). Input side of the main Argus app only. |
| #7 `feat/modulation-classifier` | 2026-11-13 | New `mod-classifier/` folder (134 commits, ~54k lines): dataset loaders, IQ windowing front end, training on CRC, trained weights, routing table, and the Burst Router UI. |

`mod-classifier/` is a **separate classifier** from the Laya / tabular-LightGBM path in the main app. It reads raw IQ: each burst is resampled to 20 MS/s and cut into 4096-sample complex64 windows.

Other docs in the folder: [README.md](README.md) (entry point), [PLAN.md](PLAN.md) (original design), [DATA_ACQUISITION.md](DATA_ACQUISITION.md), [LICENCES.md](LICENCES.md), and [docs/history/](docs/history/) (the Nov 11 handoff and the overnight training log with its morning report).

## 2. Models (wave 5, class set `v2c`)

Classes: `drone_link, wifi, bluetooth, background, gnss_jammer, lora, pulsed_radar, other_ism`.
Weights are committed in `mod-classifier/runs/crc/` (checksums in `SHA256SUMS`), each with `calibration.json` (temperature, "unknown" threshold), `metrics.json`, `config.toml` and `meta.json`.

Test set: 34,488 windows from 141 recordings no model trained on, 7 classes present.

| Model | Accuracy | Window macro-F1 | Burst macro-F1 |
|---|---|---|---|
| **2D CNN** (`cnn2d_v1_w5`, best epoch 17 of 25, RTX 6000, ~10 min/epoch) | **0.914** | **0.871** | **0.854** |
| kNN (`baseline_knn_v1_w5`) | 0.861 | 0.825 | 0.761 |
| LightGBM (`baseline_lightgbm_v1_w5`) | 0.827 | 0.758 | 0.756 |
| Majority | 0.249 | 0.057 | 0.010 |

The CNN is also exported to ONNX (`cnn2d_v1_w5/model.onnx`, max logit difference against torch 3.8e-6), so inference does not need torch.
Reproduce: commit `b4a9300`, seed 1, manifest `data/manifest_w5.jsonl` (sha256 `fdf44b09…`, **not in git**).

### How the three models differ

They see different inputs:
- **2D CNN** (`mcls/models/cnn2d.py`): the window as a 64×64 spectrogram, an image of frequency against time. It learns its own visual features from that image.
- **kNN and LightGBM** (`mcls/baselines/`): 19 hand-made measurements per window (`mcls/baselines/features.py`): bandwidth, SNR, duty cycle, burst duration, pulse interval, hopping, spectral flatness and peak count, occupied-bandwidth ratios, envelope kurtosis, peak-to-average power ratio. Anything not on that list is invisible to them.

| | 2D CNN | kNN | LightGBM |
|---|---|---|---|
| How it decides | Neural network trained on many examples | Vote of the most similar training windows | Many small decision trees, each correcting the previous ones (up to 2,000, stopped early on validation) |
| Training | Hours on a GPU (~4–5 h on CRC) | Almost none; stores the training set | Minutes on CPU |
| Saved size | ~5 MB | ~21 MB (keeps the training set) | Small |
| Strength | Fine detail in shape and timing; best overall | Simple; good when classes cluster | Fast; shows which measurements matter |
| Weakness | Needs a lot of data and a GPU; hard to explain | Slow on big data; sensitive to feature scaling | Limited to the 19 measurements |

For improving the models: more varied data helps the CNN most. Better measurements help kNN and LightGBM, for example one that tells a LoRa chirp from an rtl_433 burst.

## 3. Routing table (the current "lookup table")

`mod-classifier/runs/crc/routing_w5.json`, built by `scripts/build_routing.py` from **validation predictions only**. A burst's band picks the model; bands with fewer than 2 validation recordings or 200 windows fall back to the CNN.

| Band | Model | Val windows | Val recordings |
|---|---|---|---|
| gnss_l1 | 2D CNN | 9,363 | 2 |
| ism_2400 | LightGBM | 14,405 | 56 |
| ism_433 | 2D CNN | 5,245 | 40 |
| ism_5800 | LightGBM | 2,834 | 30 |
| ism_915 | 2D CNN | 369 | 6 |
| other | 2D CNN | 6,126 | 5 |
| srd_868 | 2D CNN | 294 | 5 |

Test scores by strategy:

| Strategy | Accuracy | Macro-F1 |
|---|---|---|
| Route by band (used) | 0.925 | 0.883 |
| Route by bandwidth | 0.923 | 0.869 |
| Average of all models | 0.915 | 0.864 |
| 2D CNN alone | 0.914 | 0.871 |
| Oracle (always picks a model that was right; not achievable) | 0.944 | 0.907 |

Caveat: each band is dominated by one dataset, so a route may reflect the dataset more than the signal type.

### Is per-band routing the right call?

**Feasible:** yes, and it already works. The UI runs all three models on every burst anyway, and picking one is a table lookup. The cost is maintenance: three models to retrain and calibrate, and the table rebuilt from validation data after every retrain.

**Standard:** partly.
- Combining models is routine in machine learning: voting, averaging, or stacking (a small extra model learns how to combine the others). A fixed per-situation rule is the simplest form.
- Band-specific handling is natural in RF, because each band is reserved for particular signal types. Detect, then band and width, then a band-specific classifier is a common staged design.
- Mixing a CNN with kNN and LightGBM is less usual. Recent RF classifiers tend to use one deep network trained on broad data, so this is closer to a patch over gaps in the training data.

**Why it is provisional:**
- The gain is small: 92.5% / 0.883 against 91.4% / 0.871 for the CNN alone, on only 141 test recordings. It may be noise.
- Several routes rest on very few validation recordings: gnss_l1 2; srd_868, ism_915 and other 5–6.
- Band and dataset are confounded. LightGBM may win at 2.4 GHz because it fits DroneRF and NIST quirks, which could flip on the team's own captures.

**How to settle it:** resample the test set by recording (bootstrap) to see whether routing beats the CNN alone beyond chance. Compare against (a) one model given the band as an input (Odon's code already has a band-prior hook, `mcls.infer.priors`) and (b) a learned combiner trained on validation. Keep routing if it still wins once the data gaps are filled.

## 4. Burst Router UI

`mod-classifier/ui/` (FastAPI + static page). Upload an IQ file (`.cu8`, `.cs8`, `.cs16`, `.cf32`, SigMF); every burst goes through all three models, the routing table picks the one that decides, and the page shows a spectrogram with the bursts marked, the confidence, and each model's answer side by side.

```bash
cd mod-classifier
.venv-ui/bin/uvicorn ui.app:app --port 8765 --app-dir .   # then open http://localhost:8765
```

`.venv-ui` must match CRC's library versions, because kNN and LightGBM are pickles: numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, onnxruntime 1.31.1 (no torch).
Results on 8 real test samples: DJI drone link, Parrot Wi-Fi, LoRa and both jammers correct; rtl_433 13 of 20 bursts `other_ism`; Yuneec 5.7 GHz called `wifi` (its label is undecided). Sample files are git-ignored and not in the repo.

## 5. The data

All public recordings, real IQ. 13 datasets in `configs/taxonomy.toml`: tampere, dronerf, nist_wifi_bt, occupancy, dronedetect, fgi_jammer, gateman, triangular_l1, loraiq, drexel_cbrs_radar, adsb (not used: needs IEEE DataPort login), nist_radar, rtl_433 (no licence stated; used by Odon's decision).

| Class | Main sources |
|---|---|
| drone_link | Tampere, DroneRF, DroneDetect |
| wifi, bluetooth | NIST Wi-Fi/BT, DroneRF |
| gnss_jammer | GATEMAN, Triangular L1, FGI |
| lora | LoRaIQ |
| pulsed_radar | NIST radar, Drexel CBRS radar |
| other_ism | rtl_433 |
| background | occupancy |

Size: 251,532 windows in the manifest; training 158,873 real + 19,535 synthetic; recordings split by recording, leakage check passed.

Test set per class (2D CNN):

| Class | Test windows | Test recordings | Precision | Recall | F1 |
|---|---|---|---|---|---|
| gnss_jammer | 9,354 | 2 | 1.00 | 1.00 | 1.00 |
| background | 3,952 | 1 | 1.00 | 1.00 | 1.00 |
| wifi | 8,581 | 56 | 0.99 | 0.93 | 0.96 |
| bluetooth | 4,547 | 27 | 0.88 | 1.00 | 0.93 |
| drone_link | 546 | 2 | 0.87 | 0.71 | 0.78 |
| lora | 4,629 | 3 | 0.92 | 0.59 | 0.72 |
| other_ism | 2,879 | 50 | 0.58 | 0.92 | 0.71 |
| pulsed_radar | 0 | 0 | – | – | – |

What stands out:
- **pulsed_radar is not tested at all** (no test recordings).
- **Few test recordings** for gnss_jammer, background, drone_link and lora (1 to 3 each): their scores rest on very little.
- **lora / other_ism confusion**: many LoRa windows are called `other_ism` (low LoRa recall, low other_ism precision).
- **Strong imbalance**: drone_link has 546 test windows against 9,354 for gnss_jammer.
- **Lab data only**: nothing from the team's own captures; no NIST Wi-Fi/BT sample in the UI.

## 6. Replicating on CRC

| Goal | Rough time | How |
|---|---|---|
| Use the trained models | < 1 hour | Build `.venv-ui` with the pinned versions, run the UI or `mcls.infer`. |
| Retrain from Odon's prepared data | ½–1 day | Read access to `~oineza/mcls_data` (or Globus copy) + `data/manifest_w5.jsonl`; `crc/setup_env.sh`, `crc/submit_wave.sh`. |
| Rebuild from scratch | 2–4 days wall clock | Downloads dominate (sources total ~1 TB if fetched whole; NIST radar alone 493 GB). Zenodo ~6 h per 50 GB file on 4 connections; FGI on Fairdata is single-stream with no resume. Then prepare array jobs (hours) and CNN training (~4–5 GPU hours plus queue). |

Blockers: home quota is 100 GB (need `/groups/...` space; raw files stream through node `/tmp`), and the exact subsets Odon downloaded are not recorded in git (`runs/file_lists/` stayed on his machines).

## 7. Next steps

1. Get from Odon: read access to `~oineza/mcls_data` on CRC, `data/manifest_w5.jsonl`, and `runs/file_lists/`.
2. Data audit, per class and per dataset: recordings, windows, SNR, bands. Biggest gaps first: pulsed_radar test recordings, more lora / drone_link / gnss_jammer / background recordings so the test scores rest on more than 1 to 3 recordings, NIST Wi-Fi/BT coverage.
3. Look at the lora ↔ other_ism confusion (likely band overlap at 868/915 MHz with rtl_433 devices).
4. Lookup table: start from `routing_w5.json` and `scripts/build_routing.py`. First bootstrap routing against the CNN alone (section 3), and compare with the band-prior and learned-combiner options. Then decide what the table keys on (band, bandwidth, both, SNR), and rebuild it on validation data only after each data change.
5. Settle open labels: Yuneec 5.7 GHz, and DroneRF as `drone_link` (Odon) vs `wifi` (Gabriel's branch).
