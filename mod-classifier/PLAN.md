# mod-classifier: build plan

**Status:** this is the original design, written Nov 11, 2026 before anything was built, and kept as written. The pipeline has since been built and trained through wave 5; for the current state read [STATUS.md](STATUS.md). Where the two differ, STATUS.md is right.
**Owner:** Odon (Ingest and Detection). **Format:** an orchestrator (the senior engineer who owns the design and every merge) directs 10 senior engineers, each owning one workstream with a written interface.

Everything under `mod-classifier/` is the training side of this feature: data fetchers, preprocessing, models, training, evaluation, export. Nothing here runs inside the live app until the integration step (section 11).

**Decisions as of Nov 11 (from Odon):**
- **No radio captures.** The team will not record with its own RTL-SDRs or HackRF for this work. Every test set is a held-out slice of public recordings, and the plan says plainly that nothing is measured on the team's own hardware.
- **Odon has a CRC account.** Training runs on CRC, where PyTorch is expected to be available; the laptop does not need PyTorch.
- **The orchestrator decides the requirements** (Python packages and versions) as part of wave 0.
- **The earlier list of open decisions is closed**; the defaults in this plan stand.

**Wave 0 (done Nov 11, orchestrator):** contracts (`mcls/contracts.py`), taxonomy (`configs/taxonomy.toml`), manifest with splits and leakage check (`mcls/manifest.py`), checked run configs (`mcls/config.py`, `configs/*.toml`), the reference Tampere loader (`mcls/sources/tampere.py`, verified identical to the dataset's own reader on the real files), the Zenodo downloader (`mcls/fetch.py`), CRC `requirements.txt`, and 42 tests. Decisions taken: configs in TOML (no extra package), manifest in JSONL, the class field is called `label`.

---

## 0. What we are building, in one paragraph

A small neural network that takes a short window of raw IQ samples and returns a probability for each signal class, using a **softmax** output layer trained by **gradient descent** (cross-entropy loss, AdamW, backpropagation). It runs on the bursts that `pipeline/features.py` already finds, so the app's order stays: detect, then classify, then Judge. The output is a class and a calibrated confidence per burst, and low confidence is never trusted (SRS NFR-4: low confidence ranks white, never blue).

### Two ideas that change the design

1. **"Demodulation" is the wrong word, and the spec forbids it.** Demodulation recovers the message. The SRS (NFR-7) says Argus measures signals and never decodes content. This model only *classifies* a waveform. Use "signal classifier" in the pitch and in code.
2. **The band is not something to learn.** The band (FM, ADS-B, 433/915 MHz, 2.4 GHz, 5.8 GHz, S-band) comes straight from the centre frequency. The model's job is to say *what kind of signal sits in the band*. The band is used afterwards as a prior (section 5.7), not as a label.

### Non-goals

- No demodulation, bit recovery, or content decoding, ever.
- No transmitting and no generating jamming. Receive-only (SRS C-1).
- No claim that this works on drones in general. It works on the emitters in its training data, and we report that plainly.
- No synthetic data in any headline number (section 3.4).

---

## 1. Decisions made up front

| Decision | Choice | Why |
| --- | --- | --- |
| Task | Window-level signal classification, aggregated to burst level | Bursts are what `detect_bursts` outputs; windows give enough training examples |
| Output | Softmax over K classes plus an "unknown" decision by confidence | One calibrated probability per class; unknown handled by thresholding, not by a fake class |
| Learning | Cross-entropy, backpropagation, AdamW (SGD with momentum as an ablation) | Standard, stable, fits a laptop |
| Framework | PyTorch, on CRC only | `laya` already depends on it; CRC's CUDA GPUs run it. The laptop never installs it |
| First model | Handcrafted features into softmax regression and LightGBM | Cheap baseline. A network that cannot beat it has not earned its complexity |
| Main model | 1D residual CNN on I/Q (about 1M parameters), with a 2D spectrogram CNN as an ablation | Small enough to train on the laptop and run on a laptop CPU |
| Labels | Real recordings only for reported results | You asked for no more simulations; section 3.4 says how synthetic data is kept out of results |
| Splits | By recording, never by slice | A random window split leaks near-identical signal into test and measures memory, not recognition |
| Compute | Laptop builds and tests the pipeline; CRC trains and prepares bulk data | The laptop has no PyTorch and keeps under about 1 GB of raw data. All training, baselines included, runs on CRC through shell scripts |

---

## 2. Classes

The classes are **physical signal families first**, because that is what IQ can actually show. Whether something is a drone is a *judgement* that the Judge makes from the class, the burst pattern and the site profile.

### 2.1 Version 1 (trainable from real data we can get)

| Class | Meaning | Main real source |
| --- | --- | --- |
| `drone_link` | Control or video link of a known consumer drone | Tampere, DroneRF, DroneDetect |
| `wifi` | Wi-Fi OFDM traffic (2.4 and 5 GHz) | NIST Wi-Fi/Bluetooth, occupancy dataset |
| `bluetooth` | Bluetooth and BLE hopping bursts | NIST Wi-Fi/Bluetooth, DroneDetect interference sets |
| `background` | Idle band: noise floor and ambient with no burst | DroneRF no-drone files, occupancy dataset idle tiles |

### 2.2 Version 2 (added when data exists, flagged)

| Class | Meaning | Source | Caveat |
| --- | --- | --- | --- |
| `pulsed_radar` | Pulsed S-band radar (the "red" scenario) | Drexel CBRS radar through a USRP (waveforms simulated, receive chain real); NIST 3.5 GHz radar (simulated) | **Emulated radar only.** No real radar recordings found. Excluded from headline metrics |
| `gnss_jammer` | Jammers on GPS L1 (tone, chirp, triangular, noise) | GATEMAN, triangular-jammed L1, FGI Jammer Test 2023 | Lab-generated jammers, no field jammers. Needs a clean L1 baseline (a candidate recording is not yet opened) |
| `adsb` | 1090 MHz ADS-B pulses (a project focus band) | 24-hour ADS-B recording (IEEE DataPort) | Small, real; login |
| `lora` | LoRa sub-GHz ISM | LoRaIQ (EPFL) | 862.5 MHz (Europe), not 915 MHz |
| `other_ism` | Consumer 433/915 MHz devices | `rtl_433_tests` | Not opened; small |

### 2.3 Label ambiguity to resolve first (taxonomy task)

- **Some drones are Wi-Fi or Bluetooth physically.** In DroneRF the Parrot Bebop and AR Drone are Wi-Fi-based, and the Parrot Mambo in Tampere is believed to use Bluetooth LE. At the signal level they look like `wifi` or `bluetooth`. To be verified, not assumed. If confirmed, those files get a *physical* class (`wifi`, `bluetooth`) plus an `emitter_id` field (the drone model). The classifier predicts the physical family; a second step or the Judge decides drone versus ordinary Wi-Fi from burst pattern, emitter match and site profile.
- **Weak labels.** NIST Wi-Fi/Bluetooth labels are per *file* (environment and band), not per burst. Windows inside them are filtered by an energy detector and treated as weakly labeled.

Output of this task: `configs/taxonomy.toml` mapping every dataset file or folder to a class, an optional `emitter_id`, and a confidence flag (`strong` / `weak`).

---

## 3. Data

### 3.1 Inventory

All sizes, licences and formats below were read from each dataset's own page on Nov 11, except where marked. **Nothing is downloaded yet; every download needs Odon's approval.**

| Dataset | Role | Real? | Labels | Size | Licence | Access | Phase |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Tampere drone recordings](https://zenodo.org/records/4264467) | `drone_link`, 2.44 and 5.8 GHz | Real (anechoic chamber) | Drone model, control vs video | 8.6 GB | CC BY 4.0 | Direct | 1 |
| [DroneRF](https://data.mendeley.com/datasets/f4c2b4n755/1) | `drone_link` and **`background`** (41 no-drone files) | Real (lab) | Drone, flight mode, present/absent | about 4 GB (per repo README) | CC BY 4.0 | Direct | 1 |
| [NIST Wi-Fi and Bluetooth I/Q](https://catalog.data.gov/dataset/wi-fi-and-bluetooth-i-q-recordings-in-the-2-4-ghz-and-5-ghz-bands-with-low-cost-software-d) | `wifi`, `bluetooth`, ambient | Real | Per file (band, indoor/outdoor, Bluetooth) | 5 files, 20.1 GB each | NIST open licence | Direct | 2 (one file first) |
| [Occupancy dataset (IEEE DataPort)](https://ieee-dataport.org/documents/multi-environment-real-world-multi-band-rf-dataset-spectrum-sensing-and-occupancy) | `background` and ambient `wifi`/`bluetooth` across 7 places | Real | Energy-detector occupancy | about 84 GB | CC BY 4.0 | IEEE login | 2 |
| [DroneDetect](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine) | `drone_link` with and without Wi-Fi/Bluetooth interference | Real | Drone, mode, interference condition | 65.8 GB | CC BY 4.0 | IEEE login | 3 (robustness) |
| [FGI Jammer Test 2023](https://research.fi/en/results/dataset/2eee8cfd-db00-4e46-bd62-d7632c09c700) | `gnss_jammer` (noise jammer, 400 s on L1, 30.69 MS/s) | Real | Scenario | not stated | CC BY 4.0 | Fairdata | 3 |
| [GATEMAN](https://zenodo.org/records/2654322) | `gnss_jammer`: tone and 10 and 20 MHz chirp jammers on GPS L1, L5, Galileo E1; 9 scenarios, 20 MS/s | Real (generator plus USRP) | Scenario folders | 48.7 GB (one zip) | CC BY 4.0 | Direct | 3 |
| [Triangular-jammed GPS L1](https://zenodo.org/records/14859381) | `gnss_jammer`: triangular jammer at 5 power levels, 10 MS/s, gzip int8 | Real | Power level | 49.9 GB (part 1) | CC BY 4.0 | Direct | 3 |
| [LoRaIQ (EPFL)](https://zenodo.org/records/17708397) | `lora`: 75,000+ frames, SigMF, 862.5 MHz | Real | Frame metadata | 49.8 GB | CC BY 4.0 | Direct | 3 |
| [Drexel CBRS radar through a USRP](https://zenodo.org/records/4521678) | `pulsed_radar` (radar vs noise), 10 MS/s | Simulated waveforms, real receiver, cabled | Radar or noise | 34.1 GB | CC BY 4.0 | Direct | 3 |
| [ADS-B 24-hour recording](https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot) | `adsb`, 130+ aircraft, 8 MS/s | Real | Flight metadata | about 254 MB raw | CC BY 4.0 | IEEE login | 3 |
| [NIST 3.5 GHz radar](https://data.nist.gov/od/id/mds2-2116) | `pulsed_radar` | **Simulated** | Waveform table | 200 files, 2.5 GB each (about 500 GB) | Public domain | Direct | 3 (small subset only) |
| Panoradio HF (already in repo scripts) | Auxiliary modulation head, HF only | Simulated conditions | Mode | about 5.7 GB | Unconfirmed | Direct | optional |
| `rtl_433_tests` (not opened yet) | `other_ism`; checks the real RTL-SDR `.cu8` layout | Real | Decoded device | small | GPL-2.0 (per QA log) | GitHub | 2 |

**Not confirmed, not planned:** CageDroneRF (reported 500 GB or more, with no-drone recordings; the abstract gives no bands, format or counts, access may need a request form, and the data licence is unconfirmed). Oregon State LoRa, the Northeastern CBRS LTE-and-radar set and the AERPAW LTE set were found but not opened.

**Excluded on purpose:** RFUAV and Spectrum Highway set 2 (CC BY-NC-SA, non-commercial), RadioML 2018 (non-commercial and reported label errors), CardRF until its licence is recorded, WireTapper (not recommended in the QA log).

### 3.2 No own captures: what that costs

There will be no recordings from the team's radios. Consequences, stated once here and repeated in the model card:

- Every public set was recorded with other radios, rates and places. The model is tested only on **held-out recordings from those sets**, so no number says how it behaves on an RTL-SDR or HackRF on campus.
- The strongest substitute is a **cross-dataset test**: train on one source of a class and test on a different source of the same class (for example Tampere drones versus DroneRF drones, NIST Wi-Fi versus the occupancy dataset). It is reported next to the in-dataset number.
- The `.cu8` (RTL-SDR) path is checked only on public sample files (`rtl_433_tests`), never on a team capture.

### 3.3 Canonical record and manifest

Every loader converts its dataset into the same thing, so the rest of the pipeline never sees dataset quirks.

**Shard format:** `complex64` NumPy arrays, one file per recording segment, already **resampled to 20 MS/s and centred on the burst** (section 4).

**Manifest** (`data/manifest.jsonl`, one row per window; a Parquet copy only if it outgrows JSONL):

| Field | Meaning |
| --- | --- |
| `window_id`, `shard`, `offset`, `length` | Where the samples are |
| `dataset`, `recording_id` | **`recording_id` is the split key** |
| `label`, `emitter_id`, `label_strength` | From the taxonomy (`strong` or `weak`) |
| `native_rate_hz`, `center_freq_hz`, `band` | From metadata, kept for priors and reports |
| `snr_db_est`, `bw_hz`, `duty_cycle` | From `pipeline/features.py` |
| `is_real` | `false` for NIST radar and anything synthetic |
| `license` | Recorded per window's dataset, so no unlicensed data reaches a shared model |
| `split` | `train`, `val`, `test`, or `cross_test` |

### 3.4 Real versus synthetic

- Real data only in `test` and `cross_test`, and in every number we report.
- Synthetic or simulated data (NIST radar, TorchSig if ever used) may appear in training, flagged `is_real=false`, and every report prints the metric with and without those rows.
- **Augmentation of real data** (noise, offsets, gain) is allowed in training because it changes real samples; it is described in section 5.4.

### 3.5 Splits and leakage rules

- Group split by `recording_id`; `val` and `test` hold whole recordings (or whole drone models for a "generalize to new drone" test, run separately).
- A drone model appears in train *or* the generalization test, never both.
- `cross_test` (a whole dataset held out for a class that has two sources) is touched once per model version, after the model is frozen.
- A leakage audit (agent J) hashes windows and checks no sample or recording appears across splits.

### 3.6 Expected volume (estimates, to be replaced by the manifest)

- Tampere: about 60 to 70 thousand 4096-sample windows at 20 MS/s before burst filtering.
- NIST Wi-Fi/Bluetooth: millions of windows; **cap each class at about 30 thousand windows per dataset** so it does not swamp the others.
- Cap per class overall and use a balanced sampler (section 5.3).

### 3.7 Space budget: keep windows, not raw files

- Raw data adds up to hundreds of GB (NIST Wi-Fi/Bluetooth about 100 GB, GATEMAN 48.7 GB, triangular 49.9 GB, LoRaIQ 49.8 GB, NIST radar about 500 GB). Raw files are **never kept** on CRC: each task downloads one file into the node's `/tmp`, cuts the windows, writes only the windows, and deletes the file (the pattern Gabriel wrote for NIST radar).
- One `complex64` window of 4096 samples is **32 KB**. 30,000 windows per class is about **1 GB per class**; 8 classes is about 8 GB. Storing int16 halves that.
- The processed set therefore fits in a 100 GB CRC home directory. Raw data does not, so it must stay transient.
- Possible shortcut, not tested: Zenodo serves files over HTTP, so a task might read single members of a large zip by range request instead of fetching the whole archive. Test before relying on it.
- **Zenodo caps each connection at about 0.6 MB/s** (measured Nov 11 from Odon's laptop, against 18 MB/s from a CDN on the same line). Four parallel byte-range connections gave 2.4 MB/s, so the cap is per connection. At one connection a 50 GB archive takes about 23 hours. Fetchers therefore download in parallel byte ranges (at most about 8 per file, to avoid being blocked), resume after drops, and verify checksums; array jobs fetch different files at the same time. Measure once from a CRC front end before planning a large download.
- Byte-order traps: GATEMAN is big-endian int16; the triangular set is gzip int8; Tampere is little-endian int16. Each loader gets a test on a real file header.

---

## 4. Front end (raw IQ to model input)

Reused code: `pipeline/iq.py` (readers) and `pipeline/features.py` (`detect_bursts`). New code: the channelizer and windowing.

1. **Read** the file in chunks (`iq.py`; add readers for HDF5 and Tampere's int16 `.bin`).
2. **Find the active sub-band.** Source rates run from 15 to 200 MS/s but the HackRF sees 20 MHz. Take a coarse STFT, find the 20 MHz slice with the most energy, shift it to baseband and **decimate to 20 MS/s** with a polyphase filter.
3. **Find bursts** with `detect_bursts` at 20 MS/s.
4. **Cut windows** of **4096 samples (about 205 microseconds)** from inside each burst, 50% overlap, at most N per burst. For `background`, cut windows from idle stretches the detector marked quiet.
5. **Normalize** each window to unit RMS. No absolute power ever enters the model (matches `features.py`: ratios only).
6. **Output** two views: raw `2 x 4096` (I, Q) and a log-magnitude STFT `1 x 64 x 64`.

Why 4096 samples: DroneRF bursts last tens of milliseconds with pulse intervals of about 0.1 ms, so one window holds real within-burst structure. A burst's final answer averages the logits of its windows.

---

## 5. Models and training

### 5.1 Baselines (agent F builds these first)

- Inputs: the features `detect_bursts` already writes (bandwidth, SNR, duty cycle, burst duration, pulse interval, hopping) plus spectral shape statistics (spectral flatness, peak count, occupied-bandwidth ratios, kurtosis of the envelope).
- Models: multinomial **softmax regression** (trained by gradient descent), **LightGBM**, **k-nearest neighbours**, and a **majority-class** floor.
- These also run the "modulation unknown" style question: how much does the classifier add over the features the Judge already has?

### 5.2 Main network: 1D residual CNN

```
input  2 x 4096 (I, Q), unit RMS
stem   Conv1d(2 -> 32, k=7, stride=2) -> BatchNorm -> GELU
stage1 2 x ResBlock(32,  k=5)  -> downsample x2
stage2 2 x ResBlock(64,  k=5)  -> downsample x2
stage3 2 x ResBlock(128, k=5)  -> downsample x2
stage4 2 x ResBlock(256, k=3)  -> downsample x2
pool   global average + global max, concatenated (512)
head   Dropout(0.2) -> Linear(512 -> K)  -> softmax
```

About 1 million parameters (about 4 MB), so the checkpoint fits in git and on a laptop CPU at well under 50 ms per burst (target; to be measured).

**Ablations (each one config file):** spectrogram CNN on the 64 x 64 view; both views fused; narrower and wider stems; with and without the auxiliary head (5.6).

### 5.3 Training recipe (softmax and gradient descent)

- **Loss:** cross-entropy on the softmax output with label smoothing 0.1; weak-label windows get weight 0.5; class weights or a balanced sampler for imbalance; focal loss as an ablation.
- **Optimizer:** AdamW, learning rate 1e-3 peak, weight decay 1e-2, one-cycle schedule (warm-up then cosine decay); SGD with Nesterov momentum 0.9 and the same schedule as the ablation.
- **Batch and length:** batch 256, 30 to 60 epochs, gradient clipping at 1.0, early stopping on validation **macro-F1**.
- **Stabilizers:** exponential moving average of weights (decay 0.999), seeded runs, 3 seeds for any number we quote.
- **Device:** `cuda` with mixed precision on CRC; `cpu` only for the tiny smoke runs in tests. No training on the laptop.
- **Reproducibility:** every run writes its config, manifest hash, git commit, seed and metrics to `runs/<name>/`.

### 5.4 Augmentation (training split only, on real windows)

- Add white noise to reach a random lower SNR (0 to 30 dB). Never raise SNR.
- Random frequency offset up to 5% of the occupied bandwidth; random time shift and crop; random global phase; random gain.
- Small resampling jitter (plus or minus 1%) to cover radio clock error.
- Mild 2 to 3 tap multipath on a fraction of windows.
- **Interference mixing (version 2):** add a window of a different class at a random signal-to-interference ratio. Label stays the stronger signal only above a set threshold; otherwise the window is dropped. Off by default.

### 5.5 Calibration and "unknown"

- **Temperature scaling** fitted on validation recordings (the same idea the Laya checkpoint uses), reporting expected calibration error before and after.
- **Unknown:** threshold on maximum softmax probability, chosen at 95% of validation windows kept. Energy score as an ablation. Below threshold, the burst is `unknown` and the Judge treats it as white. This implements SRS FR-19 and NFR-4, and gives the missing input for open issue OI-6 (the blue/white threshold): a number backed by a validation curve instead of a guess.

### 5.6 Optional auxiliary head: modulation family

Where labels exist (Panoradio), a second softmax head predicts a modulation family (on-off keying, FSK, PSK/QAM, OFDM, noise-like). Loss is masked where the label is missing. It is pretraining and a feature for the Judge; it **does not** write the `modulation` field of rows until it is evaluated on real 2.4 GHz data. Dropped if it does not help the main head.

### 5.7 Band prior at inference

Multiply class probabilities by a band-plausibility table built from the NTIA allocation chart (a 5.8 GHz drone link is plausible; a drone link at FM is not), then renormalize. The table is data, reviewed by Diana, kept in `configs/band_priors.toml`. A softer prior than a hard mask, so a surprise stays visible.

### 5.8 Domain shift between datasets

No team captures exist, so adaptation targets the gap *between* public datasets:

1. **AdaBN:** re-estimate BatchNorm statistics on unlabeled windows of the held-out dataset; report the cross-dataset score with and without it.
2. **Resampling and filtering to the HackRF's 20 MHz** (section 4) so every source looks like what a HackRF would see, even though no HackRF data is used.

---

## 6. Evaluation (agent H) and acceptance criteria

### 6.1 What gets reported, always

- Macro-F1, per-class precision and recall, confusion matrix, on `test` (real only), with and without simulated training rows.
- **Burst-level** results (windows aggregated), since bursts are what the Judge sees.
- Accuracy versus SNR curve.
- Calibration: reliability diagram, ECE before and after temperature scaling.
- Unknown handling: coverage versus accuracy curve.
- **Generalization test:** hold out one drone model entirely.
- **Cross-dataset test:** `cross_test`, a whole dataset held out for a class, run once.
- **False-alarm test:** held-out `background` recordings (DroneRF no-drone files, occupancy idle tiles); share of windows flagged as non-background.
- Baseline comparison on identical splits.
- Compute: parameters, latency per burst on laptop CPU and MPS.

### 6.2 Acceptance criteria (to agree with the team, not claimed yet)

| Gate | Criterion |
| --- | --- |
| Beats baseline | Macro-F1 on recording-held-out real data at least 5 points above the best baseline, over 3 seeds |
| Honest generalization | Held-out drone model and `cross_test` results are reported next to the headline, whatever they are |
| Calibration | ECE at most 0.05 after temperature scaling |
| Safe failure | At the chosen threshold, no class is called with confidence above threshold on `background` more than the agreed false-alarm rate |
| Latency | Under 50 ms per burst on the laptop CPU |
| Leakage | Audit reports zero shared recordings or window hashes across splits |
| Reproducible | A fresh clone plus one documented command sequence reproduces the numbers |

If the network does not beat the baseline, we ship the baseline and say so. That is a result.

---

## 7. Folder layout

```
mod-classifier/
  PLAN.md                    this file
  README.md                  how to run each stage (written with the first working code)
  configs/
    taxonomy.toml            dataset file -> class, emitter, label strength (wave 0)
    band_priors.toml         band x class plausibility
    baseline.toml  cnn1d.toml  (wave 0)   cnn2d.toml  fused.toml
  mcls/                      the Python package
    sources/                 one loader per dataset -> canonical shards
    frontend/                resample, channelize, burst windowing, STFT, normalize
    dataset/                 manifest, group splits, torch Dataset, samplers, augmentations
    models/                  baselines, cnn1d, cnn2d, fused, heads
    train/                   loop, losses, optimizers, schedules, EMA, checkpointing
    evaluate/                metrics, calibration, unknown, SNR curves, leakage audit
    export/                  weights, TorchScript, model card
    infer/                   classify bursts coming out of pipeline.ingest
  crc/                       shell scripts that run each stage on CRC (written on request)
  scripts/
    build_manifest.py  preprocess.py  train.py  evaluate.py  export.py
    crc/                     Grid Engine job scripts and a merge step
  tests/                     pytest; torch tests skip when torch is absent
  reports/                   model card, results tables, plots
  data/                      git-ignored: raw/ shards/ manifests are not committed
  runs/                      git-ignored: one folder per training run
```

Repo rules this must respect: Python 3.12; raw data under a git-ignored path; no checkpoints over 100 MB in git (ours is about 4 MB, but weights still go through a release or Drive, not the history, until the team agrees); CI runs `pytest` and has **no torch** installed, so torch-dependent tests use `pytest.importorskip("torch")`. The root CI command is `python -m pytest tests`, so either CI also runs `mod-classifier/tests` or the new tests are mirrored under `tests/`. Decide this with Gabriel.

---

## 8. Pipeline stages

```
fetch  ->  convert  ->  channelize+window  ->  manifest+split  ->  train  ->  evaluate  ->  calibrate  ->  export  ->  integrate
```

| Stage | Command (planned) | Output |
| --- | --- | --- |
| Fetch | `python -m mcls.sources.fetch --dataset tampere` | `data/raw/...` with checksums |
| Convert and window | `python -m mcls.preprocess --dataset tampere` | shards plus per-recording stats |
| Manifest and splits | `python -m mcls.dataset.build_manifest` | `data/manifest.jsonl` with a hash |
| Baseline | `python -m mcls.train --config configs/baseline.toml` | `runs/baseline_*` |
| Network | `python -m mcls.train --config configs/cnn1d.toml --seed 1` | `runs/cnn1d_*` |
| Evaluate | `python -m mcls.evaluate --run runs/cnn1d_*` | tables, plots, JSON |
| Calibrate | `python -m mcls.evaluate.calibrate --run ...` | temperature, unknown threshold |
| Export | `python -m mcls.export --run ...` | weights and `reports/model_card.md` |
| Classify | `python -m pipeline.ingest capture.cs8 ... --classify` | rows with `signal_class`, `class_confidence` |

---

## 9. Orchestration: 10 senior engineers

The orchestrator owns the contracts (shard format, manifest schema, taxonomy, config schema, the `classify(windows) -> probabilities` function), assigns work, reviews every merge, and resolves disputes. Engineers do not change a contract; they propose changes to the orchestrator.

| # | Engineer | Owns | Needs from | Delivers | Done when |
| --- | --- | --- | --- | --- | --- |
| 1 | **Taxonomy and licences** | `taxonomy.toml`, licence register, label ambiguity (2.3) | Dataset pages, QA log | Mapping for every dataset file; verified emitter protocols (Wi-Fi or Bluetooth or proprietary) | Every file in every chosen dataset has a class and a licence |
| 2 | **Data acquisition** | Per-dataset fetch lists on top of `mcls.fetch` (parallel ranges, resume, md5), CRC download jobs | Odon's approval per download | Resumable fetchers; raw data layout | Fetch is idempotent and verified by checksum |
| 3 | **Dataset loaders** | `mcls/sources/*` | 1, 2 | Each dataset to canonical `complex64` records with metadata (Tampere `.bin`, DroneRF CSV, NIST HDF5, FGI, SigMF) | Each loader has a test on a tiny fixture and a real-file smoke test |
| 4 | **DSP front end** | `mcls/frontend/*` | 3, `pipeline/features.py` | Channelizer, decimation, burst windowing, STFT, normalization | Round-trip tests; matches `detect_bursts` outputs on real files; resampling error bounded |
| 5 | **Data pipeline** | manifest, splits, torch `Dataset`, samplers, augmentation | 3, 4 | Manifest, group splits, loaders, augmentations | Leakage audit passes; deterministic given a seed |
| 6 | **Baselines** | softmax regression, LightGBM, kNN, features | 5 | Baseline numbers on final splits | Reproducible results table |
| 7 | **Neural models and training loop** | `models/`, `train/` | 5 | CNN variants, loss, AdamW and SGD, schedules, EMA, checkpoints, MPS/CUDA/CPU | Trains end to end on the laptop; resumes; seed-stable |
| 8 | **Evaluation and calibration** | `evaluate/` | 5, 6, 7 | Metrics, calibration, unknown threshold, SNR curve, generalization and cross-dataset reports | Every report in 6.1 generated by one command |
| 9 | **Infrastructure and CRC** | CRC job scripts, environment, CI, storage of weights, repo hygiene | 2, 5 | Reproducible environment; array jobs; CI-safe tests; model storage | A fresh clone reproduces a small run; CI green with no torch |
| 10 | **Integration and red team** | `infer/`, `pipeline.ingest --classify`, leakage and failure audit, API fields, handoff to the Judge | 7, 8 | Classifier callable from the app; adversarial review of splits, metrics and claims | Classified rows load in the Library; every claim in the model card traced to a run |

**Dependency order:** 1, 2 start immediately; 3 follows; 4 and 5 run in parallel once 3 has a first loader; 6, 7 start when 5 has a manifest; 8 starts when 6 or 7 has a run; 9 runs alongside everything; 10 reviews continuously and integrates last.

**Working rules for every engineer:**

- One branch per task off `feat/modulation-classifier`; small commits; no pushes without Odon's authorization; PR to the feature branch, then one PR to `main` that passes CI.
- Every module ships with pytest tests; torch tests skip without torch.
- No download without Odon's approval (filename, source, size stated first).
- Any number in a report cites a run folder and manifest hash.
- Engineer 10 may reject any merge whose claim is not backed by a run.

---

## 10. Compute and storage

- **Laptop (Apple M4, 24 GB):** writing and testing the non-PyTorch stages on small samples (under about 1 GB of raw data kept locally).
- **CRC:** all training runs, baselines included, plus bulk data conversion, each started by a shell script under `crc/`. A 1M-parameter network on a few hundred thousand windows should train in well under an hour per run on one GPU (estimate; measure on the first run).
- **PyTorch runs on CRC, not the laptop.** The orchestrator writes `mod-classifier/requirements.txt` (PyTorch and the rest, pinned to versions the CRC modules support) and keeps it out of the root `requirements-dev.txt` so the repo's CI stays light. Code that does not need PyTorch (loaders, front end, manifest) is tested on the laptop and in CI.
- **CRC:** only for bulk reading and conversion of large datasets (NIST Wi-Fi/Bluetooth, GATEMAN, triangular jammer, LoRaIQ, occupancy, DroneDetect), streamed through node-local `/tmp` as in section 3.7. It reuses the pattern Gabriel wrote for NIST radar: UGE array job, one task per file, rows returned and the download deleted. A CRC GPU is used only if a run outgrows the laptop. **Odon has a CRC account;** its quota and GPU access are checked with `quota` and `free_gpus.sh @crc_gpu` before the first job.
- **CRC storage (from the CRC quick start, FAQ and storage policy of June 2026):** home `/users/<netid>` is 100 GB and cannot be increased; group storage `/groups/<PI netid>` is 5 TB per faculty group, and accounts a faculty member sponsors get access automatically; `/goldfs` and `/bluefs` are paid legacy tiers that cannot be mounted for live jobs. Whether the Innovation Sprint Lab has a faculty sponsor is unknown. Check with `quota`, `quota /users/<netid>` and `df -h /tmp` on a front end. The size of `/tmp` on compute nodes is not documented.
- **Disk:** 90 GB free on the laptop today. Plan the raw data to stay under about 40 GB locally: Tampere (8.6 GB), DroneRF (about 4 GB), one NIST file (20 GB). Anything bigger goes to CRC and is streamed, not kept.
- **Weights:** about 4 MB each; stored as a GitHub release asset or in Drive and recorded in the model card with a checksum.

---

## 11. Integration with Argus

1. `pipeline/ingest.py` gains `--classify` (off by default). After `detect_bursts`, each burst's windows go through `mcls.infer`, which returns class probabilities.
2. Rows gain **new** fields: `signal_class`, `class_confidence`, `class_probs`. The existing `modulation` field stays `"unknown"` until section 5.6 is proven.
3. The Laya service ignores extra fields today, so nothing breaks. The Judge (Andrew) will read `signal_class` and `class_confidence`, applying the threshold from 5.5.
4. Everything stays local and receive-only; no network at inference.
5. `python -m pipeline.ingest` without `--classify` behaves exactly as before.

---

## 12. Timeline

Anchored on the team's dates: pitch and M1 on Nov 13, M2 on Nov 20, feature freeze Dec 1, demo Dec 5. Dates after tomorrow depend on the downloads being approved and on data arriving.

| When | Goal | Output |
| --- | --- | --- |
| **Nov 12 (meeting)** | Be able to show the mechanism, not a headline | Branch, this plan, scaffold, loaders and a baseline-versus-CNN run on whatever is approved and downloaded (Tampere and DroneRF are the smallest real set with drones, plus the DroneRF background class). One honest number beside the baseline. A list of what is missing |
| Nov 13 | Pitch and M1 | Model not required for M1; show the classifier as work in progress, framed as signal classification |
| Nov 14 to 18 | Negatives and splits | NIST Wi-Fi/Bluetooth on CRC, `cross_test` sets frozen |
| Nov 19 to 20 | M2 | Baseline and CNN on real data with the full report, including generalization and domain shift |
| Nov 21 to 27 | Calibration and integration | Temperature, unknown threshold, `--classify` wired in, handoff to the Judge |
| Nov 28 to Dec 1 | Freeze | Model card, tests, docs, reproduction check |
| Dec 2 to 4 | Rehearsal | Three clean runs |

**What "ready tomorrow" can honestly mean:** a working end-to-end pipeline on the data that has been approved and downloaded, a result compared with a baseline, and a clear statement of what it does not yet cover. It cannot mean a model that works on campus signals.

---

## 13. Risks

| Risk | Effect | Response |
| --- | --- | --- |
| Domain shift: public sets are lab or chamber data at other rates, and there are no team captures | Good test scores, unknown field behaviour | Cross-dataset test, AdaBN, 20 MHz front end; the model card says nothing was measured on team hardware |
| Some drones are physically Wi-Fi or Bluetooth | `drone_link` and `wifi` are indistinguishable from IQ | Physical classes plus `emitter_id`; Judge decides drone versus Wi-Fi (2.3) |
| Weak, file-level labels on Wi-Fi and Bluetooth | Label noise | Energy-detector filtering, down-weighting, confident-learning audit |
| Jammer data is lab-generated, with no field jammers and no confirmed clean L1 baseline | Weak UC2 | Keep `gnss_jammer` in version 2; add a clean L1 baseline; the noise-floor rule in the Judge covers UC2 until then |
| Radar data is simulated | Over-optimistic radar results | Flagged `is_real=false`; excluded from headline |
| Slow downloads: Zenodo caps each connection at about 0.6 MB/s | 50 GB archives take a day on one connection | Parallel byte ranges (up to about 8), resumable fetchers, array jobs over files; measure from CRC first |
| Leakage through overlapping windows | Inflated scores | Group splits by recording; leakage audit |
| Sample-rate mismatch (15 to 200 MS/s vs 20 MS/s) | Different spectral shapes | Common 20 MS/s front end; test resampling error; AdaBN |
| Class imbalance (millions of Wi-Fi windows) | Model ignores minority classes | Per-class caps, balanced sampler, per-class metrics |
| Licences | Cannot share the model | Licence per dataset in the manifest; NC-SA sources excluded |
| Scope creep into demodulation | Violates NFR-7 | Name and non-goals in section 0 |
| CI has no torch | Tests untested in CI | `importorskip`; run torch tests locally and on CRC |
| Time: Nov 12 meeting | Overpromising | Scope in section 12 |

---

## 14. Definition of done

- Every stage runs from a fresh clone with the documented commands.
- A model card states the data, the splits, the metrics (real only), the baseline comparison, the generalization and cross-dataset results, the calibration, the unknown threshold, and the known limits.
- The classifier can be switched on in `pipeline.ingest --classify` and the Judge can read its output.
- CI is green; torch tests run locally.
- Nothing in the project claims more than the model card shows.

---

## Sources

- Dataset facts: each dataset's own page, read Nov 11, 2026 (links in section 3.1). Sizes for DroneRF are from the repo's `finetune/README.md`; `rtl_433_tests` and CardRF were not opened.
- Burst features and file readers: `pipeline/features.py`, `pipeline/iq.py` on `main`.
- Requirements referenced: the team's SRS (NFR-4, NFR-7, FR-19, C-1, OI-6) as quoted in the project knowledge file.
