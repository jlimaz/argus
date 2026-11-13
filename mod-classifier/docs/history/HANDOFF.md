# Handoff: state of the mod-classifier work

> **Historical record.** Kept as written; paths are relative to `mod-classifier/` as it was then. For the current state read [STATUS.md](../../STATUS.md).

**Frozen:** Nov 11, 2026, about 23:55, at Odon's request, to continue under a Paseo orchestrator.
**Repo:** `Gabriel-smm/argus`, local checkout `/Users/odon/Desktop/DIU Project/argus`.
**Integration branch:** `feat/modulation-classifier` (local only, never pushed). Head before this file: `d132fde`.

Read in this order: this file, then [PLAN.md](../../PLAN.md) (the full design), then [README.md](../../README.md), then `mcls/contracts.py`.

---

## 1. Rules Odon set (keep them)

- **The laptop builds and tests the pipeline; it never trains.** No PyTorch on the laptop. All training, baselines included, runs on Notre Dame's CRC through shell scripts that Odon will ask for separately. Do not write training job scripts until he asks.
- **No radio captures.** Nothing is recorded with the team's RTL-SDRs or HackRF. Tests are held-out public recordings plus a cross-dataset test (PLAN.md 3.2).
- **Odon has a CRC account.** Quota and GPU access are not yet checked.
- **No large downloads on the laptop.** Keep raw data under about 1 GB. Every download needs Odon's approval, with filename, source and size stated first.
- **Never push, never touch `main`, never open a PR without Odon's say-so.** Commit locally on branches. Repo convention: `feat/<name>` branches, short imperative commit subjects (see the repo's CLAUDE.md).
- **"Demodulation" is the wrong word:** the model classifies signals and never decodes content (SRS NFR-7).
- **Do not claim anything as fact without a source you opened.**
- **The plan's "section 14 decisions" are closed;** defaults stand.

## 2. State (Nov 12, 2026): pipeline code complete, nothing run on CRC

**Branch:** `feat/modulation-classifier` (local only, never pushed) contains all ten engineers' work after three rounds. Tests: `cd mod-classifier && ../.venv/bin/python -m pytest tests -q` gives 661 passed, 10 skipped (torch, scikit-learn, lightgbm are not on the laptop); the repo's own `python -m pytest tests` gives 133 passed.

| Piece | Where | Notes |
| --- | --- | --- |
| Contracts, config, manifest | `mcls/contracts.py`, `config.py`, `manifest.py` | Round 3 added `ManifestRow.burst_id`, a `generalization` split, stratified splits, and the config keys `model.aux_head`, `optim.loss/focal_gamma/nesterov/balanced_sampler/aux_weight` |
| Taxonomy, licences | `configs/taxonomy.toml`, `LICENCES.md` | 13 datasets, drone links sourced. Mavic Mini is a weak `wifi` label. Open: Yuneec Typhoon H link, CardRF (admit or not), DroneDetect file codes |
| Download | `configs/catalog.toml`, `mcls/acquire.py`, `mcls/fetch.py`, `DATA_ACQUISITION.md` | Files have a `role` (data, metadata, doc). Fairdata: one connection, no resume, a failed checksum keeps `<name>.bad` |
| Loaders | `mcls/sources/` | Tampere, DroneRF (grouped by BUI, an inference from the paper's numbers), NIST Wi-Fi/BT, SigMF (LoRaIQ, occupancy), GATEMAN, triangular L1. No Drexel radar or ADS-B loader: formats unknown |
| Front end | `mcls/frontend/`, `mcls/preprocess.py` | numpy resampler; `windows_from_iq` is shared with inference |
| Data pipeline | `mcls/dataset/` | `build_manifest`, samplers, augmentation, torch Dataset |
| Baselines | `mcls/baselines/` | kNN and LightGBM never run (no sklearn/lightgbm here) |
| Models, training | `mcls/models/`, `mcls/train/` | **No torch code has ever run.** First CRC run will exercise it |
| Evaluation | `mcls/evaluate/` | Needs `--manifest`; `calibration.json` is the only home of the temperature |
| Inference | `mcls/infer/`, `pipeline/ingest.py --classify --model PATH` | Band priors in `configs/band_priors.toml` are a draft for Diana's review |
| CRC | `crc/`, `scripts/` | Nothing has run on CRC |

## 3. Data on the laptop (git-ignored)

`mod-classifier/data/raw/tampere/`: two files only (`DJI_mavic_mini_2G.bin`, `DJI_inspire_2_5G_1of2.bin`), md5-verified. Drone-only, so no background class on the laptop.

## 4. Next steps (all need Odon)

1. On CRC: `quota`, `free_gpus.sh @crc_gpu`; run `crc/check_env.sh`, `setup_env.sh`, `check_node.job` (see `crc/README.md`).
2. Approve the first downloads: Tampere (8.6 GB) and DroneRF (about 4 GB), with filename, source and size stated first. Also needed: the 26 KB Drexel MATLAB-scripts tar (the smallest file that reveals the radar format); the IEEE DataPort login for ADS-B, occupancy and DroneDetect.
3. Ask for the training scripts (not written by design).
4. First run: prepare, `build_manifest`, baselines, CNN, evaluate, calibrate. Expect bugs in the never-run torch code.

## 5. Known gaps and decisions

- A burst longer than about 0.1 s is split into two `burst_id`s; Tampere reads as one burst per 0.21 s block, so check burst-level scores after the first run.
- Tampere gets about 320 windows per second of recording by default; the plan expects 60 to 70 thousand windows for the whole dataset. Decide `--max-windows-per-burst` for the CRC job.
- GATEMAN's unpacked size is not in the catalog; set `MCLS_UNPACKED_GB` by hand or read the zip index first.
- Open review items (LOW or MEDIUM) are in Engineer 10's `review-e*.md` files in its worktree `.claude/worktrees/agent-a7ac187644c94ea7d/.claude/mcls-status/`. Main one: `mcls.train` `read_temperature` should call `check_calibration`.
- No result numbers exist anywhere. Nothing was measured on the team's own radios.

## 6. Rules Odon set (keep them)

See section 1. The old Wave 1 task briefs are replaced by the code above; the full design is `PLAN.md`.

## 7. Key facts

- Zenodo caps each connection at about 0.6 MB/s (0.47 measured on one, 2.44 on four); keep about 8 connections per host.
- CRC storage: home 100 GB fixed; group storage needs a faculty sponsor (unknown); raw data is streamed and deleted; windows are 32 KB each.
- Byte-order traps: Tampere int16 little-endian; GATEMAN int16 big-endian; triangular L1 gzip int8; DroneRF real floats in CSV inside RAR.
- Datasets, licences and sizes: `PLAN.md` section 3.1 and `LICENCES.md`.
