# Overnight plan, Nov 12-13 2026

> **Historical record.** Kept as written; paths are relative to `mod-classifier/` as it was then. For the current state read [STATUS.md](../../STATUS.md).

Written at Odon's request before he went to sleep. Claude follows this file step by
step; an hourly check (cron, "13 * * * *") re-reads it and continues from the
first step not marked done. Progress goes in the log at the bottom, newest last.

Odon keeps the CRC connection open and the laptop awake. Nothing is committed or
pushed (that needs Odon's OK). No torch is installed on the laptop.

## Step 1. Watch the wave 5 2D CNN until it ends (hourly)

- Job **1737114** (`cnn2d_v1_w5`, RTX 6000, about 10 min per epoch, early stopping
  after 8 epochs without a new best, at most 40 epochs). Summary job **1737551**
  emails the results and every log when it ends.
- Each hour: `ssh crc`, `qstat -u oineza`, last lines of
  `runs/crc_logs/mcls_train.o1737114` and `runs/cnn2d_v1_w5/history.jsonl`.
- If it crashed: read the log, fix the cause, resubmit with
  `qsub -q gpu@@crc_gpu,gpu@@cvrl-a10,gpu@@cvrl-rtx6k -l gpu_card=1 -m b -M oineza@nd.edu -v MCLS_MAIL_TO= crc/train.job runs/configs/cnn2d_w5.toml --resume`
  and resubmit the summary job held on the new id
  (`crc/wave_summary.job w5 <new id>,1736993,1736994,1736995 runs/crc_logs/mcls_manifest.o1736990`).
- Done when the job has left the queue and `runs/cnn2d_v1_w5/metrics.json` exists
  (and the summary job has emailed). Then go to step 2 straight away (early
  stopping is fine: it means training is done).

## Step 2. Bring everything except the training data to the laptop

- Source: `crc:~/argus/mod-classifier/`. Destination: `argus/mod-classifier/runs/crc/`
  (git-ignored) for run folders and logs; `argus/mod-classifier/data/` (git-ignored)
  for `manifest_w5.jsonl` only (the index the routing step needs, not samples).
- Copy with rsync: every `runs/<run>/` folder (checkpoint.pt, model.pkl,
  calibration.json, metrics.json, history.jsonl, predictions_*.npz, progress
  plots, config.toml, meta.json), `runs/crc_logs/`, `runs/configs/`,
  `runs/file_lists/`, the `runs/*.txt` reports.
- Leave out: `~/mcls_data/` (all shards and prepared windows), every `last.pt`
  (resume state only), `data/manifest*.jsonl` other than w5, smoke runs.
- No duplicates: rsync into the same paths (`runs/crc/<run>/...`), no `--backup`,
  no renamed copies. Files already there (the six `runs/crc/*/progress.png` pulled
  on Nov 12) are compared and replaced in place only if they changed. Do not touch
  `data/raw/` (848 MB of the laptop's own earlier test data).
- Write `runs/crc/SHA256SUMS` for every checkpoint.pt and model.pkl, and check the
  laptop copies against the CRC copies.
- Code: the laptop already has the newest code (every edit was made here first and
  copied to CRC). Diff the CRC tree against the laptop (`*.py`, `*.job`, `*.sh`,
  `*.toml`) and report any difference instead of overwriting.
- The CNN needs torch to run, and the laptop has none. On CRC, export the wave 5
  CNN's best weights to ONNX (`runs/cnn2d_v1_w5/model.onnx`, opset 17, with a
  numerical check against torch on a few test windows), and copy that too.

## Step 3. Build the UI layer on the laptop

Goal (Odon's words): "a UI layer that will allow the user to input a wave and then
you run the inferences and pick out the best model and then from the best model
you classify that waveform".

- Where: `argus/mod-classifier/ui/` (a small local web app), separate from the
  main Argus app so nothing there breaks. Runs from a separate venv
  `mod-classifier/.venv-ui` with numpy, scipy, scikit-learn, lightgbm,
  onnxruntime, fastapi, uvicorn, matplotlib. **Not torch.**
- Input: upload an IQ recording (`.cu8`, `.cs8`, `.cs16`, `.cf32` or SigMF), with
  sample rate and centre frequency (read from SigMF when present). Reuse
  `pipeline/iq.py` readers and `mcls.frontend` windowing, so it is the same code
  path as training.
- Inference: run every wave 5 model (2D CNN via ONNX, kNN, LightGBM) on each
  detected burst, with each model's calibration (temperature, unknown threshold).
- Model choice ("pick the best model"): a routing table built from the validation
  predictions only. Group by what the signal looks like before classification
  (band and bandwidth); in each group pick the model with the best validation
  macro-F1; groups with fewer than 2 validation recordings or 200 windows fall
  back to the overall best model. Report the routing table and its test score next
  to each single model and a plain average, so the choice is justified by data the
  routing never saw. Script: `mod-classifier/scripts/build_routing.py`, output
  `runs/crc/routing_w5.json`.
- Output per burst: the chosen model and why (group, its validation score), its
  class and calibrated confidence ("unknown" under the threshold), and every
  model's answer side by side, flagged when they disagree. Plus a spectrogram of
  the input with the bursts marked.
- Tests: pytest for the routing script and the inference service on small
  synthetic inputs; then start the UI with `preview_start` and check it in the
  browser pane with a real file (a Tampere or rtl_433 sample, or a synthetic one),
  screenshot included in the report.
- Docs: a short `ui/README.md` (how to start it, what it shows, its limits: lab
  data only, radar never tested, rtl_433 has no licence, models disagree means
  "check by hand").

## Additions (gaps found before Odon went to sleep)

1. **Pin the laptop's library versions to CRC's.** kNN and LightGBM are saved as
   pickles, which may refuse to load or misbehave under other versions. Install
   in `.venv-ui` exactly what `~/venvs/mcls` has on CRC (check with `pip freeze`:
   numpy 2.5.4, scikit-learn 1.9.1, lightgbm 4.7.0, scipy 1.18.1, h5py 3.16.0), and
   load each pickle once as a check.
2. **ONNX fallback.** torch 2.15's exporter may need extra packages (`onnx`,
   `onnxscript`) in the CRC venv; install them there if so (CRC only). If the
   export still fails or does not match torch to 1e-4, write a NumPy forward pass
   of the 2D CNN from the checkpoint's weights instead, checked against torch
   outputs on 64 test windows computed on CRC.
3. **Real sample inputs to test the UI.** The training data stays on CRC, so bring
   a few small real recordings (under 50 MB in total): one short slice of a
   Tampere drone file (cs16, with its rate and centre), one NIST Wi-Fi/Bluetooth
   capture slice, two rtl_433 `.cu8` files and two LoRaIQ SigMF frames (fetched
   without the full zip), into `mod-classifier/ui/samples/` (git-ignored), each
   with a note of its source, true class and whether it was a test recording.
   Prefer test-split recordings so the UI demo is honest.
4. **Things that would stop the night's work** (cannot be fixed from here, so
   log them and report them in the morning): the Claude app quitting (the hourly
   check lives in this session), the laptop sleeping (a closed lid), or the CRC
   SSH connection closing (reconnecting needs Odon's password).

## Step 4. Morning report

Write the outcome at the bottom of this file and say it in the chat: CNN test
scores against the baselines, what was copied (sizes, checksums), the routing
result, how to start the UI, what is not done or not verified. Stop the hourly
check (CronDelete) once step 3 is finished.

## Rules for the night

- Never push, never commit, never open a PR (Odon has not authorised it).
- Never delete anything on CRC except the fallback venv already removed; never
  touch `~/mcls_data/`.
- If a step is blocked (a crash I cannot fix, a missing dependency that needs a
  decision), write it in the log, skip to what can still be done, and report it.

## Log

- 23:45 Plan written. CNN at epoch 8 (val macro-F1 0.80, improving every epoch).
  Hourly check scheduled.
- 00:44 Step 1: CNN at epoch 16 of 40, still a new best every epoch (val macro-F1
  0.865 at epoch 15). Next check at about 01:13.
- 01:32 Step 1: CNN at epoch 21. Best val macro-F1 0.868 at epoch 17; epochs 18-20
  did not beat it (0.864, 0.863, 0.866). If nothing beats it, early stopping ends
  the run after epoch 25, about 02:25. No errors.
- 02:32 **Step 1 done.** Training ended at 02:06 (early stop after epoch 25, best
  weights from epoch 17, val macro-F1 0.868). Summary email sent by job 1737551.
  Test (34,488 real windows, 7 classes): 2D CNN window macro-F1 **0.871**, burst
  macro-F1 **0.854**, accuracy 0.914; kNN 0.825 / 0.761 / 0.861; LightGBM 0.758 /
  0.756 / 0.827; majority 0.057 / 0.010 / 0.249. The CNN beats every baseline.
- 02:40 **Step 2 done.** rsync of CRC `runs/` into `runs/crc/` (812 files, 434 MB;
  left out every last.pt, smoke runs, scratch samples, peek_drexel) and
  `data/manifest_w5.jsonl` (160 MB). `runs/crc/SHA256SUMS`: all 20 checkpoint.pt /
  model.pkl files match CRC. Code: all 149 code files identical on both sides
  (checked 23:55). ONNX: `scripts/export_onnx.py` (new) wrote
  `runs/cnn2d_v1_w5/model.onnx`; on 256 real test windows the largest logit
  difference against torch is 3.8e-6 and all 256 argmaxes agree. Installed onnx
  1.23.2 and onnxruntime 1.31.1 in the CRC venv. CRC versions for the laptop:
  numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, h5py 3.16.0.
- 02:55 Step 3 started. `.venv-ui` created with exactly those versions plus
  onnxruntime 1.31.1, fastapi, uvicorn, matplotlib (no torch). Root .gitignore now
  ignores `mod-classifier/.venv-ui/` and `mod-classifier/ui/samples/`.
  `scripts/build_routing.py` (new) -> `runs/crc/routing_w5.json`. Test scores:
  route by band acc 0.925 / macro-F1 0.883 (best real strategy); CNN alone 0.914 /
  0.871; plain average 0.915 / 0.864; route by bandwidth 0.923 / 0.869; oracle
  ceiling 0.944 / 0.907. Band routes: LightGBM for ism_2400 and ism_5800, the CNN
  everywhere else (gnss_l1, ism_433, ism_915, srd_868, other). The UI uses
  route-by-band.
- 03:05 Samples: `scripts/fetch_ui_samples.py` (new) fetched 8 slices of real
  test-split recordings by range request (49 MB) into `ui/samples/`. NIST left out
  (21.6 GB HDF5 cannot be sliced remotely). LoRa has 1 frame, not 2 (frame numbers
  have gaps).
- 03:15 Bug found on real data and fixed: always-on jammers gave 0 windows at
  inference (the activity test rejects them, as it did in training before the
  CONTINUOUS fix). `mcls.infer.core.burst_windows` gained `min_active`; the UI
  retries a burst with no windows at `min_active=0.0` and flags it "always-on".
- 03:35 **Step 3 done.** UI in `ui/` (service.py, app.py, static/index.html,
  README.md), served on http://localhost:8765. Checked in the browser pane at
  desktop and phone width, light and dark; raw and SigMF uploads and an error case
  checked through the API. Tests: 8 new (routing script, UI service), 696 pass in
  the main venv, 734 on CRC. Hourly check deleted.

## Morning report (Nov 13, 03:40)

**Training.** Wave 5 2D CNN (8 classes) ended at 02:06 by early stopping; best
weights from epoch 17. Test, 34,488 real windows from recordings no model trained on:

| Model | Accuracy | Window macro-F1 | Burst macro-F1 |
|---|---|---|---|
| **2D CNN** | **0.914** | **0.871** | **0.854** |
| kNN | 0.861 | 0.825 | 0.761 |
| LightGBM | 0.827 | 0.758 | 0.756 |
| Majority | 0.249 | 0.057 | 0.010 |

**Copied to the laptop** (`runs/crc/`, git-ignored): every run folder (CNN and
baseline weights, calibration, metrics, history, predictions, plots, configs),
all CRC logs and helper scripts, `data/manifest_w5.jsonl`; 434 MB + 160 MB.
`runs/crc/SHA256SUMS` lists all 21 model files (20 checkpoints/pickles + the ONNX
model); all match CRC. Not copied: training windows, every last.pt.

**Model routing.** Route by band beats every single model on test (accuracy 0.925,
macro-F1 0.883, against 0.914 / 0.871 for the CNN alone). LightGBM handles 2.4 and
5.8 GHz, the CNN everything else.

**The UI.** Start it from `mod-classifier/`:
`.venv-ui/bin/uvicorn ui.app:app --port 8765 --app-dir .` and open
http://localhost:8765. On the 8 real test samples: DJI drone link, Parrot Wi-Fi,
LoRa, both jammers correct; rtl_433 sensors 13 of 20 bursts other_ism (4 unknown,
3 called lora); Yuneec 5.7 GHz called wifi (its label is still open).

**New or changed files (nothing committed or pushed):** `ui/` (new),
`scripts/build_routing.py`, `scripts/export_onnx.py`, `scripts/fetch_ui_samples.py`
(new), `tests/test_build_routing.py`, `tests/test_ui_service.py` (new),
`mcls/infer/core.py` (min_active), root `.gitignore` (+.venv-ui, ui/samples).
On CRC only: onnx and onnxruntime installed in `~/venvs/mcls`.

**Not done or not verified:**
- NIST Wi-Fi/Bluetooth has no UI sample.
- The UI has not been tried on the team's own captures.
- The preview launcher could not start the server (no access to the Desktop
  folder), so it was started from the shell instead; a leftover
  `/Users/odon/Desktop/DIU Project/.claude/launch.json` points at it.
- Nothing committed or pushed: that waits for Odon.
