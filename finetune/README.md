# Fine-tuning Laya

How to fine-tune Laya on the RF training data on an Apple Silicon Mac (written for an M4 Pro with 24 GB). Run every command from the repository root, with the virtual environment active.

## What Laya is asked

One question, in `data/questions_20.json`: which signal type is this observation? There are nine answer options. Eight are the shortwave signal types the Panoradio dataset covers, and `other` is everything else.

| Option | Panoradio modes behind it | Training rows | Test rows |
|---|---|---|---|
| `morse` | Morse Code (CW) | 430 | 48 |
| `psk` | PSK31, PSK63, QPSK31 | 1,296 | 144 |
| `rtty` | RTTY 45 baud 170 Hz | 432 | 48 |
| `olivia` | Olivia 8/250, 16/500, 16/1000, 32/1000 | 1,728 | 192 |
| `dominoex` | DominoEX 11 | 432 | 48 |
| `mt63` | MT63-1000 | 432 | 48 |
| `navtex` | NAVTEX (SITOR-B) | 432 | 48 |
| `weather_fax` | HF weather fax | 432 | 48 |
| `other` | 290 catalogued signals from sigidwiki | 1,350 | 150 |
| Total | | 6,964 | 774 |

Modes share an option when centre frequency, bandwidth and modulation cannot tell them apart: the four Olivia variants overlap in measured bandwidth, and PSK31 and QPSK31 have the same bandwidth.

## What the accuracy means

- **Only the bandwidth of a Panoradio row is measured.** The recordings carry no centre frequency, so the converter places each one on a frequency where its mode is operated, and it writes the modulation name from the mode. A high score therefore partly shows the model recovering those assignment rules. It is not evidence that the model recognises these signals off the air.
- **The eight signal types are tested on rows of the same recordings set**, a random tenth of each. `other` is tested on 29 signal types that are not in the training data.
- **A simple reference.** A nearest-neighbour classifier on the three fields scores 82.7% on the test file (80.4% balanced). Its weak spots are `weather_fax` (24 of 48) and `dominoex` (29 of 48), whose measured bandwidths overlap other types. Laya has not been run on this data yet.

## Results so far

| Run | Question and test set | Accuracy |
|---|---|---|
| Base Laya | 12 emitter families, 2,000 mock rows | 8.6% |
| First fine-tune (2,000 items, 4 epochs) | same | 18.7% |
| Second fine-tune (2,000 items, 5 epochs) | 12 families, 482 mock rows of unseen signal types | 21.0% |
| Signal types (this setup) | 9 options, 774 test rows | not run yet |

The earlier runs answered a different question on different data, so they are not comparable with this setup. The mock observations and the hand-written drone link table they used have been removed.

## Steps

### 1. Set up

```bash
git clone https://github.com/jlimaz/undefined-sprint-hack.git
cd undefined-sprint-hack
git checkout laya
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the training items

```bash
python -m finetune.build_dataset
```

This takes a few seconds. It should report 6,964 training rows and 774 test rows, and it writes `finetune/out/train_items.pt`, `finetune/out/train_rows.jsonl` and `data/test_observations.jsonl`. The test file is committed, and the builder rewrites it identically.

### 3. Measure the baseline

```bash
python -m finetune.evaluate
```

### 4. Train

```bash
python finetune/train.py \
  --items finetune/out/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-modes \
  --epochs 3 --micro-batch 8 --grad-accum 1 --no-checkpointing
```

The first run downloads the base checkpoint (about 800 MB) into `models/laya-base`. It should print `Device: mps` and `Training items: 6564; calibration items: 400`, then a loss line every 100 batches.

**How long it takes.** The run is about 19,700 item passes (6,564 items, 3 epochs). The second fine-tune was 9,000 passes and took over 2 hours, so by item count this is 2.2 times as long. Two things should make it quicker, neither measured: each item is shorter (nine short options instead of twelve long ones), and these flags skip gradient checkpointing. For a firm figure, time the first loss line (100 batches of 8 items) and multiply by 24.6.

### 5. Check the result

```bash
python -m finetune.evaluate --checkpoint models/laya-rf-modes --baseline convaiinnovations/laya
```

It prints overall accuracy, balanced accuracy (the average over the nine options, which differ in size) and a row per option. The last line says whether the bar is met: at least 50% overall, and no single option making up more than half of the answers. The thresholds are constants at the top of `finetune/evaluate.py`.

To see how well a checkpoint fits its own training rows:

```bash
python -m finetune.evaluate --checkpoint models/laya-rf-modes --test finetune/out/train_rows.jsonl
```

### 6. Use the checkpoint

```bash
python -m pipeline
```

The pipeline and the dashboard load `models/laya-rf-modes` by default, and stop with an error when it is missing. To use it on another machine, copy the whole checkpoint folder (about 800 MB) to the same place, or put it elsewhere and point `LAYA_CHECKPOINT` at it. `models/` is not committed.

## If something goes wrong

- **Out of memory:** use `--micro-batch 4 --grad-accum 2`. If that fails too, use `--micro-batch 2 --grad-accum 4` and leave out `--no-checkpointing`.
- **Interrupted run:** the trainer saves `checkpoint_latest/` inside the output folder after every epoch. It cannot resume; evaluate that folder or start again.
- **`Device: cpu` on the Mac:** MPS is not available in that Python environment. Training on CPU will take many hours.

## The training data

| File | Rows | Source |
|---|---|---|
| `data/train/panoradio_hf.jsonl` | 6,238, all used | [Panoradio HF](https://panoradio-sdr.de/radio-signal-classification-dataset/) (S. Scholl, 2019). Bandwidth and duty cycle are measured from the IQ vectors of 13 shortwave modes; SNR comes from the dataset. The centre frequency is assigned from where each mode is really operated. |
| `data/train/sigid.jsonl` | 13,346, of which 1,500 used | [Artemis-DB](https://github.com/AresValley/Artemis-DB) v74, an export of sigidwiki.com. About 300 catalogued signals. Observations are drawn from each signal's listed frequencies, bandwidths and modulations. |

How the builder uses them:

- **`other` is capped at 1,500 rows**, spread evenly over the signal types. With every sigid row it would be two thirds of the data, and always answering `other` would already score 68%.
- **Some sigid signals are left out**: Olivia, MT63, DominoEX, DominoF, Radio Teletype (RTTY), RTTYM, Coherent CW, Phase Shift Keying (PSK), Coherent BPSK and PSK-AM. They are the same modes as the Panoradio options under a generic name, so labelling them `other` would contradict the Panoradio rows. SITOR-B (the mode NAVTEX uses) is on the list too, for when `sigid.jsonl` is regenerated; the current file does not contain it.
- **The split** is fixed by a seed: a tenth of each option's Panoradio rows, and every used row of a tenth of the sigid signal types.

Known weak spots:

- **Noisy bandwidths at low SNR.** Panoradio rows go down to 0 dB, where the measured bandwidth is unreliable.
- **`sigid.jsonl` is missing about 175 catalogued signals.** It was generated when those signals were reserved for an earlier test set. The converter no longer skips them, so regenerating the file adds them to `other` and changes the training and test rows.
- **Licences are unconfirmed** for Panoradio and the sigidwiki data. Check before publishing anything built on them.

### Rebuilding the training data

Only needed to change the sources. It downloads about 6 GB into `data/raw/`.

```bash
python -m finetune.sources.fetch
python -m finetune.sources.sigid
python -m finetune.sources.panoradio
python -m finetune.build_dataset
```

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Labels the rows with their answer option, splits them into training and test rows, and tokenises the training rows exactly as the pipeline does at inference. |
| `train.py` | Laya's official fine-tuning script (Apache-2.0), vendored, with CUDA added as a device. |
| `evaluate.py` | Accuracy per answer option for any checkpoint on the test set. |
| `sources/` | Downloaders and converters for the open data. |
