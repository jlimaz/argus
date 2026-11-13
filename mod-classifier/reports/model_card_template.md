# Model card: {{model_name}} ({{version}})

Template. Every `{{...}}` comes from the files named next to it; a number with no run folder and manifest hash behind it does not belong here. Delete this line when filling in.

| Field | Value | Source |
| --- | --- | --- |
| Run folder | `{{run_dir}}` | |
| Git commit | `{{git_commit}}` | `{{run_dir}}/meta.json` |
| Manifest | `{{manifest_path}}`, sha256 `{{manifest_sha256}}` | `{{run_dir}}/meta.json`, checked by `python -m mcls.evaluate --manifest` |
| Seed(s) | {{seeds}} | `meta.json` of each run |
| Classes | {{classes}} | `meta.json` |
| Model | {{model}}, {{parameters}} parameters | `config.toml` |
| Created | {{created}} | `meta.json` |

## What it does

Classifies a 4096-sample window (about 205 microseconds at 20 MS/s) of a burst into one physical signal family. It does not decode or demodulate content. A low top probability means `unknown`; unknown is not a class.

## Not measured on our own hardware

**No recording from the team's own radios (RTL-SDR, HackRF) was used to train or test this model.** Every number below comes from held-out recordings of public datasets, made with other radios, sample rates and places. No number here says how the model behaves on an RTL-SDR or HackRF on campus. The cross-dataset test is the closest substitute (PLAN.md 3.2).

## Training data

| Dataset | Licence | Classes | Windows (train) | Real? |
| --- | --- | --- | --- | --- |
| {{dataset}} | {{licence}} | {{classes}} | {{n_train}} | {{is_real}} |

Synthetic or simulated training rows (`is_real=false`): {{synthetic_train_rows}} (`metrics.json` `training_rows`). Results are reported for runs with and without them: {{runs_with_synthetic}} / {{runs_without_synthetic}} (`comparison.json`).

## Results (real rows only)

All from `{{run_dir}}/metrics.json` unless stated. Test and cross-test rows are real only (`real_only_verified`: {{real_only_verified}}).

| Split | Windows | Window macro-F1 | Burst macro-F1 | Recording macro-F1 |
| --- | --- | --- | --- | --- |
| test (recording held out) | {{n}} | {{f1}} | {{f1_burst}} | {{f1_recording}} |
| generalization (held-out drone model) | {{n}} | {{f1}} | {{f1_burst}} | {{f1_recording}} |
| cross_test (dataset held out) | {{n}} | {{f1}} | {{f1_burst}} | {{f1_recording}} |

Per-class precision and recall, the confusion matrix and accuracy versus estimated SNR: `metrics.json` `splits.<split>`. Burst results group windows by `burst_id` (see `mcls/evaluate/runio.py`, `burst_keys`); results after the unknown threshold: `splits.<split>.burst.after_unknown_threshold`.

### Baselines on the same splits

`{{comparison_json}}` (same manifest: {{same_manifest}})

| Run | Model | Seed | Test macro-F1 | Cross-test macro-F1 |
| --- | --- | --- | --- | --- |
| {{run}} | {{model}} | {{seed}} | {{f1}} | {{f1_cross}} |

## Calibration and unknown

From `{{run_dir}}/calibration.json`, fitted on `val` only.

- Temperature: {{temperature}}
- ECE (15 bins) before / after, val: {{ece_before_val}} / {{ece_after_val}}; test: {{ece_before_test}} / {{ece_after_test}}
- Unknown threshold on max softmax probability (after temperature), keeping {{keep_fraction}} of val windows: {{unknown_threshold}}
- Energy-score ablation, same keep fraction: {{energy_threshold}}
- False alarms on held-out `background` at that threshold: {{false_alarm_rate}} of {{n_background}} windows

## Acceptance gates (PLAN 6.2)

From `check_gates` in `metrics.json`. "Not measured" is not a pass.

| Gate | Result | Value |
| --- | --- | --- |
| {{gate}} | {{PASS / FAIL / not measured}} | {{value}} |

## Limits

- {{limits}}
- One 20 MHz slice per recording: hopping emitters outside it are not seen.
- Bursts are grouped by `burst_id`; {{windows_by_fallback}} windows had none and were grouped by the equal-stats fallback (`metrics.json` `splits.<split>.burst.grouping`).
- Burst results after the unknown threshold skip the band prior, which lives in `mcls.infer`.
