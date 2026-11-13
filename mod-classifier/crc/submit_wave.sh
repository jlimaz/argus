#!/bin/bash
# Submit one training wave as a chain the scheduler runs by itself: a manifest job
# held until the given prepare jobs end, one training job per model held until the
# manifest job ends, and a summary job held until every training job ends. Nothing
# has to stay running on a front end or a laptop.
#
# Emails: two per wave. The scheduler mails when the first training job starts, and
# crc/wave_summary.job mails the results table with every plot and log at the end.
# The training jobs themselves send none (MCLS_MAIL_TO is empty for them).
#
# Run on a front end from mod-classifier/:
#
#   bash crc/submit_wave.sh w2 1733100,1733152 nist_wifi_bt
#   bash crc/submit_wave.sh w3 none tampere,dronerf          # no prepare jobs to wait for
#
# Arguments: WAVE (name suffix), JOBS (comma-separated prepare job ids, or "none"),
# DATASETS (comma-separated datasets to merge before building the manifest).
# Writes data/manifest_<wave>.jsonl (a new file, so runs still evaluating on an
# older manifest are not disturbed) and runs/configs/<model>_<wave>.toml.
# Settings: MCLS_MODELS (default "cnn1d cnn2d fused baseline lightgbm knn majority"),
# MCLS_CLASSES (class set for the manifest and every config: v1, v2 or a named
# subset such as v2a; default v1), MCLS_SHARDS (shards dir for the manifest job;
# default ~/mcls_data/shards), MCLS_SKIP_MERGE=1 and MCLS_NO_HASH_AUDIT=1 (passed to
# the manifest job; see crc/build_manifest.job).
set -euo pipefail

WAVE="${1:?wave name, e.g. w2}"
JOBS="${2:?comma-separated prepare job ids, or none}"
DATASETS="${3:?comma-separated datasets to merge}"
MODELS="${MCLS_MODELS:-cnn1d cnn2d fused baseline lightgbm knn majority}"
MANIFEST="data/manifest_${WAVE}.jsonl"
CLASSES="${MCLS_CLASSES:-v1}"
# A10 and RTX 6000 nodes only (the general pool plus the CVRL lab's, which this account
# may use): the plain gpu queue also has TITAN Xp nodes (compute
# capability 6.1), which this torch build does not support (CUBLAS_STATUS_ARCH_MISMATCH, Nov 12).
GPU_QUEUE="${MCLS_GPU_QUEUE:-gpu@@crc_gpu,gpu@@cvrl-a10,gpu@@cvrl-rtx6k}"

[ -f crc/train.job ] || { echo "ERROR: run from mod-classifier/" >&2; exit 1; }
mkdir -p runs/crc_logs runs/configs data

hold=()
[ "$JOBS" = none ] || hold=(-hold_jid "$JOBS")
manifest_job="$(qsub -terse -q long "${hold[@]}" -v "MCLS_SHARDS=${MCLS_SHARDS:-$HOME/mcls_data/shards},MCLS_SKIP_MERGE=${MCLS_SKIP_MERGE:-0},MCLS_NO_HASH_AUDIT=${MCLS_NO_HASH_AUDIT:-0}" crc/build_manifest.job "$MANIFEST" "$DATASETS" "$CLASSES")"
manifest_job="${manifest_job%%.*}"
echo "manifest job $manifest_job -> $MANIFEST, classes $CLASSES (waits for: $JOBS)"

train_jobs=()
for m in $MODELS; do
    case "$m" in
        cnn1d | cnn2d | fused) src="configs/$m.toml"; queue=(-q "$GPU_QUEUE" -l gpu_card=1) ;;
        baseline | lightgbm | knn | majority) src="configs/baseline.toml"; queue=(-q long) ;;
        *) echo "unknown model $m, skipped"; continue ;;
    esac
    cfg="runs/configs/${m}_${WAVE}.toml"
    sed -e "s#^manifest = .*#manifest = \"$MANIFEST\"#" -e "s#^classes = .*#classes = \"$CLASSES\"#" \
        -e "0,/^name = \"\(.*\)_v1\"/s##name = \"\1_v1_${WAVE}\"#" "$src" > "$cfg"
    case "$m" in
        lightgbm | knn | majority)
            sed -i -e "s#^name = \"baseline_softmax_v1_${WAVE}\"#name = \"baseline_${m}_v1_${WAVE}\"#" \
                -e "s#^name = \"softmax_regression\"#name = \"$m\"#" "$cfg" ;;
    esac
    start_mail=()
    [ ${#train_jobs[@]} -eq 0 ] && start_mail=(-m b -M "${USER}@nd.edu")   # "wave started"
    job="$(qsub -terse "${queue[@]}" "${start_mail[@]}" -hold_jid "$manifest_job" -v MCLS_MAIL_TO= \
        crc/train.job "$cfg")"
    train_jobs+=("${job%%.*}")
    echo "train job ${job%%.*}: $m ($cfg)"
done

ids="$(IFS=,; echo "${train_jobs[*]}")"
summary="$(qsub -terse -q long -hold_jid "$ids" crc/wave_summary.job "$WAVE" "$ids" "runs/crc_logs/mcls_manifest.o$manifest_job")"
echo "summary job ${summary%%.*}: emails the results when jobs $ids have ended"
