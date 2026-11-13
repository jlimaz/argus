#!/bin/bash
# UGE array job for the Notre Dame CRC: one task per NIST radar subset.
# Each task downloads one 2.5 GB file into the node's /tmp, measures it, writes
# its rows into finetune/out/nist_radar/ and deletes the download.
#
#   cd argus && qsub finetune/crc/nist_radar.sh            # all 200 subsets
#   qsub -t 1-50 finetune/crc/nist_radar.sh                # group 1 only
#   qstat -u $USER                                         # progress
#   python finetune/crc/merge.py nist_radar                # -> data/train/nist_radar.jsonl
#
#$ -q long
#$ -pe smp 1
#$ -N nist_radar
#$ -t 1-200
#$ -tc 25
#$ -o finetune/out/crc_logs/
#$ -e finetune/out/crc_logs/
#$ -cwd
set -euo pipefail

module load python/3.12 2>/dev/null || true
source .venv/bin/activate

# Task 1-50 is group 1, 51-100 group 2, and so on.
GROUP=$(( (SGE_TASK_ID - 1) / 50 + 1 ))
SUBSET=$(( (SGE_TASK_ID - 1) % 50 + 1 ))
export ARGUS_RAW="/tmp/$USER/argus_raw"
mkdir -p "$ARGUS_RAW" finetune/out/nist_radar finetune/out/crc_logs

python -m finetune.sources.nist_radar --group "$GROUP" --subset "$SUBSET" --fetch --delete-raw \
    --output "finetune/out/nist_radar/group${GROUP}_subset_${SUBSET}.jsonl"
