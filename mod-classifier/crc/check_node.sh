#!/bin/bash
# Report what a CRC compute node offers. Run by check_node.job; can also be run
# by hand on a front end or in an interactive session (qrsh).
#
# Prints: host and queue, `df -h /tmp`, CPU count (node and granted slots),
# RAM, `nvidia-smi`, and whether torch sees CUDA inside the venv. Changes
# nothing; a missing command is reported and skipped.
#
# Settings: MCLS_VENV (default $HOME/venvs/mcls), as in setup_env.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

try() {
    if ! mcls_have "$1"; then
        echo "(command not found: $1)"
        return 0
    fi
    "$@" 2>&1 || echo "(exit status $? from: $*)"
}

mcls_heading "node"
echo "host: $(hostname)  date: $(date)"
echo "job: ${JOB_ID:-none}  queue: ${QUEUE:-none}  slots granted: ${NSLOTS:-none}"
echo "SGE_GPU / CUDA_VISIBLE_DEVICES: ${SGE_GPU:-unset} / ${CUDA_VISIBLE_DEVICES:-unset}"

mcls_heading "df -h /tmp (node-local scratch for raw downloads)"
try df -h /tmp
echo "TMPDIR (per-job scratch, if the scheduler sets one): ${TMPDIR:-unset}"
if [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR:-}" ]; then
    try df -h "$TMPDIR"
fi

mcls_heading "CPUs"
if mcls_have nproc; then
    echo "nproc (usable by this process): $(nproc)"
    echo "nproc --all (on the node): $(nproc --all)"
else
    try getconf _NPROCESSORS_ONLN
fi
if mcls_have lscpu; then
    lscpu 2>/dev/null | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core):' || true
fi

mcls_heading "RAM"
if mcls_have free; then
    free -g
elif [ -r /proc/meminfo ]; then
    grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
else
    echo "(neither free nor /proc/meminfo available)"
fi

mcls_heading "nvidia-smi"
try nvidia-smi

mcls_heading "torch inside the venv ($MCLS_VENV)"
if [ ! -r "$MCLS_VENV/bin/activate" ]; then
    echo "no venv at $MCLS_VENV: run crc/setup_env.sh on a front end first"
else
    # Load the module the venv was built with; report rather than stop on failure.
    if mcls_try_load_python; then
        echo "module: $MCLS_LOADED_MODULE"
    else
        echo "(continuing without a module; the venv's python may not start)"
    fi
    mcls_relaxed mcls_activate_venv
    python - <<'EOF' 2>&1 || echo "(torch check failed)"
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok, "devices:", torch.cuda.device_count())
if ok:
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.ones(1024, 1024, device="cuda")
    print("matmul on GPU ok:", float((x @ x)[0, 0]) == 1024.0)
EOF
fi

echo
echo "check_node.sh finished"
