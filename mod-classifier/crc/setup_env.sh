#!/bin/bash
# Build the mod-classifier Python environment on a CRC front end.
#
#   cd argus/mod-classifier && bash crc/setup_env.sh
#
# 1. loads a python/3.12 module (fails with a clear message if there is none),
# 2. creates a venv at $MCLS_VENV (default $HOME/venvs/mcls),
# 3. installs requirements.txt into it,
# 4. prints the torch version and whether CUDA is visible (no GPU on a front
#    end, so "False" is expected there: check_node.job checks a GPU node).
#
# Safe to rerun: an existing venv is reused and pip is skipped when
# requirements.txt has not changed since the last successful install.
#
# Settings (environment variables, all optional):
#   MCLS_VENV              venv path                      (default $HOME/venvs/mcls)
#   MCLS_PYTHON_MODULE     module to load                 (default newest python/3.12*)
#   MCLS_TORCH_INDEX_URL   extra pip index for a specific torch CUDA build,
#                          e.g. https://download.pytorch.org/whl/cu121
#   MCLS_FORCE_INSTALL=1   run pip even if requirements.txt is unchanged
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"
REQUIREMENTS="$(cd "$HERE/.." && pwd)/requirements.txt"
STAMP="$MCLS_VENV/mcls_requirements.sha256"

[ -r "$REQUIREMENTS" ] || mcls_die "cannot read $REQUIREMENTS"

mcls_heading "Python module"
mcls_load_python
MODULE="$MCLS_LOADED_MODULE"
echo "loaded: $MODULE"
mcls_have python3 || mcls_die "python3 not on PATH after module load $MODULE"
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "python3: $(command -v python3) ($PY_VERSION)"
[ "$PY_VERSION" = "3.12" ] || mcls_die "module $MODULE gives Python $PY_VERSION, need 3.12"

mcls_heading "venv at $MCLS_VENV"
if [ -x "$MCLS_VENV/bin/python" ]; then
    echo "exists, reusing"
else
    mkdir -p "$(dirname "$MCLS_VENV")"
    python3 -m venv "$MCLS_VENV" || mcls_die "python3 -m venv $MCLS_VENV failed"
    echo "created"
fi
echo "$MODULE" > "$MCLS_MODULE_FILE"
mcls_relaxed mcls_activate_venv

mcls_heading "requirements"
if mcls_have sha256sum; then
    WANT_HASH="$(sha256sum "$REQUIREMENTS" | cut -d' ' -f1)"
else
    WANT_HASH="$(shasum -a 256 "$REQUIREMENTS" | cut -d' ' -f1)"
fi
HAVE_HASH="$(cat "$STAMP" 2>/dev/null || true)"
if [ "$WANT_HASH" = "$HAVE_HASH" ] && [ "${MCLS_FORCE_INSTALL:-0}" != "1" ]; then
    echo "requirements.txt unchanged since the last install, skipping pip"
else
    # --no-cache-dir keeps the multi-GB torch wheels out of the 100 GB home quota.
    python -m pip install --no-cache-dir --upgrade pip
    PIP_ARGS=(--no-cache-dir -r "$REQUIREMENTS")
    if [ -n "${MCLS_TORCH_INDEX_URL:-}" ]; then
        PIP_ARGS+=(--extra-index-url "$MCLS_TORCH_INDEX_URL")
    fi
    python -m pip install "${PIP_ARGS[@]}"
    echo "$WANT_HASH" > "$STAMP"
    echo "installed"
fi

mcls_heading "torch"
python - <<'EOF'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
EOF

mcls_heading "venv size"
du -sh "$MCLS_VENV" 2>/dev/null || true
echo
echo "done: venv $MCLS_VENV with module $MODULE"
