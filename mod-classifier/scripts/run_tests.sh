#!/bin/bash
# Run the mod-classifier tests from any directory.
#
#   bash mod-classifier/scripts/run_tests.sh            # all tests, quiet
#   bash mod-classifier/scripts/run_tests.sh -k crc     # extra args go to pytest
#
# Python: $MCLS_TEST_PYTHON if set, else the repo's .venv (argus/.venv, also
# found from a git worktree), else python3 on PATH. The laptop needs only numpy
# and pytest; torch tests skip.
set -euo pipefail

MC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${MCLS_TEST_PYTHON:-}"
if [ -z "$PY" ]; then
    candidates=("$MC/../.venv/bin/python")
    # From a git worktree, the .venv lives next to the main checkout's .git.
    common="$(git -C "$MC" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [ -n "$common" ]; then
        candidates+=("$(dirname "$common")/.venv/bin/python")
    fi
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ]; then
            PY="$candidate"
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    command -v python3 >/dev/null 2>&1 || { echo "ERROR: no python found; set MCLS_TEST_PYTHON" >&2; exit 1; }
    PY="python3"
fi

cd "$MC"
echo "python: $PY"
if [ "$#" -eq 0 ]; then
    set -- -q
fi
exec "$PY" -m pytest tests "$@"
