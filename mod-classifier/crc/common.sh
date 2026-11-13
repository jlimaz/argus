#!/bin/bash
# Shared helpers for the mod-classifier CRC scripts. Sourced, never run:
#   source "$(dirname "$0")/common.sh"
#
# Settings (environment variables, all optional):
#   MCLS_VENV            the venv on CRC storage       (default: $HOME/venvs/mcls)
#   MCLS_PYTHON_MODULE   the Python module to load     (default: newest python/3.12*)
#
# setup_env.sh records the module it used in $MCLS_VENV/mcls_python_module, so
# every later job loads the same Python the venv was built with.

MCLS_VENV="${MCLS_VENV:-$HOME/venvs/mcls}"
MCLS_MODULE_FILE="$MCLS_VENV/mcls_python_module"

mcls_die() {
    echo "ERROR: $*" >&2
    exit 1
}

mcls_have() {
    command -v "$1" >/dev/null 2>&1
}

# Run a command with `set -u` relaxed: module init scripts and the `module`
# function often read unset variables, which would abort a `set -u` script.
mcls_relaxed() {
    local had_u=0 rc=0
    case "$-" in *u*) had_u=1 ;; esac
    set +u
    "$@" || rc=$?
    if [ "$had_u" = 1 ]; then set -u; fi
    return "$rc"
}

# Make the `module` command available. Batch jobs do not always get it.
mcls_init_modules() {
    if type module >/dev/null 2>&1; then
        return 0
    fi
    local init
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
        /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
        if [ -r "$init" ]; then
            # shellcheck disable=SC1090
            mcls_relaxed source "$init" >/dev/null 2>&1 || true
            if type module >/dev/null 2>&1; then
                return 0
            fi
        fi
    done
    return 1
}

# Print `module avail <name>` as plain lines (Environment Modules writes to stderr).
mcls_module_avail() {
    mcls_init_modules || return 1
    mcls_relaxed module avail "$1" 2>&1
}

# Newest module named python/3.12*, or empty. Handles Lmod and Tcl modules output.
mcls_find_python_module() {
    mcls_module_avail python 2>/dev/null \
        | tr -s ' \t' '\n\n' \
        | sed -e 's/([^)]*)//g' -e 's/:$//' \
        | grep -E '^python/3\.12([.-][^/ ]*)?$' \
        | sort -V | tail -n 1 || true
}

# Load the Python module the venv was built with (or the one requested) into
# the current shell; call it directly, never inside $(...), or the load is lost.
# Sets MCLS_LOADED_MODULE. Returns 1 with a message on stderr if it cannot.
mcls_try_load_python() {
    local wanted="${MCLS_PYTHON_MODULE:-}"
    MCLS_LOADED_MODULE=""
    if [ -z "$wanted" ] && [ -r "$MCLS_MODULE_FILE" ]; then
        wanted="$(cat "$MCLS_MODULE_FILE")"
    fi
    if ! mcls_init_modules; then
        echo "the 'module' command is not available on $(hostname)" >&2
        return 1
    fi
    if [ -z "$wanted" ]; then
        wanted="$(mcls_find_python_module)"
    fi
    if [ -z "$wanted" ]; then
        echo "no python/3.12 module found (try: module avail python; or set MCLS_PYTHON_MODULE)" >&2
        return 1
    fi
    if ! mcls_relaxed module load "$wanted"; then
        echo "module load $wanted failed" >&2
        return 1
    fi
    MCLS_LOADED_MODULE="$wanted"
}

# Same, but stop the script if no module can be loaded.
mcls_load_python() {
    mcls_try_load_python || mcls_die "cannot load a Python 3.12 module (see the message above)"
}

# Activate the venv, failing with a clear message if setup_env.sh was not run.
mcls_activate_venv() {
    [ -r "$MCLS_VENV/bin/activate" ] || mcls_die "no venv at $MCLS_VENV: run crc/setup_env.sh on a front end first (or set MCLS_VENV)"
    # shellcheck disable=SC1091
    source "$MCLS_VENV/bin/activate"
}

mcls_heading() {
    printf '\n===== %s =====\n' "$*"
}
