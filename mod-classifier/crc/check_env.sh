#!/bin/bash
# Report what a CRC front end offers, without changing anything.
#
#   bash crc/check_env.sh 2>&1 | tee check_env.txt
#
# Prints: quota (home and any /groups you belong to), groups, free space of
# $HOME, `module avail` for python/cuda/pytorch, `free_gpus.sh @crc_gpu`, and a
# 10 MB Zenodo download speed test on one connection and on four (to /dev/null).
# A missing command is reported and skipped; the script keeps going.
#
# Settings (environment variables, all optional):
#   MCLS_SPEEDTEST_URL   a large Zenodo file that honours byte ranges
#                        (default: a 2 GB Tampere file, record 4264467)
#   MCLS_SKIP_SPEEDTEST=1   skip the download test
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

# Run a command; report a missing command or a failure instead of stopping.
try() {
    if ! mcls_have "$1"; then
        echo "(command not found: $1)"
        return 0
    fi
    "$@" 2>&1 || echo "(exit status $? from: $*)"
}

mcls_heading "host"
echo "host: $(hostname)  user: $USER  date: $(date)"

# quota with human-readable sizes when supported, plain otherwise.
show_quota() {
    if ! mcls_have quota; then
        echo "(command not found: quota)"
        return 0
    fi
    quota -s "$@" 2>/dev/null || quota "$@" 2>&1 || echo "(quota $* failed)"
}

mcls_heading "groups"
try groups

mcls_heading "quota (all)"
show_quota
mcls_heading "quota for home ($HOME)"
show_quota "$HOME"

mcls_heading "/groups you belong to"
found=0
for g in $(id -Gn 2>/dev/null || true); do
    if [ -d "/groups/$g" ]; then
        found=1
        echo "--- /groups/$g"
        show_quota "/groups/$g"
        try df -h "/groups/$g"
    fi
done
if [ "$found" = 0 ]; then
    echo "none of your groups has a /groups/<name> directory"
    if [ -d /groups ]; then
        echo "readable entries under /groups (first 20):"
        find /groups -mindepth 1 -maxdepth 1 -readable 2>/dev/null | head -n 20 || true
    else
        echo "/groups does not exist on this host"
    fi
fi

mcls_heading "free space of \$HOME"
try df -h "$HOME"

mcls_heading "front-end /tmp (compute-node /tmp is checked by check_node.job)"
try df -h /tmp

for name in python cuda pytorch; do
    mcls_heading "module avail $name"
    if mcls_init_modules; then
        mcls_relaxed module avail "$name" 2>&1 || echo "(module avail $name failed)"
    else
        echo "(the 'module' command is not available)"
    fi
done
mcls_heading "python/3.12 module setup_env.sh would pick"
picked="$(mcls_find_python_module)"
echo "${picked:-none found}"

mcls_heading "free_gpus.sh @crc_gpu"
try free_gpus.sh @crc_gpu

mcls_heading "Zenodo speed test (10 MB, to /dev/null)"
URL="${MCLS_SPEEDTEST_URL:-https://zenodo.org/records/4264467/files/DJI_phantom_4_2G.bin?download=1}"
SIZE=10485760
if [ "${MCLS_SKIP_SPEEDTEST:-0}" = "1" ]; then
    echo "skipped (MCLS_SKIP_SPEEDTEST=1)"
elif ! mcls_have curl; then
    echo "(command not found: curl)"
else
    echo "url: $URL"
    # --max-filesize aborts if the server ignores the range and sends the whole file.
    fetch_range() {
        curl -sS -L -A argus-mcls-check/0.1 -r "$1-$2" --max-filesize $((SIZE + 65536)) \
            --max-time 120 -o /dev/null \
            -w '%{http_code} %{size_download} %{time_total}\n' "$URL" || echo "failed 0 0"
    }
    echo "one connection:"
    line="$(fetch_range 0 $((SIZE - 1)))"
    read -r code bytes secs <<<"$line"
    echo "  http $code, $bytes bytes in $secs s"
    awk -v b="$bytes" -v s="$secs" 'BEGIN { if (s > 0) printf "  RESULT 1 connection: %.2f MB/s\n", b / s / 1e6 }'

    echo "four connections (2.5 MB each, in parallel):"
    part=$((SIZE / 4))
    tmpdir="$(mktemp -d)"
    for i in 0 1 2 3; do
        fetch_range $((i * part)) $(((i + 1) * part - 1)) > "$tmpdir/$i" &
    done
    wait
    total=0
    slowest=0
    for i in 0 1 2 3; do
        read -r code bytes secs < "$tmpdir/$i"
        echo "  part $i: http $code, $bytes bytes in $secs s"
        case "$bytes" in '' | *[!0-9]*) bytes=0 ;; esac
        total=$((total + bytes))
        slowest="$(awk -v a="$slowest" -v b="$secs" 'BEGIN { print (b > a) ? b : a }')"
    done
    rm -rf "$tmpdir"
    awk -v b="$total" -v s="$slowest" 'BEGIN { if (s > 0) printf "  RESULT 4 connections: %.2f MB/s (all bytes / slowest part)\n", b / s / 1e6 }'
    echo "http 206 means the range was honoured; anything else is a problem worth reporting"
fi

echo
echo "check_env.sh finished"
