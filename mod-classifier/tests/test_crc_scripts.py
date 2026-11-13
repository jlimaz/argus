"""Static checks on the CRC and helper shell scripts (nothing here runs on CRC)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(
    p
    for folder in ("crc", "scripts")
    for pattern in ("*.sh", "*.job")
    for p in (ROOT / folder).glob(pattern)
)
BASH = shutil.which("bash")


def _id(path):
    return f"{path.parent.name}/{path.name}"


def test_scripts_found():
    names = {p.name for p in SCRIPTS}
    for expected in (
        "common.sh",
        "setup_env.sh",
        "check_env.sh",
        "check_node.sh",
        "check_node.job",
        "prepare_array.job",
        "run_tests.sh",
    ):
        assert expected in names


@pytest.mark.skipif(BASH is None, reason="bash not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=_id)
def test_bash_syntax(script):
    result = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=_id)
def test_shebang_and_strict_mode(script):
    text = script.read_text()
    assert text.startswith("#!/bin/bash\n")
    if script.name != "common.sh":  # sourced, so it inherits the caller's mode
        assert "set -euo pipefail" in text


def test_array_job_caps_concurrent_downloads():
    text = (ROOT / "crc" / "prepare_array.job").read_text()
    assert "#$ -tc 8" in text
    assert "${MCLS_NODE_TMP:-/tmp/$USER}" in text and "/mcls_raw/" in text
    assert "--only" in text and "--connections" in text


def test_training_job_trains_then_calibrates_then_evaluates():
    # Written on request (Nov 12): one job per config, weights saved every epoch, plot emailed.
    text = (ROOT / "crc" / "train.job").read_text()
    order = [text.index(s) for s in ("python -m mcls.train --config", "python -m mcls.evaluate.calibrate --run",
                                     "python -m mcls.evaluate --run")]
    assert order == sorted(order)
    assert "--manifest" in text  # evaluate refuses to report test without it
    assert "--resume" in text and "last.pt" in text
    assert "MCLS_EMAIL_EVERY_EPOCHS" in text and "progress.png" in text and "train_log.txt" in text


def test_wave_sends_one_start_and_one_end_email():
    text = (ROOT / "crc" / "submit_wave.sh").read_text()
    assert "-v MCLS_MAIL_TO=" in text  # training jobs stay quiet
    assert "-m b" in text and "crc/wave_summary.job" in text
    assert "mailx" in (ROOT / "crc" / "wave_summary.job").read_text()
