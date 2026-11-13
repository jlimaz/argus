"""Acceptance gates: nothing passes without a measured input."""

from mcls.evaluate.gates import check_gates, summary


def _by_name(gates):
    return {g["gate"]: g for g in gates}


def test_nothing_claimed_by_default():
    gates = check_gates({})
    assert len(gates) == 7
    assert all(g["pass"] is None for g in gates)
    assert "not measured" in summary(gates)


def test_gates_pass_and_fail_on_inputs():
    g = _by_name(check_gates({
        "model_macro_f1_seeds": [0.80, 0.82, 0.81],
        "best_baseline_macro_f1": 0.70,
        "ece_after": 0.06,
        "latency_ms_per_burst_cpu": 12.0,
        "leakage_ok": True,
        "generalization_reported": True,
        "cross_test_reported": False,
        "background_false_alarm": 0.01,
    }))
    assert g["beats_baseline"]["pass"] is True
    assert g["calibration"]["pass"] is False
    assert g["latency"]["pass"] is True
    assert g["leakage"]["pass"] is True
    assert g["honest_generalization"]["pass"] is False
    assert g["safe_failure"]["pass"] is None  # no agreed rate yet
    assert g["reproducible"]["pass"] is None


def test_beats_baseline_needs_three_seeds_and_margin():
    two = _by_name(check_gates({"model_macro_f1_seeds": [0.9, 0.9], "best_baseline_macro_f1": 0.5}))
    assert two["beats_baseline"]["pass"] is False
    close = _by_name(check_gates({"model_macro_f1_seeds": [0.74, 0.74, 0.74], "best_baseline_macro_f1": 0.70}))
    assert close["beats_baseline"]["pass"] is False


def test_safe_failure_with_agreed_rate():
    g = _by_name(check_gates({"background_false_alarm": 0.02, "agreed_false_alarm_rate": 0.01}))
    assert g["safe_failure"]["pass"] is False
    g = _by_name(check_gates({"background_false_alarm": 0.02}, {"agreed_false_alarm_rate": 0.05}))
    assert g["safe_failure"]["pass"] is True


def test_model_card_template_cites_run_and_manifest_and_states_no_own_hardware():
    from pathlib import Path
    import re

    text = (Path(__file__).resolve().parents[1] / "reports" / "model_card_template.md").read_text()
    for placeholder in ("{{run_dir}}", "{{manifest_sha256}}", "{{git_commit}}"):
        assert placeholder in text
    assert "No recording from the team's own radios" in text
    # a template carries no measured numbers: no decimals outside placeholders
    assert not re.findall(r"(?<![\w.{])0\.\d+", re.sub(r"\{\{.*?\}\}", "", text))
