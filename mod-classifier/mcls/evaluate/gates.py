"""Acceptance gates of PLAN 6.2, checked against measured inputs.

Nothing is claimed by default: a gate whose input is missing reports
`pass: None` ("not measured"), never True. The thresholds in DEFAULTS are the
plan's proposals, "to agree with the team, not claimed yet"; the false-alarm
rate has no agreed value, so that gate stays None until one is passed in.

Inputs (all optional; keys as below):
    model_macro_f1_seeds     list[float]  test macro-F1 of the network, one per seed
    best_baseline_macro_f1   float        best baseline test macro-F1, same splits
    generalization_reported  bool         `generalization` split (held-out drone model) has real windows
    cross_test_reported      bool         cross_test split has real windows
    generalization_macro_f1, cross_test_macro_f1   float   shown with the gate, not part of pass/fail
    ece_after                float        ECE after temperature scaling (test, else val)
    background_false_alarm   float        share of background windows called above the threshold
    agreed_false_alarm_rate  float        the team's agreed maximum (no default)
    latency_ms_per_burst_cpu float        measured on the laptop CPU
    leakage_ok               bool         leakage audit found nothing
    reproduced               bool         a fresh clone reproduced the numbers
"""

from __future__ import annotations

from typing import Any

DEFAULTS = {
    "beats_baseline_margin": 0.05,
    "min_seeds": 3,
    "ece_max": 0.05,
    "latency_ms_max": 50.0,
}


def _gate(name: str, criterion: str, value: Any, passed: bool | None, note: str = "") -> dict[str, Any]:
    return {"gate": name, "criterion": criterion, "value": value, "pass": passed, "note": note}


def check_gates(inputs: dict[str, Any], thresholds: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    t = {**DEFAULTS, **(thresholds or {})}
    g = inputs.get
    out = []

    seeds, base = g("model_macro_f1_seeds"), g("best_baseline_macro_f1")
    margin = t["beats_baseline_margin"]
    if seeds is None or base is None:
        out.append(_gate("beats_baseline", f"mean macro-F1 over >= {t['min_seeds']} seeds >= best baseline + {margin}",
                         None, None, "not measured"))
    else:
        mean = sum(seeds) / len(seeds) if seeds else None
        enough = len(seeds) >= t["min_seeds"]
        out.append(_gate("beats_baseline", f"mean macro-F1 over >= {t['min_seeds']} seeds >= best baseline + {margin}",
                         {"mean": mean, "min": min(seeds) if seeds else None, "seeds": len(seeds), "baseline": base},
                         bool(enough and mean is not None and mean >= base + margin),
                         "" if enough else f"only {len(seeds)} seeds"))

    gen, cross = g("generalization_reported"), g("cross_test_reported")
    out.append(_gate("honest_generalization", "held-out drone model and cross_test reported next to the headline",
                     {"generalization": gen, "cross_test": cross,
                      "generalization_macro_f1": g("generalization_macro_f1"),
                      "cross_test_macro_f1": g("cross_test_macro_f1")},
                     None if gen is None or cross is None else bool(gen and cross),
                     "pass means reported, whatever the score" if gen and cross else ""))

    e = g("ece_after")
    out.append(_gate("calibration", f"ECE after temperature scaling <= {t['ece_max']}", e,
                     None if e is None else bool(e <= t["ece_max"])))

    fa, agreed = g("background_false_alarm"), g("agreed_false_alarm_rate", t.get("agreed_false_alarm_rate"))
    out.append(_gate("safe_failure", "background false-alarm rate at the threshold <= agreed rate",
                     {"rate": fa, "agreed": agreed},
                     None if fa is None or agreed is None else bool(fa <= agreed),
                     "no agreed false-alarm rate yet" if agreed is None else ""))

    lat = g("latency_ms_per_burst_cpu")
    out.append(_gate("latency", f"< {t['latency_ms_max']} ms per burst on the laptop CPU", lat,
                     None if lat is None else bool(lat < t["latency_ms_max"])))

    leak = g("leakage_ok")
    out.append(_gate("leakage", "zero shared recordings or window hashes across splits", leak,
                     None if leak is None else bool(leak)))

    rep = g("reproduced")
    out.append(_gate("reproducible", "a fresh clone plus documented commands reproduces the numbers", rep,
                     None if rep is None else bool(rep)))
    return out


def summary(gates: list[dict[str, Any]]) -> str:
    word = {True: "PASS", False: "FAIL", None: "not measured"}
    return "\n".join(f"{x['gate']:<22} {word[x['pass']]:<13} {x['criterion']}" for x in gates)
