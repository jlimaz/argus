"""Classify the bursts `pipeline.features.detect_bursts` finds (PLAN sections 5.5, 5.7, 11).

    from mcls.infer import classify_bursts, load_classifier, load_calibration, load_priors

    classifier = load_classifier("runs/cnn1d_s0", calibration="runs/cnn1d_s0/calibration.json")
    calibration = load_calibration("runs/cnn1d_s0/calibration.json", classifier.classes)   # refuses another model's
    results = classify_bursts(iq, sample_rate_hz, center_frequency_hz, bursts, classifier,
                              calibration=calibration, priors=load_priors())

Per burst: cut 4096-sample windows at 20 MS/s with Engineer 4's
`windows_from_iq` (the preprocess code path), get each window's raw logits,
divide once by calibration.json's temperature, average them over the burst,
softmax, multiply by the band prior and renormalise, then call the burst
`unknown` when the top probability is under the threshold. `unknown` is what
the Judge ranks white (SRS NFR-4).

Nothing here imports torch: the model code is loaded only by `load_classifier`.
"""

from mcls.infer.calibration import Calibration, load_calibration
from mcls.infer.core import UNKNOWN, BurstClass, aggregate, burst_windows, classify_bursts, window_logits
from mcls.infer.loading import load_classifier
from mcls.infer.priors import DEFAULT_PRIORS, BandPriors, load_priors

__all__ = [
    "UNKNOWN",
    "BurstClass",
    "Calibration",
    "BandPriors",
    "DEFAULT_PRIORS",
    "aggregate",
    "burst_windows",
    "classify_bursts",
    "load_calibration",
    "load_classifier",
    "load_priors",
    "window_logits",
]
