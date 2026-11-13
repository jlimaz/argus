"""Baselines on hand-made window features (PLAN.md section 5.1).

    from mcls.baselines import train, load
    train(config, run_dir)            # fit, select on val, write the run folder
    clf = load(run_dir)               # a contracts.Classifier: clf.classify(windows) -> probabilities

Models: majority, softmax_regression (numpy), knn (scikit-learn), lightgbm.
scikit-learn and lightgbm are imported only when those models are fitted.
"""

from mcls.baselines.classifier import FeatureClassifier, load
from mcls.baselines.run import train

__all__ = ["FeatureClassifier", "load", "train"]
