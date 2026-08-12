"""Hierarchical image classification with human-in-the-loop active learning.

    split_data     split the class-foldered images into an init set and a test set
    initialize     fit the global scaler, the hierarchical SGD classifiers and the
                   per-class anomaly detectors on the init set
    active_learner run the test set through the hierarchy, asking the human only
                   when confidence or anomaly thresholds are breached
    benchmark      image-level vs patch-level vs human-in-the-loop ablation
"""

from .active_learner import SequentialActiveLearner

__all__ = ["SequentialActiveLearner"]
