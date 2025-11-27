from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def train_svm(X_train: np.ndarray, y_train, kernel: str = "linear") -> SVC:
    """Train a linear SVM classifier.

    This corresponds to the classifier used in the Real Experiments.
    """
    clf = SVC(kernel=kernel, probability=False)
    clf.fit(X_train, y_train)
    return clf


def svm_margin_scores(clf: SVC, X: np.ndarray) -> np.ndarray:
    """Compute margin scores for margin-based active learning.

    Lower absolute scores indicate higher uncertainty.
    """
    scores = clf.decision_function(X)
    return np.abs(scores)
