from __future__ import annotations

import numpy as np
from sklearn.utils import shuffle

from .models import train_svm, svm_margin_scores


def query_by_margin(clf, X_pool: np.ndarray, k: int = 1) -> np.ndarray:
    """Return indices of the k most uncertain points in the pool.

    Uncertainty is approximated by the smallest decision margin.
    """
    margins = svm_margin_scores(clf, X_pool)
    return np.argsort(margins)[:k]


def centralized_active_learning_loop(
    X: np.ndarray,
    y,
    initial_labeled: int = 10,
    query_size: int = 1,
    rounds: int = 10,
    random_state: int = 42,
):
    """Simple centralized active learning loop (AL-NoFL baseline).

    This is intentionally lightweight. Extend it with logging or metrics
    per round if you want to reproduce detailed curves.
    """
    X_shuf, y_shuf = shuffle(X, y, random_state=random_state)
    idx_all = np.arange(len(X_shuf))

    L_idx = idx_all[:initial_labeled]
    U_idx = idx_all[initial_labeled:]

    X_L, y_L = X_shuf[L_idx], y_shuf[L_idx]
    X_U, y_U = X_shuf[U_idx], y_shuf[U_idx]

    for _ in range(rounds):
        if len(X_U) == 0:
            break

        clf = train_svm(X_L, y_L)

        q = min(query_size, len(X_U))
        query_indices = query_by_margin(clf, X_U, k=q)

        X_new, y_new = X_U[query_indices], y_U[query_indices]
        X_L = np.vstack([X_L, X_new])
        y_L = np.concatenate([y_L, y_new])

        mask = np.ones(len(X_U), dtype=bool)
        mask[query_indices] = False
        X_U, y_U = X_U[mask], y_U[mask]

    final_clf = train_svm(X_L, y_L)
    return final_clf
