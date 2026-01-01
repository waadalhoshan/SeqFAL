import numpy as np

def query_by_random(U_idx, k, rng):
    k = min(k, len(U_idx))
    if k <= 0:
        return np.array([], dtype=int)
    return rng.choice(U_idx, size=k, replace=False)

def query_by_least_confidence(clf, X, U_idx, k):
    k = min(k, len(U_idx))
    if k <= 0:
        return np.array([], dtype=int)
    proba = clf.predict_proba(X[U_idx])
    maxp = np.max(proba, axis=1)
    order = np.argsort(maxp)
    return U_idx[order[:k]]

def query_by_margin(clf, X, U_idx, k):
    k = min(k, len(U_idx))
    if k <= 0:
        return np.array([], dtype=int)
    scores = clf.decision_function(X[U_idx])
    margins = np.abs(scores)
    order = np.argsort(margins)
    return U_idx[order[:k]]
