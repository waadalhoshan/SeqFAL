import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def extract_linear_params(clf):
    if isinstance(clf, CalibratedClassifierCV):
        estimators = [cc.estimator for cc in clf.calibrated_classifiers_]
        ws, bs = [], []
        for est in estimators:
            ws.append(est.coef_.reshape(-1))
            bs.append(float(est.intercept_.reshape(-1)[0]))
        w = np.mean(np.vstack(ws), axis=0)
        b = float(np.mean(bs))
        return w, b

    w = clf.coef_.reshape(-1)
    b = float(clf.intercept_.reshape(-1)[0])
    return w, b

def fedavg_linear(local_params, weights):
    W = np.array(weights, dtype=float)
    W = W / np.sum(W)
    ws = np.vstack([p[0] for p in local_params])
    bs = np.array([p[1] for p in local_params], dtype=float)
    w_global = np.sum(ws * W[:, None], axis=0)
    b_global = float(np.sum(bs * W))
    return w_global, b_global

def linear_predict_from_params(X, w, b):
    scores = X @ w + b
    return (scores >= 0).astype(int)
