from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def make_classifier(name: str, seed: int, need_proba: bool, calib_cv: int = 2):
    name = name.lower()

    if name == "logreg":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            max_iter=1000,
            tol=1e-3,
            random_state=seed
        )

    if name == "linearsvc":
        base = LinearSVC(random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=calib_cv) if need_proba else base

    if name == "sgd_hinge":
        base = SGDClassifier(loss="hinge", random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=calib_cv) if need_proba else base

    raise ValueError(f"Unknown classifier: {name}")
