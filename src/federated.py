from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle

from .models import train_svm
from .active_learning import query_by_margin
from .config import seqfal_config


def federated_avg(models, client_sizes):
    """Average linear SVM parameters (coef_ and intercept_).

    This mirrors a FedAvg-style aggregation for linear SVMs.
    """
    ws = [m.coef_.flatten() for m in models]
    bs = [m.intercept_[0] for m in models]

    total = float(sum(client_sizes))
    weights = [s / total for s in client_sizes]

    w_global = np.sum([a * w for a, w in zip(weights, ws)], axis=0)
    b_global = np.sum([a * b for a, b in zip(weights, bs)], axis=0)
    return w_global, b_global


def global_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    scores = X @ w + b
    # Binary 0/1 predictions; adapt if you have more than two classes
    return (scores >= 0).astype(int)


def run_seqfal_experiment(
    X: np.ndarray,
    y,
    num_clients: int,
    query_size: int,
    rounds: int,
    initial_labels_per_client: int | None = None,
    random_state: int | None = None,
):
    """Core SeqFAL experiment loop (federated active learning).

    - Splits the dataset into `num_clients` partitions.
    - Each client starts with `initial_labels_per_client` seed labels.
    - At each round:
        * Each client trains locally.
        * Performs margin-based querying on its own pool.
        * The server aggregates client SVMs via `federated_avg`.
    - At the end, the aggregated global model is evaluated on the full data.
    """
    if initial_labels_per_client is None:
        initial_labels_per_client = seqfal_config.initial_labels_per_client
    if random_state is None:
        random_state = seqfal_config.seed

    X_shuf, y_shuf = shuffle(X, y, random_state=random_state)
    indices = np.arange(len(X_shuf))

    client_indices = np.array_split(indices, num_clients)
    client_data = [(X_shuf[idx], np.array(y_shuf)[idx]) for idx in client_indices]

    clients = []
    for Xc, yc in client_data:
        idx = np.arange(len(Xc))
        np.random.shuffle(idx)

        L_idx = idx[:initial_labels_per_client]
        U_idx = idx[initial_labels_per_client:]

        X_L, y_L = Xc[L_idx], yc[L_idx]
        X_U, y_U = Xc[U_idx], yc[U_idx]

        clients.append({"L": (X_L, y_L), "U": (X_U, y_U)})

    global_w, global_b = None, None

    for _ in range(rounds):
        local_models = []
        client_sizes = []

        for c in clients:
            X_L, y_L = c["L"]
            X_U, y_U = c["U"]

            if len(X_L) == 0:
                continue

            clf = train_svm(X_L, y_L, kernel=seqfal_config.svm_kernel)
            local_models.append(clf)
            client_sizes.append(len(y_L))

            if len(X_U) > 0 and query_size > 0:
                q = min(query_size, len(X_U))
                idx_q = query_by_margin(clf, X_U, k=q)

                X_new, y_new = X_U[idx_q], y_U[idx_q]
                X_L = np.vstack([X_L, X_new])
                y_L = np.concatenate([y_L, y_new])

                mask = np.ones(len(X_U), dtype=bool)
                mask[idx_q] = False
                X_U, y_U = X_U[mask], y_U[mask]

                c["L"] = (X_L, y_L)
                c["U"] = (X_U, y_U)

        if not local_models:
            continue

        global_w, global_b = federated_avg(local_models, client_sizes)

    if global_w is None or global_b is None:
        raise RuntimeError("Federated training did not produce a global model.")

    y_pred = global_predict(X_shuf, global_w, global_b)
    report = classification_report(y_shuf, y_pred, output_dict=True)
    accuracy = accuracy_score(y_shuf, y_pred)
    return report, accuracy
