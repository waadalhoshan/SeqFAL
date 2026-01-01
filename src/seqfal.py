import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

from .utils import make_rng
from .partition import dirichlet_label_skew_partition, summarize_client_pools
from .logging_utils import log_round_state, has_two_classes, enforce_two_classes
from .models import make_classifier
from .query import query_by_random, query_by_margin, query_by_least_confidence
from .fedavg import extract_linear_params, fedavg_linear, linear_predict_from_params

def run_seqfal(
    X_train, y_train, X_test, y_test,
    K, alpha, rounds, q,
    clf_name, query_strategy,
    seed,
    init_per_client=2,
    enforce_max_tries=5,
    calib_cv=2
):
    rng_global = make_rng(seed)

    clients_indices = dirichlet_label_skew_partition(y_train, K=K, alpha=alpha, seed=seed)
    pool_dist_df = summarize_client_pools(y_train, clients_indices, seed, K, alpha)

    L_sets, U_sets = [], []
    round_state_rows = []

    for cid, idx in enumerate(clients_indices):
        perm = idx.copy()
        rng_global.shuffle(perm)

        L = perm[:min(init_per_client, len(perm))]
        U = perm[min(init_per_client, len(perm)):]

        L = np.asarray(L, dtype=int)
        U = np.asarray(U, dtype=int)

        L_sets.append(L)
        U_sets.append(U)

        round_state_rows.append(
            log_round_state(y_train, seed, K, alpha, round_id=0, client_id=cid, L_idx=L, U_idx=U, forced_random_sampling=0)
        )

    w_global = None
    b_global = 0.0

    for r in range(1, rounds + 1):
        local_params = []
        local_weights = []

        for cid in range(K):
            L_idx = L_sets[cid]
            U_idx = U_sets[cid]

            original_L_idx = L_idx.copy()
            forced_sampling = not has_two_classes(y_train, original_L_idx)

            rng_local = make_rng(seed * 10_000 + r * 100 + cid)

            if forced_sampling:
                L_new, U_new = enforce_two_classes(
                    y_train, L_idx, U_idx, rng=rng_local, max_tries=enforce_max_tries
                )
                L_sets[cid] = L_new
                U_sets[cid] = U_new
                L_idx, U_idx = L_new, U_new

            if not has_two_classes(y_train, L_idx):
                if w_global is not None:
                    local_params.append((w_global, b_global))
                    local_weights.append(len(L_idx))
                    round_state_rows.append(
                        log_round_state(y_train, seed, K, alpha, round_id=r, client_id=cid, L_idx=L_idx, U_idx=U_idx, forced_random_sampling=forced_sampling)
                    )
                    continue
                else:
                    round_state_rows.append(
                        log_round_state(y_train, seed, K, alpha, round_id=r, client_id=cid, L_idx=L_idx, U_idx=U_idx, forced_random_sampling=forced_sampling)
                    )
                    continue

            need_proba = (query_strategy == "lc")
            clf = make_classifier(clf_name, seed + r * 100 + cid, need_proba=need_proba, calib_cv=calib_cv)

            if need_proba and isinstance(clf, CalibratedClassifierCV):
                yL = y_train[L_idx]
                _, counts = np.unique(yL, return_counts=True)
                if np.min(counts) < calib_cv:
                    clf = make_classifier(clf_name, seed + r * 100 + cid, need_proba=False, calib_cv=calib_cv)

            clf.fit(X_train[L_idx], y_train[L_idx])

            if len(U_idx) > 0 and q > 0:
                if query_strategy == "random":
                    chosen = query_by_random(U_idx, q, rng_local)
                elif query_strategy == "margin":
                    chosen = query_by_margin(clf, X_train, U_idx, q)
                elif query_strategy == "lc":
                    if hasattr(clf, "predict_proba"):
                        chosen = query_by_least_confidence(clf, X_train, U_idx, q)
                    else:
                        chosen = query_by_random(U_idx, q, rng_local)
                else:
                    raise ValueError(f"Unknown query_strategy: {query_strategy}")

                chosen = np.asarray(chosen, dtype=int)
                L_idx = np.asarray(L_idx, dtype=int)
                U_idx = np.asarray(U_idx, dtype=int)

                chosen_set = set(chosen.tolist())
                new_L = np.concatenate([L_idx, chosen]).astype(int) if chosen.size > 0 else L_idx.astype(int)
                new_U = np.asarray([int(i) for i in U_idx if int(i) not in chosen_set], dtype=int)

                L_sets[cid] = new_L
                U_sets[cid] = new_U
                L_idx, U_idx = new_L, new_U

                clf.fit(X_train[L_idx], y_train[L_idx])

            w, b = extract_linear_params(clf)
            local_params.append((w, b))
            local_weights.append(len(L_idx))

            round_state_rows.append(
                log_round_state(y_train, seed, K, alpha, round_id=r, client_id=cid, L_idx=L_idx, U_idx=U_idx, forced_random_sampling=forced_sampling)
            )

        if len(local_params) > 0:
            w_global, b_global = fedavg_linear(local_params, local_weights)

    y_pred = linear_predict_from_params(X_test, w_global, b_global)
    acc = accuracy_score(y_test, y_pred)
    p = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    r_ = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    result_row = {
        "seed": seed,
        "K": K,
        "alpha": alpha,
        "rounds": rounds,
        "q": q,
        "init_per_client": init_per_client,
        "classifier": clf_name,
        "query": query_strategy,
        "accuracy": acc,
        "f1": f1,
        "p": p,
        "r": r_,
    }

    round_state_df = pd.DataFrame(round_state_rows)
    return result_row, pool_dist_df, round_state_df
