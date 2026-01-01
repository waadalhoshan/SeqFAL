import numpy as np

def log_round_state(y_train, seed, K, alpha, round_id, client_id, L_idx, U_idx, forced_random_sampling):
    yL = y_train[L_idx]
    yU = y_train[U_idx]
    return {
        "seed": seed,
        "K": K,
        "alpha": alpha,
        "round": round_id,
        "client_id": client_id,
        "L_size": int(len(L_idx)),
        "L_n0": int(np.sum(yL == 0)),
        "L_n1": int(np.sum(yL == 1)),
        "U_size": int(len(U_idx)),
        "U_n0": int(np.sum(yU == 0)),
        "U_n1": int(np.sum(yU == 1)),
        "forced_random_sampling": int(forced_random_sampling),
    }

def has_two_classes(y, idx):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return False
    return np.unique(y[idx]).size >= 2

def enforce_two_classes(y, L_idx, U_idx, rng, max_tries=5):
    L = list(np.asarray(L_idx, dtype=int))
    U = list(np.asarray(U_idx, dtype=int))
    tries = 0
    while np.unique(y[np.asarray(L, dtype=int)]).size < 2 and len(U) > 0 and tries < max_tries:
        pick = int(rng.choice(U))
        U.remove(pick)
        L.append(pick)
        tries += 1
    return np.asarray(L, dtype=int), np.asarray(U, dtype=int)
