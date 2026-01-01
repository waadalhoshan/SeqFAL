import numpy as np
import pandas as pd
from .utils import make_rng

def dirichlet_label_skew_partition(y_train, K: int, alpha: float, seed: int):
    rng = make_rng(seed)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    class_indices = {c: np.where(y_train == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(K)]

    for c in classes:
        idx_c = class_indices[c]
        if len(idx_c) == 0:
            continue

        proportions = rng.dirichlet(alpha * np.ones(K))
        counts = (proportions * len(idx_c)).astype(int)

        diff = len(idx_c) - int(np.sum(counts))
        if diff > 0:
            for i in rng.choice(np.arange(K), size=diff, replace=True):
                counts[i] += 1
        elif diff < 0:
            for i in rng.choice(np.where(counts > 0)[0], size=-diff, replace=True):
                counts[i] -= 1

        start = 0
        for k in range(K):
            take = int(counts[k])
            if take > 0:
                client_indices[k].extend(idx_c[start:start+take].tolist())
                start += take

    out = []
    for k in range(K):
        idx = np.array(client_indices[k], dtype=int)
        rng.shuffle(idx)
        out.append(idx)
    return out

def summarize_client_pools(y_train, clients_indices, seed, K, alpha):
    rows = []
    for cid, idx in enumerate(clients_indices):
        y_local = y_train[idx]
        n0 = int(np.sum(y_local == 0))
        n1 = int(np.sum(y_local == 1))
        rows.append({
            "seed": seed,
            "K": K,
            "alpha": alpha,
            "client_id": cid,
            "pool_size": int(len(idx)),
            "pool_n0": n0,
            "pool_n1": n1,
            "pool_p0": n0 / max(1, len(idx)),
            "pool_p1": n1 / max(1, len(idx)),
        })
    return pd.DataFrame(rows)
