from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..data_loading import read_dataset, get_xy
from ..embeddings import EmbeddingModel
from ..federated import run_seqfal_experiment

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def main():
    df = read_dataset()
    texts, labels = get_xy(df)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    embedder = EmbeddingModel()
    X = embedder.encode(texts)

    query_sizes = [1, 2, 3, 4, 5]
    num_clients = 3
    rounds = 10

    records: list[dict] = []

    for q in query_sizes:
        rep, acc = run_seqfal_experiment(
            X, y,
            num_clients=num_clients,
            query_size=q,
            rounds=rounds,
        )

        row = {
            "query_size": q,
            "clients": num_clients,
            "rounds": rounds,
            "Accu": rep.get("accuracy", np.nan),
            "macro_F1": rep.get("macro avg", {}).get("f1-score", np.nan),
            "wP": rep.get("weighted avg", {}).get("precision", np.nan),
            "wR": rep.get("weighted avg", {}).get("recall", np.nan),
            "wF1": rep.get("weighted avg", {}).get("f1-score", np.nan),
        }
        records.append(row)

    df_results = pd.DataFrame(records)
    out_path = RESULTS_DIR / "rq2_query_size_results.xlsx"
    df_results.to_excel(out_path, index=False)
    print(f"Saved RQ2 results to {out_path}")


if __name__ == "__main__":
    main()
