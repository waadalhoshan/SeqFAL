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

    rounds_list = [5, 10, 15, 20]
    num_clients = 3
    query_size = 3

    records: list[dict] = []

    for r in rounds_list:
        rep, acc = run_seqfal_experiment(
            X, y,
            num_clients=num_clients,
            query_size=query_size,
            rounds=r,
        )

        row = {
            "query_size": query_size,
            "clients": num_clients,
            "rounds": r,
            "Accu": rep.get("accuracy", np.nan),
            "macro_F1": rep.get("macro avg", {}).get("f1-score", np.nan),
            "wP": rep.get("weighted avg", {}).get("precision", np.nan),
            "wR": rep.get("weighted avg", {}).get("recall", np.nan),
            "wF1": rep.get("weighted avg", {}).get("f1-score", np.nan),
        }
        records.append(row)

    df_results = pd.DataFrame(records)
    out_path = RESULTS_DIR / "rq4_num_rounds_results.xlsx"
    df_results.to_excel(out_path, index=False)
    print(f"Saved RQ4 results to {out_path}")


if __name__ == "__main__":
    main()
