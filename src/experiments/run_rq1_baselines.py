from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from ..data_loading import read_dataset, get_xy
from ..embeddings import EmbeddingModel
from ..models import train_svm
from ..active_learning import centralized_active_learning_loop
from ..federated import run_seqfal_experiment

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def extract_report(report: dict, model_name: str, query: int, rounds: int, clients: int) -> dict:
    """Flatten a classification_report dict into a single result row."""
    row = {
        "Model": model_name,
        "Query": query,
        "Rounds": rounds,
        "Clients": clients,
        "Accu": report.get("accuracy", np.nan),
        "macro_F1": report.get("macro avg", {}).get("f1-score", np.nan),
        "wP": report.get("weighted avg", {}).get("precision", np.nan),
        "wR": report.get("weighted avg", {}).get("recall", np.nan),
        "wF1": report.get("weighted avg", {}).get("f1-score", np.nan),
    }
    return row


def main():
    df = read_dataset()
    texts, labels = get_xy(df)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    embedder = EmbeddingModel()
    X = embedder.encode(texts)

    baseline_results: list[dict] = []

    queries = [1, 2, 3, 4, 5]
    rounds_list = [5, 10, 15, 20]
    clients_list = [2, 3, 4, 5]

    # 1) Centralized Upper-Bound
    clf = train_svm(X, y)
    y_pred = clf.predict(X)
    rep = classification_report(y, y_pred, output_dict=True)
    baseline_results.append(extract_report(rep, "Upper-Bound", query=0, rounds=0, clients=1))

    # 2) AL-NoFL (centralized active learning)
    for q in queries:
        for r in rounds_list:
            clf_al = centralized_active_learning_loop(
                X, y,
                initial_labeled=10,
                query_size=q,
                rounds=r,
            )
            y_pred = clf_al.predict(X)
            rep = classification_report(y, y_pred, output_dict=True)
            baseline_results.append(extract_report(rep, "AL-NoFL", query=q, rounds=r, clients=1))

    # 3) FL-NoAL (federated, no active querying: query_size=0)
    for c in clients_list:
        for r in rounds_list:
            rep, _ = run_seqfal_experiment(
                X, y,
                num_clients=c,
                query_size=0,
                rounds=r,
            )
            baseline_results.append(extract_report(rep, "FL-NoAL", query=0, rounds=r, clients=c))

    df_baselines = pd.DataFrame(baseline_results)
    out_path = RESULTS_DIR / "rq1_baselines.xlsx"
    df_baselines.to_excel(out_path, index=False)
    print(f"Saved RQ1 baseline results to {out_path}")


if __name__ == "__main__":
    main()
