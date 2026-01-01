import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ALPHA, TEST_SIZE, SEEDS, K_LIST, Q_LIST, ROUNDS_LIST, CLASSIFIERS, QUERIES,
    INIT_PER_CLIENT, EMBED_MODEL_NAME, DEFAULT_OUTPUT_DIR, CALIB_CV
)
from .data import load_dataset_csv, encode_labels
from .embeddings import embed_requirements
from .seqfal import run_seqfal

def parse_args():
    p = argparse.ArgumentParser(description="Run SeqFAL experiment grid (modular version of v2_seqfal.py).")
    p.add_argument("--dataset_csv", default=os.path.join("dataset", "dataset.csv"))
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--test_size", type=float, default=TEST_SIZE)
    p.add_argument("--embed_model", default=EMBED_MODEL_NAME)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset_csv(args.dataset_csv)
    requirements = df["RequirementText"].tolist()
    labels = df["Label"].tolist()

    y, le, mapping = encode_labels(labels)
    print("Label mapping:", mapping)

    X = embed_requirements(requirements, model_name=args.embed_model)

    all_results = []
    all_pool_dists = []
    all_round_logs = []

    grid_size = len(SEEDS) * len(K_LIST) * len(Q_LIST) * len(ROUNDS_LIST) * len(CLASSIFIERS) * len(QUERIES)
    print("Grid size:", grid_size)

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=seed
        )

        for K in K_LIST:
            for q in Q_LIST:
                for rounds in ROUNDS_LIST:
                    for clf_name in CLASSIFIERS:
                        for query in QUERIES:
                            exp_no = len(all_results) + 1
                            print()
                            print(f"Experiment no. {exp_no}>> K={K}, q={q}, rounds={rounds}, clf={clf_name}, query={query}, seed={seed}")

                            row, pool_df, round_df = run_seqfal(
                                X_train=X_tr, y_train=y_tr,
                                X_test=X_te, y_test=y_te,
                                K=K, alpha=args.alpha, rounds=rounds, q=q,
                                clf_name=clf_name, query_strategy=query,
                                seed=seed,
                                init_per_client=INIT_PER_CLIENT,
                                calib_cv=CALIB_CV
                            )

                            all_results.append(row)
                            all_pool_dists.append(pool_df)
                            all_round_logs.append(round_df)

    results_df = pd.DataFrame(all_results)
    pool_dist_df = pd.concat(all_pool_dists, ignore_index=True) if all_pool_dists else pd.DataFrame()
    round_log_df = pd.concat(all_round_logs, ignore_index=True) if all_round_logs else pd.DataFrame()

    print("Done.")
    print("results_df:", results_df.shape)
    print("pool_dist_df:", pool_dist_df.shape)
    print("round_log_df:", round_log_df.shape)

    group_cols = ["K", "alpha", "rounds", "q", "init_per_client", "classifier", "query"]
    summary = (
        results_df
        .groupby(group_cols)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            p_mean=("p", "mean"),
            p_std=("p", "std"),
            r_mean=("r", "mean"),
            r_std=("r", "std"),
            n_runs=("accuracy", "count"),
        )
        .reset_index()
    )

    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=True)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=True)
    pool_dist_df.to_csv(os.path.join(args.output_dir, "pool_dist.csv"), index=True)
    round_log_df.to_csv(os.path.join(args.output_dir, "round_log.csv"), index=True)

    print("Saved outputs to:", args.output_dir)

if __name__ == "__main__":
    main()
