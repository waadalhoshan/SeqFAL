# SeqFAL: Sequential Federated Active Learning for Security Requirements

This repository provides the **replication package** for the paper:

**SeqFAL: Sequential Federated Active Learning for Security Requirements**

The package enables full reproduction of all experimental results reported in the paper under varying federation sizes, query strategies, classifiers, communication rounds, and random seeds, while preserving data privacy.

---

## Overview

SeqFAL is a **Federated Active Learning (FAL)** framework for security requirements classification in distributed organizational settings.  
Each client performs **local uncertainty-based active learning**, while **only model parameters** are shared with a central coordinator for aggregation. Raw requirement text never leaves the client.

---

## Repository Structure

```
SeqFAL-Replication/
├── README.md
├── requirements.txt
├── dataset/
│   ├── dataset.csv
│   └── dataset_schema.md
├── src/
│   ├── v2_seqfal.py        # complete SeqFAL implementation
│   └── run_experiments.py # experiment entry point
└── outputs/
    ├── results.csv
    └── round_log.csv
```

---

## Dataset

The experiments expect a CSV file located at:

```
dataset/dataset.csv
```

**Required columns:**

| Column | Description |
|------|-------------|
| `Requirement` | Natural-language requirement text |
| `Label` | Binary class (Security / Non-Security) |
| `ProjectID` | Project identifier |
| `DatasetSource` | Dataset origin |

A strict **70/30 held-out test split** is applied per random seed and is never queried during active learning.

---

## Experimental Setup

- **Sentence embeddings:** Frozen MiniLM (SentenceTransformers)
- **Classifiers:** LinearSVC, SGD-LR, SGD-SVM
- **Query strategies:** Margin (proposed), Least-Confidence, Random
- **Client partitioning:** Dirichlet label-skew (non-IID)
- **Federated aggregation:** FedAvg over linear model parameters
- **Random seeds:** 10 runs (42–51)

---

## Baselines

All baselines reuse the same embeddings and classifiers:

- **Upper-Bound:** Centralized training
- **AL-NoFL:** Active learning without federated aggregation
- **FL-NoAL:** Federated learning without active querying

---

## Running the Experiments

```bash
pip install -r requirements.txt
python src/run_experiments.py
```

Outputs are saved to the `outputs/` directory and can be directly used for statistical analysis and figure generation.

---

## Reproducibility Statement

This replication package:
- Uses the **exact SeqFAL implementation** described in the paper
- Introduces **no additional models or heuristics**
- Preserves data locality and privacy throughout training

---

## Citation

If you use this package, please cite the paper: TO BE UPDATED LATER
