# SeqFAL: Sequential Federated Active Learning for Security Requirements

This repository contains a code skeleton for the **Real Experiments** reported in the SeqFAL paper.
It is intended as a clean, GitHub-ready starting point rather than a full reproduction of all analyses.

The skeleton includes:

- Sentence-level embeddings using `sentence-transformers`.
- A linear SVM classifier for security requirement classification.
- Centralized active learning (AL-NoFL).
- Federated learning with simple parameter averaging (FL-NoAL).
- Federated active learning (SeqFAL).
- Scripts to run the main experiment configurations (RQ1–RQ4).

> Note: Statistical significance testing (e.g., Friedman / post-hoc tests) is intentionally **omitted**.

Please see comments and TODOs in the code for guidance on how to adapt it to your own datasets.
