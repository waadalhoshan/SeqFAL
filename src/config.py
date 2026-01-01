"""Experiment configuration (matches v2_seqfal.py defaults)."""

ALPHA = 0.3
TEST_SIZE = 0.3

SEEDS = list(range(42, 52))  # 42..51 (10 seeds)
K_LIST = [2, 3, 4, 5, 6, 7, 8, 9]
Q_LIST = [1, 2, 3, 4, 5]
ROUNDS_LIST = [5, 10, 15, 20]
CLASSIFIERS = ["linearsvc", "sgd_hinge", "logreg"]
QUERIES = ["random", "lc", "margin"]

INIT_PER_CLIENT = 2
CALIB_CV = 2  # keep 2 for speed (as in original)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_OUTPUT_DIR = "outputs"
