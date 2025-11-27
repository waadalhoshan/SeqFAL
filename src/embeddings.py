from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import embedding_config


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer.

    In the SeqFAL experiments, 'all-MiniLM-L6-v2' is used by default.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or embedding_config.model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a 2D embedding array."""
        embeddings = self.model.encode(texts)
        return np.asarray(embeddings)
