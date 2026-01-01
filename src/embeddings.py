import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def embed_requirements(texts, model_name: str = "all-MiniLM-L6-v2"):
    """Frozen sentence embeddings (as in original code)."""
    device = get_device()
    model = SentenceTransformer(model_name, device=str(device))
    X = np.array(model.encode(list(texts), show_progress_bar=True))
    return X
