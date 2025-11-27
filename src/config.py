from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    """Configuration for dataset paths and loading.

    - local_excel_path: path to a local Excel file with requirements.
    - google_drive_url: optional Google Drive URL for remote loading.
    """
    local_excel_path: Path = PROJECT_ROOT / "data" / "requirements.xlsx"
    google_drive_url: str | None = None


@dataclass
class EmbeddingConfig:
    """Configuration for the sentence embedding model."""
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class SeqFALConfig:
    """High-level hyperparameters used in the SeqFAL experiments."""
    seed: int = 42
    initial_labels_per_client: int = 2
    svm_kernel: str = "linear"


data_config = DataConfig()
embedding_config = EmbeddingConfig()
seqfal_config = SeqFALConfig()
