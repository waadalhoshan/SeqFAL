import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any

def load_dataset_csv(path: str) -> pd.DataFrame:
    """Load dataset.csv (local). Required cols:
    DatasetName, ProjectID, RequirementText, Label.
    """
    df = pd.read_csv(path)
    required = {"DatasetName", "ProjectID", "RequirementText", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv missing columns: {sorted(missing)}")
    df["Label"] = df["Label"].astype(str).str.strip()
    return df

def encode_labels(labels) -> Tuple[Any, LabelEncoder, Dict[str, int]]:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    mapping = {cls: int(val) for cls, val in zip(le.classes_, le.transform(le.classes_))}
    return y, le, mapping
