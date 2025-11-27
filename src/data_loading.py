from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import data_config


def read_dataset() -> pd.DataFrame:
    """Load the requirements dataset.

    Priority:
    1. If `data_config.local_excel_path` exists, load from there.
    2. Else, if `data_config.google_drive_url` is set, load from Google Drive.
    3. Otherwise, raise a helpful error.
    """
    path: Path = data_config.local_excel_path

    if path.exists():
        return pd.read_excel(path)

    if data_config.google_drive_url:
        url = data_config.google_drive_url
        # Expect a Google Sheets URL; extract file id and build a direct download link
        file_id = url.split("/")[-2]
        direct_url = f"https://drive.google.com/uc?id={file_id}"
        return pd.read_excel(direct_url)

    raise FileNotFoundError(
        "Could not find dataset. Please either:\n"
        f"  - place an Excel file at: {path}\n"
        "  - or set `data_config.google_drive_url` in config.py"
    )


def get_xy(
    df: pd.DataFrame,
    text_col: str = "RequirementText",
    label_col: str = "Label",
):
    """Extract X (texts) and y (labels) from the dataframe.

    Adjust `text_col` and `label_col` to match your dataset.
    """
    X = df[text_col].astype(str).tolist()
    y = df[label_col].tolist()
    return X, y
