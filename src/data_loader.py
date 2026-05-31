import pandas as pd
from pathlib import Path
from typing import Tuple


def load_content(path: str | Path = "data/content.csv") -> pd.DataFrame:
    """Load content metadata CSV and return a DataFrame."""
    return pd.read_csv(path)


def load_interactions(path: str | Path = "data/interactions.csv") -> pd.DataFrame:
    """Load user-item interactions CSV and return a DataFrame."""
    return pd.read_csv(path)
